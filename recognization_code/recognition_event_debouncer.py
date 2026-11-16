"""Thread-safe, latest-event-wins delivery for portal recognition events."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DeliveryResult:
    success: bool
    retryable: bool = False


@dataclass
class _Pending:
    event: dict
    deadline: float
    generation: int
    retry_at: float | None = None
    attempts: int = 0
    sending: bool = False


class RecognitionEventDebouncer:
    """One fixed debounce window and latest event per authenticated tenant.

    ``deliver`` is called outside the lock.  A generation number ensures an
    older in-flight delivery can never clear or retry over a newer event.
    """

    def __init__(self, deliver: Callable[[dict], DeliveryResult], interval_seconds: float = 120,
                 max_retries: int = 3, clock: Callable[[], float] = time.monotonic,
                 max_pending_users: int = 10_000):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._deliver = deliver
        self._interval = interval_seconds
        self._max_retries = max_retries
        self._clock = clock
        self._max_pending_users = max_pending_users
        self._pending: dict[str, _Pending] = {}
        self._generation = 0
        self._condition = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.log = logging.getLogger("debounce")

    @staticmethod
    def _fields(user_id: str, event: dict) -> dict:
        return {"userId": user_id, "cameraId": event.get("camera_id"), "personId": event.get("person_id")}

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, daemon=True, name="recognition-event-debounce")
            self._thread.start()

    def submit(self, user_id: str, event: dict) -> None:
        """Accept an event without blocking the caller or resetting its window."""
        if not user_id:
            self.log.warning("[DEBOUNCE] Event discarded: missing portal user ID | cameraId=%s", event.get("camera_id"))
            return
        now = self._clock()
        with self._condition:
            current = self._pending.get(user_id)
            self._generation += 1
            generation = self._generation
            if current is None or current.sending:
                # An in-flight send belongs to an expired window. A concurrent
                # event begins the next independent fixed window.
                if current is None and len(self._pending) >= self._max_pending_users:
                    oldest = min(self._pending, key=lambda key: self._pending[key].deadline)
                    dropped = self._pending.pop(oldest)
                    self.log.warning("[DEBOUNCE] Pending state evicted at capacity | userId=%s | cameraId=%s", oldest, dropped.event.get("camera_id"))
                self._pending[user_id] = _Pending(event=dict(event), deadline=now + self._interval, generation=generation)
                self.log.info("[DEBOUNCE] First recognition received; timer started | %s | windowSeconds=%s", self._context(self._fields(user_id, event)), self._interval)
            else:
                # A retry is already for an expired original window. Retain its
                # retry deadline; a new event replaces its payload but never
                # makes a new window because delivery has not succeeded yet.
                remaining = max(0.0, current.deadline - now)
                current.event = dict(event)
                current.generation = generation
                self.log.info("[DEBOUNCE] Pending event replaced by newer recognition | %s | secondsRemaining=%.3f | retryAttempt=%d", self._context(self._fields(user_id, event)), remaining, current.attempts)
            self._condition.notify_all()

    @staticmethod
    def _context(fields: dict) -> str:
        return " | ".join(f"{key}={value}" for key, value in fields.items())

    def _due(self, state: _Pending, now: float) -> bool:
        return not state.sending and (state.retry_at is not None and state.retry_at <= now or state.retry_at is None and state.deadline <= now)

    def process_due(self) -> int:
        """Deliver due events once; public for deterministic tests."""
        now = self._clock()
        due: list[tuple[str, int, dict, int]] = []
        with self._condition:
            for user_id, state in self._pending.items():
                if self._due(state, now):
                    state.sending = True
                    due.append((user_id, state.generation, dict(state.event), state.attempts))
        for user_id, generation, event, attempt in due:
            self.log.info("[DEBOUNCE] User debounce window expired; sending latest recognition event | %s | retryAttempt=%d", self._context(self._fields(user_id, event)), attempt)
            try:
                result = self._deliver(event)
            except Exception as error:  # delivery functions must not kill the scheduler
                self.log.exception("[DEBOUNCE] Portal request failed unexpectedly | %s | errorType=%s", self._context(self._fields(user_id, event)), type(error).__name__)
                result = DeliveryResult(success=False, retryable=True)
            self._finish(user_id, generation, event, result)
        return len(due)

    def _finish(self, user_id: str, generation: int, event: dict, result: DeliveryResult) -> None:
        now = self._clock()
        with self._condition:
            state = self._pending.get(user_id)
            # A submit during the HTTP request installed a newer generation.
            # Never clear it or schedule an old event over it.
            if state is None or state.generation != generation:
                return
            if result.success:
                del self._pending[user_id]
                self.log.info("[DEBOUNCE] Portal request successful; pending state cleared | %s", self._context(self._fields(user_id, event)))
            elif result.retryable and state.attempts < self._max_retries:
                state.sending = False
                state.attempts += 1
                delay = min(30.0, 2.0 ** (state.attempts - 1))
                state.retry_at = now + delay
                self.log.warning("[DEBOUNCE] Portal request failed; retry scheduled | %s | retryAttempt=%d | retryDelaySeconds=%.1f", self._context(self._fields(user_id, event)), state.attempts, delay)
            else:
                del self._pending[user_id]
                self.log.error("[DEBOUNCE] Portal request failed permanently; pending state cleared | %s | retryAttempt=%d", self._context(self._fields(user_id, event)), state.attempts)
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
                now = self._clock()
                wakeups = [state.retry_at if state.retry_at is not None else state.deadline for state in self._pending.values() if not state.sending]
                timeout = max(0.0, min(wakeups) - now) if wakeups else None
                self._condition.wait(timeout)
                if self._stopped:
                    return
            self.process_due()

    def shutdown(self, flush: bool = True) -> None:
        """Stop scheduling and make one bounded best-effort final delivery per user."""
        with self._condition:
            self._stopped = True
            pending = [(user_id, dict(state.event)) for user_id, state in self._pending.items() if not state.sending]
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if flush:
            for user_id, event in pending:
                try:
                    result = self._deliver(event)
                    self.log.info("[DEBOUNCE] Shutdown delivery %s | %s", "succeeded" if result.success else "failed", self._context(self._fields(user_id, event)))
                except Exception:
                    self.log.exception("[DEBOUNCE] Shutdown delivery failed | %s", self._context(self._fields(user_id, event)))
        with self._condition:
            self._pending.clear()
