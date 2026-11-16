import unittest

from recognition_event_debouncer import DeliveryResult, RecognitionEventDebouncer


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def event(user, camera, person):
    return {"_tenant_id": user, "camera_id": camera, "person_id": person, "track_id": 1, "timestamp": "2026-11-15T00:00:00+00:00"}


class RecognitionEventDebouncerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sent = []
        self.debouncer = RecognitionEventDebouncer(lambda item: self.sent.append(item) or DeliveryResult(True), interval_seconds=120, clock=self.clock)

    def test_one_event_is_sent_at_fixed_deadline(self):
        self.debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(119.9)
        self.assertEqual(self.debouncer.process_due(), 0)
        self.clock.advance(.1)
        self.assertEqual(self.debouncer.process_due(), 1)
        self.assertEqual([item["person_id"] for item in self.sent], ["100"])

    def test_newer_events_replace_without_resetting_original_deadline(self):
        self.debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(60)
        self.debouncer.submit("1", event("1", "11", "101"))
        self.clock.advance(60)
        self.debouncer.process_due()
        self.assertEqual([(item["camera_id"], item["person_id"]) for item in self.sent], [("11", "101")])

    def test_different_users_have_independent_windows(self):
        self.debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(30)
        self.debouncer.submit("2", event("2", "20", "200"))
        self.clock.advance(90)
        self.debouncer.process_due()
        self.assertEqual([item["person_id"] for item in self.sent], ["100"])
        self.clock.advance(30)
        self.debouncer.process_due()
        self.assertEqual([item["person_id"] for item in self.sent], ["100", "200"])

    def test_new_event_during_send_starts_next_window_and_is_not_lost(self):
        holder = {}
        sent = []
        def deliver(item):
            sent.append(item)
            if item["person_id"] == "100":
                holder["debouncer"].submit("1", event("1", "11", "101"))
            return DeliveryResult(True)
        debouncer = RecognitionEventDebouncer(deliver, interval_seconds=120, clock=self.clock)
        holder["debouncer"] = debouncer
        debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(120)
        debouncer.process_due()
        self.clock.advance(120)
        debouncer.process_due()
        self.assertEqual([item["person_id"] for item in sent], ["100", "101"])

    def test_retry_preserves_latest_event_and_uses_bounded_backoff(self):
        attempts = []
        def deliver(item):
            attempts.append(item)
            return DeliveryResult(False, retryable=True) if len(attempts) == 1 else DeliveryResult(True)
        debouncer = RecognitionEventDebouncer(deliver, interval_seconds=120, clock=self.clock)
        debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(120)
        debouncer.process_due()
        debouncer.submit("1", event("1", "11", "101"))
        self.clock.advance(1)
        debouncer.process_due()
        self.assertEqual([item["person_id"] for item in attempts], ["100", "101"])

    def test_no_duplicate_delivery_while_request_is_in_flight(self):
        calls = []
        def deliver(item):
            calls.append(item)
            self.assertEqual(self.debouncer.process_due(), 0)
            return DeliveryResult(True)
        self.debouncer = RecognitionEventDebouncer(deliver, interval_seconds=120, clock=self.clock)
        self.debouncer.submit("1", event("1", "10", "100"))
        self.clock.advance(120)
        self.debouncer.process_due()
        self.assertEqual(len(calls), 1)

    def test_shutdown_flushes_latest_pending_event_once(self):
        self.debouncer.submit("1", event("1", "10", "100"))
        self.debouncer.submit("1", event("1", "11", "101"))
        self.debouncer.shutdown(flush=True)
        self.assertEqual([item["person_id"] for item in self.sent], ["101"])


if __name__ == "__main__":
    unittest.main()
