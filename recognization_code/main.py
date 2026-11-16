"""
main.py - program entry point.

Usage:
    python main.py                     run every enabled camera in cameras.yaml
    python main.py --camera entrance   run only the camera with this id
    python main.py --headless          no preview windows (for servers)

How it works: the main process is a "supervisor". It starts one separate
process per camera (so cameras run in parallel on different CPU cores) and
restarts any camera process that crashes.
"""
import argparse
import base64
import logging
import multiprocessing as mp
import json
import os
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import cv2
import numpy as np

from config import WORKER_TOKEN_SOURCE, settings
from camera_source import CameraConfig, load_cameras, open_source
from face_engine import FaceEngine, MODEL_VERSION
from pipeline import CameraPipeline
from recognition_event_debouncer import DeliveryResult, RecognitionEventDebouncer

LOG_FORMAT = "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"


class PortalCameraLoadError(RuntimeError):
    """A sanitized, actionable portal camera configuration failure."""


def _worker_token():
    """Canonical worker token: UTF-8 text without surrounding whitespace."""
    return os.getenv("RECOGNITION_WORKER_TOKEN", "").strip()


def draw_overlay(frame, cam, fps, recognitions):
    """Draw the small info panel (camera name, FPS, ...) in the top-left corner."""
    h, w = frame.shape[:2]
    lines = [
        (cam.name, (255, 255, 255)),
        (f"FPS: {fps:.1f}", (0, 255, 0)),
        (f"Recognitions: {recognitions}", (0, 255, 255)),
        (f"{w}x{h}", (255, 255, 255)),
    ]
    cv2.rectangle(frame, (10, 10), (260, 20 + 28 * len(lines)), (0, 0, 0), -1)  # black panel
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 38 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _post_event(event):
    """Post one event once; retry policy belongs to the per-user debouncer."""
    endpoint = os.getenv("PORTAL_API_URL", "").strip().rstrip("/") + "/api/recognition-events"
    token = _worker_token()
    context = "eventId=%s | cameraId=%s | trackId=%s | eventStatus=%s | submittedAt=%s"
    context_values = (event.get("event_id", "unassigned"), event.get("camera_id", "unknown"), event.get("track_id", "unknown"), event.get("status", "unknown"), event.get("timestamp", "unknown"))
    if not endpoint.strip("/") or not token:
        logging.getLogger("event").error("Recognition event not delivered: portal API/token is not configured | " + context, *context_values)
        return DeliveryResult(success=False, retryable=False)
    # Local paths and worker-only tenant routing metadata never leave this
    # process. A completed, bounded JPEG is encoded only on this background
    # delivery path, never on the camera processing thread.
    api_event = {key: value for key, value in event.items() if key not in {"snapshot", "snapshot_path", "_tenant_id"}}
    snapshot_path = event.get("snapshot_path")
    if event.get("status") == "unknown" and isinstance(snapshot_path, str):
        try:
            path = os.path.realpath(snapshot_path)
            root = os.path.realpath(str(settings.snapshot_dir)) + os.sep
            if path.startswith(root) and path.lower().endswith(".jpg") and os.path.getsize(path) <= 1_000_000:
                with open(path, "rb") as snapshot_file:
                    jpeg = snapshot_file.read()
                if jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"):
                    api_event["snapshot_jpeg"] = base64.b64encode(jpeg).decode("ascii")
        except (OSError, ValueError):
            # Snapshot failure must never prevent recording the unknown event.
            logging.getLogger("event").warning("Unknown snapshot unavailable; submitting event without image | eventId=%s", event.get("event_id"))
    payload = json.dumps(api_event).encode("utf-8")
    try:
        req = urllib.request.Request(endpoint, payload, {
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
            "X-Request-ID": str(event.get("event_id") or "unassigned"),
        }, method="POST")
        with urllib.request.urlopen(req, timeout=3) as response:
            result = json.load(response)
            if not isinstance(result, dict):
                raise json.JSONDecodeError("Expected an object response", "", 0)
            logging.getLogger("event").info("Recognition event accepted | " + context + " | httpStatus=%s | databaseEventId=%s | duplicate=%s", *context_values, response.status, result.get("eventId"), result.get("duplicate", False))
            return DeliveryResult(success=True)
    except urllib.error.HTTPError as error:
        logging.getLogger("event").error("Recognition event rejected by portal | " + context + " | httpStatus=%s | backendError=%s", *context_values, error.code, _safe_event_api_error(error))
        return DeliveryResult(success=False, retryable=error.code >= 500)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.getLogger("event").error("Portal returned invalid JSON for recognition event | " + context, *context_values)
        return DeliveryResult(success=False, retryable=True)
    except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
        logging.getLogger("event").warning("Recognition event delivery failed | " + context + " | errorType=%s", *context_values, type(error).__name__)
        return DeliveryResult(success=False, retryable=True)


def _request_door_access(event):
    """Immediately ask the portal to authorize one recognized event for access.

    This is deliberately independent of debounced recognition-history delivery.
    The portal owns all authorization and never trusts worker tenant metadata.
    """
    if event.get("status") != "recognized" or not event.get("person_id"):
        return
    base = os.getenv("PORTAL_API_URL", "").strip().rstrip("/")
    token = _worker_token()
    event_id = str(event.get("event_id") or "")
    if not base or not token or not event_id:
        logging.getLogger("door_access").warning("[DOOR_ACCESS] Access request not sent: portal configuration is unavailable | cameraId=%s", event.get("camera_id"))
        return
    payload = {key: value for key, value in event.items() if key not in {"snapshot", "_tenant_id"}}
    try:
        logging.getLogger("door_access").info("[DOOR_ACCESS] Recognition received; requesting portal authorization | eventId=%s | personId=%s | cameraId=%s", event_id, event.get("person_id"), event.get("camera_id"))
        request = urllib.request.Request(
            f"{base}/api/door-access", json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "X-Request-ID": event_id}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.load(response)
            if not isinstance(body, dict) or body.get("accepted") is not True:
                raise json.JSONDecodeError("Expected accepted door access response", "", 0)
            logging.getLogger("door_access").info("[DOOR_ACCESS] Portal completed access evaluation | eventId=%s | personId=%s | cameraId=%s | granted=%s | reason=%s", event_id, event.get("person_id"), event.get("camera_id"), body.get("granted"), body.get("reason"))
    except urllib.error.HTTPError as error:
        logging.getLogger("door_access").warning("[DOOR_ACCESS] Portal denied or rejected access request | eventId=%s | personId=%s | cameraId=%s | httpStatus=%s | error=%s", event_id, event.get("person_id"), event.get("camera_id"), error.code, _safe_event_api_error(error))
    except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, UnicodeDecodeError) as error:
        # Never retry an unlock automatically: an interrupted request may have
        # reached the controller. The ESP32 command ID still makes duplicates safe.
        logging.getLogger("door_access").warning("[DOOR_ACCESS] Portal unavailable; access command not retried | eventId=%s | personId=%s | cameraId=%s | errorType=%s", event_id, event.get("person_id"), event.get("camera_id"), type(error).__name__)


def _deliver_events(events):
    """Legacy non-portal delivery worker; portal mode uses the shared debouncer."""
    while True:
        _post_event(events.get())


def _safe_event_api_error(error):
    """Read only a short, sanitized error/code pair from the API response."""
    try:
        body = error.read(2048)
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            return "request rejected"
        message = parsed.get("error")
        code = parsed.get("code")
        pieces = [value for value in (code, message) if isinstance(value, str)]
        safe = " | ".join(pieces) if pieces else "request rejected"
        safe = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", safe)
        safe = re.sub(r"(?i)rtsps?://\S+", "[redacted-camera-url]", safe)
        safe = re.sub(r"(?i)mysql://\S+", "mysql://[redacted]", safe)
        return safe.replace("\n", " ").replace("\r", " ")[:300]
    except (OSError, ValueError, TypeError):
        return "request rejected (response body unavailable or invalid)"


def _safe_worker_api_error(error):
    """Extract only safe API diagnostics; never print bearer values or hashes."""
    try:
        parsed = json.loads(error.read(2048))
        if not isinstance(parsed, dict):
            return "request rejected"
        pieces = [value for value in (parsed.get("code"), parsed.get("error")) if isinstance(value, str)]
        safe = " | ".join(pieces) if pieces else "request rejected"
        safe = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", safe)
        safe = re.sub(r"(?i)rtsps?://\S+", "[redacted-camera-url]", safe)
        safe = re.sub(r"(?i)mysql://\S+", "mysql://[redacted]", safe)
        safe = re.sub(r"(?i)(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])", "[redacted-hash]", safe)
        return safe.replace("\n", " ").replace("\r", " ")[:300]
    except (OSError, ValueError, TypeError):
        return "request rejected (response body unavailable or invalid)"


def handle_event(event, events, owner_id=None):
    """
    Called whenever a person is recognised / an unknown face is confirmed.

    Queue the event so an HTTP request cannot stall frame processing.
    """
    logging.getLogger("event").info(
        "Recognition event queued | eventId=%s | cameraId=%s | trackId=%s | status=%s | submittedAt=%s",
        event.get("event_id", "unassigned"), event["camera_id"], event["track_id"], event.get("status", "unknown"), event.get("timestamp", "unknown"),
    )
    try:
        queued_event = dict(event)
        if owner_id:
            queued_event["_tenant_id"] = str(owner_id)
        events.put_nowait(queued_event)
    except queue.Full:
        logging.getLogger("event").error("Recognition delivery queue is full; dropping event")


def load_portal_embeddings(camera_id):
    """Fetch only the identities authorized for this database camera."""
    base = os.getenv("PORTAL_API_URL", "").strip().rstrip("/")
    token = _worker_token()
    if not base or not token:
        raise PortalCameraLoadError("Portal embedding authentication is not configured")
    records = []
    offset = 0
    while True:
        request = urllib.request.Request(
            f"{base}/api/worker/person-embeddings?camera_id={camera_id}&offset={offset}&limit=1000",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            logging.getLogger("worker_api").error("Identity refresh rejected | cameraId=%s | httpStatus=%s", camera_id, error.code)
            raise PortalCameraLoadError(f"Portal identity API returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as error:
            logging.getLogger("worker_api").warning("Identity refresh failed | cameraId=%s | errorType=%s", camera_id, type(error).__name__)
            raise PortalCameraLoadError("Portal identity API request failed") from None
        if not isinstance(payload, dict) or payload.get("camera_id") != str(camera_id) or not isinstance(payload.get("embeddings"), list):
            raise PortalCameraLoadError("Portal returned an invalid identity response")
        page = payload["embeddings"]
        for row in page:
            if not isinstance(row, dict) or row.get("encoding") != "float32le" or not isinstance(row.get("embedding"), str):
                raise PortalCameraLoadError("Portal returned an invalid embedding record")
            if row.get("model_version") != MODEL_VERSION:
                raise PortalCameraLoadError("Portal identity embeddings use an incompatible model version")
            if not re.fullmatch(r"[1-9]\d*", str(row.get("person_id", ""))) or not isinstance(row.get("name"), str):
                raise PortalCameraLoadError("Portal returned an invalid person identity")
            try:
                vector = np.frombuffer(base64.b64decode(row["embedding"], validate=True), dtype="<f4")
            except (ValueError, TypeError):
                raise PortalCameraLoadError("Portal returned an invalid embedding vector") from None
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise PortalCameraLoadError("Portal returned an invalid embedding vector")
            records.append({"person_id": str(row["person_id"]), "name": str(row["name"]), "embedding": vector})
        next_offset = payload.get("nextOffset")
        if next_offset is None:
            break
        if not isinstance(next_offset, int) or next_offset <= offset:
            raise PortalCameraLoadError("Portal returned an invalid embedding pagination cursor")
        offset = next_offset
    logging.getLogger("worker_api").info("Identity dataset loaded | cameraId=%s | persons=%d | embeddings=%d", camera_id, len({r["person_id"] for r in records}), len(records))
    return records


def _refresh_identities(engine, camera_id, stop_event):
    log = logging.getLogger("worker_api")
    while not stop_event.wait(settings.identity_sync_interval):
        try:
            records = load_portal_embeddings(camera_id)
            engine.load_identity_embeddings(records)
            log.info("Identity embedding refresh completed | cameraId=%s | embeddings=%d", camera_id, len(records))
        except PortalCameraLoadError:
            log.warning("Identity embedding refresh failed; retaining previous authorized index | cameraId=%s", camera_id)


def run_camera(cam, show, portal_mode=False, stop_event=None, shared_events=None):
    """Runs inside its own process: one camera, from connection to shutdown."""
    # Child processes start fresh, so logging must be configured again here.
    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("camera")

    # Each process loads its own models (they can't be shared between processes).
    identities = load_portal_embeddings(cam.id) if portal_mode else []
    engine = FaceEngine(settings, identities)
    if stop_event is None:
        stop_event = threading.Event()
    if portal_mode:
        threading.Thread(target=_refresh_identities, args=(engine, cam.id, stop_event), daemon=True, name=f"identity-refresh-{cam.id}").start()
    events = shared_events if shared_events is not None else queue.Queue(maxsize=100)
    if shared_events is None:
        threading.Thread(target=_deliver_events, args=(events,), daemon=True, name="event-delivery").start()
    pipeline = CameraPipeline(cam, settings, engine, on_event=lambda event: handle_event(event, events, cam.owner_id if portal_mode else None))
    reader = open_source(cam)

    fps, prev = 0.0, time.time()
    try:
        while not stop_event.is_set():
            ok, frame = reader.read()
            if not ok:
                if reader.ended:           # a non-looping video file finished
                    log.info("[%s] video ended", cam.id)
                    break
                continue                   # stream is down; reader is reconnecting

            pipeline.process(frame, draw=show)  # detect -> track -> recognise -> draw

            # Smoothed FPS (90% old value + 10% new) so the number doesn't jitter.
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            prev = now

            if show:
                draw_overlay(frame, cam, fps, pipeline.recognition_count)
                cv2.imshow(cam.name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # press q in the window to quit
                    break
    except KeyboardInterrupt:  # Ctrl+C
        pass
    finally:
        reader.release()
        cv2.destroyAllWindows()


def load_portal_cameras():
    """Load only registered, enabled cameras using the worker credential."""
    log = logging.getLogger("worker_api")
    base = os.getenv("PORTAL_API_URL", "").strip().rstrip("/")
    token = _worker_token()
    if not base:
        log.error("PORTAL_API_URL is missing from recognization_code/.env or the process environment")
        raise PortalCameraLoadError("Portal API URL is not configured")
    if not token:
        log.error("RECOGNITION_WORKER_TOKEN is missing from recognization_code/.env or the process environment")
        raise PortalCameraLoadError("Worker authentication is not configured")
    log.info("Worker token loaded | source=%s", WORKER_TOKEN_SOURCE)

    parsed_base = urlsplit(base)
    if parsed_base.scheme not in ("http", "https") or not parsed_base.hostname or parsed_base.username or parsed_base.password or parsed_base.query or parsed_base.fragment:
        log.error("PORTAL_API_URL must be an http(s) origin/path without embedded credentials or query parameters")
        raise PortalCameraLoadError("Portal API URL is invalid")
    if parsed_base.scheme == "http" and parsed_base.hostname.lower() not in {"localhost", "127.0.0.1", "::1"}:
        log.error("PORTAL_API_URL must use HTTPS when the portal is not on this host")
        raise PortalCameraLoadError("Portal API URL must use HTTPS outside loopback")

    endpoint = f"{base}/api/worker/cameras"
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    log.info("Portal mode enabled")
    log.info("Connecting to portal API host %s", parsed_base.hostname)
    log.info("Fetching registered camera configurations")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            log.info("Portal API connection successful")
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        api_error = _safe_worker_api_error(error)
        if error.code == 401:
            log.error("Portal API returned HTTP Status: 401 | API Error: %s", api_error)
        elif error.code >= 500:
            log.error("Portal API returned HTTP Status: %s | API Error: %s", error.code, api_error)
        else:
            log.error("Portal API returned HTTP Status: %s | API Error: %s", error.code, api_error)
        log.debug("Portal HTTP response exception details", exc_info=True)
        raise PortalCameraLoadError(f"Portal API returned HTTP {error.code}") from None
    except (TimeoutError, socket.timeout):
        log.error("Portal API request timed out after 5 seconds")
        log.debug("Portal timeout exception details", exc_info=True)
        raise PortalCameraLoadError("Portal API request timed out") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            log.error("Portal API request timed out after 5 seconds")
            log.debug("Portal timeout exception details", exc_info=True)
            raise PortalCameraLoadError("Portal API request timed out") from None
        log.error("Could not connect to the portal API at host %s", parsed_base.hostname)
        log.debug("Portal connection exception details", exc_info=True)
        raise PortalCameraLoadError("Portal API is unreachable") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.error("Portal API returned invalid JSON")
        log.debug("Portal JSON exception details", exc_info=True)
        raise PortalCameraLoadError("Portal API response was not valid JSON") from None
    except OSError:
        log.exception("Portal API connection failed unexpectedly")
        raise PortalCameraLoadError("Portal API connection failed") from None

    if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), list):
        log.error("Portal API response is missing a valid cameras array")
        raise PortalCameraLoadError("Portal API response has an invalid shape")

    cameras = []
    for index, item in enumerate(payload["cameras"], start=1):
        if not isinstance(item, dict) or not item.get("id") or not isinstance(item.get("name"), str) or not isinstance(item.get("source"), str) or not re.fullmatch(r"[1-9]\d*", str(item.get("owner_id", ""))) or not isinstance(item.get("config_version"), str):
            log.error("Portal API returned an invalid camera configuration at index %d", index)
            raise PortalCameraLoadError("Portal API returned an invalid camera configuration")
        source = item["source"]
        kind = "stream" if source.lower().startswith(("rtsp://", "rtsps://")) else "file"
        if kind != "stream":
            log.error("Portal returned an unsupported camera source type for cameraId=%s", item["id"])
            raise PortalCameraLoadError("Portal returned an unsupported camera source")
        cameras.append(CameraConfig(id=str(item["id"]), name=str(item["name"]), source=source, kind=kind, owner_id=str(item["owner_id"]), config_version=item["config_version"]))
    log.info("Received %d camera configurations", len(cameras))
    return cameras


class PortalCameraSupervisor:
    """Reconcile portal cameras without interrupting unaffected streams."""
    def __init__(self, context, show, start_target=run_camera, shared_events=None):
        self.context = context
        self.show = show
        self.start_target = start_target
        self.shared_events = shared_events
        self.procs = {}  # camera id -> (CameraConfig, Process, Event)
        self.log = logging.getLogger("supervisor")

    @staticmethod
    def changed(old, new):
        return old.source != new.source or old.name != new.name or old.config_version != new.config_version

    def start(self, camera):
        stop_event = self.context.Event()
        process = self.context.Process(target=self.start_target, args=(camera, self.show, True, stop_event, self.shared_events), name=camera.id)
        process.start()
        self.procs[camera.id] = (camera, process, stop_event)
        self.log.info("Started camera process | cameraId=%s", camera.id)

    def stop(self, camera_id):
        entry = self.procs.pop(camera_id, None)
        if not entry:
            return
        _, process, stop_event = entry
        stop_event.set()
        process.join(timeout=8)
        if process.is_alive():
            self.log.warning("Camera did not stop cleanly; terminating | cameraId=%s", camera_id)
            process.terminate()
            process.join(timeout=5)
        self.log.info("Stopped camera process | cameraId=%s", camera_id)

    def reconcile(self, cameras):
        desired = {camera.id: camera for camera in cameras if camera.enabled}
        for camera_id in list(self.procs):
            if camera_id not in desired:
                self.stop(camera_id)
        for camera_id, camera in desired.items():
            existing = self.procs.get(camera_id)
            if existing is None:
                self.start(camera)
            elif self.changed(existing[0], camera):
                self.log.info("Camera configuration changed; reconnecting | cameraId=%s", camera_id)
                self.stop(camera_id)
                self.start(camera)

    def restart_crashed(self):
        for camera_id, (camera, process, _stop_event) in list(self.procs.items()):
            if not process.is_alive():
                self.procs.pop(camera_id, None)
                self.log.error("Camera process exited; restarting | cameraId=%s | exitCode=%s", camera_id, process.exitcode)
                self.start(camera)

    def shutdown(self):
        for camera_id in list(self.procs):
            self.stop(camera_id)


def _dispatch_portal_events(events, debouncer, stop_event):
    """Bridge child camera-process events into the single tenant debouncer."""
    log = logging.getLogger("event")
    while not stop_event.is_set():
        try:
            event = events.get(timeout=0.5)
        except queue.Empty:
            continue
        if not isinstance(event, dict):
            log.warning("Ignoring malformed recognition event from camera process")
            continue
        tenant_id = event.get("_tenant_id")
        if not isinstance(tenant_id, str) or not re.fullmatch(r"[1-9]\d*", tenant_id):
            log.error("Ignoring recognition event without portal tenant metadata | cameraId=%s", event.get("camera_id"))
            continue
        _request_door_access(event)
        debouncer.submit(tenant_id, event)


def main():
    parser = argparse.ArgumentParser(description="Face recognition security system")
    parser.add_argument("--cameras", default=str(settings.cameras_file), help="path to cameras.yaml")
    parser.add_argument("--camera", action="append", help="only run this camera id (can repeat)")
    parser.add_argument("--headless", action="store_true", help="do not open preview windows")
    parser.add_argument("--portal", action="store_true", help="run the dynamic central worker using the portal API")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("supervisor")
    log.info("Recognition worker starting")

    show = settings.show_window and not args.headless

    # "spawn" behaves the same on macOS, Windows and Linux.
    ctx = mp.get_context("spawn")

    # Portal mode is a persistent central worker: an empty account/camera set
    # is normal, and temporary API failures retain already-running cameras.
    if args.portal:
        event_queue = ctx.Queue(maxsize=1000)
        debounce_stop = threading.Event()
        debouncer = RecognitionEventDebouncer(_post_event, interval_seconds=settings.recognition_debounce_seconds)
        debouncer.start()
        dispatcher = threading.Thread(target=_dispatch_portal_events, args=(event_queue, debouncer, debounce_stop), daemon=True, name="portal-event-dispatch")
        dispatcher.start()
        supervisor = PortalCameraSupervisor(ctx, show, shared_events=event_queue)
        try:
            while True:
                try:
                    cameras = load_portal_cameras()
                    if args.camera:
                        cameras = [camera for camera in cameras if camera.id in args.camera]
                    supervisor.reconcile(cameras)
                except PortalCameraLoadError:
                    log.warning("Portal camera synchronization failed; retaining current camera processes")
                supervisor.restart_crashed()
                time.sleep(max(5, settings.portal_sync_interval))
        except KeyboardInterrupt:
            log.info("Shutting down central worker...")
        finally:
            supervisor.shutdown()
            debounce_stop.set()
            dispatcher.join(timeout=2)
            debouncer.shutdown(flush=True)
        return 0

    # File mode preserves the previous one-time YAML workflow.
    cameras = [c for c in load_cameras(args.cameras) if c.enabled and (not args.camera or c.id in args.camera)]
    if not cameras:
        log.error("No enabled cameras. Edit cameras.yaml and set enabled: true.")
        return 1

    procs = {}  # camera id -> (CameraConfig, Process)

    def start(cam):
        log.info("Initializing camera | cameraId=%s", cam.id)
        p = ctx.Process(target=run_camera, args=(cam, show, args.portal), name=cam.id)
        p.start()
        procs[cam.id] = (cam, p)
        log.info("Started camera process | cameraId=%s", cam.id)

    for cam in cameras:
        start(cam)
    log.info("Camera initialization completed | count=%d", len(cameras))

    # Supervisor loop: watch the camera processes.
    try:
        while procs:
            time.sleep(1)
            for cid, (cam, p) in list(procs.items()):
                if p.is_alive():
                    continue
                if p.exitcode == 0:  # clean exit (user pressed q / video ended) - don't restart
                    log.info("Camera '%s' finished", cid)
                    del procs[cid]
                else:                # crashed - wait a little, then restart it
                    log.error("Camera '%s' crashed (exit code %s) - restarting in 5s", cid, p.exitcode)
                    time.sleep(5)
                    start(cam)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        for _, p in procs.values():
            p.terminate()
        for _, p in procs.values():
            p.join(timeout=5)


# This guard is REQUIRED when using multiprocessing with "spawn".
if __name__ == "__main__":
    raise SystemExit(main())
