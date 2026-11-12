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
import logging
import multiprocessing as mp
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

import cv2

from config import settings
from camera_source import CameraConfig, load_cameras, open_source
from face_engine import FaceEngine
from pipeline import CameraPipeline

LOG_FORMAT = "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"


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


def _deliver_events(events):
    endpoint = os.getenv("PORTAL_API_URL", "").rstrip("/") + "/api/recognition-events"
    token = os.getenv("RECOGNITION_WORKER_TOKEN", "")
    while True:
        event = events.get()
        if not endpoint.strip("/") or not token:
            logging.getLogger("event").warning("Recognition event not delivered: portal API/token is not configured")
            continue
        payload = json.dumps(event).encode("utf-8")
        for attempt in range(3):
            try:
                req = urllib.request.Request(endpoint, payload, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=3) as response:
                    if response.status < 300:
                        break
                    raise RuntimeError(f"portal returned HTTP {response.status}")
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                if attempt == 2:
                    logging.getLogger("event").error("Could not deliver recognition event: %s", exc)
                else:
                    time.sleep(0.25 * (2 ** attempt))


def handle_event(event, events):
    """
    Called whenever a person is recognised / an unknown face is confirmed.

    Queue the event so an HTTP request cannot stall frame processing.
    """
    logging.getLogger("event").info(
        "%s | camera=%s | track=%s | similarity=%.2f",
        event["person_name"], event["camera_id"], event["track_id"], event["similarity"],
    )
    try:
        events.put_nowait(event)
    except queue.Full:
        logging.getLogger("event").error("Recognition delivery queue is full; dropping event")


def run_camera(cam, show):
    """Runs inside its own process: one camera, from connection to shutdown."""
    # Child processes start fresh, so logging must be configured again here.
    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("camera")

    # Each process loads its own models (they can't be shared between processes).
    engine = FaceEngine(settings)
    events = queue.Queue(maxsize=100)
    threading.Thread(target=_deliver_events, args=(events,), daemon=True, name="event-delivery").start()
    pipeline = CameraPipeline(cam, settings, engine, on_event=lambda event: handle_event(event, events))
    reader = open_source(cam)

    fps, prev = 0.0, time.time()
    try:
        while True:
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
    base = os.getenv("PORTAL_API_URL", "").rstrip("/")
    token = os.getenv("RECOGNITION_WORKER_TOKEN", "")
    if not base or not token:
        raise RuntimeError("PORTAL_API_URL and RECOGNITION_WORKER_TOKEN are required for portal cameras")
    request = urllib.request.Request(f"{base}/api/worker/cameras", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.load(response)
    cameras = []
    for item in payload.get("cameras", []):
        source = str(item["source"])
        kind = "stream" if source.lower().startswith(("rtsp://", "rtsps://")) else "file"
        cameras.append(CameraConfig(id=str(item["id"]), name=str(item["name"]), source=source, kind=kind))
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Face recognition security system")
    parser.add_argument("--cameras", default=str(settings.cameras_file), help="path to cameras.yaml")
    parser.add_argument("--camera", action="append", help="only run this camera id (can repeat)")
    parser.add_argument("--headless", action="store_true", help="do not open preview windows")
    parser.add_argument("--portal", action="store_true", help="load registered cameras from the portal API")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    log = logging.getLogger("supervisor")

    # Keep only enabled cameras (and only the requested ones, if --camera was used).
    configured_cameras = load_portal_cameras() if args.portal else load_cameras(args.cameras)
    cameras = [c for c in configured_cameras
               if c.enabled and (not args.camera or c.id in args.camera)]
    if not cameras:
        sys.exit("No enabled cameras. Edit cameras.yaml and set enabled: true.")

    show = settings.show_window and not args.headless

    # "spawn" behaves the same on macOS, Windows and Linux.
    ctx = mp.get_context("spawn")
    procs = {}  # camera id -> (CameraConfig, Process)

    def start(cam):
        p = ctx.Process(target=run_camera, args=(cam, show), name=cam.id)
        p.start()
        procs[cam.id] = (cam, p)
        log.info("Started camera '%s' (%s)", cam.id, cam.safe_source)

    for cam in cameras:
        start(cam)

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
    main()
