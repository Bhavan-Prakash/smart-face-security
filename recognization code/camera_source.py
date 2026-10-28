"""
camera_source.py - everything about getting frames from cameras / videos.

- load_cameras()   reads cameras.yaml and returns a list of CameraConfig.
- StreamReader     live source (RTSP / webcam). Always returns the NEWEST frame
                   and reconnects automatically if the stream drops.
- VideoFileReader  plays a video file at its natural speed (for testing).
- open_source()    picks the right reader for a camera.
"""
import os

# IMPORTANT: this must be set before OpenCV opens any stream.
# RTSP over TCP is much more reliable than UDP (fewer corrupted/grey frames).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

log = logging.getLogger(__name__)

# A source starting with one of these is treated as a network stream.
_STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


@dataclass(frozen=True)
class CameraConfig:
    """One camera entry from cameras.yaml, after validation."""
    id: str
    name: str
    source: str   # final RTSP URL, file path, or webcam index (as text)
    kind: str     # "stream" | "file" | "webcam"
    enabled: bool = True
    loop: bool = True  # only used for video files

    @property
    def safe_source(self) -> str:
        """Source with the password hidden - always use this when logging."""
        return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", self.source)


def load_cameras(path) -> list:
    """Read cameras.yaml and return a list of CameraConfig objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras, seen_ids = [], set()
    for i, item in enumerate(data.get("cameras", [])):
        cam_id = str(item.get("id") or f"camera{i + 1}")
        if cam_id in seen_ids:
            raise ValueError(f"Duplicate camera id '{cam_id}' in {path}")
        seen_ids.add(cam_id)

        if "source" not in item:
            raise ValueError(f"Camera '{cam_id}' has no 'source' in {path}")

        # Replace ${VARIABLES} in the source with values from .env / environment.
        raw = str(item["source"]).strip()
        source = os.path.expandvars(raw)

        # If a ${VARIABLE} is still there, it was not defined -> tell the user clearly.
        if re.search(r"\$\{?\w+\}?", source):
            raise ValueError(
                f"Camera '{cam_id}': an environment variable in 'source' is not set. "
                f"Define it in your .env file."
            )

        # Work out what kind of source this is.
        if source.isdigit():
            kind = "webcam"
        elif source.lower().startswith(_STREAM_PREFIXES):
            kind = "stream"
        else:
            kind = "file"
            # Relative file paths are relative to cameras.yaml's folder.
            p = Path(source).expanduser()
            source = str(p if p.is_absolute() else path.parent / p)

        cameras.append(CameraConfig(
            id=cam_id,
            name=str(item.get("name") or cam_id),
            source=source,
            kind=kind,
            enabled=bool(item.get("enabled", True)),
            loop=bool(item.get("loop", True)),
        ))
    return cameras


class StreamReader:
    """
    Live source (RTSP / webcam).

    A background thread reads frames non-stop and keeps ONLY the newest one.
    Why: if your processing is slower than the camera, frames would otherwise
    pile up and the video would lag further and further behind real time.
    """
    ended = False  # live streams never "end" - they just reconnect

    def __init__(self, source, name="camera", reconnect_delay=3.0):
        # A digit string means webcam index; OpenCV needs it as an int.
        self.source = int(source) if str(source).isdigit() else source
        self.name = name
        self.delay = reconnect_delay

        self._frame = None                   # newest frame
        self._lock = threading.Lock()        # protects _frame between threads
        self._new = threading.Event()        # set when an unread frame exists
        self._stop = threading.Event()       # set to stop the background thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        """Try to open the stream; return the capture or None."""
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep OpenCV's internal buffer tiny
            return cap
        cap.release()
        return None

    def _run(self):
        """Background loop: connect, read frames, reconnect on failure."""
        cap = None
        while not self._stop.is_set():
            # Not connected -> (re)connect, wait and retry if it fails.
            if cap is None:
                cap = self._open()
                if cap is None:
                    log.warning("[%s] cannot open source, retrying in %.0fs", self.name, self.delay)
                    self._stop.wait(self.delay)
                    continue
                log.info("[%s] connected", self.name)

            ok, frame = cap.read()
            if not ok:  # stream dropped
                log.warning("[%s] stream lost, reconnecting...", self.name)
                cap.release()
                cap = None
                self._stop.wait(self.delay)
                continue

            # Store the newest frame, replacing any older unread one.
            with self._lock:
                self._frame = frame
                self._new.set()

        if cap is not None:
            cap.release()

    def read(self, timeout=5.0):
        """Return (True, newest_frame), or (False, None) if nothing arrived in `timeout` s."""
        if not self._new.wait(timeout):
            return False, None
        with self._lock:
            frame = self._frame
            self._new.clear()  # mark as consumed so we never process the same frame twice
        return True, frame

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)


class VideoFileReader:
    """Plays a video file at its real speed. Handy for testing without a camera."""

    def __init__(self, path, loop=True, name="video"):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {path}")
        self.loop = loop
        self.name = name
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.ended = False             # becomes True when a non-looping video finishes
        self._next = time.monotonic()  # when the next frame should be shown

    def read(self, timeout=5.0):
        ok, frame = self.cap.read()

        # End of file: jump back to frame 0 if looping.
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if not ok:
            self.ended = True
            return False, None

        # Slow down to the video's real FPS (otherwise it plays super fast).
        now = time.monotonic()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next, now) + 1.0 / self.fps
        return True, frame

    def release(self):
        self.cap.release()


def open_source(cam: CameraConfig):
    """Return the right reader (file or live stream) for this camera."""
    log.info("[%s] opening %s source: %s", cam.id, cam.kind, cam.safe_source)
    if cam.kind == "file":
        return VideoFileReader(cam.source, loop=cam.loop, name=cam.id)
    return StreamReader(cam.source, name=cam.id)