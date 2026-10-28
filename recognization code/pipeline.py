"""
pipeline.py - the logic for ONE camera:

    frame -> detect faces -> track them (ByteTrack) -> recognise each track
          -> decide when something is worth reporting -> draw boxes on the frame

Key idea: recognition is slow, tracking is fast. So we recognise a face once,
remember the result under its track ID, and only re-check every few seconds.
"""
import logging
import time
from collections import Counter, deque
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from face_engine import UNKNOWN

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class TrackerResults:
    """
    BYTETracker was made for YOLO, so it expects an object with .xywh, .conf
    and .cls that can also be filtered with a mask. This is the smallest
    object that satisfies it.
    """
    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return TrackerResults(self.xywh[mask], self.conf[mask], self.cls[mask])


def to_tracker_results(faces):
    """Convert YuNet detections (x, y, w, h at top-left) to the tracker's format (centre x, y, w, h)."""
    if len(faces) == 0:
        return TrackerResults(np.empty((0, 4), np.float32),
                              np.empty((0,), np.float32),
                              np.empty((0,), np.float32))
    x, y, w, h = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3]
    xywh = np.stack([x + w / 2, y + h / 2, w, h], axis=1).astype(np.float32)
    conf = faces[:, -1].astype(np.float32)   # last column = detection confidence
    cls = np.zeros(len(faces), np.float32)   # only one class: "face"
    return TrackerResults(xywh, conf, cls)


def iou(a, b):
    """
    Intersection-over-Union of two boxes [x1, y1, x2, y2].
    1.0 = identical boxes, 0.0 = no overlap. Used to link a track to a detection.
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def majority_vote(votes):
    """Most common name in the recent results. On a tie, the newest result wins."""
    counts = Counter(votes)
    top = max(counts.values())
    tied = {name for name, c in counts.items() if c == top}
    return votes[-1] if votes[-1] in tied else next(iter(tied))


# ----------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------
class CameraPipeline:
    def __init__(self, camera, settings, engine, on_event=None):
        """
        camera    - CameraConfig for this camera
        settings  - the shared Settings object
        engine    - FaceEngine (detector + recognizer)
        on_event  - function called with a dict whenever something is worth
                    reporting. This is where you'll connect a backend later.
        """
        self.cam = camera
        self.s = settings
        self.engine = engine
        self.on_event = on_event or (lambda event: None)

        # ByteTrack gives each face a stable ID across frames.
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=0.5,  # detections above this are matched first
            track_low_thresh=0.1,   # weaker detections can still keep an existing track alive
            new_track_thresh=0.6,   # confidence needed to start a NEW track
            track_buffer=30,        # frames to keep a lost track before deleting it
            match_thresh=0.8,       # how strict track<->detection matching is
            fuse_score=True,
        ))

        self.memory = {}          # track_id -> what we know about that face
        self.last_event = {}      # who was last reported, and when (for the cooldown)
        self.recognition_count = 0  # total recognitions done (shown on screen)

    # ------------------------------------------------------------------
    # Called once per frame
    # ------------------------------------------------------------------
    def process(self, frame, draw=True):
        now = time.time()

        faces = self.engine.detect(frame)                                  # 1. detect
        tracks = self.tracker.update(to_tracker_results(faces), frame)     # 2. track
        # Each track row = [x1, y1, x2, y2, track_id, score, class, detection_index]

        results = []
        for t in tracks:
            track_id = int(t[4])
            idx = self._detection_index(t, faces)  # which detection belongs to this track?
            mem = self.memory.get(track_id)

            # 3. recognise (only if the face was actually detected this frame)
            if idx is not None:
                mem = self._update_identity(frame, faces[idx], track_id, mem, now)
            if mem is not None:
                mem["last_seen"] = now
            results.append((t[:4], track_id, mem))

        self._expire(now)  # forget tracks that disappeared

        # 4. draw AFTER all work is done, so boxes never end up inside saved snapshots
        if draw:
            for box, track_id, mem in results:
                self._draw(frame, box, track_id, mem)

    # ------------------------------------------------------------------
    # Linking tracks to detections
    # ------------------------------------------------------------------
    @staticmethod
    def _detection_index(track, faces):
        """Find which detected face this track corresponds to (needed for its landmarks)."""
        box = track[:4]

        # Fast path: the tracker already tells us the detection index (last column).
        # We double-check it by making sure the boxes overlap properly.
        if len(track) > 7:
            i = int(track[7])
            if 0 <= i < len(faces):
                f = faces[i]
                if iou(box, (f[0], f[1], f[0] + f[2], f[1] + f[3])) >= 0.3:
                    return i

        # Fallback: pick the detection that overlaps the track box the most.
        best, best_iou = None, 0.3
        for i, f in enumerate(faces):
            v = iou(box, (f[0], f[1], f[0] + f[2], f[1] + f[3]))
            if v > best_iou:
                best, best_iou = i, v
        return best  # None = no matching detection this frame

    # ------------------------------------------------------------------
    # Identity (who is this track?)
    # ------------------------------------------------------------------
    def _update_identity(self, frame, face, track_id, mem, now):
        """Create or refresh the identity of a track, recognising it only when needed."""
        too_small = min(face[2], face[3]) < self.s.min_face_size

        if mem is None:
            # Brand-new track.
            if too_small:
                return None  # skip tiny faces - can't be recognised reliably
            mem = {
                "name": UNKNOWN,
                "similarity": 0.0,
                "votes": deque(maxlen=self.s.vote_window),  # last few recognition results
                "reads": 0,           # how many recognitions we've done for this track
                "last_read": 0.0,     # time of the last recognition
                "announced": None,    # last name we reported for this track
                "last_seen": now,
            }
            self.memory[track_id] = mem
            due = True  # recognise immediately
        else:
            # Existing track: re-recognise only every `recognition_interval` seconds.
            due = (not too_small) and (now - mem["last_read"] >= self.s.recognition_interval)

        if due:
            self._read_identity(frame, face, mem, now)
            self._maybe_emit(frame, face, track_id, mem, now)
        return mem

    def _read_identity(self, frame, face, mem, now):
        """Run recognition once and update the track's name by majority vote."""
        mem["last_read"] = now  # set first, so a failure doesn't retry on every frame

        emb = self.engine.embed(frame, self.engine.landmarks(face))
        if emb is None:
            return
        self.recognition_count += 1

        name, sim = self.engine.match(emb)
        if sim < self.s.recognition_threshold:
            name = UNKNOWN  # best match isn't good enough

        mem["votes"].append(name)
        mem["reads"] += 1
        mem["similarity"] = sim
        mem["name"] = majority_vote(mem["votes"])  # smooths out one-frame mistakes

    # ------------------------------------------------------------------
    # Events (what's worth reporting)
    # ------------------------------------------------------------------
    def _maybe_emit(self, frame, face, track_id, mem, now):
        """Report an event when a track's confirmed identity changes."""
        name = mem["name"]

        if name == mem["announced"]:
            return  # already reported this identity for this track

        # Don't shout "Unknown" after a single bad frame; wait for a few reads.
        if name == UNKNOWN and mem["reads"] < self.s.unknown_confirm_reads:
            return

        mem["announced"] = name

        # Cooldown: don't report the same person again within `event_cooldown` seconds.
        # Unknown people are keyed by track so two different strangers both get reported.
        key = name if name != UNKNOWN else f"unknown:{track_id}"
        if now - self.last_event.get(key, 0.0) < self.s.event_cooldown:
            return
        self.last_event[key] = now

        event = {
            "camera_id": self.cam.id,
            "person_name": name,
            "similarity": float(mem["similarity"]),
            "track_id": int(track_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if name == UNKNOWN:
            event["snapshot"] = self._save_snapshot(frame, face, track_id)

        self.on_event(event)  # hand it to whoever is listening (log now, backend later)

    def _save_snapshot(self, frame, face, track_id):
        """Save a crop of an unknown face to disk as evidence. Returns the file path."""
        try:
            x, y, w, h = (int(v) for v in face[:4])
            pad = int(0.25 * max(w, h))  # include a bit of surroundings
            crop = frame[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
            if crop.size == 0:
                return None
            folder = self.s.snapshot_dir / self.cam.id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{datetime.now():%Y%m%d_%H%M%S}_track{track_id}.jpg"
            cv2.imwrite(str(path), crop)
            return str(path)
        except Exception:
            log.exception("Could not save snapshot")
            return None

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def _expire(self, now):
        """Delete memory of tracks we haven't seen recently, and old cooldown entries."""
        stale = [t for t, m in self.memory.items()
                 if now - m["last_seen"] > self.s.track_memory_timeout]
        for tid in stale:
            del self.memory[tid]

        if len(self.last_event) > 500:  # stop this dict growing forever
            self.last_event = {k: v for k, v in self.last_event.items()
                               if now - v < self.s.event_cooldown}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    @staticmethod
    def _draw(frame, box, track_id, mem):
        """Draw the box and label: green = known person, red = unknown, grey = not yet analysed."""
        x1, y1, x2, y2 = (int(v) for v in box)

        if mem is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 2)
            return

        color = (0, 0, 255) if mem["name"] == UNKNOWN else (0, 200, 0)  # BGR colours
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label_y = max(y1 - 50, 25)  # keep text on screen even if the face is at the top
        cv2.putText(frame, mem["name"], (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Sim: {mem['similarity']:.2f}", (x1, label_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, label_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)