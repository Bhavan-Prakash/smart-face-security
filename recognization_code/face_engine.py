"""
face_engine.py - the machine-learning part.

FaceEngine does four things:
  1. detect()      find faces in a frame (YuNet)
  2. landmarks()   get the 5 key points (eyes, nose, mouth corners)
  3. embed()       align the face and turn it into a 512-number "fingerprint"
  4. match()       compare a fingerprint to the known people
"""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

UNKNOWN = "Unknown"
MODEL_VERSION = "w600k_mbf-v1"

# Where the 5 landmarks should land in a standard 112x112 ArcFace face image.
# Warping every face onto this template makes recognition much more accurate.
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


class FaceEngine:
    """Owns the two ONNX models. Create one per process."""

    def __init__(self, settings, identities=None):
        # Fail early with a clear message if a model file is missing.
        for p in (settings.detector_model, settings.recognizer_model):
            if not Path(p).exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        # YuNet face detector. The (320, 320) size is a placeholder; detect()
        # updates it to the real frame size.
        self.detector = cv2.FaceDetectorYN.create(
            str(settings.detector_model), "", (320, 320),
            score_threshold=settings.detect_score_threshold,
        )
        # Recognition model: face image in -> 512-number embedding out.
        self.recognizer = cv2.dnn.readNetFromONNX(str(settings.recognizer_model))
        self._input_size = (320, 320)

        # One row per database embedding. Identity IDs, never filenames/names,
        # are the matching keys. Enrollment uses this same model instance.
        self._identity_index = (
            np.empty((0, 512), dtype=np.float32),
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )
        self._identity_lock = threading.RLock()
        self.load_identity_embeddings(identities or [])

    # ------------------------------------------------------------------
    # Known faces
    # ------------------------------------------------------------------
    def load_identity_embeddings(self, records):
        """Atomically replace the in-memory index from authorized database rows."""
        vectors, person_ids, names = [], [], []
        for record in records:
            vector = np.asarray(record["embedding"], dtype=np.float32).reshape(-1)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError("Worker returned an invalid face embedding")
            norm = float(np.linalg.norm(vector))
            if not 0.99 <= norm <= 1.01:
                raise ValueError("Worker returned a non-normalized face embedding")
            vectors.append(vector / norm)
            person_ids.append(str(record["person_id"]))
            names.append(str(record["name"]))
        matrix = np.vstack(vectors) if vectors else np.empty((0, 512), dtype=np.float32)
        # Readers take the same lock, so a refresh can never expose partially
        # rebuilt tenant data to a frame-processing thread.
        with getattr(self, "_identity_lock", threading.RLock()):
            self._identity_index = (matrix, np.asarray(person_ids, dtype=object), np.asarray(names, dtype=object))
        log.info("Loaded %d database embeddings for %d people", len(vectors), len(set(person_ids)))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Find faces. Returns an array with one row per face, 15 columns:
        [x, y, w, h, 5 landmark (x,y) pairs (10 numbers), confidence]
        """
        h, w = frame.shape[:2]
        if (w, h) != self._input_size:       # YuNet must be told the frame size
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, faces = self.detector.detect(frame)
        # OpenCV returns None when no face is found; return an empty array instead.
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    @staticmethod
    def landmarks(face):
        """Extract the 5 landmark points from one detection row, shape (5, 2)."""
        return np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def embed(self, frame, landmarks):
        """
        Turn a face into a 512-number embedding (unit length).
        Steps: align -> BGR to RGB -> scale to [-1, 1] -> HWC to CHW -> network.
        Returns None if alignment fails.
        """
        # 1. Find the rotate/scale/shift that moves our landmarks onto the template.
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARC_FACE_TEMPLATE)
        if matrix is None:
            return None

        # 2. Warp the face into a straight, centred 112x112 image.
        face = cv2.warpAffine(frame, matrix, (112, 112))

        # 3. OpenCV is BGR, the model expects RGB; also convert to float.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 4. Scale pixel values from 0..255 to about -1..1.
        face = (face - 127.5) / 127.5

        # 5. Height-Width-Channel -> Channel-Height-Width, then add a batch dimension.
        face = np.expand_dims(face.transpose(2, 0, 1), axis=0)

        # 6. Run the network and normalise the result to length 1
        #    (so a dot product equals cosine similarity).
        self.recognizer.setInput(face)
        emb = self.recognizer.forward().flatten()
        return emb / (np.linalg.norm(emb) + 1e-12)

    def match(self, embedding):
        """
        Compare one embedding to all known ones.
        Returns (database_person_id, current_name, similarity).
        """
        with getattr(self, "_identity_lock", threading.RLock()):
            matrix, person_ids, names = self._identity_index
            if len(person_ids) == 0:
                return None, UNKNOWN, 0.0
            sims = matrix @ embedding   # cosine similarity against everyone at once
            i = int(np.argmax(sims))         # index of the best match
            return str(person_ids[i]), str(names[i]), float(sims[i])
