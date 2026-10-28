"""
config.py - central settings.

Every tunable number or path lives here (or in the .env file), so you never
have to hunt through the code to change a threshold.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Folder that contains this file. All relative paths are resolved from here,
# so the project works no matter where you launch it from.
BASE_DIR = Path(__file__).resolve().parent

# Load variables from the .env file (if python-dotenv is installed).
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


# ---- small helpers that read an environment variable with a default value ----
def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default)).expanduser()
    return p if p.is_absolute() else BASE_DIR / p


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)  # frozen = settings can't be changed accidentally while running
class Settings:
    # ---------------- files / folders ----------------
    detector_model: Path = _path("DETECTOR_MODEL", "models/face_detection_yunet_2026may.onnx")
    recognizer_model: Path = _path("RECOGNIZER_MODEL", "models/w600k_mbf.onnx")
    embeddings_dir: Path = _path("EMBEDDINGS_DIR", "embeddings_for_known_faces")
    cameras_file: Path = _path("CAMERAS_FILE", "cameras.yaml")
    snapshot_dir: Path = _path("SNAPSHOT_DIR", "snapshots")  # unknown-face crops are saved here

    # ---------------- detection / recognition ----------------
    # Minimum confidence for YuNet to accept something as a face.
    detect_score_threshold: float = _float("DETECT_SCORE_THRESHOLD", 0.9)
    # Cosine similarity needed to say "this is person X". Higher = stricter
    # (fewer false matches, more missed matches). Tune on real footage!
    recognition_threshold: float = _float("RECOGNITION_THRESHOLD", 0.45)
    # A tracked face is re-checked every N seconds (saves CPU vs. every frame).
    recognition_interval: float = _float("RECOGNITION_INTERVAL", 2.0)
    # Faces smaller than this (pixels) are too blurry to recognise reliably.
    min_face_size: int = _int("MIN_FACE_SIZE", 40)
    # The last N recognition results "vote" on who a track is (stops name flicker).
    vote_window: int = _int("VOTE_WINDOW", 3)
    # "Unknown" is only reported after this many reads (avoids false alarms
    # from one bad-angle frame).
    unknown_confirm_reads: int = _int("UNKNOWN_CONFIRM_READS", 2)

    # ---------------- tracking / events ----------------
    # Forget a track's identity if it has not been seen for this many seconds.
    track_memory_timeout: float = _float("TRACK_MEMORY_TIMEOUT", 2.0)
    # Don't report the same person again within this many seconds.
    event_cooldown: float = _float("EVENT_COOLDOWN", 30.0)

    # ---------------- runtime ----------------
    show_window: bool = _bool("SHOW_WINDOW", True)  # show live preview window(s)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# One shared instance imported by the other files: `from config import settings`
settings = Settings()