"""Private asynchronous embedding-generation API. Keep on a trusted network."""
import hmac
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile

from config import settings
from face_engine import MODEL_VERSION, FaceEngine

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("enrollment")
app = FastAPI(title="Private Face Enrollment Service", docs_url=None, redoc_url=None)
engine = FaceEngine(settings)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
inference_lock = threading.Lock()
JOB_TTL_SECONDS = 15 * 60

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _authorize(authorization: str | None) -> None:
    expected = os.getenv("ENROLLMENT_SERVICE_TOKEN", "")
    supplied = authorization[7:] if authorization and authorization[:7].lower() == "bearer " else ""
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Enrollment service authentication failed")


def _purge_expired_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        for key, value in list(jobs.items()):
            if value.get("updated_at", time.time()) < cutoff:
                del jobs[key]


def _log(level: int, event: str, **fields: object) -> None:
    """Emit one safe, structured event without image bytes, vectors, or credentials."""
    log.log(level, "%s %s", event, json.dumps(fields, default=str, sort_keys=True))


@app.get("/health")
def health():
    return {"ok": True, "model_version": MODEL_VERSION}


def _generate_job(job_id: str, images: list[tuple[str, bytes] | tuple[str, bytes, str]], request_id: str | None = None) -> None:
    with jobs_lock:
        jobs[job_id] = {"status": "processing", "updated_at": time.time()}
    outputs = []
    stats = {
        "images_received": len(images), "images_processed": 0, "images_decoded": 0,
        "faces_detected": 0, "images_skipped": 0, "embeddings_generated": 0,
    }
    _log(logging.INFO, "enrollment_started", enrollment_id=job_id, request_id=request_id, image_count=len(images))
    try:
        for index, item in enumerate(images, start=1):
            content_type, raw = item[0], item[1]
            filename = item[2] if len(item) > 2 else None
            stats["images_processed"] += 1
            image_fields = {"enrollment_id": job_id, "request_id": request_id, "image_index": index, "filename": filename}
            try:
                if content_type not in ALLOWED_TYPES:
                    raise ValueError("unsupported image MIME type")
                encoded = np.frombuffer(raw, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("file is not a supported image")
                height, width = image.shape[:2]
                stats["images_decoded"] += 1
                _log(logging.INFO, "image_decoded", **image_fields, width=width, height=height)
                if width < 64 or height < 64 or width * height > 24_000_000:
                    raise ValueError("image dimensions are outside the supported range")
                _log(logging.INFO, "face_detection_started", **image_fields)
                with inference_lock:
                    faces = engine.detect(image)
                face_count = len(faces)
                _log(logging.INFO, "face_detection_completed", **image_fields, face_count=face_count)
                if len(faces) == 0:
                    raise ValueError("no face detected")
                if len(faces) != 1:
                    raise ValueError("exactly one face is required")
                face = faces[0]
                if min(float(face[2]), float(face[3])) < settings.min_face_size:
                    raise ValueError("detected face is too small")
                stats["faces_detected"] += 1
                _log(logging.INFO, "embedding_generation_started", **image_fields)
                with inference_lock:
                    embedding = engine.embed(image, engine.landmarks(face))
                if embedding is None or embedding.shape != (512,) or not np.isfinite(embedding).all():
                    raise ValueError("face alignment or embedding generation failed")
                norm = float(np.linalg.norm(embedding))
                if not 0.99 <= norm <= 1.01:
                    raise ValueError("embedding failed normalization validation")
                outputs.append(embedding.astype(np.float32).tolist())
                stats["embeddings_generated"] += 1
                _log(logging.INFO, "embedding_generated", **image_fields)
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id]["updated_at"] = time.time()
            except ValueError as error:
                stats["images_skipped"] += 1
                _log(logging.WARNING, "image_skipped", **image_fields, reason=str(error))
            except Exception as error:
                # A detector/alignment failure for one image must not discard
                # embeddings already generated from other images.
                stats["images_skipped"] += 1
                _log(logging.WARNING, "image_processing_failed", **image_fields, error_type=type(error).__name__, error_message=str(error))
        if not outputs:
            error_message = "No valid face embeddings could be generated from the uploaded images."
            with jobs_lock:
                jobs[job_id] = {"status": "failed", "error": error_message, **stats, "updated_at": time.time()}
            _log(logging.WARNING, "enrollment_failed_no_embeddings", enrollment_id=job_id, request_id=request_id, **stats)
            return
        with jobs_lock:
            jobs[job_id] = {
                "status": "completed", "model_version": MODEL_VERSION, "dimension": 512,
                "embeddings": outputs, **stats, "updated_at": time.time(),
            }
        _log(logging.INFO, "enrollment_embeddings_generated", enrollment_id=job_id, request_id=request_id, **stats, final_status="completed")
    except Exception:
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": "Face embedding generation failed", **stats, "updated_at": time.time()}
        log.exception("Enrollment embedding generation failed | enrollment_id=%s", job_id)


@app.post("/v1/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    enrollment_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    authorization: Annotated[str | None, Header()] = None,
    x_request_id: Annotated[str | None, Header()] = None,
):
    _authorize(authorization)
    _log(logging.INFO, "enrollment_upload_received", enrollment_id=enrollment_id, request_id=x_request_id, declared_image_count=len(files))
    if not re.fullmatch(r"[1-9]\d*", enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    if not 1 <= len(files) <= 10:
        raise HTTPException(status_code=400, detail="Upload between 1 and 10 images")
    _purge_expired_jobs()
    images: list[tuple[str, bytes, str]] = []
    hashes = set()
    for index, upload in enumerate(files, start=1):
        if upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail=f"Image {index}: use JPG, PNG, or WEBP")
        raw = await upload.read(MAX_IMAGE_BYTES + 1)
        if not raw:
            raise HTTPException(status_code=400, detail=f"Image {index}: file is empty")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"Image {index}: maximum size is 8 MB")
        digest = hashlib.sha256(raw).digest()
        if digest in hashes:
            raise HTTPException(status_code=400, detail=f"Image {index}: duplicate image selected")
        hashes.add(digest)
        images.append((upload.content_type, raw, (upload.filename or "")[:255]))
        await upload.close()
    with jobs_lock:
        jobs[enrollment_id] = {"status": "queued", "updated_at": time.time()}
    background_tasks.add_task(_generate_job, enrollment_id, images, x_request_id)
    _log(logging.INFO, "enrollment_job_accepted", enrollment_id=enrollment_id, request_id=x_request_id, image_count=len(images))
    return {"enrollment_id": enrollment_id, "status": "queued"}


@app.get("/v1/jobs/{enrollment_id}")
def get_job(enrollment_id: str, authorization: Annotated[str | None, Header()] = None):
    _authorize(authorization)
    if not re.fullmatch(r"[1-9]\d*", enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    _purge_expired_jobs()
    with jobs_lock:
        job = jobs.get(enrollment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Enrollment job not found")
    return {"enrollment_id": enrollment_id, **{key: value for key, value in job.items() if key != "updated_at"}}


@app.delete("/v1/jobs/{enrollment_id}")
def delete_job(enrollment_id: str, authorization: Annotated[str | None, Header()] = None):
    _authorize(authorization)
    with jobs_lock:
        jobs.pop(enrollment_id, None)
    return {"deleted": True}


def _cleanup_loop() -> None:
    while True:
        time.sleep(60)
        _purge_expired_jobs()


threading.Thread(target=_cleanup_loop, daemon=True, name="enrollment-job-cleanup").start()
