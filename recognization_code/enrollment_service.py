"""Private asynchronous embedding-generation API. Keep on a trusted network."""
import hmac
import hashlib
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


@app.get("/health")
def health():
    return {"ok": True, "model_version": MODEL_VERSION}


def _generate_job(job_id: str, images: list[tuple[str, bytes]]) -> None:
    with jobs_lock:
        jobs[job_id] = {"status": "processing", "updated_at": time.time()}
    outputs = []
    try:
        for index, (content_type, raw) in enumerate(images, start=1):
            with inference_lock:
                if content_type not in ALLOWED_TYPES:
                    raise ValueError(f"Image {index}: use JPG, PNG, or WEBP")
                encoded = np.frombuffer(raw, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Image {index}: file is not a supported image")
                height, width = image.shape[:2]
                if width < 64 or height < 64 or width * height > 24_000_000:
                    raise ValueError(f"Image {index}: image dimensions are outside the supported range")
                faces = engine.detect(image)
                if len(faces) == 0:
                    raise ValueError(f"Image {index}: no face detected")
                if len(faces) != 1:
                    raise ValueError(f"Image {index}: exactly one face is required")
                face = faces[0]
                if min(float(face[2]), float(face[3])) < settings.min_face_size:
                    raise ValueError(f"Image {index}: detected face is too small")
                embedding = engine.embed(image, engine.landmarks(face))
                if embedding is None or embedding.shape != (512,) or not np.isfinite(embedding).all():
                    raise ValueError(f"Image {index}: face alignment or embedding generation failed")
                norm = float(np.linalg.norm(embedding))
                if not 0.99 <= norm <= 1.01:
                    raise ValueError(f"Image {index}: embedding failed normalization validation")
                outputs.append(embedding.astype(np.float32).tolist())
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id]["updated_at"] = time.time()
        with jobs_lock:
            jobs[job_id] = {"status": "completed", "model_version": MODEL_VERSION, "dimension": 512, "embeddings": outputs, "updated_at": time.time()}
        log.info("Enrollment embeddings generated | enrollment_id=%s | image_count=%d | embedding_count=%d", job_id, len(images), len(outputs))
    except ValueError as error:
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": str(error), "updated_at": time.time()}
        log.info("Enrollment image validation rejected | enrollment_id=%s | error=%s", job_id, str(error))
    except Exception:
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": "Face embedding generation failed", "updated_at": time.time()}
        log.exception("Enrollment embedding generation failed | enrollment_id=%s", job_id)


@app.post("/v1/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    enrollment_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize(authorization)
    if not re.fullmatch(r"[1-9]\d*", enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    if not 1 <= len(files) <= 10:
        raise HTTPException(status_code=400, detail="Upload between 1 and 10 images")
    _purge_expired_jobs()
    images = []
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
        images.append((upload.content_type, raw))
        await upload.close()
    with jobs_lock:
        jobs[enrollment_id] = {"status": "queued", "updated_at": time.time()}
    background_tasks.add_task(_generate_job, enrollment_id, images)
    log.info("Enrollment job accepted | enrollment_id=%s | image_count=%d", enrollment_id, len(images))
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
