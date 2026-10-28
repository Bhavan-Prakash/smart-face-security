from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from sqlalchemy.orm import Session
from sqlalchemy import func

from datetime import datetime

from .database import engine, Base, get_db, SessionLocal
from .models import Camera, RecognitionEvent
from .schemas import RecognitionCreate
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


def initialize_cameras():
    db = SessionLocal()

    try:
        cameras = [
            {
                "camera_id": "entrance",
                "name": "Entrance",
                "location": "Main Entrance"
            },
            {
                "camera_id": "floor1",
                "name": "Floor 1",
                "location": "First Floor"
            },
            {
                "camera_id": "floor2",
                "name": "Floor 2",
                "location": "Second Floor"
            }
        ]

        for camera_data in cameras:

            existing_camera = (
                db.query(Camera)
                .filter(
                    Camera.camera_id ==
                    camera_data["camera_id"]
                )
                .first()
            )

            if existing_camera is None:

                camera = Camera(
                    camera_id=camera_data["camera_id"],
                    name=camera_data["name"],
                    location=camera_data["location"]
                )

                db.add(camera)

        db.commit()

    finally:
        db.close()


initialize_cameras()


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Home Security Face Recognition"
)

app.mount(
    "/static",
    StaticFiles(directory="backend/static"),
    name="static"
)

# --------------------------------------------------
# TEMPLATES
# --------------------------------------------------

templates = Jinja2Templates(
    directory="backend/templates"
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    db: Session = Depends(get_db)
):

    cameras = db.query(Camera).all()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "cameras": cameras
        }
    )


# --------------------------------------------------
# CAMERA PAGE
# --------------------------------------------------

@app.get(
    "/camera/{camera_id}",
    response_class=HTMLResponse
)
def camera_page(
    request: Request,
    camera_id: str,
    db: Session = Depends(get_db)
):

    camera = (
        db.query(Camera)
        .filter(Camera.camera_id == camera_id)
        .first()
    )

    if camera is None:
        return HTMLResponse(
            "Camera not found",
            status_code=404
        )

    return templates.TemplateResponse(
        request=request,
        name="camera.html",
        context={
            "request": request,
            "camera": camera
        }
    )


# --------------------------------------------------
# GET CAMERAS
# --------------------------------------------------

@app.get("/api/cameras")
def get_cameras(
    db: Session = Depends(get_db)
):

    cameras = db.query(Camera).all()

    return [
        {
            "camera_id": camera.camera_id,
            "name": camera.name,
            "location": camera.location
        }
        for camera in cameras
    ]


# --------------------------------------------------
# ADD RECOGNITION EVENT
# --------------------------------------------------

@app.post("/api/recognitions")
def add_recognition(
    recognition: RecognitionCreate,
    db: Session = Depends(get_db)
):

    event = RecognitionEvent(
        camera_id=recognition.camera_id,
        person_name=recognition.person_name,
        similarity=recognition.similarity,
        track_id=recognition.track_id,
        recognized_at=datetime.now()
    )

    db.add(event)
    db.commit()
    db.refresh(event)

    return {
        "success": True,
        "id": event.id
    }


# --------------------------------------------------
# GET TODAY'S RECOGNITIONS
# --------------------------------------------------

@app.get(
    "/api/cameras/{camera_id}/recognitions/today"
)
def get_today_recognitions(
    camera_id: str,
    db: Session = Depends(get_db)
):

    today = datetime.now().date()

    events = (
        db.query(RecognitionEvent)
        .filter(
            RecognitionEvent.camera_id == camera_id,
            func.date(
                RecognitionEvent.recognized_at
            ) == today
        )
        .order_by(
            RecognitionEvent.recognized_at.desc()
        )
        .all()
    )

    return [
        {
            "id": event.id,
            "person_name": event.person_name,
            "similarity": event.similarity,
            "track_id": event.track_id,
            "time": event.recognized_at.strftime(
                "%H:%M:%S"
            )
        }
        for event in events
    ]


# --------------------------------------------------
# TEMPORARY CAMERA SETUP
# --------------------------------------------------

@app.post("/api/cameras")
def add_camera(
    camera_id: str,
    name: str,
    location: str = "",
    db: Session = Depends(get_db)
):

    existing = (
        db.query(Camera)
        .filter(Camera.camera_id == camera_id)
        .first()
    )

    if existing:
        return {
            "success": False,
            "message": "Camera already exists"
        }

    camera = Camera(
        camera_id=camera_id,
        name=name,
        location=location
    )

    db.add(camera)
    db.commit()
    db.refresh(camera)

    return {
        "success": True,
        "camera_id": camera.camera_id
    }