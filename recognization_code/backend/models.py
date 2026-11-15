from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

from .database import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)

    camera_id = Column(
        String,
        unique=True,
        nullable=False,
        index=True
    )

    name = Column(
        String,
        nullable=False
    )

    location = Column(
        String,
        nullable=True
    )


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    camera_id = Column(
        String,
        nullable=False,
        index=True
    )

    person_name = Column(
        String,
        nullable=False
    )

    similarity = Column(
        Float,
        nullable=False
    )

    track_id = Column(
        Integer,
        nullable=False
    )

    recognized_at = Column(
        DateTime,
        default=datetime.now,
        index=True
    )