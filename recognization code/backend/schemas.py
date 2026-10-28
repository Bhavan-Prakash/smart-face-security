from pydantic import BaseModel


class RecognitionCreate(BaseModel):
    camera_id: str
    person_name: str
    similarity: float
    track_id: int