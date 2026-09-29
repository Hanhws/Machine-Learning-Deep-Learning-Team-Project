from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import Camera


def get_camera_or_404(db: Session, camera_id: int) -> Camera:
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(404, "카메라를 찾을 수 없습니다")
    return cam
