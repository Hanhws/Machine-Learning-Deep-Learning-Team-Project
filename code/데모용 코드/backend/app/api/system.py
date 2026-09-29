from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import CONGESTION_LEVELS, PROJECT_ROOT, get_settings
from app.db.models import Camera, Direction
from app.db.session import get_db
from app.ml.registry import model_status
from app.ml.vehicle_seg import VEHICLE_CLASSES
from app.services import media
from app.services.worker_manager import manager

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def status():
    s = get_settings()
    return {
        "app": s.app_name,
        "its_configured": bool(s.its_api_key),
        "thresholds": s.congestion_thresholds,
        "levels": CONGESTION_LEVELS,
        "vehicle_classes": VEHICLE_CLASSES,
        "infer_fps": s.infer_fps,
        "default_infer_interval_s": s.default_infer_interval_s,
        "models": model_status(),
        "workers": {cid: w.state.status for cid, w in manager._workers.items() if w.is_alive()},
    }


DEMO_VIDEO = PROJECT_ROOT / "data/uploads/sample_highway.mp4"


@router.post("/demo", status_code=201)
def create_demo(db: Session = Depends(get_db)):
    """샘플 영상으로 데모 카메라(양방향 도로 마스크 포함)를 만들고 모니터링을 시작한다."""
    import numpy as np
    import cv2

    if not DEMO_VIDEO.exists():
        raise HTTPException(400, f"샘플 영상이 없습니다: {DEMO_VIDEO}")
    existing = db.query(Camera).filter(Camera.meta["demo"].as_boolean().is_(True)).first() if False else None
    for c in db.query(Camera).all():
        if (c.meta or {}).get("demo"):
            existing = c
            break
    if existing:
        manager.start(existing.id)
        return {"camera_id": existing.id, "created": False}
    frame = media.grab_snapshot(str(DEMO_VIDEO), skip=3)
    h, w = frame.shape[:2]
    cam = Camera(
        name="샘플: 호남선 태인 졸음쉼터",
        source_type="upload",
        video_path=str(DEMO_VIDEO),
        lon=126.95,
        lat=35.65,
        route="호남선",
        region="전북 정읍",
        section="태인 졸음쉼터",
        meta={"demo": True, "note": "샘플 영상(반복 재생). AI-Hub 고속도로 CCTV 프레임"},
        infer_interval_s=get_settings().default_infer_interval_s,
        enabled=True,
    )
    db.add(cam)
    db.flush()
    cam.snapshot_path = str(media.save_snapshot(cam.id, frame))
    cam.frame_width, cam.frame_height = w, h
    sx, sy = w / 1920, h / 1080
    label = np.zeros((h, w), np.uint8)
    left = (np.array([[470, 1080], [1000, 1080], [600, 175], [515, 175]], np.float32) * [sx, sy]).astype(np.int32)
    right = (np.array([[1015, 1080], [1920, 1080], [1920, 900], [650, 175], [610, 175]], np.float32) * [sx, sy]).astype(np.int32)
    cv2.fillPoly(label, [left], 1)
    cv2.fillPoly(label, [right], 2)
    cam.mask_path = str(media.save_mask(cam.id, label))
    cam.directions.append(Direction(index=1, name="서울 방면", color="#2a78d6"))
    cam.directions.append(Direction(index=2, name="광주 방면", color="#eb6834"))
    cam.meta = dict(cam.meta, road_px={"1": int((label == 1).sum()), "2": int((label == 2).sum())}, road_coverage=float((label > 0).mean()))
    db.commit()
    manager.start(cam.id)
    return {"camera_id": cam.id, "created": True}
