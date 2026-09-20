"""SAM3 기반 도로 segmentation (등록 편집기용)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_camera_or_404
from app.api.snapshots import snapshot_path
from app.db.session import get_db
from app.ml.registry import get_road_segmenter
from app.schemas import MaskOut, SegmentPromptIn, SegmentTextIn
from app.services import media
from app.utils.images import mask_to_b64

router = APIRouter(prefix="/segment", tags=["segment"])


def _load_image(db: Session, camera_id: int | None, snapshot_id: str | None) -> np.ndarray:
    if snapshot_id:
        return media.load_image(snapshot_path(snapshot_id))
    if camera_id is not None:
        cam = get_camera_or_404(db, camera_id)
        if not cam.snapshot_path:
            raise HTTPException(400, "카메라 스냅샷이 없습니다")
        return media.load_image(cam.snapshot_path)
    raise HTTPException(400, "camera_id 또는 snapshot_id 가 필요합니다")


def _out(mask: np.ndarray) -> MaskOut:
    h, w = mask.shape
    return MaskOut(
        mask_png_base64=mask_to_b64(mask),
        coverage=float(mask.mean()),
        width=w,
        height=h,
        backend=get_road_segmenter().backend_name,
    )


@router.get("/status")
def segment_status():
    return get_road_segmenter().status()


@router.post("/text", response_model=MaskOut)
def segment_text(body: SegmentTextIn, db: Session = Depends(get_db)):
    img = _load_image(db, body.camera_id, body.snapshot_id)
    seg = get_road_segmenter()
    if not seg.has_sam3:
        raise HTTPException(
            400, "텍스트 프롬프트는 SAM3 가중치가 필요합니다. scripts/download_sam3.py 로 받거나 점/박스 프롬프트를 사용하세요."
        )
    try:
        mask = seg.segment_text(img, body.text, body.conf)
    except Exception as e:
        raise HTTPException(500, f"SAM3 텍스트 추론 실패: {e}")
    return _out(mask)


@router.post("/prompt", response_model=MaskOut)
def segment_prompt(body: SegmentPromptIn, db: Session = Depends(get_db)):
    img = _load_image(db, body.camera_id, body.snapshot_id)
    if not body.points and not body.boxes:
        raise HTTPException(400, "points 또는 boxes 가 필요합니다")
    try:
        mask = get_road_segmenter().segment_prompt(img, body.points, body.labels, body.boxes)
    except Exception as e:
        raise HTTPException(500, f"SAM 프롬프트 추론 실패: {e}")
    return _out(mask)
