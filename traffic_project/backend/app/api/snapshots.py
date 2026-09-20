"""등록 전 임시 스냅샷 / 파일 업로드."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import get_settings
from app.services import media

router = APIRouter(tags=["snapshots"])

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".webm"}


def snapshot_path(snapshot_id: str) -> Path:
    p = get_settings().snapshots_dir / f"{snapshot_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "스냅샷이 없습니다")
    return p


class SnapshotFromSource(BaseModel):
    source: str  # HLS/RTSP URL 또는 로컬 파일 경로


@router.post("/snapshots/from-source")
def snapshot_from_source(body: SnapshotFromSource):
    try:
        frame = media.grab_snapshot(body.source)
    except Exception as e:
        raise HTTPException(400, f"프레임을 가져오지 못했습니다: {e}")
    sid = uuid.uuid4().hex[:12]
    p = get_settings().snapshots_dir / f"{sid}.jpg"
    cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    h, w = frame.shape[:2]
    return {"snapshot_id": sid, "width": w, "height": h, "url": f"/api/snapshots/{sid}.jpg"}


@router.post("/uploads")
async def upload_media(file: UploadFile = File(...)):
    """영상(mp4 등) 또는 이미지 업로드 → 저장 경로 + 스냅샷."""
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(400, f"지원하지 않는 확장자: {ext}")
    uid = uuid.uuid4().hex[:8]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file.filename).stem)[:60]
    dest = s.uploads_dir / f"{uid}_{safe}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    kind = "image" if ext in IMAGE_EXT else "video"
    try:
        if kind == "image":
            frame = media.load_image(dest)
            info = {"fps": 0, "frames": 1, "width": frame.shape[1], "height": frame.shape[0]}
        else:
            frame = media.grab_snapshot(str(dest), skip=3)
            info = media.video_info(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"파일을 읽을 수 없습니다: {e}")
    sid = uuid.uuid4().hex[:12]
    cv2.imwrite(str(s.snapshots_dir / f"{sid}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return {
        "kind": kind,
        "path": str(dest),
        "filename": dest.name,
        "snapshot_id": sid,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "info": info,
        "url": f"/api/snapshots/{sid}.jpg",
    }


@router.get("/uploads")
def list_uploads():
    s = get_settings()
    out = []
    for p in sorted(s.uploads_dir.iterdir()):
        if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
            out.append({"path": str(p), "filename": p.name, "size": p.stat().st_size})
    return out


@router.get("/snapshots/{snapshot_id}.jpg")
def get_snapshot(snapshot_id: str):
    return FileResponse(snapshot_path(snapshot_id), media_type="image/jpeg")
