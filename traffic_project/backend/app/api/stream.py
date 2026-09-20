"""실시간 출력: MJPEG 스트림, 단일 프레임, 최신 지표."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_camera_or_404
from app.db.session import SessionLocal, get_db
from app.ml.render import encode_jpeg, render_overlay
from app.services import media
from app.services.worker_manager import manager

router = APIRouter(prefix="/stream", tags=["stream"])
MODES = "^(class|vehicle|road|none)$"


def _fallback_frame(db: Session, camera_id: int, mode: str) -> bytes:
    cam = get_camera_or_404(db, camera_id)
    if not cam.snapshot_path:
        raise HTTPException(404, "프레임 없음")
    img = media.load_image(cam.snapshot_path)
    road = media.load_mask(cam.mask_path) if cam.mask_path else None
    return encode_jpeg(render_overlay(img, None, road, mode="road" if mode != "none" else "none"), 75)


@router.get("/{camera_id}/frame.jpg")
def frame(camera_id: int, mode: str = Query("class", pattern=MODES), hud: bool = True, db: Session = Depends(get_db)):
    w = manager.get(camera_id)
    data = w.render_jpeg(mode, hud) if w and w.is_alive() else None
    if data is None:
        data = _fallback_frame(db, camera_id, mode)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/{camera_id}/live")
def live(camera_id: int, db: Session = Depends(get_db)):
    w = manager.get(camera_id)
    if not w or not w.is_alive():
        get_camera_or_404(db, camera_id)
        return {"camera_id": camera_id, "status": "stopped", "directions": []}
    return w.latest_json()


@router.get("/{camera_id}/mjpeg")
async def mjpeg(
    camera_id: int,
    mode: str = Query("class", pattern=MODES),
    hud: bool = True,
    max_fps: float = Query(8.0, ge=0.5, le=30),
    db: Session = Depends(get_db),
):
    get_camera_or_404(db, camera_id)
    boundary = "frame"

    async def gen():
        last_seq = -1
        last_send = 0.0
        min_interval = 1.0 / max_fps
        while True:
            w = manager.get(camera_id)
            now = time.time()
            if w and w.is_alive() and w.state.seq != last_seq and now - last_send >= min_interval:
                data = w.render_jpeg(mode, hud)
                if data:
                    last_seq, last_send = w.state.seq, now
                    yield b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            elif not (w and w.is_alive()):
                # 워커가 없으면 정지 프레임을 1초마다
                if now - last_send >= 1.0:
                    db2 = SessionLocal()
                    try:
                        data = _fallback_frame(db2, camera_id, mode)
                    except Exception:
                        break
                    finally:
                        db2.close()
                    last_send = now
                    yield b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")
