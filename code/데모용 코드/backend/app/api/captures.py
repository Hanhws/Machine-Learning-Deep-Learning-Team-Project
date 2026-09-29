"""현재 프레임 저장(캡처) — 세그멘테이션 오버레이 JPEG + 그 시점 지표 JSON 을 카메라 폴더에 보관."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import get_camera_or_404
from app.db.session import get_db
from app.services import media
from app.services.worker_manager import manager

router = APIRouter(prefix="/cameras", tags=["captures"])
KST = timezone(timedelta(hours=9))


def _dir(camera_id: int) -> Path:
    d = media.camera_dir(camera_id) / "captures"
    d.mkdir(exist_ok=True)
    return d


@router.post("/{camera_id}/captures", status_code=201)
def create_capture(camera_id: int, mode: str = Query("class", pattern="^(class|vehicle|road|none)$"), note: str = "", db: Session = Depends(get_db)):
    get_camera_or_404(db, camera_id)
    w = manager.get(camera_id)
    if not w or not w.is_alive() or w.state.frame is None:
        raise HTTPException(400, "모니터링 중인 프레임이 없습니다")
    data = w.render_jpeg(mode, True)
    name = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    d = _dir(camera_id)
    (d / f"{name}.jpg").write_bytes(data)
    meta = {"name": name, "ts": datetime.now(timezone.utc).isoformat(), "mode": mode, "note": note, "live": w.latest_json()}
    (d / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


@router.get("/{camera_id}/captures")
def list_captures(camera_id: int, db: Session = Depends(get_db)):
    get_camera_or_404(db, camera_id)
    out = []
    for p in sorted(_dir(camera_id).glob("*.json"), reverse=True)[:100]:
        try:
            m = json.loads(p.read_text())
            dirs = m.get("live", {}).get("directions", [])
            overall = next((x for x in dirs if x["direction_index"] == 0), None)
            out.append({"name": m["name"], "ts": m["ts"], "mode": m.get("mode"), "note": m.get("note", ""), "occupancy": overall["occupancy"] if overall else None, "level": overall["level"] if overall else None, "n_vehicles": overall["n_vehicles"] if overall else None, "url": f"/api/cameras/{camera_id}/captures/{m['name']}.jpg"})
        except Exception:
            continue
    return out


@router.get("/{camera_id}/captures/{name}.jpg")
def get_capture(camera_id: int, name: str, db: Session = Depends(get_db)):
    p = _dir(camera_id) / f"{Path(name).name}.jpg"
    if not p.exists():
        raise HTTPException(404, "캡처 없음")
    return FileResponse(p, media_type="image/jpeg")


@router.delete("/{camera_id}/captures/{name}", status_code=204)
def delete_capture(camera_id: int, name: str, db: Session = Depends(get_db)):
    d = _dir(camera_id)
    for ext in (".jpg", ".json"):
        (d / f"{Path(name).name}{ext}").unlink(missing_ok=True)
    return Response(status_code=204)
