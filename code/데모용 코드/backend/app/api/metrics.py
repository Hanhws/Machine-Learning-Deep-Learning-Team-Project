"""점유율 지표: 실시간 전체, 시계열, 지역/노선/구간 집계."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import CONGESTION_LEVELS, congestion_level, get_settings
from app.db.models import Camera, Direction, OccupancySample
from app.db.session import get_db
from app.services.worker_manager import manager

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _cam_brief(c: Camera) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "source_type": c.source_type,
        "lon": c.lon,
        "lat": c.lat,
        "route": c.route,
        "region": c.region,
        "section": c.section,
        "enabled": c.enabled,
        "has_mask": bool(c.mask_path),
        "infer_interval_s": c.infer_interval_s,
        "directions": [
            {
                "index": d.index, "name": d.name, "color": d.color, "road": d.road, "destination": d.destination,
                "heading_deg": d.heading_deg, "heading_source": d.heading_source, "lat": d.lat, "lon": d.lon,
                "measured": bool(((c.meta or {}).get("road_px") or {}).get(str(d.index))),
            }
            for d in c.directions
        ],
    }


@router.get("/live")
def live_all(db: Session = Depends(get_db)):
    """모든 카메라의 최신 상태 (지도/그리드 폴링용)."""
    s = get_settings()
    latest = manager.all_latest()
    cams = db.query(Camera).order_by(Camera.id).all()
    out = []
    for c in cams:
        lv = latest.get(c.id)
        overall = None
        if lv and lv.get("directions"):
            overall = next((d for d in lv["directions"] if d["direction_index"] == 0), None)
        out.append(
            {
                **_cam_brief(c),
                "running": manager.is_running(c.id),
                "live": lv,
                "occupancy": overall["occupancy"] if overall else None,
                "level": overall["level"] if overall else None,
                "n_vehicles": overall["n_vehicles"] if overall else None,
            }
        )
    return {"thresholds": s.congestion_thresholds, "levels": CONGESTION_LEVELS, "cameras": out}


@router.get("/history")
def history(
    camera_id: int,
    direction: int = Query(0, ge=0),
    minutes: int = Query(60, ge=1, le=60 * 24 * 30),
    bucket: int = Query(60, ge=1, le=3600),
    source: str = Query("live", pattern="^(live|file)$"),
    db: Session = Depends(get_db),
):
    """시계열. live → 벽시계 기준(최근 minutes 분), file → 영상 내 시각 기준(전체)."""
    if db.get(Camera, camera_id) is None:
        raise HTTPException(404, "카메라 없음")
    S = OccupancySample
    if source == "live":
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        tkey = (func.floor(func.extract("epoch", S.ts) / bucket) * bucket).label("t")
        q = db.query(
            tkey,
            func.avg(S.occupancy),
            func.max(S.occupancy),
            func.avg(S.n_car + S.n_bus + S.n_truck),
            func.avg(S.n_car),
            func.avg(S.n_bus),
            func.avg(S.n_truck),
            func.sum(S.n_frames),
        ).filter(S.camera_id == camera_id, S.direction_index == direction, S.source == "live", S.ts >= since)
    else:
        tkey = (func.floor(S.video_time / bucket) * bucket).label("t")
        q = db.query(
            tkey,
            func.avg(S.occupancy),
            func.max(S.occupancy),
            func.avg(S.n_car + S.n_bus + S.n_truck),
            func.avg(S.n_car),
            func.avg(S.n_bus),
            func.avg(S.n_truck),
            func.sum(S.n_frames),
        ).filter(S.camera_id == camera_id, S.direction_index == direction, S.source == "file")
    rows = q.group_by(tkey).order_by(tkey).all()
    thr = get_settings().congestion_thresholds
    points = []
    for t, occ, mx, nv, nc, nb, nt, nf in rows:
        points.append(
            {
                "t": datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat() if source == "live" else float(t),
                "occupancy": float(occ or 0),
                "max": float(mx or 0),
                "n_vehicles": float(nv or 0),
                "n_car": float(nc or 0),
                "n_bus": float(nb or 0),
                "n_truck": float(nt or 0),
                "frames": int(nf or 0),
                "level": congestion_level(float(occ or 0), thr),
            }
        )
    return {"camera_id": camera_id, "direction": direction, "source": source, "bucket": bucket, "points": points}


@router.get("/summary")
def summary(
    by: str = Query("route", pattern="^(route|region|section|camera)$"),
    minutes: int = Query(15, ge=1, le=60 * 24 * 7),
    db: Session = Depends(get_db),
):
    """최근 minutes 분 평균 점유율을 지역/노선/구간별로 집계 (방향별 포함)."""
    s = get_settings()
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    S = OccupancySample
    rows = (
        db.query(
            S.camera_id,
            S.direction_index,
            func.avg(S.occupancy),
            func.max(S.occupancy),
            func.avg(S.n_car + S.n_bus + S.n_truck),
            func.count(S.id),
        )
        .filter(S.source == "live", S.ts >= since)
        .group_by(S.camera_id, S.direction_index)
        .all()
    )
    cams = {c.id: c for c in db.query(Camera).all()}
    dir_rows = {(d.camera_id, d.index): d for d in db.query(Direction).all()}
    dir_names = {k: d.name for k, d in dir_rows.items()}
    per_cam: dict[int, dict] = {}
    for cid, d, occ, mx, nv, n in rows:
        c = cams.get(cid)
        if not c:
            continue
        pc = per_cam.setdefault(
            cid,
            {**_cam_brief(c), "overall": None, "directions": []},
        )
        item = {
            "direction_index": d,
            "name": "전체" if d == 0 else dir_names.get((cid, d), f"방향 {d}"),
            "road": None if d == 0 else getattr(dir_rows.get((cid, d)), "road", None),
            "destination": None if d == 0 else getattr(dir_rows.get((cid, d)), "destination", None),
            "heading_deg": None if d == 0 else getattr(dir_rows.get((cid, d)), "heading_deg", None),
            "occupancy": float(occ),
            "max": float(mx),
            "n_vehicles": float(nv or 0),
            "samples": int(n),
            "level": congestion_level(float(occ), s.congestion_thresholds),
        }
        if d == 0:
            pc["overall"] = item
        else:
            pc["directions"].append(item)
    for pc in per_cam.values():
        pc["directions"].sort(key=lambda x: x["direction_index"])

    live = manager.all_latest()
    for cid, pc in per_cam.items():
        lv = live.get(cid)
        pc["live_occupancy"] = None
        if lv and lv.get("directions"):
            o = next((x for x in lv["directions"] if x["direction_index"] == 0), None)
            pc["live_occupancy"] = o["occupancy"] if o else None

    if by == "camera":
        groups = [{"key": pc["name"], "cameras": [pc]} for pc in per_cam.values()]
    else:
        gmap: dict[str, list] = {}
        for pc in per_cam.values():
            gmap.setdefault(pc.get(by) or "(미지정)", []).append(pc)
        groups = [{"key": k, "cameras": v} for k, v in gmap.items()]
    for g in groups:
        occs = [pc["overall"]["occupancy"] for pc in g["cameras"] if pc["overall"]]
        g["occupancy"] = sum(occs) / len(occs) if occs else None
        g["max"] = max(occs) if occs else None
        g["n_cameras"] = len(g["cameras"])
        g["level"] = congestion_level(g["occupancy"], s.congestion_thresholds)
        g["n_vehicles"] = sum(pc["overall"]["n_vehicles"] for pc in g["cameras"] if pc["overall"])
    groups.sort(key=lambda g: -(g["occupancy"] or 0))
    return {"by": by, "minutes": minutes, "thresholds": s.congestion_thresholds, "groups": groups}
