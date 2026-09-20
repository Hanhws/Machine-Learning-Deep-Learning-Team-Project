"""응용 페이지: 카메라 묶음(그룹) 분석. 기본 프리셋은 '한강 대교'."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, literal_column
from sqlalchemy.orm import Session

from app.config import congestion_level, get_settings
from app.db.models import AppGroup, AppGroupCamera, Camera, Direction, OccupancySample
from app.db.session import get_db
from app.schemas import GroupCreate, GroupOut, GroupUpdate
from app.services.worker_manager import manager

router = APIRouter(prefix="/apps", tags=["apps"])

# 한강 교량 (서쪽 → 동쪽). 좌표는 교량 중앙 근사값이며 ITS CCTV 탐색 중심으로 쓴다.
HAN_RIVER_BRIDGES = [
    {"name": "일산대교", "lat": 37.6648, "lon": 126.7418},
    {"name": "김포대교", "lat": 37.6215, "lon": 126.7860},
    {"name": "행주대교", "lat": 37.5985, "lon": 126.8195},
    {"name": "방화대교", "lat": 37.5850, "lon": 126.8300},
    {"name": "가양대교", "lat": 37.5750, "lon": 126.8560},
    {"name": "성산대교", "lat": 37.5525, "lon": 126.8905},
    {"name": "양화대교", "lat": 37.5430, "lon": 126.9020},
    {"name": "서강대교", "lat": 37.5410, "lon": 126.9250},
    {"name": "마포대교", "lat": 37.5340, "lon": 126.9400},
    {"name": "원효대교", "lat": 37.5280, "lon": 126.9490},
    {"name": "한강대교", "lat": 37.5170, "lon": 126.9580},
    {"name": "동작대교", "lat": 37.5090, "lon": 126.9800},
    {"name": "반포대교", "lat": 37.5130, "lon": 126.9960},
    {"name": "한남대교", "lat": 37.5260, "lon": 127.0110},
    {"name": "동호대교", "lat": 37.5350, "lon": 127.0200},
    {"name": "성수대교", "lat": 37.5375, "lon": 127.0330},
    {"name": "영동대교", "lat": 37.5310, "lon": 127.0440},
    {"name": "청담대교", "lat": 37.5265, "lon": 127.0590},
    {"name": "잠실대교", "lat": 37.5215, "lon": 127.0850},
    {"name": "올림픽대교", "lat": 37.5270, "lon": 127.1030},
    {"name": "천호대교", "lat": 37.5395, "lon": 127.1120},
    {"name": "광진교", "lat": 37.5450, "lon": 127.1130},
    {"name": "구리암사대교", "lat": 37.5590, "lon": 127.1230},
    {"name": "강동대교", "lat": 37.5530, "lon": 127.1480},
    {"name": "미사대교", "lat": 37.5700, "lon": 127.1860},
    {"name": "팔당대교", "lat": 37.5340, "lon": 127.2400},
]


@router.get("/presets/han-river")
def han_river_preset():
    return {"name": "한강 대교", "bridges": HAN_RIVER_BRIDGES}


def _group_out(g: AppGroup) -> GroupOut:
    out = GroupOut.model_validate(g)
    for m_out, m in zip(out.members, g.members):
        m_out.camera_name = m.camera.name if m.camera else None
    return out


@router.get("/groups", response_model=list[GroupOut])
def list_groups(db: Session = Depends(get_db)):
    return [_group_out(g) for g in db.query(AppGroup).order_by(AppGroup.id).all()]


@router.post("/groups", response_model=GroupOut, status_code=201)
def create_group(body: GroupCreate, db: Session = Depends(get_db)):
    g = AppGroup(name=body.name, kind=body.kind, description=body.description)
    for m in body.members:
        if db.get(Camera, m.camera_id) is None:
            raise HTTPException(400, f"카메라 {m.camera_id} 없음")
        g.members.append(AppGroupCamera(camera_id=m.camera_id, label=m.label, order=m.order))
    db.add(g)
    db.commit()
    db.refresh(g)
    return _group_out(g)


@router.get("/groups/{group_id}", response_model=GroupOut)
def get_group(group_id: int, db: Session = Depends(get_db)):
    g = db.get(AppGroup, group_id)
    if not g:
        raise HTTPException(404, "그룹 없음")
    return _group_out(g)


@router.patch("/groups/{group_id}", response_model=GroupOut)
def update_group(group_id: int, body: GroupUpdate, db: Session = Depends(get_db)):
    g = db.get(AppGroup, group_id)
    if not g:
        raise HTTPException(404, "그룹 없음")
    if body.name is not None:
        g.name = body.name
    if body.description is not None:
        g.description = body.description
    if body.members is not None:
        g.members.clear()
        db.flush()
        for m in body.members:
            g.members.append(AppGroupCamera(camera_id=m.camera_id, label=m.label, order=m.order))
    db.commit()
    db.refresh(g)
    return _group_out(g)


@router.delete("/groups/{group_id}", status_code=204)
def delete_group(group_id: int, db: Session = Depends(get_db)):
    g = db.get(AppGroup, group_id)
    if g:
        db.delete(g)
        db.commit()
    return Response(status_code=204)


@router.get("/groups/{group_id}/report")
def group_report(
    group_id: int,
    minutes: int = Query(60, ge=1, le=60 * 24 * 30),
    bucket: int = Query(300, ge=30, le=3600),
    db: Session = Depends(get_db),
):
    """그룹 리포트: 멤버별 현재/평균/최대 점유율, 방향별 비교, 시간대(0~23시) 프로필, 시계열."""
    g = db.get(AppGroup, group_id)
    if not g:
        raise HTTPException(404, "그룹 없음")
    s = get_settings()
    thr = s.congestion_thresholds
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    S = OccupancySample
    cam_ids = [m.camera_id for m in g.members]
    dir_names = {(d.camera_id, d.index): d.name for d in db.query(Direction).filter(Direction.camera_id.in_(cam_ids or [0])).all()}
    live = manager.all_latest()

    # 멤버별 창 평균
    agg = (
        db.query(S.camera_id, S.direction_index, func.avg(S.occupancy), func.max(S.occupancy), func.avg(S.n_car + S.n_bus + S.n_truck), func.count(S.id))
        .filter(S.camera_id.in_(cam_ids or [0]), S.source == "live", S.ts >= since)
        .group_by(S.camera_id, S.direction_index)
        .all()
    )
    agg_map: dict[tuple[int, int], tuple] = {(cid, d): (occ, mx, nv, n) for cid, d, occ, mx, nv, n in agg}

    # 시간대 프로필 (전체 기간, 방향 0)
    hour_expr = func.extract("hour", func.timezone(literal_column("'Asia/Seoul'"), S.ts)).label("hr")
    hour_rows = (
        db.query(S.camera_id, hour_expr, func.avg(S.occupancy))
        .filter(S.camera_id.in_(cam_ids or [0]), S.source == "live", S.direction_index == 0)
        .group_by(S.camera_id, hour_expr)
        .all()
    )
    hourly: dict[int, dict[int, float]] = {}
    for cid, hr, occ in hour_rows:
        hourly.setdefault(cid, {})[int(hr)] = float(occ)

    # 시계열 (창 내, 방향 0)
    tkey = (func.floor(func.extract("epoch", S.ts) / bucket) * bucket).label("t")
    ts_rows = (
        db.query(S.camera_id, tkey, func.avg(S.occupancy))
        .filter(S.camera_id.in_(cam_ids or [0]), S.source == "live", S.direction_index == 0, S.ts >= since)
        .group_by(S.camera_id, tkey)
        .order_by(tkey)
        .all()
    )
    series: dict[int, list] = {}
    for cid, t, occ in ts_rows:
        series.setdefault(cid, []).append({"t": datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(), "occupancy": float(occ)})

    members = []
    for m in g.members:
        c = m.camera
        if not c:
            continue
        lv = live.get(c.id)
        live_overall = None
        live_dirs = []
        if lv and lv.get("directions"):
            live_overall = next((d for d in lv["directions"] if d["direction_index"] == 0), None)
            live_dirs = [d for d in lv["directions"] if d["direction_index"] != 0]
        o = agg_map.get((c.id, 0))
        dirs = []
        for d in c.directions:
            a = agg_map.get((c.id, d.index))
            lvd = next((x for x in live_dirs if x["direction_index"] == d.index), None)
            dirs.append(
                {
                    "index": d.index,
                    "name": d.name,
                    "color": d.color,
                    "mean": float(a[0]) if a else None,
                    "max": float(a[1]) if a else None,
                    "n_vehicles": float(a[2]) if a else None,
                    "live": lvd["occupancy"] if lvd else None,
                    "level": congestion_level(float(a[0]), thr) if a else None,
                }
            )
        members.append(
            {
                "camera_id": c.id,
                "label": m.label or c.name,
                "name": c.name,
                "lon": c.lon,
                "lat": c.lat,
                "running": manager.is_running(c.id),
                "live": live_overall["occupancy"] if live_overall else None,
                "live_level": live_overall["level"] if live_overall else None,
                "mean": float(o[0]) if o else None,
                "max": float(o[1]) if o else None,
                "n_vehicles": float(o[2]) if o else None,
                "samples": int(o[3]) if o else 0,
                "level": congestion_level(float(o[0]), thr) if o else None,
                "directions": dirs,
                "hourly": [hourly.get(c.id, {}).get(h) for h in range(24)],
                "series": series.get(c.id, []),
            }
        )
    means = [m["mean"] for m in members if m["mean"] is not None]
    ranked = sorted([m for m in members if m["mean"] is not None], key=lambda m: -m["mean"])
    return {
        "group": _group_out(g).model_dump(mode="json"),
        "minutes": minutes,
        "bucket": bucket,
        "thresholds": thr,
        "overall": {
            "mean": sum(means) / len(means) if means else None,
            "level": congestion_level(sum(means) / len(means), thr) if means else None,
            "most_congested": ranked[0]["label"] if ranked else None,
            "least_congested": ranked[-1]["label"] if ranked else None,
            "n_members": len(members),
            "n_running": sum(1 for m in members if m["running"]),
        },
        "members": members,
    }


@router.get("/groups/{group_id}/daily")
def group_daily(group_id: int, days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    """일별 리포트: 멤버별 날짜별 평균/최대/피크 시간(KST), 방향별 평균."""
    from datetime import timedelta as _td

    g = db.get(AppGroup, group_id)
    if not g:
        raise HTTPException(404, "그룹 없음")
    thr = get_settings().congestion_thresholds
    kst = timezone(_td(hours=9))
    since = datetime.now(timezone.utc) - _td(days=days)
    cam_ids = [m.camera_id for m in g.members]
    S = OccupancySample
    rows = (
        db.query(S.camera_id, S.direction_index, S.ts, S.occupancy, S.n_car + S.n_bus + S.n_truck)
        .filter(S.camera_id.in_(cam_ids or [0]), S.source == "live", S.ts >= since)
        .all()
    )
    acc: dict[tuple[int, int, str], dict] = {}
    hours: dict[tuple[int, str], dict[int, list]] = {}
    for cid, d, ts, occ, nv in rows:
        k = ts.astimezone(kst)
        day = k.strftime("%Y-%m-%d")
        a = acc.setdefault((cid, d, day), {"sum": 0.0, "n": 0, "max": 0.0, "veh": 0.0, "jam": 0})
        a["sum"] += occ
        a["n"] += 1
        a["max"] = max(a["max"], occ)
        a["veh"] += float(nv or 0)
        if occ >= thr[-1]:
            a["jam"] += 1
        if d == 0:
            hours.setdefault((cid, day), {}).setdefault(k.hour, []).append(occ)
    labels = {m.camera_id: (m.label or (m.camera.name if m.camera else str(m.camera_id))) for m in g.members}
    dnames = {(d.camera_id, d.index): d.name for d in db.query(Direction).filter(Direction.camera_id.in_(cam_ids or [0])).all()}
    days_sorted = sorted({k[2] for k in acc})
    members = []
    for cid in cam_ids:
        per_day = []
        for day in days_sorted:
            a = acc.get((cid, 0, day))
            if not a:
                per_day.append({"day": day, "mean": None})
                continue
            hh = hours.get((cid, day), {})
            peak_hour = max(hh, key=lambda h: sum(hh[h]) / len(hh[h])) if hh else None
            dirs = []
            for (c2, d), name in dnames.items():
                if c2 != cid:
                    continue
                ad = acc.get((cid, d, day))
                dirs.append({"index": d, "name": name, "mean": ad["sum"] / ad["n"] if ad else None})
            per_day.append(
                {
                    "day": day,
                    "mean": a["sum"] / a["n"],
                    "max": a["max"],
                    "level": congestion_level(a["sum"] / a["n"], thr),
                    "peak_hour": peak_hour,
                    "jam_ratio": a["jam"] / a["n"],
                    "n_vehicles": a["veh"] / a["n"],
                    "samples": a["n"],
                    "directions": sorted(dirs, key=lambda x: x["index"]),
                }
            )
        members.append({"camera_id": cid, "label": labels.get(cid), "days": per_day})
    return {"group_id": group_id, "days": days_sorted, "members": members, "thresholds": thr}
