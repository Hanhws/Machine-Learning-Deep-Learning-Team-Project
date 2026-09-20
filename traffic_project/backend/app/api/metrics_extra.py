"""지표 확장: 타임라인(지도 재생), 경보 구간, 요일×시간 히트맵, CSV 내보내기."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import congestion_level, get_settings
from app.db.models import Camera, Direction, OccupancySample
from app.db.session import get_db

router = APIRouter(prefix="/metrics", tags=["metrics"])
KST = timezone(timedelta(hours=9))


@router.get("/timeline")
def timeline(
    minutes: int = Query(60, ge=5, le=60 * 24 * 7),
    bucket: int = Query(60, ge=10, le=3600),
    direction: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """모든 카메라의 시간 버킷별 평균 점유율 (지도 시간 슬라이더용)."""
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    S = OccupancySample
    tkey = (func.floor(func.extract("epoch", S.ts) / bucket) * bucket).label("t")
    rows = (
        db.query(S.camera_id, tkey, func.avg(S.occupancy), func.avg(S.n_car + S.n_bus + S.n_truck))
        .filter(S.source == "live", S.direction_index == direction, S.ts >= since)
        .group_by(S.camera_id, tkey)
        .all()
    )
    t0 = int(since.timestamp() // bucket * bucket)
    t1 = int(datetime.now(timezone.utc).timestamp() // bucket * bucket)
    buckets = list(range(t0, t1 + bucket, bucket))
    idx = {t: i for i, t in enumerate(buckets)}
    cams: dict[int, list] = {}
    veh: dict[int, list] = {}
    for cid, t, occ, nv in rows:
        i = idx.get(int(t))
        if i is None:
            continue
        cams.setdefault(cid, [None] * len(buckets))[i] = round(float(occ), 4)
        veh.setdefault(cid, [None] * len(buckets))[i] = round(float(nv or 0), 2)
    return {
        "bucket": bucket,
        "buckets": [datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in buckets],
        "cameras": cams,
        "vehicles": veh,
        "thresholds": get_settings().congestion_thresholds,
    }


@router.get("/alerts")
def alerts(
    minutes: int = Query(60 * 24, ge=5, le=60 * 24 * 30),
    camera_id: int | None = None,
    min_level: int = Query(2, ge=1, le=3),  # 1=서행 2=지체 3=정체 이상
    min_duration: int = Query(30, ge=5, le=3600),  # 초
    db: Session = Depends(get_db),
):
    """임계값 이상이 min_duration 초 이상 이어진 구간(에피소드)."""
    s = get_settings()
    thr = s.congestion_thresholds[min_level - 1]
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    S = OccupancySample
    q = db.query(S.camera_id, S.direction_index, S.ts, S.occupancy, S.n_car + S.n_bus + S.n_truck).filter(S.source == "live", S.ts >= since)
    if camera_id is not None:
        q = q.filter(S.camera_id == camera_id)
    rows = q.order_by(S.camera_id, S.direction_index, S.ts).all()
    cams = {c.id: c.name for c in db.query(Camera).all()}
    dnames = {(d.camera_id, d.index): d.name for d in db.query(Direction).all()}
    episodes = []
    cur = None
    gap = timedelta(seconds=max(s.sample_write_interval * 3, 20))

    def close(c):
        dur = (c["end"] - c["start"]).total_seconds()
        if dur >= min_duration and c["n"] >= 2:
            episodes.append(
                {
                    "camera_id": c["cid"],
                    "camera_name": cams.get(c["cid"], str(c["cid"])),
                    "direction_index": c["d"],
                    "direction_name": "전체" if c["d"] == 0 else dnames.get((c["cid"], c["d"]), f"방향 {c['d']}"),
                    "start": c["start"].isoformat(),
                    "end": c["end"].isoformat(),
                    "duration_s": int(dur),
                    "peak": round(c["peak"], 4),
                    "mean": round(c["sum"] / c["n"], 4),
                    "level": congestion_level(c["peak"], s.congestion_thresholds),
                    "n_vehicles": round(c["veh"] / c["n"], 1),
                    "ongoing": (datetime.now(timezone.utc) - c["end"]) < gap,
                }
            )

    for cid, d, ts, occ, nv in rows:
        key = (cid, d)
        above = occ >= thr
        if cur and (cur["key"] != key or not above or ts - cur["end"] > gap):
            close(cur)
            cur = None
        if above:
            if cur is None:
                cur = {"key": key, "cid": cid, "d": d, "start": ts, "end": ts, "peak": occ, "sum": 0.0, "n": 0, "veh": 0.0}
            cur["end"] = ts
            cur["peak"] = max(cur["peak"], occ)
            cur["sum"] += occ
            cur["n"] += 1
            cur["veh"] += float(nv or 0)
    if cur:
        close(cur)
    episodes.sort(key=lambda e: e["start"], reverse=True)
    return {"threshold": thr, "min_level": min_level, "count": len(episodes), "episodes": episodes[:200]}


@router.get("/heatmap")
def heatmap(camera_id: int, days: int = Query(7, ge=1, le=90), direction: int = Query(0, ge=0), db: Session = Depends(get_db)):
    """요일(0=월)×시간(0~23) 평균 점유율 (KST)."""
    if db.get(Camera, camera_id) is None:
        raise HTTPException(404, "카메라 없음")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    S = OccupancySample
    rows = db.query(S.ts, S.occupancy).filter(S.camera_id == camera_id, S.direction_index == direction, S.source == "live", S.ts >= since).all()
    acc = [[[0.0, 0] for _ in range(24)] for _ in range(7)]
    for ts, occ in rows:
        k = ts.astimezone(KST)
        a = acc[k.weekday()][k.hour]
        a[0] += occ
        a[1] += 1
    grid = [[(round(a[0] / a[1], 4) if a[1] else None) for a in row] for row in acc]
    return {"camera_id": camera_id, "direction": direction, "days": days, "weekdays": ["월", "화", "수", "목", "금", "토", "일"], "grid": grid}


def _csv_response(rows: list[list], header: list[str], filename: str) -> StreamingResponse:
    buf = io.StringIO()
    buf.write("﻿")  # 엑셀 한글 BOM
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/history.csv")
def history_csv(camera_id: int, minutes: int = Query(60 * 24, ge=1, le=60 * 24 * 90), source: str = Query("live", pattern="^(live|file)$"), db: Session = Depends(get_db)):
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(404, "카메라 없음")
    dnames = {d.index: d.name for d in cam.directions}
    S = OccupancySample
    q = db.query(S).filter(S.camera_id == camera_id, S.source == source)
    if source == "live":
        q = q.filter(S.ts >= datetime.now(timezone.utc) - timedelta(minutes=minutes))
    rows = []
    for r in q.order_by(S.ts, S.direction_index).all():
        rows.append(
            [
                r.ts.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S"),
                r.video_time if r.video_time is not None else "",
                r.direction_index,
                "전체" if r.direction_index == 0 else dnames.get(r.direction_index, r.direction_index),
                round(r.occupancy, 5),
                congestion_level(r.occupancy, get_settings().congestion_thresholds),
                round(r.n_car + r.n_bus + r.n_truck, 2),
                round(r.n_car, 2),
                round(r.n_bus, 2),
                round(r.n_truck, 2),
                int(r.vehicle_px),
                int(r.road_px),
                r.n_frames,
            ]
        )
    header = ["time_kst", "video_time_s", "direction_index", "direction", "occupancy", "level", "n_vehicles", "n_car", "n_bus", "n_truck", "vehicle_px", "road_px", "n_frames"]
    return _csv_response(rows, header, f"camera{camera_id}_{source}_{datetime.now(KST):%Y%m%d_%H%M}.csv")


@router.get("/summary.csv")
def summary_csv(by: str = Query("route", pattern="^(route|region|section|camera)$"), minutes: int = Query(15, ge=1, le=60 * 24 * 7), db: Session = Depends(get_db)):
    from app.api.metrics import summary

    data = summary(by=by, minutes=minutes, db=db)
    rows = []
    for g in data["groups"]:
        for c in g["cameras"]:
            o = c["overall"] or {}
            rows.append([g["key"], c["name"], "전체", round(o.get("occupancy", 0), 5), o.get("level", ""), round(o.get("max", 0), 5), round(o.get("n_vehicles", 0), 2), o.get("samples", 0)])
            for d in c["directions"]:
                rows.append([g["key"], c["name"], d["name"], round(d["occupancy"], 5), d["level"], round(d["max"], 5), round(d["n_vehicles"], 2), d["samples"]])
    return _csv_response(rows, [by, "camera", "direction", "mean_occupancy", "level", "max_occupancy", "mean_vehicles", "samples"], f"summary_{by}_{minutes}m.csv")
