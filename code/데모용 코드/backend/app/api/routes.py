"""노선 선 · 방면 해석 API (전국 현황 지도용)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Camera
from app.db.session import get_db
from app.services import routes as rs

router = APIRouter(prefix="/routes", tags=["routes"])


class ResolveIn(BaseModel):
    route: str | None = None
    lat: float | None = None
    lon: float | None = None
    road_type: str | None = None
    destinations: list[str] = []


@router.post("/resolve")
def resolve(body: ResolveIn):
    """방면(목적지) 이름들 → 진행 방향. 등록·방면 편집 화면의 '자동 계산'."""
    return rs.resolve_many(body.route, body.lat, body.lon, body.destinations, body.road_type)


@router.get("/near")
def near(route: str, lat: float, lon: float, road_type: str | None = None, span_m: float = Query(6000, ge=500, le=30000)):
    """카메라 주변 노선 선 (방면 편집기 지도용). 선이 증가하는 방향 = sign +1."""
    found = rs.find_chain(route, lat, lon, road_type)
    if not found:
        raise HTTPException(404, f"'{route}' 노선 선을 찾지 못했습니다")
    ch, s, d = found
    coords = ch.slice(s - span_m, s + span_m)
    return {"route": route, "distance_m": round(d), "coords": [[round(a, 6), round(b, 6)] for a, b in coords], "heading_forward": round(ch.heading_at(s, 1), 1)}


@router.get("/view")
def view(with_lines: bool = Query(True), db: Session = Depends(get_db)):
    """전국 현황용: 등록 카메라가 있는 노선의 선(회색 바탕) + 카메라·방면별 칠할 구간."""
    cams = db.query(Camera).all()
    try:
        segs = rs.camera_segments(cams)
    except Exception as e:
        raise HTTPException(502, f"노선 정보를 만들지 못했습니다: {e}")
    lines = {}
    if with_lines:
        routes = sorted({(d.road or c.route) for c in cams for d in c.directions if (d.road or c.route)})
        lines = rs.route_lines(routes, {c.route: c.its_road_type for c in cams if c.route})
    return {"lines": lines, "segments": segs}
