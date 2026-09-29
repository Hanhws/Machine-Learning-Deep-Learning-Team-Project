"""ITS CCTV Open API 프록시 (프론트가 키를 직접 다루지 않도록)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.services import its_client

router = APIRouter(prefix="/its", tags=["its"])


@router.get("/status")
def its_status():
    s = get_settings()
    return {"configured": bool(s.its_api_key), "base_url": s.its_base_url}


@router.get("/search")
def its_search(
    road_type: str = Query("all", pattern="^(ex|its|all)$"),
    cctv_type: str = Query("1", pattern="^[1-5]$"),
    min_x: float = Query(...),
    max_x: float = Query(...),
    min_y: float = Query(...),
    max_y: float = Query(...),
):
    try:
        bbox = dict(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        if road_type == "all":
            items = its_client.search_both(cctv_type, **bbox)
        else:
            items = its_client.search(road_type, cctv_type, **bbox)
    except its_client.ItsError as e:
        raise HTTPException(502, str(e))
    except Exception as e:  # 네트워크 등
        raise HTTPException(502, f"ITS 호출 실패: {e}")
    return {"count": len(items), "items": [dict(it.to_dict(), route=its_client.parse_route(it.name)) for it in items]}


@router.get("/suggest-heading")
def suggest_heading(
    lat: float = Query(...),
    lon: float = Query(...),
    route: str | None = Query(None, description="노선 이름 (예: 경부선). 같은 노선의 이웃 CCTV 만 사용"),
    road_type: str = Query("ex", pattern="^(ex|its)$"),
    radius_deg: float = Query(0.06, ge=0.01, le=0.3),
):
    """이웃 CCTV 좌표로 도로가 뻗은 축(0~180°)을 추정한다.

    같은 노선의 가까운 CCTV 들은 도로를 따라 늘어서 있으므로, 현재 지점 + 이웃들의 좌표에
    주성분(PCA)을 적용하면 도로 축 방향이 나온다. 진행 방향은 이 축과 그 반대(+180°) 두 개다.
    """
    import math

    try:
        items = its_client.search(road_type, "1", lon - radius_deg, lon + radius_deg, lat - radius_deg, lat + radius_deg)
    except its_client.ItsError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(502, f"ITS 호출 실패: {e}")

    def dist(it):
        return (it.lat - lat) ** 2 + ((it.lon - lon) * math.cos(math.radians(lat))) ** 2

    cands = [it for it in items if dist(it) > 1e-10]
    same = [it for it in cands if route and its_client.parse_route(it.name) == route]
    used_route = bool(same)
    pool = sorted(same if same else cands, key=dist)[:4]
    if not pool:
        raise HTTPException(404, "주변에 이웃 CCTV 가 없어 도로 축을 추정할 수 없습니다. 지도에서 직접 지정하세요.")

    ky = 110540.0
    kx = 111320.0 * math.cos(math.radians(lat))
    pts = [(0.0, 0.0)] + [((it.lon - lon) * kx, (it.lat - lat) * ky) for it in pool]
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)  # 주축 (x=동, y=북 기준 각도)
    vx, vy = math.cos(theta), math.sin(theta)
    axis = math.degrees(math.atan2(vx, vy)) % 180.0  # 방위각 (북=0, 시계방향), 0~180
    # 축의 신뢰도: 주축 분산 비율 (1 에 가까울수록 일직선)
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0))
    linearity = lam1 / tr if tr > 0 else 0.0
    return {
        "axis_deg": round(axis, 1),
        "opposite_deg": round((axis + 180) % 360, 1),
        "linearity": round(linearity, 3),
        "used_route": used_route,
        "route": route if used_route else None,
        "neighbors": [{"name": it.name, "lat": it.lat, "lon": it.lon} for it in pool],
    }
