"""노선 선(line) 과 방면 → 진행 방향 계산.

ITS 는 노선마다 CCTV 를 1~2km 간격으로 촘촘히 두므로, 같은 노선 이름의 CCTV 좌표를 순서대로 이으면
그 노선의 선이 된다(곡선은 각지게 나오지만 전국 그림에는 충분하다).

- 노선 선 위에서 카메라 위치와 방면(목적지) 위치를 각각 투영해, 목적지가 선의 앞쪽이면 +1, 뒤쪽이면 -1.
  진행 방향(heading) 은 카메라 지점의 선 접선을 그 부호 방향으로 본 방위각이다.
- 지도에는 각 카메라·방면의 측정값을 카메라 앞뒤 일정 구간(다음 측정 카메라와의 중간까지, 최대 CAP_M)에 칠한다.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.services import geocode, its_client

log = logging.getLogger(__name__)

TTL_S = 24 * 3600
DEDUP_M = 60.0  # 같은 지점에 여러 CCTV(상·하행 카메라) → 하나로
GAP_M = 15000.0  # 이보다 멀면 선을 끊는다
CAP_M = 8000.0  # 측정값을 칠하는 최대 거리 (카메라 앞뒤 각각)
MIN_M = 800.0

_lock = threading.Lock()
_points: dict[str, tuple[float, list[dict]]] = {}  # road_type → (fetched_at, [{name, lat, lon}])
_chains: dict[tuple[str, str], list["Chain"]] = {}


def _hav(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


@dataclass
class Chain:
    route: str
    pts: list[tuple[float, float]]
    cum: list[float] = field(default_factory=list)  # 누적 거리(m)

    def __post_init__(self):
        self.cum = [0.0]
        for i in range(1, len(self.pts)):
            self.cum.append(self.cum[-1] + _hav(self.pts[i - 1], self.pts[i]))

    @property
    def length(self) -> float:
        return self.cum[-1]

    def project(self, p: tuple[float, float]) -> tuple[float, float]:
        """p 를 선에 투영 → (선 위 거리 s, p 와 선의 거리 m)."""
        best = (0.0, float("inf"))
        kx = math.cos(math.radians(p[0])) * 111320.0
        ky = 110540.0
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ax, ay = (a[1] - p[1]) * kx, (a[0] - p[0]) * ky
            bx, by = (b[1] - p[1]) * kx, (b[0] - p[0]) * ky
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
            d = math.hypot(ax + t * dx, ay + t * dy)
            if d < best[1]:
                best = (self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), d)
        if len(self.pts) == 1:
            best = (0.0, _hav(p, self.pts[0]))
        return best

    def point_at(self, s: float) -> tuple[float, float]:
        s = max(0.0, min(self.length, s))
        for i in range(len(self.pts) - 1):
            if self.cum[i + 1] >= s:
                seg = self.cum[i + 1] - self.cum[i]
                t = 0.0 if seg == 0 else (s - self.cum[i]) / seg
                a, b = self.pts[i], self.pts[i + 1]
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return self.pts[-1]

    def slice(self, s0: float, s1: float) -> list[tuple[float, float]]:
        s0, s1 = max(0.0, min(s0, s1)), min(self.length, max(s0, s1))
        out = [self.point_at(s0)]
        out += [p for p, c in zip(self.pts, self.cum) if s0 < c < s1]
        out.append(self.point_at(s1))
        return out

    def heading_at(self, s: float, sign: int, span: float = 400.0) -> float:
        a = self.point_at(s - span)
        b = self.point_at(s + span)
        h = _bearing(a, b)
        return h if sign >= 0 else (h + 180) % 360


# ---------------- ITS 좌표 적재 ----------------
def _disk_path(road_type: str):
    return get_settings().data_dir / f"its_points_{road_type}.json"


def _load_points(road_type: str) -> list[dict]:
    now = time.time()
    hit = _points.get(road_type)
    if hit and now - hit[0] < TTL_S:
        return hit[1]
    p = _disk_path(road_type)
    try:
        j = json.loads(p.read_text())
        if now - j["fetched_at"] < TTL_S:
            _points[road_type] = (j["fetched_at"], j["points"])
            return j["points"]
    except Exception:
        pass
    items = its_client.search(road_type, "1", 124.5, 131.0, 33.0, 38.7)
    pts = [{"name": it.name, "lat": it.lat, "lon": it.lon} for it in items]
    _points[road_type] = (now, pts)
    try:
        p.write_text(json.dumps({"fetched_at": now, "points": pts}, ensure_ascii=False))
    except Exception:
        pass
    # 좌표가 바뀌었으면 선도 다시 만든다
    for k in [k for k in _chains if k[0] == road_type]:
        _chains.pop(k, None)
    log.info("ITS %s CCTV 좌표 %d개 적재", road_type, len(pts))
    return pts


def _build_chains(route: str, pts: list[tuple[float, float]]) -> list[Chain]:
    # 중복 제거
    uniq: list[tuple[float, float]] = []
    for p in pts:
        if all(_hav(p, q) > DEDUP_M for q in uniq):
            uniq.append(p)
    chains: list[Chain] = []
    left = uniq[:]
    while left:
        cy = sum(p[0] for p in left) / len(left)
        cx = sum(p[1] for p in left) / len(left)
        start = max(left, key=lambda p: _hav(p, (cy, cx)))  # 끝점에서 시작
        path = [start]
        left.remove(start)
        while left:
            cur = path[-1]
            nxt = min(left, key=lambda p: _hav(cur, p))
            if _hav(cur, nxt) > GAP_M:
                break
            path.append(nxt)
            left.remove(nxt)
        chains.append(Chain(route, path))
    return chains


def chains_for(route: str, road_type: str = "ex") -> list[Chain]:
    key = (road_type, route)
    with _lock:
        if key in _chains:
            return _chains[key]
        pts = [(p["lat"], p["lon"]) for p in _load_points(road_type) if its_client.parse_route(p["name"]) == route]
        _chains[key] = _build_chains(route, pts) if pts else []
        return _chains[key]


def find_chain(route: str, lat: float, lon: float, road_type: str | None = None, max_dist: float = 3000.0):
    """(chain, s, dist) — 카메라에 가장 가까운 선. road_type 을 모르면 ex → its 순서로."""
    for rt in ([road_type] if road_type else []) + [t for t in ("ex", "its") if t != road_type]:
        try:
            chains = chains_for(route, rt)
        except Exception as e:
            log.warning("노선 선 적재 실패(%s/%s): %s", rt, route, e)
            continue
        best = None
        for ch in chains:
            s, d = ch.project((lat, lon))
            if best is None or d < best[2]:
                best = (ch, s, d)
        if best and best[2] <= max_dist:
            return best
    return None


def resolve(route: str | None, lat: float | None, lon: float | None, destination: str | None, road_type: str | None = None) -> dict:
    """노선·위치·방면 → {ok, heading, sign, reason, confidence}."""
    if not route or lat is None or lon is None:
        return {"ok": False, "reason": "노선과 CCTV 좌표가 필요합니다"}
    if not destination:
        return {"ok": False, "reason": "방면(목적지)을 입력하세요"}
    found = find_chain(route, lat, lon, road_type)
    if not found:
        return {"ok": False, "reason": f"ITS 에서 '{route}' 노선 선을 찾지 못했습니다 (노선 이름 확인)"}
    ch, s_cam, _ = found
    dest = geocode.geocode(destination)
    if not dest:
        return {"ok": False, "reason": f"'{destination}' 위치를 찾지 못했습니다"}
    s_dest, off = ch.project(dest)
    delta = s_dest - s_cam
    dest_at_end = s_dest <= 1 or s_dest >= ch.length - 1
    if abs(delta) < 300:
        # 카메라가 선 끝 근처이고 목적지가 그 끝 너머(선 밖 멀리)에 있으면 → 그 끝 쪽
        if dest_at_end and off > 2000:
            sign = 1 if s_dest > ch.length / 2 else -1
            heading = ch.heading_at(s_cam, sign)
            return {"ok": True, "heading": round(heading, 1), "sign": sign, "confidence": "medium", "reason": f"{destination} 은(는) {route} 선 끝 너머 → 그 끝 방향"}
        return {"ok": False, "reason": f"'{destination}' 이(가) 이 지점과 너무 가까워 방향을 정할 수 없습니다"}
    sign = 1 if delta > 0 else -1
    heading = ch.heading_at(s_cam, sign)
    # 목적지가 노선에서 멀고, 투영 지점이 선 끝이면 신뢰도 낮음
    at_end = s_dest <= 1 or s_dest >= ch.length - 1
    confidence = "high" if off < 15000 or not at_end else "medium"
    return {"ok": True, "heading": round(heading, 1), "sign": sign, "confidence": confidence, "reason": f"{route} 선 위에서 {destination} 쪽({abs(delta) / 1000:.0f}km)"}


def resolve_many(route: str | None, lat, lon, destinations: list[str], road_type: str | None = None) -> dict[str, dict]:
    """같은 노선의 방면들을 한꺼번에. 하나만 풀리고 방면이 2개면 나머지는 반대 방향으로 채운다."""
    out = {d: resolve(route, lat, lon, d, road_type) for d in destinations}
    ok = [d for d in destinations if out[d].get("ok")]
    bad = [d for d in destinations if not out[d].get("ok")]
    if len(destinations) == 2 and len(ok) == 1 and bad:
        r = out[ok[0]]
        out[bad[0]] = {"ok": True, "heading": round((r["heading"] + 180) % 360, 1), "sign": -r["sign"], "confidence": "medium", "reason": f"'{ok[0]}' 의 반대 방향 (원래 사유: {out[bad[0]]['reason']})"}
    return out


def auto_resolve_directions(cam) -> int:
    """카메라의 방향들 중 방면이 있고 수동 지정이 아닌 것의 진행 방향을 채운다. 바뀐 개수."""
    n = 0
    by_route: dict[str, list] = {}
    for d in cam.directions:
        if d.heading_source == "manual" or not d.destination:
            continue
        by_route.setdefault(d.road or cam.route or "", []).append(d)
    for route, dirs in by_route.items():
        res = resolve_many(route, cam.lat, cam.lon, [d.destination for d in dirs], cam.its_road_type)
        for d in dirs:
            r = res.get(d.destination) or {}
            if r.get("ok"):
                d.heading_deg, d.heading_source, d.lat, d.lon = r["heading"], "auto", None, None
                n += 1
    return n


def camera_segments(cams) -> list[dict]:
    """측정 중인 (카메라, 방향) 마다 지도에 칠할 구간."""
    items = []
    for c in cams:
        if c.lat is None or c.lon is None:
            continue
        measured = (c.meta or {}).get("road_px") or {}
        for d in c.directions:
            route = d.road or c.route
            if not route or d.heading_deg is None or not measured.get(str(d.index)):
                continue
            found = find_chain(route, d.lat if d.lat is not None else c.lat, d.lon if d.lon is not None else c.lon, c.its_road_type)
            if not found:
                continue
            ch, s, _ = found
            # 선 접선과 진행 방향 비교 → 부호
            tan = ch.heading_at(s, 1)
            diff = abs(((d.heading_deg - tan) + 180) % 360 - 180)
            sign = 1 if diff <= 90 else -1
            items.append({"camera_id": c.id, "camera_name": c.name, "direction_index": d.index, "name": d.name, "destination": d.destination, "route": route, "chain": id(ch), "_ch": ch, "s": s, "sign": sign})
    # 같은 선·같은 부호의 이웃 측정 지점과의 중간까지만 칠한다
    out = []
    for it in items:
        same = sorted(x["s"] for x in items if x["chain"] == it["chain"] and x["sign"] == it["sign"] and x is not it)
        before = [v for v in same if v < it["s"]]
        after = [v for v in same if v > it["s"]]
        back = min(CAP_M, (it["s"] - before[-1]) / 2) if before else CAP_M
        fwd = min(CAP_M, (after[0] - it["s"]) / 2) if after else CAP_M
        back, fwd = max(MIN_M, back), max(MIN_M, fwd)
        ch: Chain = it.pop("_ch")
        coords = ch.slice(it["s"] - back, it["s"] + fwd)
        if it["sign"] < 0:
            coords = coords[::-1]  # 진행 방향 순서로
        out.append({**{k: v for k, v in it.items() if k not in ("chain", "s")}, "coords": [[round(a, 6), round(b, 6)] for a, b in coords], "extent_m": [round(back), round(fwd)]})
    return out


def route_lines(routes: list[str], road_types: dict[str, str | None]) -> dict[str, list[list[list[float]]]]:
    out = {}
    for r in routes:
        for rt in ([road_types.get(r)] if road_types.get(r) else []) + ["ex", "its"]:
            try:
                chs = chains_for(r, rt)
            except Exception:
                chs = []
            if chs:
                out[r] = [[[round(a, 5), round(b, 5)] for a, b in ch.pts] for ch in chs if len(ch.pts) >= 2]
                break
    return out
