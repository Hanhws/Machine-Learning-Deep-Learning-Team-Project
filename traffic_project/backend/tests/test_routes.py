"""노선 선 만들기 · 방면 → 진행 방향 (네트워크 없이 합성 데이터로)."""

import app.services.routes as rs


def _north_south_route():
    # 남(35.0) → 북(36.0) 으로 1km 남짓 간격의 CCTV, 일부러 섞어서 넣는다
    pts = [(35.0 + i * 0.01, 127.0) for i in range(101)]
    return pts[::2] + pts[1::2]


def test_chain_is_ordered_and_deduped():
    pts = _north_south_route() + [(35.5, 127.0001)]  # 60m 이내 중복
    chains = rs._build_chains("테스트선", pts)
    assert len(chains) == 1
    ch = chains[0]
    assert len(ch.pts) == 101
    lats = [p[0] for p in ch.pts]
    assert lats == sorted(lats) or lats == sorted(lats, reverse=True)
    assert 110_000 < ch.length < 112_000


def test_gap_splits_chain():
    pts = [(35.0 + i * 0.01, 127.0) for i in range(10)] + [(36.0 + i * 0.01, 127.0) for i in range(10)]
    assert len(rs._build_chains("테스트선", pts)) == 2


def test_resolve_by_destination(monkeypatch):
    ch = rs._build_chains("테스트선", _north_south_route())
    monkeypatch.setattr(rs, "find_chain", lambda route, lat, lon, rt=None, max_dist=3000.0: (ch[0], ch[0].project((lat, lon))[0], 0.0))
    monkeypatch.setattr(rs.geocode, "geocode", lambda name: {"북쪽시": (36.2, 127.05), "남쪽시": (34.8, 126.95)}.get(name))
    north = rs.resolve("테스트선", 35.5, 127.0, "북쪽시")
    south = rs.resolve("테스트선", 35.5, 127.0, "남쪽시")
    assert north["ok"] and south["ok"]
    assert north["heading"] < 5 or north["heading"] > 355  # 북쪽
    assert 175 < south["heading"] < 185  # 남쪽
    assert north["sign"] == -south["sign"]


def test_resolve_many_fills_opposite(monkeypatch):
    ch = rs._build_chains("테스트선", _north_south_route())
    monkeypatch.setattr(rs, "find_chain", lambda route, lat, lon, rt=None, max_dist=3000.0: (ch[0], ch[0].project((lat, lon))[0], 0.0))
    monkeypatch.setattr(rs.geocode, "geocode", lambda name: {"북쪽시": (36.2, 127.05)}.get(name))
    out = rs.resolve_many("테스트선", 35.5, 127.0, ["북쪽시", "모르는곳"])
    assert out["모르는곳"]["ok"]
    assert abs(((out["모르는곳"]["heading"] - out["북쪽시"]["heading"]) % 360) - 180) < 1


def test_offset_side_is_right_of_travel():
    # 북쪽으로 가는 선을 오른쪽(동쪽)으로 밀면 경도가 커져야 한다 — 프론트 offsetLine 과 같은 규칙을 파이썬으로 확인
    import math

    lat0 = 35.0
    kx = math.cos(math.radians(lat0)) * 111320
    dx, dy = 0.0, 1.0  # 북쪽 진행
    x_off = dy * 10  # 오른쪽 법선 (dy, -dx)
    assert x_off / kx > 0
