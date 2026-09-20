from app.config import congestion_level


def test_congestion_level():
    thr = [0.08, 0.15, 0.25]
    assert congestion_level(0.0, thr) == "원활"
    assert congestion_level(0.079, thr) == "원활"
    assert congestion_level(0.08, thr) == "서행"
    assert congestion_level(0.2, thr) == "지체"
    assert congestion_level(0.9, thr) == "정체"
    assert congestion_level(None, thr) is None
