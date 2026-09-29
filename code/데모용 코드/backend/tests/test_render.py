import numpy as np

from app.ml.render import encode_jpeg, render_overlay


def test_render_modes():
    frame = np.zeros((60, 80, 3), np.uint8)
    road = np.zeros((60, 80), np.uint8)
    road[30:, :40] = 1
    road[30:, 40:] = 2
    veh = np.zeros((60, 80), np.uint8)
    veh[40:50, 10:20] = 1
    for mode in ("class", "vehicle", "road", "none"):
        out = render_overlay(frame, veh, road, mode=mode, hud=["ALL 10%"])
        assert out.shape == frame.shape
        assert len(encode_jpeg(out)) > 100
    # 'none' 은 HUD 만 그린다: 도로 영역 픽셀이 검정 그대로
    out = render_overlay(frame, veh, road, mode="none")
    assert out[45, 15].tolist() == [0, 0, 0]
    out = render_overlay(frame, veh, road, mode="vehicle")
    assert out[45, 15].tolist() != [0, 0, 0]


def test_road_layer_cache_matches_inline():
    """미리 만든 RoadLayer 로 그린 결과가 그때그때 만든 것과 같다."""
    from app.ml.render import RoadLayer

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    road = np.zeros((60, 80), np.uint8)
    road[30:, :40] = 1
    road[30:, 40:] = 2
    veh = np.zeros((60, 80), np.uint8)
    veh[40:50, 10:20] = 2
    layer = RoadLayer(road, None)
    for mode in ("class", "vehicle", "road"):
        a = render_overlay(frame, veh, road, mode=mode)
        b = render_overlay(frame, veh, road, mode=mode, road_layer=layer)
        assert np.array_equal(a, b)


def test_mismatched_layer_is_ignored():
    """프레임 크기가 바뀌면 캐시된 레이어를 쓰지 않는다 (크기 오류 방지)."""
    from app.ml.render import RoadLayer

    frame = np.zeros((60, 80, 3), np.uint8)
    stale = RoadLayer(np.ones((30, 40), np.uint8), None)
    out = render_overlay(frame, None, None, mode="road", road_layer=stale)
    assert out.shape == frame.shape
