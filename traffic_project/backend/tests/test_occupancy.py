import numpy as np

from app.ml.occupancy import compute_occupancy
from app.ml.vehicle_seg import VehicleInstance, VehicleResult


def _road(h=100, w=200):
    road = np.zeros((h, w), np.uint8)
    road[50:, :100] = 1  # 방향 1: 왼쪽 아래
    road[50:, 100:] = 2  # 방향 2: 오른쪽 아래
    return road


def test_occupancy_per_direction():
    road = _road()
    vlabel = np.zeros_like(road)
    vlabel[60:70, 10:30] = 1  # car 200px in dir1
    vlabel[60:80, 150:160] = 3  # truck 200px in dir2
    vlabel[0:10, 0:10] = 2  # bus outside road (ignored in ratio)
    inst = [
        VehicleInstance("car", 0.9, 200, 20, 65, (10, 60, 30, 70)),
        VehicleInstance("truck", 0.9, 200, 155, 70, (150, 60, 160, 80)),
        VehicleInstance("bus", 0.9, 100, 5, 5, (0, 0, 10, 10)),
    ]
    res = compute_occupancy(VehicleResult(label_map=vlabel, instances=inst), road, 2)
    assert res[1].road_px == 50 * 100 and res[2].road_px == 50 * 100
    assert abs(res[1].occupancy - 200 / 5000) < 1e-9
    assert abs(res[2].occupancy - 200 / 5000) < 1e-9
    assert abs(res[0].occupancy - 400 / 10000) < 1e-9
    assert res[1].counts == {"car": 1, "bus": 0, "truck": 0}
    assert res[2].counts == {"car": 0, "bus": 0, "truck": 1}
    assert res[0].counts["bus"] == 0  # 도로 밖 차량은 대수에서 제외
    assert res[1].class_px["car"] == 200 and res[2].class_px["truck"] == 200


def test_empty_road_gives_zero():
    road = np.zeros((10, 10), np.uint8)
    vlabel = np.ones((10, 10), np.uint8)
    res = compute_occupancy(VehicleResult(label_map=vlabel), road, 0)
    assert res[0].occupancy == 0.0 and res[0].road_px == 0


def test_roi_gives_same_result():
    """도로 바운딩 박스만 세도 전체를 센 것과 결과가 같다."""
    from app.ml.roi import roi_from_mask

    road = _road()
    vlabel = np.zeros_like(road)
    vlabel[60:70, 10:30] = 1
    vlabel[60:80, 150:160] = 3
    res = VehicleResult(label_map=vlabel, instances=[])
    full = compute_occupancy(res, road, 2)
    windowed = compute_occupancy(res, road, 2, roi_from_mask(road, 4))
    for d in full:
        assert (full[d].occupancy, full[d].vehicle_px, full[d].road_px, full[d].class_px) == (
            windowed[d].occupancy,
            windowed[d].vehicle_px,
            windowed[d].road_px,
            windowed[d].class_px,
        )


def test_many_directions_uses_bincount_path():
    """방향 수가 많아 uint8 로 합칠 수 없을 때(64방향 이상)도 정확해야 한다."""
    road = np.zeros((10, 200), np.uint8)
    for d in range(1, 101):
        road[:, (d - 1) * 2 : d * 2] = d
    vlabel = np.zeros_like(road)
    vlabel[0:5, 0:2] = 1  # 방향 1 에 car 10px
    res = compute_occupancy(VehicleResult(label_map=vlabel, instances=[]), road, 100)
    assert res[1].road_px == 20 and res[1].class_px["car"] == 10
    assert abs(res[1].occupancy - 0.5) < 1e-9
    assert res[100].vehicle_px == 0
