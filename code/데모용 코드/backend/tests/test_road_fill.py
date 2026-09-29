import numpy as np

from app.ml.road_fill import fill_binary, fill_label


def test_car_hole_inside_road_is_filled():
    road = np.zeros((100, 100), bool)
    road[20:80, 10:90] = True
    car = np.zeros_like(road)
    car[40:50, 40:55] = True
    road[car] = False  # SAM 이 차량 자리를 뺀 상태
    out, added = fill_binary(road, car)
    assert out[car].all() and added == car.sum()


def test_car_on_edge_notch_is_filled():
    road = np.zeros((100, 100), bool)
    road[20:80, 10:90] = True
    car = np.zeros_like(road)
    car[70:90, 30:45] = True  # 도로 가장자리에 걸친 차량 → 홈
    road[car] = False
    out, _ = fill_binary(road, car)
    assert out[car].all()


def test_car_far_from_road_is_not_added():
    road = np.zeros((100, 100), bool)
    road[0:30, :] = True
    car = np.zeros_like(road)
    car[80:90, 40:50] = True  # 주차장 차량
    out, added = fill_binary(road, car)
    assert not out[car].any() and added == 0


def test_label_goes_to_touching_direction():
    lab = np.zeros((100, 100), np.uint8)
    lab[:, 0:50] = 1
    lab[:, 50:100] = 2
    car = np.zeros((100, 100), bool)
    car[40:50, 60:75] = True
    lab[car] = 0
    out, _ = fill_label(lab, car)
    assert (out[car] == 2).all()
    assert (out[:, :50] == 1).all()


def test_big_enclosed_region_not_filled():
    road = np.ones((200, 200), bool)
    road[60:140, 60:140] = False  # 16% 크기 섬(중앙분리대 등)은 메우지 않는다
    out, added = fill_binary(road, np.zeros_like(road))
    assert not out[80:120, 80:120].any() and added == 0
