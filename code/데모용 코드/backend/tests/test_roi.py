import numpy as np

from app.ml.roi import infer_in_roi, roi_from_mask
from app.ml.vehicle_seg import VehicleInstance, VehicleResult


class FakeSeg:
    """주어진 프레임 크기 전체에 차량 2대(왼쪽 위·오른쪽 아래)를 그리는 가짜 세그멘터."""

    def __init__(self):
        self.calls = []

    def infer(self, frame):
        h, w = frame.shape[:2]
        self.calls.append((h, w))
        lab = np.zeros((h, w), np.uint8)
        lab[0:2, 0:2] = 1  # 왼쪽 위 모서리 차량 (잘린 영역의 여유(pad) 부분 = 도로 밖)
        lab[h - 6 : h - 2, w - 6 : w - 2] = 3  # 오른쪽 아래 트럭
        inst = [
            VehicleInstance("car", 0.9, 4, 1.0, 1.0, (0, 0, 2, 2)),
            VehicleInstance("truck", 0.9, 16, w - 4.5, h - 4.5, (w - 6, h - 6, w - 2, h - 2)),
        ]
        return VehicleResult(label_map=lab, instances=inst)


def test_roi_bbox_and_masking():
    road = np.zeros((40, 60), np.uint8)
    road[20:40, 30:60] = 1  # 오른쪽 아래 사분면만 도로
    roi = roi_from_mask(road, pad=2)
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (28, 18, 60, 40)
    seg = FakeSeg()
    frame = np.zeros((40, 60, 3), np.uint8)
    res = infer_in_roi(seg, frame, road, roi, crop=True)
    # 잘린 영역만 추론했는지
    assert seg.calls == [(22, 32)]
    # 도로 밖 픽셀은 제거, 도로 안 차량만 남음
    assert res.label_map[:20, :].sum() == 0 and (res.label_map == 1).sum() == 0
    assert (res.label_map == 3).sum() == 16
    assert [i.cls for i in res.instances] == ["truck"]
    # 좌표가 원본 기준으로 복원됐는지
    t = res.instances[0]
    assert 50 < t.cx < 60 and 30 < t.cy < 40


def test_roi_none_when_empty_mask():
    assert roi_from_mask(np.zeros((10, 10), np.uint8)) is None


def test_no_crop_still_masks():
    road = np.zeros((40, 60), np.uint8)
    road[:10, :10] = 2  # 왼쪽 위만 도로
    seg = FakeSeg()
    res = infer_in_roi(seg, np.zeros((40, 60, 3), np.uint8), road, roi_from_mask(road), crop=False)
    assert seg.calls == [(40, 60)]
    assert [i.cls for i in res.instances] == ["car"]
    assert (res.label_map == 3).sum() == 0
