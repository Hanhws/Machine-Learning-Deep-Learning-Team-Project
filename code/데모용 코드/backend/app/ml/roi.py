"""도로 마스크 영역(ROI) 안에서만 차량을 세그멘테이션한다.

- 마스크의 바운딩 박스(+여유)만 잘라 YOLO 에 넣는다 → 마스크 밖 도로/주차장/건물의 차량은 애초에 보지 않고, 입력이 작아져 추론도 빨라진다.
- 결과 라벨맵을 원본 좌표로 되돌린 뒤 마스크 밖 픽셀을 0 으로 지운다.
- 인스턴스는 마스크 안에 절반 이상 걸친 것만 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.vehicle_seg import VehicleInstance, VehicleResult, VehicleSegmenter


@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def roi_from_mask(road_label: np.ndarray, pad: int = 32) -> Roi | None:
    ys, xs = np.nonzero(road_label)
    if len(ys) == 0:
        return None
    h, w = road_label.shape
    return Roi(max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def infer_in_roi(seg: VehicleSegmenter, frame: np.ndarray, road_label: np.ndarray, roi: Roi | None, crop: bool = True) -> VehicleResult:
    """road_label 이 0 인 곳의 차량 픽셀/인스턴스를 제거한 VehicleResult."""
    h, w = frame.shape[:2]
    if crop and roi is not None and (roi.width < w or roi.height < h):
        sub = seg.infer(frame[roi.y0 : roi.y1, roi.x0 : roi.x1])
        label = np.zeros((h, w), dtype=np.uint8)
        label[roi.y0 : roi.y1, roi.x0 : roi.x1] = sub.label_map
        insts = [
            VehicleInstance(cls=i.cls, conf=i.conf, area=i.area, cx=i.cx + roi.x0, cy=i.cy + roi.y0, box=(i.box[0] + roi.x0, i.box[1] + roi.y0, i.box[2] + roi.x0, i.box[3] + roi.y0))
            for i in sub.instances
        ]
        res = VehicleResult(label_map=label, instances=insts, infer_ms=sub.infer_ms)
    else:
        res = seg.infer(frame)

    inside = road_label > 0
    res.label_map[~inside] = 0
    kept: list[VehicleInstance] = []
    for i in res.instances:
        x1, y1, x2, y2 = (int(round(v)) for v in i.box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
        patch_in = inside[y1:y2, x1:x2]
        # 박스 안 도로 비율 또는 무게중심으로 판단 (절반 이상 걸치면 유지)
        cx, cy = int(min(max(i.cx, 0), w - 1)), int(min(max(i.cy, 0), h - 1))
        if patch_in.size and (patch_in.mean() >= 0.5 or inside[cy, cx]):
            kept.append(i)
    res.instances = kept
    return res
