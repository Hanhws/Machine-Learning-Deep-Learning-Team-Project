"""점유율 계산.

occupancy(d) = |vehicle ∩ road_d| / |road_d|
  - road_d : 도로 라벨맵에서 값이 d 인 픽셀 (d=1..N 방향), d=0 은 모든 방향 합집합
  - vehicle: YOLO 라벨맵 > 0
차량 대수는 인스턴스 무게중심이 속한 방향으로 센다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.ml.vehicle_seg import VEHICLE_CLASSES, VehicleResult

if TYPE_CHECKING:
    from app.ml.roi import Roi


@dataclass
class DirectionMetrics:
    direction_index: int
    occupancy: float
    vehicle_px: int
    road_px: int
    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in VEHICLE_CLASSES})
    class_px: dict[str, int] = field(default_factory=lambda: {c: 0 for c in VEHICLE_CLASSES})

    def to_dict(self) -> dict:
        return {
            "direction_index": self.direction_index,
            "occupancy": round(self.occupancy, 5),
            "vehicle_px": self.vehicle_px,
            "road_px": self.road_px,
            "n_vehicles": int(sum(self.counts.values())),
            "counts": self.counts,
            "class_px": self.class_px,
        }


def _count_table(road_label: np.ndarray, vlabel: np.ndarray, stride: int, n_cls: int, road_max: int) -> np.ndarray:
    """table[d][c] = 방향 d 안에서 차량라벨이 c 인 픽셀 수 (c=0 은 빈 도로).

    방향×차종을 하나의 값(d*stride + c)으로 합쳐 히스토그램 한 번으로 전부 센다.
    방향마다 마스크를 따로 만들면 프레임 전체를 20번 넘게 훑게 된다.
    """
    veh = np.minimum(vlabel, n_cls)
    if (road_max + 1) * stride <= 256:  # uint8 로 합칠 수 있으면 calcHist 가 가장 빠르다
        combined = cv2.add(road_label * stride, veh)
        hist = cv2.calcHist([combined], [0], None, [256], [0, 256]).ravel()
        return hist[: (road_max + 1) * stride].astype(np.int64).reshape(-1, stride)
    combined = road_label.astype(np.int32) * stride + veh
    return np.bincount(combined.ravel(), minlength=(road_max + 1) * stride).reshape(-1, stride)


def compute_occupancy(vehicle: VehicleResult, road_label: np.ndarray, n_directions: int, roi: "Roi | None" = None) -> dict[int, DirectionMetrics]:
    """모든 방향(0=전체, 1..N)에 대한 지표.

    roi 를 주면 그 안에서만 센다. 도로 마스크의 바운딩 박스면 결과는 같고(바깥은 전부 배경)
    훑는 픽셀 수가 줄어 더 빠르다.
    """
    out: dict[int, DirectionMetrics] = {}
    vlabel = vehicle.label_map
    n_cls = len(VEHICLE_CLASSES)
    stride = n_cls + 1  # 차량 라벨 0(배경)..n_cls

    win_road, win_veh = road_label, vlabel
    if roi is not None:
        win_road = road_label[roi.y0 : roi.y1, roi.x0 : roi.x1]
        win_veh = vlabel[roi.y0 : roi.y1, roi.x0 : roi.x1]
    road_max = int(win_road.max()) if win_road.size else 0
    table = _count_table(win_road, win_veh, stride, n_cls, road_max)

    for d in [0] + list(range(1, n_directions + 1)):
        row = table[1 : road_max + 1].sum(0) if d == 0 else (table[d] if d <= road_max else np.zeros(stride, np.int64))
        road_px = int(row.sum())
        vehicle_px = int(row[1:].sum())
        m = DirectionMetrics(
            direction_index=d,
            occupancy=(vehicle_px / road_px) if road_px > 0 else 0.0,
            vehicle_px=vehicle_px,
            road_px=road_px,
        )
        for i, c in enumerate(VEHICLE_CLASSES):
            m.class_px[c] = int(row[i + 1])
        out[d] = m

    # 대수: 무게중심이 속한 방향
    h, w = road_label.shape
    for inst in vehicle.instances:
        x, y = int(min(max(inst.cx, 0), w - 1)), int(min(max(inst.cy, 0), h - 1))
        d = int(road_label[y, x])
        if d == 0:
            # 도로 마스크 바깥의 무게중심: 마스크와 겹치는지 대략 확인 (박스 중심 대신 박스 내부 도로 라벨 최빈값)
            x1, y1, x2, y2 = (int(v) for v in inst.box)
            patch = road_label[max(y1, 0) : max(y2, y1 + 1), max(x1, 0) : max(x2, x1 + 1)]
            if patch.size and patch.max() > 0:
                vals, cnts = np.unique(patch[patch > 0], return_counts=True)
                d = int(vals[np.argmax(cnts)])
        if d > 0 and d in out:
            out[d].counts[inst.cls] = out[d].counts.get(inst.cls, 0) + 1
        if d > 0 or (road_label[y, x] > 0):
            out[0].counts[inst.cls] = out[0].counts.get(inst.cls, 0) + 1
    return out
