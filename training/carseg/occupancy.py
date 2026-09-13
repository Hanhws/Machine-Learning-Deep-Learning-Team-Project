"""차량 픽셀 통계와 도로 대비 점유율.

    union_mask = masks.any(axis=0)          # 겹친 픽셀은 한 번만 센다
    vehicle_px = union_mask.sum()
    occupancy  = vehicle_px(도로 안) / road_px

도로 모델이 아직 없으면 road_mask=None → 이미지 전체 대비 면적비만 계산한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from . import CLASS_NAMES
from .models.base import InstancePrediction


@dataclass
class VehicleStats:
    image_pixels: int
    num_vehicles: int                                # score ≥ union_conf 인스턴스 수
    counts: Dict[str, int] = field(default_factory=dict)          # 클래스별 대수
    class_pixels: Dict[str, int] = field(default_factory=dict)    # 클래스별 합집합 픽셀
    vehicle_pixels: int = 0                          # 전체 차량 합집합 픽셀 (클래스 무관)
    overlap_pixels: int = 0                          # 인스턴스끼리 겹쳐 중복 제거된 픽셀 수
    image_ratio: float = 0.0                         # vehicle_pixels / image_pixels
    road_pixels: Optional[int] = None
    vehicle_on_road_pixels: Optional[int] = None
    occupancy: Optional[float] = None                # vehicle_on_road_pixels / road_pixels

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def flat(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.to_dict().items() if not isinstance(v, dict)}
        d.update({f"count_{k}": v for k, v in self.counts.items()})
        d.update({f"pixels_{k}": v for k, v in self.class_pixels.items()})
        return d


def union_mask(pred: InstancePrediction) -> np.ndarray:
    if len(pred) == 0:
        return np.zeros(pred.image_shape, bool)
    return pred.masks.any(axis=0)


def compute_vehicle_stats(pred: InstancePrediction, score_thr: float = 0.5,
                          road_mask: Optional[np.ndarray] = None) -> VehicleStats:
    pred = pred.above(score_thr)
    h, w = pred.image_shape
    union = union_mask(pred)
    vehicle_px = int(union.sum())
    stats = VehicleStats(
        image_pixels=h * w,
        num_vehicles=len(pred),
        counts={n: int((pred.labels == i).sum()) for i, n in enumerate(CLASS_NAMES)},
        class_pixels={n: int(pred.masks[pred.labels == i].any(axis=0).sum()) if (pred.labels == i).any() else 0
                      for i, n in enumerate(CLASS_NAMES)},
        vehicle_pixels=vehicle_px,
        overlap_pixels=int(pred.masks.sum()) - vehicle_px if len(pred) else 0,
        image_ratio=vehicle_px / float(h * w),
    )
    if road_mask is not None:
        if road_mask.shape != union.shape:
            raise ValueError(f"도로 마스크 크기 {road_mask.shape} ≠ 이미지 {union.shape}")
        road = road_mask.astype(bool)
        stats.road_pixels = int(road.sum())
        stats.vehicle_on_road_pixels = int(np.logical_and(union, road).sum())
        stats.occupancy = stats.vehicle_on_road_pixels / stats.road_pixels if stats.road_pixels else None
    return stats
