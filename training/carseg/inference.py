"""학습이 끝난 모델을 혼잡도 시스템에서 쓰기 위한 고수준 API.

    from carseg.inference import VehicleSegmenter
    seg = VehicleSegmenter.from_run("runs/yolo11s_seg_1280")     # config.yaml + best 가중치 자동
    out = seg.run(image_rgb, road_mask=None)
    out.stats.counts          # {'car': 12, 'bus': 1, 'truck': 3}
    out.stats.vehicle_pixels  # 합집합 픽셀 수 (겹침 1회만)
    out.union_mask            # (H, W) bool
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ExperimentConfig
from .models import build_predictor
from .models.base import InstancePrediction, read_rgb
from .occupancy import VehicleStats, compute_vehicle_stats, union_mask


@dataclass
class SegmentationOutput:
    prediction: InstancePrediction
    union_mask: np.ndarray
    stats: VehicleStats


class VehicleSegmenter:
    def __init__(self, cfg: ExperimentConfig, weights: str, label_mode: str = "finetuned",
                 score_thr: Optional[float] = None):
        self.cfg = cfg
        self.predictor = build_predictor(cfg, weights, label_mode)
        self.score_thr = cfg.eval.union_conf if score_thr is None else score_thr

    @classmethod
    def from_run(cls, run_dir: str | Path, weights: Optional[str] = None, **kwargs) -> "VehicleSegmenter":
        """train.py 가 남긴 run 폴더(config.yaml, train_result.json)에서 바로 불러온다."""
        run_dir = Path(run_dir)
        cfg = ExperimentConfig.load(run_dir / "config.yaml")
        if weights is None:
            result = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
            weights = result["best_weights"]
        return cls(cfg, weights, **kwargs)

    def run(self, image: np.ndarray | str | Path, road_mask: Optional[np.ndarray] = None) -> SegmentationOutput:
        if not isinstance(image, np.ndarray):
            image = read_rgb(image)
        pred = self.predictor.predict(image)
        stats = compute_vehicle_stats(pred, self.score_thr, road_mask)
        return SegmentationOutput(prediction=pred, union_mask=union_mask(pred.above(self.score_thr)),
                                  stats=stats)
