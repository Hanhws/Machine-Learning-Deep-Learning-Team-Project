"""모델 계열 레지스트리 — 설정의 model.family 문자열로 Trainer/Predictor 를 고른다.

무거운 라이브러리(ultralytics, transformers, rfdetr)는 실제로 쓸 때만 import 한다.
"""
from __future__ import annotations

import importlib
from typing import Tuple

from ..config import ExperimentConfig
from .base import InstancePrediction, Predictor, Trainer, TrainResult

FAMILIES = {
    #  family        모듈                 Trainer                 Predictor               pip 패키지
    "ultralytics": ("ultralytics_seg", "UltralyticsTrainer", "UltralyticsPredictor", "ultralytics"),
    "maskrcnn":    ("maskrcnn",        "MaskRCNNTrainer",    "MaskRCNNPredictor",    "torchvision"),
    "mask2former": ("mask2former",     "Mask2FormerTrainer", "Mask2FormerPredictor", "transformers"),
    "rfdetr":      ("rfdetr_seg",      "RFDETRSegTrainer",   "RFDETRSegPredictor",   "rfdetr"),
}


def _classes(family: str) -> Tuple[type, type]:
    if family not in FAMILIES:
        raise ValueError(f"알 수 없는 model.family={family!r} — 가능: {list(FAMILIES)}")
    module, trainer, predictor, package = FAMILIES[family]
    try:
        mod = importlib.import_module(f"{__name__}.{module}")
        return getattr(mod, trainer), getattr(mod, predictor)
    except ImportError as e:
        raise ImportError(f"{family} 계열에 필요한 패키지를 불러오지 못했습니다 "
                          f"(pip install {package}): {e}") from e


def build_trainer(cfg: ExperimentConfig) -> Trainer:
    return _classes(cfg.model.family)[0](cfg)


def build_predictor(cfg: ExperimentConfig, weights: str, label_mode: str = "finetuned") -> Predictor:
    """label_mode='coco' → COCO 사전학습 모델을 그대로 평가(car/bus/truck 만 추림)."""
    return _classes(cfg.model.family)[1](cfg, weights, label_mode)


__all__ = ["FAMILIES", "build_trainer", "build_predictor", "InstancePrediction", "Predictor",
           "Trainer", "TrainResult"]
