"""모든 모델 계열이 따르는 공통 인터페이스.

- Predictor.predict(image) → InstancePrediction   : 평가·추론이 계열을 몰라도 되게 한다
- Trainer.train()          → TrainResult           : train.py 가 계열을 몰라도 되게 한다

사전학습(COCO 80종) 모델을 바로 평가할 때는 label_mode="coco" 로 두면
예측 클래스 이름이 car/bus/truck 인 것만 남기고 우리 번호로 바꿔 준다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .. import CLASS_NAMES
from ..config import ExperimentConfig


@dataclass
class InstancePrediction:
    """이미지 한 장의 인스턴스 예측. 모든 배열은 원본 이미지 해상도 기준."""

    boxes: np.ndarray    # (N, 4) float32 xyxy (픽셀)
    scores: np.ndarray   # (N,) float32
    labels: np.ndarray   # (N,) int64 — 0=car 1=bus 2=truck
    masks: np.ndarray    # (N, H, W) bool

    @classmethod
    def empty(cls, height: int, width: int) -> "InstancePrediction":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                   np.zeros((0,), np.int64), np.zeros((0, height, width), bool))

    def __len__(self) -> int:
        return int(len(self.scores))

    @property
    def image_shape(self) -> tuple:
        return tuple(self.masks.shape[1:])

    def filter(self, keep: np.ndarray) -> "InstancePrediction":
        return InstancePrediction(self.boxes[keep], self.scores[keep], self.labels[keep], self.masks[keep])

    def above(self, score: float) -> "InstancePrediction":
        return self.filter(self.scores >= score)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """마스크만 주는 모델(Mask2Former)을 위해 마스크 외접 상자를 구한다."""
    boxes = np.zeros((len(masks), 4), np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(xs):
            boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes


class LabelMapper:
    """모델의 클래스 번호 → 우리 클래스 번호(0=car 1=bus 2=truck). 해당 없음은 -1.

    - finetuned: 모델이 이미 우리 3클래스로 학습됨 → 이름으로 맞춰 본다(순서가 달라도 안전)
    - coco     : COCO 사전학습 모델 → 'car','bus','truck' 이름만 골라 쓴다
    """

    def __init__(self, model_names: Mapping[int, str], mode: str = "finetuned",
                 offset: int = 0):
        self.mode = mode
        wanted = {n: i for i, n in enumerate(CLASS_NAMES)}
        self.table: Dict[int, int] = {}
        for idx, name in model_names.items():
            key = str(name).strip().lower()
            if key in wanted:
                self.table[int(idx)] = wanted[key]
        if not self.table and mode == "finetuned":
            # 이름 정보가 없는 체크포인트 — 번호 순서가 같다고 가정
            self.table = {i + offset: i for i in range(len(CLASS_NAMES))}
        if not self.table:
            raise ValueError(f"모델 클래스 이름에서 {CLASS_NAMES} 를 찾지 못했습니다: {dict(model_names)}")

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        return np.asarray([self.table.get(int(l), -1) for l in labels], np.int64)

    def describe(self) -> str:
        return ", ".join(f"{k}→{CLASS_NAMES[v]}" for k, v in sorted(self.table.items()))


class Predictor(ABC):
    """학습된(또는 사전학습) 모델로 이미지 한 장을 예측한다."""

    def __init__(self, cfg: ExperimentConfig, weights: str, label_mode: str = "finetuned"):
        self.cfg = cfg
        self.weights = weights
        self.label_mode = label_mode

    @abstractmethod
    def predict(self, image: np.ndarray) -> InstancePrediction:
        """image: (H, W, 3) uint8 **RGB**."""

    def predict_path(self, path: str | Path) -> InstancePrediction:
        return self.predict(read_rgb(path))

    def _finalize(self, boxes, scores, labels, masks, height: int, width: int) -> InstancePrediction:
        """라벨 매핑 → 해당 없는 클래스 제거 → 점수 내림차순 정렬 → max_det 자르기."""
        if len(scores) == 0:
            return InstancePrediction.empty(height, width)
        mapped = self.mapper(np.asarray(labels))
        keep = mapped >= 0
        pred = InstancePrediction(
            boxes=np.asarray(boxes, np.float32)[keep],
            scores=np.asarray(scores, np.float32)[keep],
            labels=mapped[keep],
            masks=np.asarray(masks).astype(bool)[keep],
        )
        order = np.argsort(-pred.scores)[: self.cfg.eval.max_det]
        return pred.filter(order)

    mapper: LabelMapper


@dataclass
class TrainResult:
    best_weights: str
    last_weights: str = ""
    run_dir: str = ""
    train_seconds: float = 0.0
    epochs_done: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    @abstractmethod
    def train(self) -> TrainResult:
        ...


def read_rgb(path: str | Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
