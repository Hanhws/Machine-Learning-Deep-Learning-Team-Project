"""YOLOv8-Seg / YOLO11-Seg / YOLO26-Seg (Ultralytics).

가중치 이름만 바꾸면 세 버전을 같은 코드로 학습한다.
    yolov8{n,s,m,l,x}-seg.pt · yolo11{n,s,m,l,x}-seg.pt · yolo26{n,s,m,l,x}-seg.pt

학습은 Ultralytics 내장 트레이너(YOLO 라벨 사용)에 맡기고,
평가는 carseg.coco_eval 로 다른 계열과 같은 기준으로 다시 채점한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config import ExperimentConfig
from ..data import CarSegData
from ..utils import Timer, resolve_device, set_seed
from .base import InstancePrediction, LabelMapper, Predictor, Trainer, TrainResult


def _device_arg(device: str) -> Any:
    d = resolve_device(device)
    return 0 if d == "cuda" else d


class UltralyticsTrainer(Trainer):
    def train(self) -> TrainResult:
        from ultralytics import YOLO

        cfg, t = self.cfg, self.cfg.train
        set_seed(t.seed)
        data = CarSegData(cfg.data.root)
        data_yaml = data.write_data_yaml(cfg.run_dir / "data.yaml")

        if t.resume:
            model = YOLO(t.resume)
            kwargs: Dict[str, Any] = {"resume": True}
        else:
            model = YOLO(cfg.model.weights)
            kwargs = {
                "data": str(data_yaml),
                "epochs": t.epochs,
                "imgsz": t.imgsz,
                "batch": t.batch,
                "lr0": t.lr0,
                "lrf": t.lrf,
                "optimizer": t.optimizer,
                "momentum": t.momentum,
                "weight_decay": t.weight_decay,
                "warmup_epochs": t.warmup_epochs,
                "workers": t.workers,
                "device": _device_arg(t.device),
                "amp": t.amp,
                "seed": t.seed,
                "patience": t.patience or t.epochs,
                "freeze": t.freeze or None,
                "fliplr": t.hflip,
                "flipud": t.vflip,
                "mosaic": t.mosaic,
                "mixup": t.mixup,
                "copy_paste": t.copy_paste,
                "degrees": t.degrees,
                "translate": t.translate,
                "scale": t.scale,
                "box": t.box_gain,
                "cls": t.cls_gain,
                "dfl": t.dfl_gain,
                "fraction": _fraction(cfg, data),
                "project": str(cfg.run_dir.resolve()),
                "name": "ultralytics",
                "exist_ok": True,
                "plots": True,
                "verbose": True,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if t.accumulate > 1:
                # ultralytics 는 nbs(명목 배치)로 누적을 제어한다: 누적 = nbs / batch
                kwargs["nbs"] = t.batch * t.accumulate
            kwargs.update(cfg.model.extra.get("train_kwargs", {}))  # 설정 파일에서 직접 추가 인자

        with Timer() as timer:
            model.train(**kwargs)

        trainer = model.trainer
        best = Path(trainer.best) if Path(trainer.best).exists() else Path(trainer.last)
        return TrainResult(
            best_weights=str(best),
            last_weights=str(trainer.last),
            run_dir=str(trainer.save_dir),
            train_seconds=timer.seconds,
            epochs_done=int(getattr(trainer, "epoch", t.epochs - 1)) + 1,
            extra={"ultralytics_metrics": _clean(getattr(trainer, "metrics", {}) or {})},
        )


def _fraction(cfg: ExperimentConfig, data: CarSegData) -> float | None:
    if not cfg.data.train_limit:
        return None
    n = len(data.image_paths(cfg.data.train_split))
    return max(min(cfg.data.train_limit / max(n, 1), 1.0), 1e-4)


def _clean(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = round(float(v), 5)
        except (TypeError, ValueError):
            pass
    return out


class UltralyticsPredictor(Predictor):
    def __init__(self, cfg: ExperimentConfig, weights: str, label_mode: str = "finetuned"):
        super().__init__(cfg, weights, label_mode)
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = _device_arg(cfg.train.device)
        self.mapper = LabelMapper(self.model.names, mode=label_mode)
        print(f"[ultralytics] {weights}  클래스 매핑: {self.mapper.describe()}")

    def predict(self, image: np.ndarray) -> InstancePrediction:
        e = self.cfg.eval
        h, w = image.shape[:2]
        # ultralytics 는 numpy 입력을 BGR 로 해석한다
        result = self.model.predict(
            source=image[:, :, ::-1], imgsz=self.cfg.train.imgsz, conf=e.conf, iou=e.iou,
            max_det=e.max_det, augment=e.tta, retina_masks=True, device=self.device, verbose=False,
        )[0]
        if result.masks is None or len(result.boxes) == 0:
            return InstancePrediction.empty(h, w)

        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):  # retina_masks 가 안 먹는 버전 대비
            import cv2

            masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in masks])
        return self._finalize(
            boxes=result.boxes.xyxy.cpu().numpy(),
            scores=result.boxes.conf.cpu().numpy(),
            labels=result.boxes.cls.cpu().numpy().astype(np.int64),
            masks=masks > e.mask_threshold,
            height=h, width=w,
        )
