"""RF-DETR Segmentation (Roboflow `rfdetr` 패키지).

    model.weights : 사전학습이면 비워 두거나 "DEFAULT", 학습 후에는 checkpoint_best_total.pth 경로
    model.extra   :
        variant       RFDETRSegNano | RFDETRSegSmall | RFDETRSegMedium (기본) | RFDETRSegLarge | RFDETRSegXLarge
        resolution    입력 해상도 (지정 안 하면 변형별 기본값 — 모델마다 고정 해상도로 사전학습됨)
        train_kwargs  model.train() 에 그대로 넘길 추가 인자 (예: {early_stopping: true})

RF-DETR 은 Roboflow 형식 COCO 폴더를 요구한다:
    <dataset_dir>/{train,valid,test}/_annotations.coco.json + 이미지
→ prepare_dataset() 가 car_seg_split 을 심볼릭 링크로 이 형식에 맞춰 준다(용량 추가 거의 없음).
imgsz 는 쓰지 않는다(resolution 사용). batch 는 batch_size 로 전달.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .. import CLASS_NAMES
from ..config import ExperimentConfig
from ..data import CarSegData
from ..utils import Timer, set_seed
from .base import InstancePrediction, LabelMapper, Predictor, Trainer, TrainResult

DEFAULT_VARIANT = "RFDETRSegMedium"
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def _model_class(cfg: ExperimentConfig):
    import rfdetr

    name = cfg.model.extra.get("variant", DEFAULT_VARIANT)
    if not hasattr(rfdetr, name):
        seg = [n for n in dir(rfdetr) if n.startswith("RFDETRSeg")]
        raise ImportError(f"rfdetr 에 {name} 이 없습니다. 설치된 버전의 세그멘테이션 모델: {seg}")
    return getattr(rfdetr, name)


def _build(cfg: ExperimentConfig, weights: str = ""):
    kwargs: Dict[str, Any] = {}
    if cfg.model.extra.get("resolution"):
        kwargs["resolution"] = int(cfg.model.extra["resolution"])
    if weights and weights != "DEFAULT":
        kwargs["pretrain_weights"] = str(weights)
    return _model_class(cfg)(**kwargs)


def prepare_dataset(data: CarSegData, dest: str | Path, splits=("train", "val", "test"),
                    limit: int = 0) -> Path:
    """car_seg_split → Roboflow COCO 폴더 (이미지는 심볼릭 링크)."""
    dest = Path(dest)
    for split in splits:
        out = dest / SPLIT_DIRS[split]
        out.mkdir(parents=True, exist_ok=True)
        coco = json.loads(data.ann_file(split).read_text(encoding="utf-8"))
        if limit and split == "train":
            keep_ids = {im["id"] for im in coco["images"][:limit]}
            coco["images"] = [im for im in coco["images"] if im["id"] in keep_ids]
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        # Roboflow 내보내기 형식: id 0 은 상위 카테고리, 실제 클래스는 1부터
        coco["categories"] = [{"id": 0, "name": "vehicles", "supercategory": "none"}] + [
            {"id": i + 1, "name": n, "supercategory": "vehicles"} for i, n in enumerate(CLASS_NAMES)]
        for im in coco["images"]:
            link = out / im["file_name"]
            if not link.exists():
                os.symlink(data.images_dir(split) / im["file_name"], link)
        (out / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"[rfdetr] 데이터셋 준비 완료: {dest}")
    return dest


class RFDETRSegTrainer(Trainer):
    def train(self) -> TrainResult:
        cfg, t = self.cfg, self.cfg.train
        set_seed(t.seed)
        data = CarSegData(cfg.data.root)
        dataset_dir = prepare_dataset(data, cfg.run_dir / "rfdetr_dataset", limit=cfg.data.train_limit)
        out_dir = cfg.run_dir / "rfdetr"
        model = _build(cfg, cfg.model.weights)

        kwargs: Dict[str, Any] = {
            "dataset_dir": str(dataset_dir),
            "epochs": t.epochs,
            "batch_size": t.batch,
            "grad_accum_steps": max(t.accumulate, 1),
            "lr": t.lr0,
            "weight_decay": t.weight_decay,
            "output_dir": str(out_dir),
        }  # 장치는 rfdetr 가 자동 선택. 그 밖의 인자는 model.extra.train_kwargs 로
        if t.resume:  # 세션이 끊겼을 때: rfdetr/last.ckpt (또는 checkpoint.pth)
            kwargs["resume"] = str(t.resume)
        if t.patience:
            kwargs.update({"early_stopping": True, "early_stopping_patience": t.patience})
        kwargs.update(cfg.model.extra.get("train_kwargs", {}))

        with Timer() as timer:
            model.train(**kwargs)

        for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                     "checkpoint_best_regular.pth", "checkpoint.pth"):
            if (out_dir / name).exists():
                best = out_dir / name
                break
        else:
            raise FileNotFoundError(f"RF-DETR 체크포인트를 찾지 못했습니다: {out_dir}")
        return TrainResult(best_weights=str(best), run_dir=str(out_dir),
                           train_seconds=timer.seconds, epochs_done=t.epochs)


def _coco_names() -> Dict[int, str]:
    for mod in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
        try:
            return dict(__import__(mod, fromlist=["COCO_CLASSES"]).COCO_CLASSES)
        except (ImportError, AttributeError):
            continue
    raise ImportError("rfdetr COCO_CLASSES 를 찾지 못했습니다")


class RFDETRSegPredictor(Predictor):
    def __init__(self, cfg: ExperimentConfig, weights: str = "", label_mode: str = "finetuned"):
        super().__init__(cfg, weights, label_mode)
        is_pretrained = label_mode == "coco" or weights in ("", "DEFAULT")
        self.model = _build(cfg, "" if is_pretrained else weights)
        if is_pretrained:
            self.label_mode, names = "coco", _coco_names()
        else:
            names = getattr(self.model, "class_names", None) or {}
            if isinstance(names, (list, tuple)):
                names = dict(enumerate(names))
            if not any(str(v).lower() in CLASS_NAMES for v in dict(names).values()):
                names = {i + 1: n for i, n in enumerate(CLASS_NAMES)}  # prepare_dataset 의 번호 규칙
        self.mapper = LabelMapper(names, mode=self.label_mode, offset=1)
        print(f"[rfdetr] {weights or 'COCO 사전학습'}  클래스 매핑: {self.mapper.describe()}")

    def predict(self, image: np.ndarray) -> InstancePrediction:
        from PIL import Image

        h, w = image.shape[:2]
        det = self.model.predict(Image.fromarray(image), threshold=self.cfg.eval.conf)
        if det is None or len(det) == 0 or det.mask is None:
            return InstancePrediction.empty(h, w)
        names = (det.data or {}).get("class_name")
        if names is not None and len(names) == len(det):
            # rfdetr 가 붙여 주는 클래스 이름으로 직접 매핑 (번호 규칙이 버전마다 달라도 안전)
            wanted = {n: i for i, n in enumerate(CLASS_NAMES)}
            labels = np.asarray([wanted.get(str(n).lower(), -1) for n in names], np.int64)
            keep = labels >= 0
            det, labels = det[keep], labels[keep]
            if len(det) == 0:
                return InstancePrediction.empty(h, w)
            self.mapper = LabelMapper({i: n for i, n in enumerate(CLASS_NAMES)})
        else:
            labels = det.class_id
        return self._finalize(boxes=det.xyxy, scores=det.confidence, labels=labels,
                              masks=det.mask, height=h, width=w)
