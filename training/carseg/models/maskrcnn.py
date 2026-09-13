"""Mask R-CNN (torchvision).

설치가 필요 없는(torchvision 내장) 2-stage 기준 모델.
    model.weights : "DEFAULT"(COCO 사전학습) 또는 학습된 체크포인트 경로(.pt)
    model.extra   :
        arch                       maskrcnn_resnet50_fpn_v2 (기본) | maskrcnn_resnet50_fpn
        trainable_backbone_layers  0~5 (기본 3) — 클수록 백본을 더 많이 학습
        max_size                   긴 변 상한 (기본 imgsz*2 → 16:9 FHD 가 잘리지 않게)
        anchor_sizes               [[16],[32],[64],[128],[256]] 처럼 작은 앵커 (소형 차량 83%)
        box_nms_thresh             ROI NMS IoU (기본 0.5)
imgsz 는 짧은 변 크기(min_size)로 쓴다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .. import CLASS_NAMES
from ..config import ExperimentConfig
from ..data import CarSegData, CocoInstanceDataset
from ..utils import resolve_device, set_seed
from .base import InstancePrediction, LabelMapper, Predictor, Trainer, TrainResult

NUM_CLASSES = len(CLASS_NAMES) + 1  # + background
ARCHES = ("maskrcnn_resnet50_fpn_v2", "maskrcnn_resnet50_fpn")


def _weights_enum(arch: str):
    from torchvision.models import detection as D

    return {"maskrcnn_resnet50_fpn_v2": D.MaskRCNN_ResNet50_FPN_V2_Weights,
            "maskrcnn_resnet50_fpn": D.MaskRCNN_ResNet50_FPN_Weights}[arch].DEFAULT


def build_model(cfg: ExperimentConfig, *, coco_head: bool, load_coco: bool = True):
    """coco_head=True 면 91클래스(COCO) 그대로, False 면 우리 4클래스(배경 포함) 헤드로 교체."""
    from torchvision.models import detection as D
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    ex = cfg.model.extra
    arch = ex.get("arch", ARCHES[0])
    if arch not in ARCHES:
        raise ValueError(f"model.extra.arch 는 {ARCHES} 중 하나: {arch}")
    imgsz = cfg.train.imgsz
    kwargs: Dict[str, Any] = {
        "min_size": imgsz,
        "max_size": int(ex.get("max_size", imgsz * 2)),
        "box_detections_per_img": cfg.eval.max_det,
        "box_score_thresh": cfg.eval.conf,
        "box_nms_thresh": float(ex.get("box_nms_thresh", 0.5)),
    }
    builder = getattr(D, arch)
    model = builder(weights=_weights_enum(arch) if load_coco else None,
                    weights_backbone=None,
                    trainable_backbone_layers=int(ex.get("trainable_backbone_layers", 3)),
                    **kwargs)
    if ex.get("anchor_sizes") and not coco_head:
        # 생성 후 교체 (v2 빌더는 앵커 생성기를 내부에서 넘기므로 kwargs 로는 못 바꾼다).
        # 위치당 앵커 수(비율 3개)가 같아서 사전학습 RPN 헤드를 그대로 쓸 수 있다.
        # COCO 헤드 그대로 평가(zero-shot)할 때는 사전학습 앵커를 유지한다.
        sizes = tuple(tuple(int(v) for v in s) for s in ex["anchor_sizes"])
        model.rpn.anchor_generator = AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * len(sizes))
    if not coco_head:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)
    return model


def load_state(model, state: Dict[str, Any]) -> None:
    """체크포인트 로드. 사전학습 없이 만든 모델은 BatchNorm 이 Frozen 이 아니어서
    num_batches_tracked 키만 다를 수 있다 — 그 외 불일치는 오류로 알린다."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.endswith("num_batches_tracked")]
    if bad:
        raise RuntimeError(f"체크포인트와 모델 구조가 다릅니다 (arch/anchor_sizes 확인): {bad[:8]}")


def _to_tensor_batch(samples) -> tuple:
    import torch

    images, targets = [], []
    for s in samples:
        images.append(torch.from_numpy(s.image).permute(2, 0, 1).float().div_(255.0))
        targets.append({
            "boxes": torch.from_numpy(s.boxes).float(),
            "labels": torch.from_numpy(s.labels + 1).long(),   # 0 은 배경
            "masks": torch.from_numpy(s.masks).to(torch.uint8),
        })
    return images, targets


def _collate(batch):
    return batch


class MaskRCNNTrainer(Trainer):
    def train(self) -> TrainResult:
        import torch
        from torch.utils.data import DataLoader

        from .torch_loop import TorchLoop

        cfg, t = self.cfg, self.cfg.train
        set_seed(t.seed)
        data = CarSegData(cfg.data.root)
        ds = CocoInstanceDataset(data, cfg.data.train_split, limit=cfg.data.train_limit, augment=True,
                                 hflip=t.hflip, vflip=t.vflip, color_jitter=t.color_jitter, seed=t.seed)
        loader = DataLoader(ds, batch_size=t.batch, shuffle=True, num_workers=t.workers,
                            collate_fn=_collate, pin_memory=True, drop_last=True,
                            persistent_workers=t.workers > 0)

        weights = cfg.model.weights
        if weights and weights != "DEFAULT" and Path(weights).exists():
            model = build_model(cfg, coco_head=False, load_coco=True)  # 학습 때와 같은 구조(FrozenBN)
            load_state(model, torch.load(weights, map_location="cpu", weights_only=False)["model"])
            print(f"[maskrcnn] 체크포인트에서 시작: {weights}")
        else:
            model = build_model(cfg, coco_head=False, load_coco=True)
            print("[maskrcnn] COCO 사전학습 가중치에서 시작 (헤드는 3+1 클래스로 교체)")

        def step_fn(m, batch, device):
            images, targets = _to_tensor_batch(batch)
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in tg.items()} for tg in targets]
            losses = m(images, targets)
            losses["loss"] = sum(losses.values())
            return losses

        meta = {"family": "maskrcnn", "arch": cfg.model.extra.get("arch", ARCHES[0]),
                "num_classes": NUM_CLASSES, "class_names": list(CLASS_NAMES)}
        loop = TorchLoop(cfg, model, loader, step_fn,
                         make_predictor=lambda m: MaskRCNNPredictor(cfg, model=m), checkpoint_meta=meta)
        return loop.run()


class MaskRCNNPredictor(Predictor):
    def __init__(self, cfg: ExperimentConfig, weights: str = "DEFAULT", label_mode: str = "finetuned",
                 model=None):
        super().__init__(cfg, weights, label_mode)
        import torch

        self.device = resolve_device(cfg.train.device)
        if model is not None:                                   # 학습 루프에서 넘겨받은 모델
            self.model = model
            names = {i + 1: n for i, n in enumerate(CLASS_NAMES)}
        elif label_mode == "coco" or weights in ("", "DEFAULT"):
            self.model = build_model(cfg, coco_head=True, load_coco=True)
            arch = cfg.model.extra.get("arch", ARCHES[0])
            names = dict(enumerate(_weights_enum(arch).meta["categories"]))
            self.label_mode = "coco"
        else:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            if ckpt.get("arch"):
                cfg.model.extra.setdefault("arch", ckpt["arch"])
            self.model = build_model(cfg, coco_head=False, load_coco=False)
            load_state(self.model, ckpt["model"])
            names = {i + 1: n for i, n in enumerate(ckpt.get("class_names", CLASS_NAMES))}
        # 추론 설정 반영 (학습 때와 달라도 됨)
        self.model.roi_heads.score_thresh = cfg.eval.conf
        self.model.roi_heads.detections_per_img = cfg.eval.max_det
        self.model.to(self.device).eval()
        self.mapper = LabelMapper(names, mode=self.label_mode)
        if model is None:
            print(f"[maskrcnn] {weights}  클래스 매핑: {self.mapper.describe()}")

    def predict(self, image: np.ndarray) -> InstancePrediction:
        import torch

        h, w = image.shape[:2]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).to(self.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        if len(out["scores"]) == 0:
            return InstancePrediction.empty(h, w)
        return self._finalize(
            boxes=out["boxes"].cpu().numpy(),
            scores=out["scores"].cpu().numpy(),
            labels=out["labels"].cpu().numpy(),
            masks=(out["masks"][:, 0] > self.cfg.eval.mask_threshold).cpu().numpy(),
            height=h, width=w,
        )
