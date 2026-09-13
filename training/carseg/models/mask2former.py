"""Mask2Former (HuggingFace transformers).

    model.weights : 허브 ID 또는 학습된 체크포인트 폴더
        facebook/mask2former-swin-tiny-coco-instance
        facebook/mask2former-swin-small-coco-instance   (기본)
        facebook/mask2former-swin-base-coco-instance
        facebook/mask2former-swin-large-coco-instance
    model.extra   :
        image_size  [H, W]  학습·추론 입력 크기 (기본 [imgsz, imgsz*16/9 를 32 배수로])
                    — 고정 크기로 리사이즈하므로 16:9 가 아닌 영상(세로 1080×1920)은 비율이 바뀐다
        topk        인스턴스 후보 수 (기본 eval.max_det)

추론 후처리는 HF 의 post_process_instance_segmentation 대신 직접 구현한다.
HF 구현은 마스크를 384×384 로 줄였다 키우기 때문에 소형 차량(전체의 83%)이 뭉개진다.
여기서는 입력 해상도로 올린 뒤 원본 크기로 맞춘다 (Mask2Former 논문의 instance inference 와 동일:
query×class top-k, 점수 = 클래스 확률 × 마스크 내부 평균 확률).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .. import CLASS_NAMES
from ..config import ExperimentConfig
from ..data import CarSegData, CocoInstanceDataset, letterbox_free_resize
from ..utils import resolve_device, set_seed
from .base import (InstancePrediction, LabelMapper, Predictor, Trainer, TrainResult,
                   boxes_from_masks)

DEFAULT_WEIGHTS = "facebook/mask2former-swin-small-coco-instance"


def input_size(cfg: ExperimentConfig) -> Tuple[int, int]:
    size = cfg.model.extra.get("image_size")
    if size:
        return int(size[0]), int(size[1])
    h = cfg.train.imgsz
    w = int(round(h * 16 / 9 / 32) * 32)
    return h, w


def _norm_stats(weights: str) -> Tuple[np.ndarray, np.ndarray]:
    from transformers import AutoImageProcessor

    try:
        proc = AutoImageProcessor.from_pretrained(weights)
        mean, std = proc.image_mean, proc.image_std
    except Exception:  # 체크포인트 폴더에 전처리 설정이 없을 때 — ImageNet 기본값
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return np.asarray(mean, np.float32), np.asarray(std, np.float32)


def _to_pixel_values(image: np.ndarray, mean: np.ndarray, std: np.ndarray):
    import torch

    x = (image.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


class Mask2FormerTrainer(Trainer):
    def train(self) -> TrainResult:
        import torch
        from torch.utils.data import DataLoader
        from transformers import Mask2FormerForUniversalSegmentation

        from .torch_loop import TorchLoop

        cfg, t = self.cfg, self.cfg.train
        set_seed(t.seed)
        weights = cfg.model.weights or DEFAULT_WEIGHTS
        size = input_size(cfg)
        mean, std = _norm_stats(weights)

        id2label = dict(enumerate(CLASS_NAMES))
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            weights, id2label=id2label, label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,  # 분류 헤드 80+1 → 3+1
        )
        if t.freeze:
            for p in model.model.pixel_level_module.encoder.parameters():
                p.requires_grad = False
            print("[mask2former] 백본(encoder) 동결")

        data = CarSegData(cfg.data.root)
        ds = CocoInstanceDataset(data, cfg.data.train_split, limit=cfg.data.train_limit, augment=True,
                                 hflip=t.hflip, vflip=t.vflip, color_jitter=t.color_jitter, seed=t.seed)

        def collate(samples):
            pixel_values, mask_labels, class_labels = [], [], []
            for s in samples:
                if len(s.labels) == 0:          # 정답 없는 이미지는 매칭 손실이 정의되지 않음
                    continue
                img, masks = letterbox_free_resize(s.image, s.masks, size)
                keep = masks.reshape(len(masks), -1).any(axis=1)  # 축소 후 사라진 초소형 객체 제거
                if not keep.any():
                    continue
                pixel_values.append(_to_pixel_values(img, mean, std))
                mask_labels.append(torch.from_numpy(masks[keep]).float())
                class_labels.append(torch.from_numpy(s.labels[keep]).long())
            if not pixel_values:
                return None
            return {"pixel_values": torch.stack(pixel_values),
                    "mask_labels": mask_labels, "class_labels": class_labels}

        loader = DataLoader(ds, batch_size=t.batch, shuffle=True, num_workers=t.workers,
                            collate_fn=collate, pin_memory=True, drop_last=True,
                            persistent_workers=t.workers > 0)

        def step_fn(m, batch, device):
            if batch is None:
                return None
            out = m(pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    mask_labels=[x.to(device) for x in batch["mask_labels"]],
                    class_labels=[x.to(device) for x in batch["class_labels"]])
            return {"loss": out.loss}

        meta = {"family": "mask2former", "base_weights": weights, "image_size": list(size),
                "class_names": list(CLASS_NAMES)}
        loop = TorchLoop(cfg, model, loader, step_fn,
                         make_predictor=lambda m: Mask2FormerPredictor(cfg, weights, model=m),
                         checkpoint_meta=meta)
        result = loop.run()

        # HF 형식 폴더로도 저장 → from_pretrained 로 바로 불러올 수 있게
        best = torch.load(result.best_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        hf_dir = cfg.run_dir / "weights" / "best_hf"
        model.save_pretrained(hf_dir)
        try:
            from transformers import AutoImageProcessor

            AutoImageProcessor.from_pretrained(weights).save_pretrained(hf_dir)
        except Exception:
            pass
        result.extra["hf_dir"] = str(hf_dir)
        result.best_weights = str(hf_dir)
        return result


class Mask2FormerPredictor(Predictor):
    def __init__(self, cfg: ExperimentConfig, weights: str = DEFAULT_WEIGHTS,
                 label_mode: str = "finetuned", model=None):
        super().__init__(cfg, weights, label_mode)
        from transformers import Mask2FormerForUniversalSegmentation

        self.device = resolve_device(cfg.train.device)
        weights = weights or DEFAULT_WEIGHTS
        verbose = model is None  # 학습 중 채점용으로 만들 때는 조용히
        if model is None:
            if str(weights).endswith((".pt", ".pth")):       # TorchLoop 체크포인트
                import torch

                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
                base = ckpt.get("base_weights", DEFAULT_WEIGHTS)
                id2label = dict(enumerate(ckpt.get("class_names", CLASS_NAMES)))
                model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    base, id2label=id2label, label2id={v: k for k, v in id2label.items()},
                    ignore_mismatched_sizes=True)
                model.load_state_dict(ckpt["model"])
                weights_for_norm = base
            else:                                             # 허브 ID 또는 save_pretrained 폴더
                model = Mask2FormerForUniversalSegmentation.from_pretrained(weights)
                weights_for_norm = weights
        else:
            weights_for_norm = weights
        self.model = model.to(self.device).eval()
        self.mean, self.std = _norm_stats(weights_for_norm)
        self.size = input_size(cfg)
        self.mapper = LabelMapper(self.model.config.id2label, mode=label_mode)
        if verbose:
            print(f"[mask2former] {weights}  입력 {self.size}  클래스 매핑: {self.mapper.describe()}")

    def predict(self, image: np.ndarray) -> InstancePrediction:
        import cv2
        import torch
        import torch.nn.functional as F

        e = self.cfg.eval
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        pixel_values = _to_pixel_values(resized, self.mean, self.std)[None].to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=pixel_values)

        class_logits = out.class_queries_logits[0]          # (Q, K+1)
        mask_logits = out.masks_queries_logits[0]           # (Q, h/4, w/4)
        num_q, num_k = class_logits.shape[0], class_logits.shape[1] - 1
        probs = class_logits.softmax(-1)[:, :-1]            # no-object 제외
        topk = min(int(self.cfg.model.extra.get("topk", e.max_det)), num_q * num_k)
        scores, flat_idx = probs.flatten().topk(topk)
        query_idx = torch.div(flat_idx, num_k, rounding_mode="floor")
        labels = flat_idx % num_k

        keep = scores >= e.conf
        scores, labels, query_idx = scores[keep], labels[keep], query_idx[keep]
        if len(scores) == 0:
            return InstancePrediction.empty(h, w)

        masks_out, final_scores = [], []
        for chunk in torch.split(torch.arange(len(scores), device=scores.device), 16):
            logits = mask_logits[query_idx[chunk]][None]    # (1, n, h/4, w/4)
            logits = F.interpolate(logits, size=self.size, mode="bilinear", align_corners=False)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)[0]
            prob = logits.sigmoid()
            binary = prob > e.mask_threshold
            area = binary.flatten(1).sum(1).clamp(min=1)
            mask_score = (prob * binary).flatten(1).sum(1) / area
            final_scores.append(scores[chunk] * mask_score)
            masks_out.append(binary.cpu())
        masks = torch.cat(masks_out).numpy()
        final = torch.cat(final_scores).cpu().numpy()
        nonempty = masks.reshape(len(masks), -1).any(axis=1)
        masks, final, labels_np = masks[nonempty], final[nonempty], labels.cpu().numpy()[nonempty]
        return self._finalize(boxes=boxes_from_masks(masks), scores=final, labels=labels_np,
                              masks=masks, height=h, width=w)
