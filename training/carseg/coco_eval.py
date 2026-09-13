"""통합 평가기 — 어떤 계열이든 Predictor 만 있으면 같은 기준으로 채점한다.

1) COCO mAP (pycocotools COCOeval)
   - segm(마스크) · bbox 각각 AP, AP50, AP75, AP_small/medium/large, 클래스별 AP
   - ultralytics 내장 val 과 수치가 조금 다를 수 있다(보간 방식 차이). 모델 간 비교는 이 값으로 한다.
2) 점유율 지표 (프로젝트 목적에 직결)
   - 예측 차량 합집합 마스크 vs 정답 마스크(masks/*.png > 0)
   - vehicle_IoU        : 데이터셋 전체 합산 IoU
   - area_ratio_MAE_pp  : |예측 면적비 − 정답 면적비| 평균 (%p, 면적비 = 차량 픽셀 / 이미지 픽셀)
   - count_MAE          : |예측 대수 − 정답 대수| 평균
   union_conf 이상인 예측만 사용 (mAP 용 낮은 conf 로 합집합을 만들면 과대 추정되므로).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import CLASS_NAMES
from .data import CLASS_TO_CATID, CarSegData
from .models.base import InstancePrediction, Predictor, read_rgb
from .utils import save_json

_STAT_NAMES = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
               "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def encode_prediction(pred: InstancePrediction, image_id: int) -> List[Dict[str, Any]]:
    """InstancePrediction → COCO results 형식 (마스크는 RLE)."""
    from pycocotools import mask as mask_util

    out = []
    for box, score, label, mask in zip(pred.boxes, pred.scores, pred.labels, pred.masks):
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        x1, y1, x2, y2 = [float(v) for v in box]
        out.append({
            "image_id": int(image_id),
            "category_id": CLASS_TO_CATID[int(label)],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
            "segmentation": rle,
        })
    return out


def run_cocoeval(coco_gt, results: List[Dict[str, Any]], iou_type: str,
                 img_ids: List[int]) -> Dict[str, Any]:
    """COCOeval 을 돌려 요약 지표 + 클래스별 AP 를 dict 로 돌려준다."""
    from pycocotools.cocoeval import COCOeval

    if not results:
        metrics = {k: 0.0 for k in _STAT_NAMES}
        metrics.update({f"AP_{n}": 0.0 for n in CLASS_NAMES})
        return metrics

    dets = results
    if iou_type == "bbox":  # bbox 평가에 segmentation 이 있으면 pycocotools 가 상자를 다시 계산한다
        dets = [{k: v for k, v in r.items() if k != "segmentation"} for r in results]
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metrics = {name: float(v) for name, v in zip(_STAT_NAMES, ev.stats)}
    precision = ev.eval["precision"]  # [T IoU, R recall, K class, A area, M maxDets]
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        name = coco_gt.cats[cat_id]["name"]
        metrics[f"AP_{name}"] = float(p.mean()) if p.size else 0.0
        p50 = precision[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        metrics[f"AP50_{name}"] = float(p50.mean()) if p50.size else 0.0
    return metrics


class _UnionMeter:
    """합집합 마스크 기반 점유율 지표 누적기."""

    def __init__(self):
        self.inter = 0
        self.union = 0
        self.ratio_err: List[float] = []
        self.count_err: List[int] = []

    def update(self, pred: InstancePrediction, gt_mask: np.ndarray, gt_count: int) -> None:
        pred_union = pred.masks.any(axis=0) if len(pred) else np.zeros(gt_mask.shape, bool)
        if pred_union.shape != gt_mask.shape:
            return
        self.inter += int(np.logical_and(pred_union, gt_mask).sum())
        self.union += int(np.logical_or(pred_union, gt_mask).sum())
        area = gt_mask.size
        self.ratio_err.append(abs(pred_union.sum() - gt_mask.sum()) / area * 100.0)
        self.count_err.append(abs(len(pred) - gt_count))

    def result(self) -> Dict[str, float]:
        if not self.ratio_err:
            return {}
        return {
            "vehicle_IoU": self.inter / max(self.union, 1),
            "area_ratio_MAE_pp": float(np.mean(self.ratio_err)),
            "count_MAE": float(np.mean(self.count_err)),
        }


def evaluate(predictor: Predictor, data: CarSegData, split: str, *,
             limit: int = 0, out_dir: Optional[str | Path] = None,
             save_predictions: bool = False, progress: bool = True) -> Dict[str, Any]:
    """split 전체(또는 앞 limit 장)를 예측하고 채점한다.

    반환 dict 키 예: mask_mAP, mask_mAP50, box_mAP, mask_AP_car, vehicle_IoU, fps ...
    """
    import cv2

    coco_gt = data.coco(split)
    img_ids = sorted(coco_gt.imgs)
    if limit:
        img_ids = img_ids[:limit]
    union_conf = predictor.cfg.eval.union_conf
    masks_dir = data.masks_dir(split)

    iterator = img_ids
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(img_ids, desc=f"eval[{split}]", unit="img", dynamic_ncols=True)

    results: List[Dict[str, Any]] = []
    meter = _UnionMeter()
    infer_seconds = 0.0
    for image_id in iterator:
        info = coco_gt.imgs[image_id]
        image = read_rgb(data.images_dir(split) / info["file_name"])
        t0 = time.time()
        pred = predictor.predict(image)
        infer_seconds += time.time() - t0
        results.extend(encode_prediction(pred, image_id))

        gt_path = masks_dir / f"{Path(info['file_name']).stem}.png"
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
            gt_count = len(coco_gt.getAnnIds(imgIds=image_id, iscrowd=False))
            meter.update(pred.above(union_conf), gt, gt_count)

    metrics: Dict[str, Any] = {"split": split, "n_images": len(img_ids), "n_predictions": len(results)}
    for iou_type, prefix in (("segm", "mask"), ("bbox", "box")):
        print(f"\n===== COCOeval {iou_type} ({split}) =====")
        for k, v in run_cocoeval(coco_gt, results, iou_type, img_ids).items():
            metrics[f"{prefix}_{k}"] = round(v, 4)
    metrics.update({k: round(v, 4) for k, v in meter.result().items()})
    metrics["fps"] = round(len(img_ids) / max(infer_seconds, 1e-9), 2)

    if out_dir:
        out_dir = Path(out_dir)
        save_json(metrics, out_dir / f"metrics_{split}.json")
        if save_predictions:
            save_json(results, out_dir / f"predictions_{split}.json")
    print_metrics(metrics)
    return metrics


def print_metrics(m: Dict[str, Any]) -> None:
    keys = ["mask_mAP", "mask_mAP50", "mask_mAP75", "mask_mAP_small", "box_mAP", "box_mAP50",
            *[f"mask_AP_{n}" for n in CLASS_NAMES], "vehicle_IoU", "area_ratio_MAE_pp", "count_MAE", "fps"]
    print("\n" + "-" * 44)
    print(f" 평가 요약 — split={m.get('split')}  images={m.get('n_images')}")
    print("-" * 44)
    for k in keys:
        if k in m:
            print(f" {k:<22} {m[k]}")
    print("-" * 44)
