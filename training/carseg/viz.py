"""추론 결과 시각화 — 인스턴스 마스크·상자·라벨 + 요약 패널."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import CLASS_NAMES
from .models.base import InstancePrediction
from .occupancy import VehicleStats

# RGB — car 파랑, bus 주황, truck 초록 (data_processing 마스크 팔레트와 구분되는 선명한 색)
CLASS_COLORS = {0: (40, 120, 255), 1: (255, 150, 30), 2: (40, 200, 90)}


def draw_prediction(image: np.ndarray, pred: InstancePrediction, *, score_thr: float = 0.5,
                    stats: Optional[VehicleStats] = None, road_mask: Optional[np.ndarray] = None,
                    alpha: float = 0.45, show_union: bool = False) -> np.ndarray:
    """image: RGB uint8. 반환도 RGB."""
    import cv2

    canvas = image.copy()
    pred = pred.above(score_thr)
    if road_mask is not None:  # 도로는 옅은 보라
        canvas[road_mask] = (canvas[road_mask] * 0.75 + np.array([150, 80, 200]) * 0.25).astype(np.uint8)
    overlay = canvas.copy()
    for i in np.argsort(pred.scores):  # 낮은 점수부터 칠해 높은 점수가 위에 오게
        overlay[pred.masks[i]] = CLASS_COLORS[int(pred.labels[i])]
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    if show_union and len(pred):
        contours, _ = cv2.findContours(pred.masks.any(0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)

    thick = max(1, round(min(image.shape[:2]) / 540))
    for box, score, label in zip(pred.boxes, pred.scores, pred.labels):
        color = CLASS_COLORS[int(label)]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        text = f"{CLASS_NAMES[int(label)]} {score:.2f}"
        cv2.putText(canvas, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick, color, thick, cv2.LINE_AA)

    if stats is not None:
        lines = [f"vehicles: {stats.num_vehicles}  " +
                 "  ".join(f"{k}={v}" for k, v in stats.counts.items()),
                 f"vehicle px: {stats.vehicle_pixels:,}  ({stats.image_ratio * 100:.2f}% of image)"]
        if stats.occupancy is not None:
            lines.append(f"road px: {stats.road_pixels:,}  occupancy: {stats.occupancy * 100:.2f}%")
        scale = 0.5 * thick
        pad, lh = 8 * thick, int(22 * thick)
        box_w = int(max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines) + 2 * pad)
        panel = canvas[: lh * len(lines) + pad, :box_w]
        canvas[: panel.shape[0], : panel.shape[1]] = (panel * 0.35).astype(np.uint8)
        for j, l in enumerate(lines):
            cv2.putText(canvas, l, (pad, lh * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
    return canvas


def save_rgb(image: np.ndarray, path: str | Path) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path
