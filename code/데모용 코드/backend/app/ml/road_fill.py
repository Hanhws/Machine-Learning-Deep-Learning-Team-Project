"""도로 마스크에 차량 자리를 채운다.

점유율의 분모는 '차량이 덮을 수 있는 노면 전체' 여야 한다. SAM3 의 "road" 는 보이는 노면만 잡으므로
스냅샷에 서 있던 차량 자리가 구멍/홈으로 빠지고, 막힐수록 분모가 작아져 점유율이 부풀려진다.
그래서 도로 단계는 SAM3 만으로: 한 번의 호출로 road 와 car/truck/bus 를 같이 찾고,
도로에 닿아 있는 차량만 도로에 합친다(주차장·측도의 차량은 제외). 도로로 둘러싸인 작은 구멍도 메운다.
"""

from __future__ import annotations

import cv2
import numpy as np

TOUCH_PX = 12  # 차량과 도로 사이 이 정도 틈은 닿은 것으로 본다
MIN_TOUCH = 0.10  # 차량 둘레 중 도로와 닿은 비율 하한
HOLE_MAX_FRAC = 0.02  # 이보다 큰 구멍(프레임 대비)은 중앙분리대 등일 수 있어 메우지 않는다
VEHICLE_CONCEPTS = ["car", "truck", "bus"]


def road_with_vehicles(img: np.ndarray, road_text: str = "road", conf: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SAM3 한 번 호출 → (도로 마스크, 차량 마스크). road_text 는 쉼표로 여러 개 가능."""
    from app.ml.registry import get_road_segmenter

    road_terms = [t.strip() for t in road_text.split(",") if t.strip()] or ["road"]
    road_terms = [t for t in road_terms if t not in VEHICLE_CONCEPTS] or ["road"]
    masks = get_road_segmenter().segment_concepts(img, road_terms + VEHICLE_CONCEPTS, conf)
    road = np.zeros(img.shape[:2], bool)
    for t in road_terms:
        road |= masks[t]
    veh = np.zeros(img.shape[:2], bool)
    for t in VEHICLE_CONCEPTS:
        veh |= masks[t]
    return road, veh


def detect_vehicles(img: np.ndarray) -> np.ndarray:
    """스냅샷의 차량 합집합 마스크 (SAM3: car, truck, bus)."""
    from app.ml.registry import get_road_segmenter

    masks = get_road_segmenter().segment_concepts(img, VEHICLE_CONCEPTS)
    out = np.zeros(img.shape[:2], bool)
    for m in masks.values():
        out |= m
    return out


def _components(mask: np.ndarray):
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield i, cc, (x, y, w, h), area


def _ring_labels(label: np.ndarray, comp: np.ndarray, box, pad: int) -> tuple[int, float]:
    """차량 컴포넌트 둘레(pad px) 에서 가장 많이 닿은 도로 라벨과 그 비율."""
    x, y, w, h = box
    H, W = label.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    c = comp[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    ring &= c == 0
    if not ring.any():
        return 0, 0.0
    labs = label[y0:y1, x0:x1][ring]
    nz = labs[labs > 0]
    if nz.size == 0:
        return 0, 0.0
    vals, cnts = np.unique(nz, return_counts=True)
    k = int(np.argmax(cnts))
    return int(vals[k]), float(cnts[k] / ring.sum())


def _fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """mask 로 완전히 둘러싸인 작은 구멍을 메운다 (테두리에 닿은 빈 곳은 그대로)."""
    inv = (~mask).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = mask.shape
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if area <= max_area:
            out[cc == i] = True
    return out


def fill_label(label: np.ndarray, vehicles: np.ndarray) -> tuple[np.ndarray, int]:
    """방향 라벨맵(0=도로 아님, 1..N) 에 차량 자리를 채운다. 각 차량은 가장 많이 닿은 방향 라벨로. (결과, 추가된 픽셀 수)"""
    out = label.copy()
    before = int((out > 0).sum())
    for i, cc, box, area in _components(vehicles & (label == 0)):
        comp = cc == i
        lab, frac = _ring_labels(out, comp, box, TOUCH_PX)
        if lab and frac >= MIN_TOUCH:
            out[comp] = lab
    # 방향별로 완전히 둘러싸인 구멍 메우기
    max_area = int(HOLE_MAX_FRAC * label.size)
    for d in [int(v) for v in np.unique(out) if v > 0]:
        m = out == d
        filled = _fill_holes(m, max_area)
        out[filled & (out == 0)] = d
    return out, int((out > 0).sum()) - before


def fill_binary(road: np.ndarray, vehicles: np.ndarray) -> tuple[np.ndarray, int]:
    lab, added = fill_label(road.astype(np.uint8), vehicles)
    return lab > 0, added
