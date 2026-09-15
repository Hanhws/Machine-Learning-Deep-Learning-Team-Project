"""2) 도로 · 방향별 도로(carriageway) · 차로를 라벨 없이 자동 추출.

1. 차 없는 배경 사진(s1)에 Mapillary Vistas 사전학습 Mask2Former → 도로, 차선 도색, 중앙분리대
2. 차선 도색 · 분리대 직선으로 소실점(VP) 추정 (도로가 곧은 화면 아래쪽 직선에 가중치)
3. 픽셀을 "소실점에서 본 각도(θ)"로 바꾸면 평행한 차선이 각각 θ 하나로 표현된다
4. θ 마다 도로 비율과 분리대 비율을 재서, 분리대로 끊기는 θ 구간 = 방향별 도로
5. 방향별 도로 안의 차선 도색 θ 봉우리 = 차로 경계 (도색이 안 보이면 차량 바닥점 봉우리 = 차로 중심)
6. 우측통행: 화면 왼쪽 도로는 카메라로 다가오고(toward), 오른쪽 도로는 멀어진다(away)
7. 운전자 기준 바깥쪽 가장자리 칸에 차가 거의 없으면 갓길

출력 (outputs/s2/<clip_id>/)
  seg.png        Mask2Former 클래스 지도 (작업 해상도)
  lanes.npz      lane_map (0=차로 아님, k=차로 id), theta, vp
  lanes.json     방향별 도로·차로 정보
  lanes_vis.jpg  확인용 그림
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks

from common import OUT, Clip, list_clips

M2F_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
# Road, 차선 도색, 횡단보도 도색, Service Lane, Bike Lane, 맨홀, 배수구, 포트홀
ROAD_IDS = [13, 24, 23, 14, 7, 41, 36, 43]
MARKING_ID = 24
# Guard Rail, Barrier, Curb, Wall, Fence
SEPARATOR_IDS = [4, 5, 2, 6, 3]

THETA_BIN = 0.25  # 도
ROAD_COVER_MIN = 0.3  # 이 θ 방향 광선의 30% 이상이 도로여야 도로 구간
SEP_COVER_CUT = 0.1  # 광선의 10% 이상이 분리대면 방향별 도로를 끊는다
# 실선/점선: Mask2Former 는 점선 사이를 이어 칠하므로 배경 사진 밝기로 판정한다
PAINT_BAND_DEG = 0.35
PAINT_MIN_COV = 0.15  # 경계를 따라 밝은 구간이 이보다 적으면 가짜 경계
SOLID_COV = 0.85  # 밝은 구간이 이 이상이고 끊김이 2토막 이하면 실선(가장자리선)


class Segmenter:
    def __init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(M2F_ID)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_ID).to("mps").eval()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = bgr.shape[:2]
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # 기본 384×384 는 가는 점선이 끊긴다 → 원본 해상도(작업 해상도 2배)로 넣는다
        inp = self.proc(images=img, return_tensors="pt", size={"height": h * 2, "width": w * 2}).to("mps")
        with self.torch.no_grad():
            out = self.model(**inp)
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
        return seg.cpu().numpy().astype(np.uint8)


def foot_points(dets: list[dict]) -> np.ndarray:
    """차량 마스크의 가장 아래쪽 중앙 = 바퀴가 닿는 위치."""
    pts = []
    for d in dets:
        p = d["poly"]
        if len(p) < 3:
            continue
        ymax = p[:, 1].max()
        pts.append((p[p[:, 1] >= ymax - 3, 0].mean(), ymax))
    return np.array(pts, np.float32).reshape(-1, 2)


# ---------------------------------------------------------------- 소실점

def _hough(img: np.ndarray, min_len: int) -> np.ndarray | None:
    lines = cv2.HoughLinesP(img, 1, np.pi / 360, threshold=25, minLineLength=min_len, maxLineGap=10)
    return None if lines is None else lines.reshape(-1, 4).astype(np.float64)  # OpenCV 4: (N,1,4), 5: (N,4)


def _ransac_vp(S: np.ndarray, h: int, w: int, seed: int = 0):
    p1, p2 = S[:, :2], S[:, 2:]
    d = p2 - p1
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    keep = (L >= 15) & (ang > 12) & (ang < 168)  # 수평에 가까운 선(육교, 정지선)은 제외
    p1, d, L = p1[keep], d[keep], L[keep]
    if len(L) < 4:
        return None
    u = d / L[:, None]
    mid = p1 + d / 2
    nrm = np.c_[-u[:, 1], u[:, 0]]
    c = (nrm * mid).sum(1)
    # 휘어진 도로에서도 화면 아래쪽(가까운 곳)은 곧다 → 아래쪽 직선에 가중치
    wt = L * (mid[:, 1] / h).clip(0.05, 1) ** 2

    def score(vp):
        r = vp[None, :] - mid
        cos = np.abs((r * u).sum(1)) / (np.hypot(r[:, 0], r[:, 1]) + 1e-6)
        inl = np.degrees(np.arccos(np.clip(cos, -1, 1))) < 1.5
        return (wt * inl).sum(), inl

    rng = np.random.default_rng(seed)
    prob = wt / wt.sum()
    best, best_s = None, 0.0
    for _ in range(3000):
        i, j = rng.choice(len(L), 2, replace=False, p=prob)
        A = np.array([nrm[i], nrm[j]])
        if abs(np.linalg.det(A)) < 1e-3:
            continue
        vp = np.linalg.solve(A, [c[i], c[j]])
        if not (-3 * h < vp[1] < 0.75 * h and -2 * w < vp[0] < 3 * w):
            continue
        s, _ = score(vp)
        if s > best_s:
            best, best_s = vp, s
    if best is None:
        return None
    _, inl = score(best)
    ww = wt[inl]
    vp = np.linalg.lstsq(nrm[inl] * ww[:, None], c[inl] * ww, rcond=None)[0]
    return float(vp[0]), float(vp[1]), float(best_s / wt.sum()), float(L[inl].sum())


def estimate_vp(marking: np.ndarray, separator: np.ndarray, road: np.ndarray):
    h, w = marking.shape
    mk = _hough(marking * 255, 20)
    sp = _hough(cv2.Canny(separator * 255, 50, 150), 40)
    rd = _hough(cv2.Canny(road * 255, 50, 150), 60)
    res = None
    # 차선 도색·분리대가 충분하면 그것만 쓰고, 부족할 때만 도로 경계선을 더한다
    for group in ((mk, sp), (mk, sp, rd)):
        segs = [g for g in group if g is not None]
        if not segs:
            continue
        res = _ransac_vp(np.concatenate(segs), h, w)
        if res is not None and res[3] >= 400:
            break
    return None if res is None else res[:3]


# ---------------------------------------------------------------- 차로

def elongated_markings(marking: np.ndarray, theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """소실점 방향으로 길쭉한 도색만 남긴다 (노면 글씨·화면 자막 제거)."""
    n, lab = cv2.connectedComponents(marking, connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < 6:
            continue
        th, r = theta[ys, xs], radius[ys, xs]
        t_spread = np.radians(np.percentile(th, 95) - np.percentile(th, 5))
        r_spread = np.percentile(r, 95) - np.percentile(r, 5)
        keep[k] = r_spread / max(r.mean() * t_spread, 1.0) >= 2.5
    return keep[lab]


def theta_runs(ok: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """True 가 이어진 bin 구간들 (max_gap 이하의 끊김은 잇는다). 끝 index 는 포함하지 않는다."""
    runs, start, gap = [], None, 0
    for i, v in enumerate(ok):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(ok) - gap))
    return runs


def peak_positions(values: np.ndarray, weights: np.ndarray | None, lo: float, hi: float,
                   sigma: float, prominence: float, min_sep_deg: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(lo, hi + THETA_BIN, THETA_BIN)
    if len(values) == 0 or len(edges) < 3:
        return np.array([]), np.array([])
    hist, _ = np.histogram(values, bins=edges, weights=weights)
    hs = gaussian_filter1d(hist.astype(np.float64), sigma)
    if hs.max() <= 0:
        return np.array([]), np.array([])
    peaks, _ = find_peaks(hs, prominence=prominence * hs.max(), distance=max(2, int(min_sep_deg / THETA_BIN)))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers[peaks], hs[peaks]


def drop_narrow(bounds: list[float], strength: dict[float, float]) -> list[float]:
    """너무 좁은 칸을 만드는 경계를 없앤다."""
    b = sorted(bounds)
    while len(b) > 2:
        widths = np.diff(b)
        i = int(np.argmin(widths))
        if widths[i] >= 0.4 * np.median(widths):
            break
        if i == 0:
            b.pop(0)
        elif i == len(widths) - 1:
            b.pop()
        else:
            b.pop(i if strength.get(b[i], 0) < strength.get(b[i + 1], 0) else i + 1)
    return b


def boundary_paint(t: float, theta: np.ndarray, radius: np.ndarray, road_zone: np.ndarray,
                   gray: np.ndarray, asphalt: float) -> tuple[float, int]:
    """경계 θ 를 따라 4px 간격으로 가장 밝은 값을 보고 (도색 비율, 밝은 토막 수)."""
    band = road_zone & (np.abs(theta - t) <= PAINT_BAND_DEG)
    if band.sum() < 20:
        return 0.0, 0
    rb = (radius[band] // 4).astype(np.int64)
    vals = gray[band]
    order = np.argsort(rb, kind="stable")
    _, start = np.unique(rb[order], return_index=True)
    bright = np.maximum.reduceat(vals[order], start) > asphalt * 1.25 + 8
    runs = int(np.count_nonzero(np.diff(np.r_[0, bright.astype(np.int8)]) == 1))
    return float(bright.mean()), runs


def lane_geometry(seg: np.ndarray, pts: np.ndarray, bg: np.ndarray, row_min: int = 0) -> dict:
    """row_min > 0 이면 그 행보다 아래(카메라 가까운 곳)만 보고 소실점·방향별 도로·차로를 정한다.
    휜 도로에서는 먼 곳 차선이 소실점을 끌어당기므로, 곧은 가까운 구간만 쓰기 위한 옵션."""
    h, w = seg.shape
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road = cv2.morphologyEx(np.isin(seg, ROAD_IDS).astype(np.uint8), cv2.MORPH_CLOSE, k5)
    marking = (seg == MARKING_ID).astype(np.uint8)
    sep = np.isin(seg, SEPARATOR_IDS).astype(np.uint8)
    if row_min > 0:
        pts = pts[pts[:, 1] >= row_min]
        below = (np.arange(h)[:, None] >= row_min).astype(np.uint8)
        vp = estimate_vp(marking * below, sep * below, road * below)
    else:
        vp = estimate_vp(marking, sep, road)
    if vp is None:
        return {"ok": False, "reason": "소실점을 찾지 못함"}
    vx, vy, vp_support = vp

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.degrees(np.arctan2(xx - vx, yy - vy))
    radius = np.hypot(xx - vx, yy - vy)
    # 소실점 근처는 차로가 몇 픽셀로 뭉개져 각도가 불안정하다
    zone = (yy > vy) & (radius > 0.12 * max(h, w)) & (yy >= row_min)
    # 광선 위 픽셀 수는 거리에 비례해 늘어나므로 1/거리 가중치로 광선 길이를 고르게 센다
    wr = 1.0 / np.maximum(yy - vy, 1.0)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    road_zone = (road > 0) & zone
    asphalt_px = (seg == 13) & zone
    asphalt = float(np.median(gray[asphalt_px])) if asphalt_px.any() else float(np.median(gray))

    # --- 방향별 도로: θ 마다 도로 비율·분리대 비율
    tz = theta[zone]
    lo, hi = np.floor(tz.min()), np.ceil(tz.max())
    edges = np.arange(lo, hi + THETA_BIN, THETA_BIN)
    centers = (edges[:-1] + edges[1:]) / 2

    def prof(m):
        sel = m & zone
        return gaussian_filter1d(np.histogram(theta[sel], bins=edges, weights=wr[sel])[0], 1.0)

    Z = prof(np.ones_like(zone))
    road_cov = prof(road > 0) / np.maximum(Z, 1e-9)
    sep_cov = prof(sep > 0) / np.maximum(Z, 1e-9)
    ok_bins = (road_cov >= ROAD_COVER_MIN) & (sep_cov < SEP_COVER_CUT) & (Z > 1e-3 * Z.max())

    near_road = distance_transform_edt(road == 0)
    pi = np.clip(np.round(pts).astype(int), [0, 0], [w - 1, h - 1])
    pt_ok = near_road[pi[:, 1], pi[:, 0]] <= 12
    pt_theta = np.degrees(np.arctan2(pts[:, 0] - vx, pts[:, 1] - vy))

    min_pts = max(8, 0.03 * len(pts))
    cws = []
    for a, b in theta_runs(ok_bins):
        tmin, tmax = float(edges[a]), float(edges[b])
        if tmax - tmin < 1.0:
            continue
        in_c = pt_ok & (pt_theta >= tmin) & (pt_theta < tmax)
        if in_c.sum() < min_pts:
            continue
        # 인접한 방향별 도로 사이에 분리대가 있었는지 (중앙분리대 후보)
        cws.append({"theta": [tmin, tmax], "n_points": int(in_c.sum()),
                    "sep_before": bool(cws and (sep_cov[int((cws[-1]['theta'][1] - lo) / THETA_BIN):a] >= SEP_COVER_CUT).any())})
    if not cws:
        return {"ok": False, "reason": "차량이 다니는 도로를 찾지 못함", "vp": [vx, vy]}
    for i, c in enumerate(cws):
        c["id"] = i

    # --- 기하 규칙 방향: 분리대로 나뉜 경계 중 양쪽 차량 수가 가장 고른 곳 = 중앙분리대
    total = sum(c["n_points"] for c in cws)
    split, best = None, 0.0
    for i in range(1, len(cws)):
        if not cws[i]["sep_before"]:
            continue
        left = sum(c["n_points"] for c in cws[:i])
        bal = min(left, total - left) / total
        if bal > best:
            split, best = i, bal
    geo_two_way = split is not None and best >= 0.15
    for i, c in enumerate(cws):
        c["geo_direction"] = ("toward" if i < split else "away") if geo_two_way else None

    # --- 차로
    mk = elongated_markings(marking, theta, radius) & zone
    mk_road = mk & (cv2.dilate(road, np.ones((9, 9), np.uint8)) > 0)
    mys, mxs = np.nonzero(mk_road)
    m_theta, m_w = theta[mys, mxs], wr[mys, mxs]

    lane_map = np.zeros((h, w), np.int16)
    lanes = []
    for c in cws:
        tmin, tmax = c["theta"]
        sel = (m_theta >= tmin - 0.5) & (m_theta <= tmax + 0.5)
        pos, strg = peak_positions(m_theta[sel], m_w[sel], tmin - 1, tmax + 1, 1.5, 0.12, max(1.0, (tmax - tmin) / 12))
        bounds = [tmin, tmax] + [float(p) for p in pos if tmin < p < tmax]
        strength = {float(p): float(s) for p, s in zip(pos, strg)}
        bounds = drop_narrow(bounds, strength)
        c["lane_source"] = "markings"

        cpts = pt_theta[pt_ok & (pt_theta >= tmin) & (pt_theta < tmax)]
        if len(bounds) == 2 and len(cpts) >= 30:
            # 차선 도색이 안 보이면(눈·안개·야간) 차량이 몰린 θ 를 차로 중심으로 쓴다
            ctr, _ = peak_positions(cpts, None, tmin, tmax, 2.0, 0.15, max(1.0, (tmax - tmin) / 8))
            if len(ctr) >= 2:
                bounds = [tmin] + [float((x + y) / 2) for x, y in zip(ctr[:-1], ctr[1:])] + [tmax]
                c["lane_source"] = "vehicles"

        if c["lane_source"] == "markings":
            # 배경 사진에서 도색이 거의 안 보이는 경계(유도선 색띠·그림자 등으로 생긴 봉우리)는 없앤다
            bounds = [bounds[0]] + [t for t in bounds[1:-1]
                                    if boundary_paint(t, theta, radius, road_zone, gray, asphalt)[0] >= PAINT_MIN_COV] + [bounds[-1]]
        paints = [boundary_paint(t, theta, radius, road_zone, gray, asphalt) for t in bounds[1:-1]]
        solid = [cov >= SOLID_COV and runs <= 2 for cov, runs in paints]
        c["boundaries"] = [{"theta": round(t, 2), "paint_cov": round(cov, 2), "runs": runs, "solid": s}
                           for t, (cov, runs), s in zip(bounds[1:-1], paints, solid)]

        n_int = len(bounds) - 1
        counts = [int(((cpts >= bounds[j]) & (cpts < bounds[j + 1])).sum()) for j in range(n_int)]
        busiest = max(max(counts), 1)
        # 가장자리부터 안쪽으로 한 칸씩 벗겨낸다 (갓길이 가장자리선·요철 포장으로 두 칸이 되기도 한다).
        # 차량 수만으로는 안 된다: 갓길로 바퀴 위치가 번진 경우(15~38%)와 차가 적은 바깥 차로(32%)가 겹친다.
        # → 안쪽 경계가 실선(가장자리선)이면 느슨하게, 점선이면 엄격하게 뺀다.
        # (실선일 때, 아닐 때) 가장 붐비는 차로 대비 이용률 한도. 바깥쪽 = 운전자 오른쪽.
        outer_lim, median_lim = (0.6, 0.2), (0.4, 0.1)
        left_lim, right_lim = {"toward": (outer_lim, median_lim),
                               "away": (median_lim, outer_lim)}.get(c["geo_direction"], (outer_lim, outer_lim))
        not_lane = set()
        for j in range(n_int):  # 화면 왼쪽 끝부터
            inner_solid = j < n_int - 1 and solid[j]
            if len(not_lane) >= n_int - 1 or counts[j] >= left_lim[0 if inner_solid else 1] * busiest:
                break
            not_lane.add(j)
        for j in range(n_int - 1, -1, -1):  # 화면 오른쪽 끝부터
            inner_solid = j > 0 and solid[j - 1]
            if j in not_lane or len(not_lane) >= n_int - 1 or counts[j] >= right_lim[0 if inner_solid else 1] * busiest:
                break
            not_lane.add(j)
        c["lanes"] = []
        for j in range(n_int):
            a, b = bounds[j], bounds[j + 1]
            cnt = counts[j]
            share = cnt / max(len(cpts), 1)
            shoulder = j in not_lane
            lane_id = len(lanes) + 1
            lanes.append({"id": lane_id, "carriageway": c["id"], "theta": [a, b], "n_points": cnt,
                          "share": round(share, 3), "shoulder": bool(shoulder),
                          "geo_direction": c["geo_direction"]})
            c["lanes"].append(lane_id)
            lane_map[(road > 0) & zone & (theta >= a) & (theta < b)] = lane_id
        c["n_lanes"] = sum(not lanes[i - 1]["shoulder"] for i in c["lanes"])

    return {"ok": True, "vp": [vx, vy], "vp_support": round(vp_support, 3), "n_points": int(len(pts)),
            "geo_two_way": geo_two_way, "carriageways": cws, "lanes": lanes,
            "lane_map": lane_map, "theta": theta.astype(np.float32),
            "profile": {"theta": centers.tolist(), "road_cov": road_cov.tolist(), "sep_cov": sep_cov.tolist()}}


# ---------------------------------------------------------------- 그림 · 평가

TOWARD_COLORS = [(40, 40, 230), (0, 140, 255), (0, 215, 255), (80, 255, 180), (180, 120, 255)]
AWAY_COLORS = [(230, 90, 30), (255, 200, 0), (200, 255, 0), (255, 120, 180), (160, 60, 120)]
NEUTRAL_COLORS = [(60, 200, 60), (160, 220, 60), (60, 160, 160), (200, 160, 60), (120, 120, 220)]


def draw(bg: np.ndarray, geo: dict, pts: np.ndarray, clip: Clip) -> np.ndarray:
    vis = bg.copy()
    if geo["ok"]:
        over = bg.copy()
        count = {"toward": 0, "away": 0, None: 0}
        for lane in geo["lanes"]:
            m = geo["lane_map"] == lane["id"]
            if lane["shoulder"]:
                over[m] = (128, 128, 128)
                continue
            d = lane["geo_direction"]
            pal = TOWARD_COLORS if d == "toward" else AWAY_COLORS if d == "away" else NEUTRAL_COLORS
            over[m] = pal[count[d] % len(pal)]
            count[d] += 1
        vis = cv2.addWeighted(bg, 0.45, over, 0.55, 0)
        vx, vy = geo["vp"]
        R = 4 * max(vis.shape)
        for lane in geo["lanes"]:
            for t in lane["theta"]:
                e = (int(vx + R * np.sin(np.radians(t))), int(vy + R * np.cos(np.radians(t))))
                cv2.line(vis, (int(vx), int(vy)), e, (255, 255, 255), 1, cv2.LINE_AA)
        if 0 <= vx < vis.shape[1] and 0 <= vy < vis.shape[0]:
            cv2.circle(vis, (int(vx), int(vy)), 6, (0, 0, 255), 2)
        arrow = {"toward": "<", "away": ">", None: "?"}
        summary = " | ".join(f"C{c['id']}:{c['n_lanes']}{arrow[c['geo_direction']]}{'v' if c['lane_source'] == 'vehicles' else ''}"
                             for c in geo["carriageways"])
    else:
        summary = "FAIL " + geo["reason"]
    for x, y in pts.astype(int):
        cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 255), -1)
    text = f"{clip.lane_token}  detected {summary}"
    cv2.rectangle(vis, (0, vis.shape[0] - 30), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, vis.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def evaluate(clip: Clip, geo: dict) -> dict:
    row = {"clip_id": clip.clip_id, "camera": clip.camera_id, "token": clip.lane_token, "weather": clip.weather,
           "ok": geo["ok"]}
    if not geo["ok"]:
        return row | {"reason": geo["reason"]}
    # 차량이 가장 많이 다닌 방향별 도로를 파일명 차로 수와 비교 (양방향이면 2개)
    main = sorted(geo["carriageways"], key=lambda c: -c["n_points"])[: 2 if clip.two_way else 1]
    detected = [c["n_lanes"] for c in main]
    want = 2 if clip.two_way else 1
    return row | {
        "n_carriageways": len(geo["carriageways"]),
        "geo_two_way": geo["geo_two_way"],
        "two_way_match": geo["geo_two_way"] == clip.two_way,
        "main_lanes": "/".join(map(str, detected)),
        "lane_exact": len(detected) == want and all(n == clip.lanes_per_direction for n in detected),
        "lane_within1": len(detected) == want and all(abs(n - clip.lanes_per_direction) <= 1 for n in detected),
        "lane_source": "/".join(c["lane_source"] for c in main),
        "vp_support": geo["vp_support"],
        "n_points": geo["n_points"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reseg", action="store_true", help="Mask2Former 를 다시 돌린다")
    args = ap.parse_args()

    # s1b 가 끝난 클립만: 회전형 카메라는 화면(장면)마다 따로 처리한다
    clips = [c for c in list_clips() if (OUT / "s1" / c.clip_id / "views.json").exists()]
    if args.only:
        clips = [c for c in clips if any(c.clip_id.startswith(o) for o in args.only)]

    segmenter = None
    rows = []
    for k, clip in enumerate(clips, 1):
        s1 = OUT / "s1" / clip.clip_id
        with open(s1 / "views.json") as f:
            views = json.load(f)["views"]
        with open(s1 / "dets.pkl", "rb") as f:
            dets = pickle.load(f)["dets"]
        if len(views) > 1 and (OUT / "s2" / clip.clip_id).exists():
            shutil.rmtree(OUT / "s2" / clip.clip_id)  # 화면이 섞인 채로 만든 옛 결과

        for view in views:
            scene_id = view["scene_id"]
            fids = set(view["frames"])
            pts = foot_points([d for d in dets if d["frame"] in fids])
            out = OUT / "s2" / scene_id
            out.mkdir(parents=True, exist_ok=True)
            bg = cv2.imread(str(s1 / view["background"]))
            seg_path = out / "seg.png"
            if args.reseg or not seg_path.exists():
                segmenter = segmenter or Segmenter()
                cv2.imwrite(str(seg_path), segmenter(bg))
            seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)

            geo = lane_geometry(seg, pts, bg)
            cv2.imwrite(str(out / "lanes_vis.jpg"), draw(bg, geo, pts, clip), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if geo["ok"]:
                np.savez_compressed(out / "lanes.npz", lane_map=geo["lane_map"], theta=geo["theta"],
                                    vp=np.array(geo["vp"]))
            with open(out / "lanes.json", "w") as f:
                json.dump({k2: v for k2, v in geo.items() if k2 not in ("lane_map", "theta")}
                          | {"clip_id": clip.clip_id, "scene_id": scene_id, "frames": sorted(fids)},
                          f, ensure_ascii=False)
            row = {"scene_id": scene_id, "n_views": len(views)} | evaluate(clip, geo)
            rows.append(row)
            print(f"[{k}/{len(clips)}] {scene_id} {row.get('main_lanes', row.get('reason'))} "
                  f"token={clip.lane_token} exact={row.get('lane_exact')} src={row.get('lane_source')}", flush=True)

    fields = list(dict.fromkeys(key for r in rows for key in r))
    with open(OUT / "s2_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    ok = [r for r in rows if r["ok"]]
    if ok:
        print(f"\n성공 {len(ok)}/{len(rows)} | 차로 수 정확 {np.mean([r['lane_exact'] for r in ok]):.2f} "
              f"| ±1 {np.mean([r['lane_within1'] for r in ok]):.2f} | 양방향 판정 {np.mean([r['two_way_match'] for r in ok]):.2f}")


if __name__ == "__main__":
    main()
