"""2d) 휘어진 도로 대응 (시험): 가까운 구간에서 차로를 정하고, 경계선을 곡선으로 따라 올라간다.

지금 방식은 모든 차선을 소실점 하나에서 뻗은 직선으로 본다. 도로가 휘면 먼 곳 차선이 소실점을 끌어당겨
가까운 곳 차로까지 틀어진다 (먼 곳이 4° 넘게 휜 94장면: 정확 22~39%, 곧은 도로 57%).

1. 가까운 구간(도로가 보이는 행의 아래쪽 NEAR_FRAC)만으로 소실점·방향별 도로·차로 수를 정한다 — 가까운 곳은 곧다
2. 경계마다 가까운 구간의 직선에서 출발해 STEP px 씩 위로 올라가며, 예측 위치 근처의
   차선 도색(안쪽 경계) 또는 도로 가장자리·분리대(바깥 경계)를 찾아 곡선을 다시 맞춘다 (점선 빈칸은 예측으로 건넘)
3. 행마다 곡선 경계 사이의 도로 픽셀을 차로로 칠한다

실행
  python s2d_curves.py eval                 # 전체 장면: 가까운 구간 방식의 차로 수 정확도를 휜 정도별로 비교
  python s2d_curves.py demo <scene_id> ...  # 곡선 추적 그림 → outputs/s2d/<scene>.jpg
"""
from __future__ import annotations

import json
import pickle
import sys

import cv2
import numpy as np
import pandas as pd

import s2_road_lanes as s2
from common import OUT, list_clips

NEAR_FRAC = 0.5
STEP = 16
MAX_MISS = 3
WIN_RATIO = 0.35  # 탐색 폭 = 그 행의 차로 폭 × 비율
LOCAL_SPAN = 140  # 곡선을 맞출 때 쓰는 최근 구간 길이(px)
MAX_TURN_DEG = 15.0  # 한 걸음(STEP)에 허용하는 차선 방향 변화(도)
EXTEND_PX = 80  # 추적이 멈춘 뒤 마지막 곡선으로 더 이어 그리는 길이


def scene_inputs(scene_id: str, clips: dict):
    clip_id = scene_id.split("__view")[0]
    views = json.loads((OUT / "s1" / clip_id / "views.json").read_text())["views"]
    view = next(v for v in views if v["scene_id"] == scene_id)
    bg = cv2.imread(str(OUT / "s1" / clip_id / view["background"]))
    seg = cv2.imread(str(OUT / "s2" / scene_id / "seg.png"), cv2.IMREAD_UNCHANGED)
    with open(OUT / "s1" / clip_id / "dets.pkl", "rb") as f:
        frames = set(view["frames"])
        pts = s2.foot_points([d for d in pickle.load(f)["dets"] if d["frame"] in frames])
    return bg, seg, pts, clips[clip_id]


def road_rows(seg: np.ndarray) -> tuple[int, int]:
    rows = np.nonzero(np.isin(seg, s2.ROAD_IDS).sum(1) > 20)[0]
    return (int(rows.min()), int(rows.max())) if len(rows) else (0, seg.shape[0] - 1)


def marking_deviation(seg: np.ndarray, vp) -> float:
    """먼 곳(도로 행의 위쪽 절반) 차선 도색 조각이 소실점 방향과 어긋난 각도(길이 가중 평균)."""
    vx, vy = vp
    y_top, y_bot = road_rows(seg)
    y_top = max(y_top, vy)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((seg == s2.MARKING_ID).astype(np.uint8), 8)
    devs, wts = [], []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 25:
            continue
        ys, xs = np.nonzero(lab == k)
        c = np.array([xs.mean(), ys.mean()])
        if c[1] <= vy or (c[1] - y_top) / max(y_bot - y_top, 1) >= 0.5:
            continue
        ev, evec = np.linalg.eigh(np.cov(np.c_[xs, ys].T.astype(float)))
        if ev[1] < 4 * max(ev[0], 1e-6):
            continue
        r = c - np.array([vx, vy])
        devs.append(np.degrees(np.arccos(min(1.0, abs(evec[:, 1] @ (r / np.linalg.norm(r)))))))
        wts.append(np.sqrt(ev[1]))
    return float(np.average(devs, weights=wts)) if devs else float("nan")


def near_geometry(seg, pts, bg, frac: float = NEAR_FRAC) -> tuple[dict, int]:
    y_top, y_bot = road_rows(seg)
    row_min = int(y_bot - frac * (y_bot - y_top))
    return s2.lane_geometry(seg, pts, bg, row_min=row_min), row_min


# ---------------------------------------------------------------- 곡선 추적

def candidate_rows(mask: np.ndarray) -> dict[int, np.ndarray]:
    ys, xs = np.nonzero(mask)
    out: dict[int, list] = {}
    for y, x in zip(ys, xs):
        out.setdefault(int(y), []).append(int(x))
    return {y: np.array(v) for y, v in out.items()}


def local_fit(fit_y: list, fit_x: list, y_new: int) -> np.ndarray:
    """최근 구간만으로 다시 맞춰 곡선을 따라간다 (점이 적으면 1차)."""
    sel = [i for i, v in enumerate(fit_y) if v <= y_new + LOCAL_SPAN]
    yy_fit, xx_fit = np.array(fit_y)[sel], np.array(fit_x)[sel]
    deg = 2 if len(sel) >= 12 and np.ptp(yy_fit) > 60 else 1
    return np.polyfit(yy_fit, xx_fit, deg)


def slope(coef: np.ndarray, y: float) -> float:
    return float(np.polyval(np.polyder(coef), y))


def track_boundary(theta: float, kind: str, geo: dict, row_min: int, y_top: int, y_bot: int,
                   cands: dict, lane_w_ref: float, y_ref: float, road: np.ndarray) -> dict[int, float]:
    """경계 하나를 아래(가까운 곳)에서 위로 따라가며 행 → x 위치를 돌려준다.

    안전장치: 한 걸음에 방향(dx/dy)이 MAX_SLOPE_JUMP 넘게 꺾이거나, 예측 위치 근처에 도로가 없으면
    방금 찾은 점을 버리고 놓친 것으로 센다 (가드레일·옆 도로 가장자리로 새는 것 방지)."""
    vx, vy = geo["vp"]
    t = np.tan(np.radians(theta))
    ray = lambda y: vx + (y - vy) * t
    w = road.shape[1]
    path = {y: ray(y) for y in range(row_min, y_bot + 1)}
    fit_y = list(range(row_min, min(y_bot, row_min + LOCAL_SPAN), 4))
    fit_x = [ray(y) for y in fit_y]
    coef = np.polyfit(fit_y, fit_x, 1)
    y, misses = row_min, 0
    while y - STEP > max(y_top, vy + 8):
        y_new = y - STEP
        fy, fx = [], []
        for yy in range(y_new, y, 2):
            pred = float(np.polyval(coef, yy))
            win = max(3.0, WIN_RATIO * lane_w_ref * (yy - vy) / max(y_ref - vy, 1))
            xs = cands[kind].get(yy)
            if xs is None:
                continue
            near = xs[np.abs(xs - pred) <= win]
            if len(near):
                fy.append(yy)
                fx.append(float(near[np.argmin(np.abs(near - pred))]))
        accepted = False
        if len(fy) >= 3:
            new_coef = local_fit(fit_y + fy, fit_x + fx, y_new)
            # 원근 때문에 비스듬한 차선은 기울기(dx/dy) 값이 커서, 기울기 차이 대신 각도 차이로 꺾임을 본다
            turn = abs(np.degrees(np.arctan(slope(new_coef, y_new)) - np.arctan(slope(coef, y))))
            if turn <= MAX_TURN_DEG:
                fit_y += fy
                fit_x += fx
                coef = new_coef
                accepted = True
        win_here = max(6.0, WIN_RATIO * lane_w_ref * (y_new - vy) / max(y_ref - vy, 1))
        x_mid = int(round(float(np.polyval(coef, y_new))))
        lo, hi = max(0, int(x_mid - win_here)), min(w, int(x_mid + win_here) + 1)
        on_road = 0 <= x_mid < w and road[y_new, lo:hi].any()
        misses = 0 if (accepted and on_road) else misses + 1
        if misses > MAX_MISS:
            break
        for yy in range(y_new, y):
            path[yy] = float(np.polyval(coef, yy))
        y = y_new
    # 멈춘 경계는 마지막 곡선으로 도로 위에서 조금 더 이어 그린다 (짝 경계가 멈춰 차로가 끊기지 않게)
    y_end = y
    for yy in range(y_end - 1, max(y_top, int(vy) + 8, y_end - EXTEND_PX) - 1, -1):
        x = float(np.polyval(coef, yy))
        xi = int(round(x))
        if not (0 <= xi < w) or not road[yy, max(0, xi - 8):min(w, xi + 9)].any():
            break
        path[yy] = x
    return path


def curved_lane_map(geo: dict, seg: np.ndarray, row_min: int) -> tuple[np.ndarray, dict]:
    h, w = seg.shape
    road = np.isin(seg, s2.ROAD_IDS).astype(np.uint8)
    edge = (road - cv2.erode(road, np.ones((3, 3), np.uint8))) | np.isin(seg, s2.SEPARATOR_IDS).astype(np.uint8)
    cands = {"marking": candidate_rows(seg == s2.MARKING_ID), "edge": candidate_rows(edge > 0)}
    y_top, y_bot = road_rows(seg)
    vx, vy = geo["vp"]
    lane_map = np.zeros((h, w), np.int16)
    paths: dict[tuple, dict] = {}
    xs_row = np.arange(w)
    for c in geo["carriageways"]:
        lanes = [geo["lanes"][i - 1] for i in c["lanes"]]
        bounds = sorted({round(l["theta"][0], 4) for l in lanes} | {round(lanes[-1]["theta"][1], 4)})
        y_ref = float(y_bot)
        widths = [abs((y_ref - vy) * (np.tan(np.radians(b)) - np.tan(np.radians(a)))) for a, b in zip(bounds[:-1], bounds[1:])]
        lane_w_ref = float(np.median(widths)) if widths else 40.0
        bpaths = []
        for j, b in enumerate(bounds):
            kind = "edge" if j in (0, len(bounds) - 1) else "marking"
            p = track_boundary(b, kind, geo, row_min, y_top, y_bot, cands, lane_w_ref, y_ref, road)
            bpaths.append(p)
            paths[(c["id"], j)] = p
        for j, lane in enumerate(lanes):
            left, right = bpaths[j], bpaths[j + 1]
            for y in range(min(min(left), min(right)), y_bot + 1):
                if y not in left or y not in right:
                    continue
                xl, xr = sorted((left[y], right[y]))
                if xr - xl < 1.5:
                    continue
                m = (xs_row >= xl) & (xs_row < xr) & (road[y] > 0) & (lane_map[y] == 0)
                lane_map[y, m] = lane["id"]
    return lane_map, paths


# ---------------------------------------------------------------- 공통 휨 곡선 (방향별 도로 하나 = 곡선 하나)
# 경계를 따로 추적하면 옆 차선 도색에 붙어 서로 엇갈린다. 실제 차선은 평행하게 함께 휘므로,
#   x_j(y) = 소실점 + (y − 소실점y)·tanθ_j  +  g(y)
# 로 두고, 방향별 도로의 모든 경계가 휨 곡선 g 하나를 공유한다. 경계 간격은 원근 비율대로 유지돼 엇갈리지 않고,
# 곧은 도로에서는 g ≈ 0 이 되어 원래 직선 결과와 같다.

def robust_poly(u: np.ndarray, r: np.ndarray, w: np.ndarray, deg: int, ridge: float = 0.5) -> np.ndarray:
    """가중 릿지 다항식(낮은 차수부터) 을 두 번 맞추며 튀는 점을 버린다."""
    keep = np.ones(len(u), bool)
    coef = np.zeros(4)
    for _ in range(2):
        A = np.vander(u[keep], deg + 1, increasing=True) * w[keep, None]
        b = r[keep] * w[keep]
        reg = ridge * np.eye(deg + 1)
        reg[0, 0] = reg[1, 1] = 1e-3  # 이동·기울기(소실점 오차)는 자유롭게, 휨(2·3차)만 규제
        c = np.linalg.solve(A.T @ A + reg, A.T @ b)
        coef = np.zeros(4)
        coef[: deg + 1] = c
        res = r - np.polyval(coef[::-1], u)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) <= max(3.0, 3.0 * 1.4826 * mad)
    return coef


def fit_carriageway_curve(geo: dict, c: dict, cands: dict, y_top: int, y_bot: int):
    vx, vy = geo["vp"]
    lanes = [geo["lanes"][i - 1] for i in c["lanes"]]
    bounds = sorted({round(l["theta"][0], 4) for l in lanes} | {round(lanes[-1]["theta"][1], 4)})
    t = np.tan(np.radians(bounds))
    kinds = ["edge"] + ["marking"] * (len(bounds) - 2) + ["edge"]
    y_ref = float(y_bot)
    lane_w_ref = float(np.median(np.abs(np.diff(t)))) * (y_ref - vy) if len(t) > 1 else 40.0
    coef = np.zeros(4)
    du, dr, dw = [], [], []
    y, misses, y_stop = y_bot, 0, y_bot
    while y - STEP > max(y_top, vy + 8):
        y_new = y - STEP
        got = 0
        for yy in range(y_new, y, 2):
            u = (y_ref - yy) / 100.0
            g = float(np.polyval(coef[::-1], u))
            scale = yy - vy
            win = max(3.0, WIN_RATIO * lane_w_ref * scale / max(y_ref - vy, 1))
            for tj, kind in zip(t, kinds):
                xs = cands[kind].get(yy)
                if xs is None:
                    continue
                pred = vx + scale * tj + g
                near = xs[np.abs(xs - pred) <= win]
                if len(near):
                    xb = float(near[np.argmin(np.abs(near - pred))])
                    du.append(u)
                    dr.append(xb - (vx + scale * tj))
                    dw.append(1.0 if kind == "marking" else 0.5)
                    got += 1
        if got >= max(4, len(t)):
            misses, y_stop = 0, y_new
            span = (y_ref - y_new) / 100.0
            deg = 1 if span < 0.8 else 2 if span < 1.6 else 3
            coef = robust_poly(np.array(du), np.array(dr), np.array(dw), deg)
        else:
            misses += 1
            if misses > MAX_MISS:
                break
        y = y_new
    return t, coef, y_stop, lanes


def joint_lane_map(geo: dict, seg: np.ndarray) -> tuple[np.ndarray, dict, int]:
    h, w = seg.shape
    road = np.isin(seg, s2.ROAD_IDS).astype(np.uint8)
    edge = (road - cv2.erode(road, np.ones((3, 3), np.uint8))) | np.isin(seg, s2.SEPARATOR_IDS).astype(np.uint8)
    cands = {"marking": candidate_rows(seg == s2.MARKING_ID), "edge": candidate_rows(edge > 0)}
    y_top, y_bot = road_rows(seg)
    vx, vy = geo["vp"]
    lane_map = np.zeros((h, w), np.int16)
    paths: dict[tuple, dict] = {}
    xs_row = np.arange(w)
    straight = geo.get("lane_map")
    for c in geo["carriageways"]:
        # 방향별 도로마다 실제로 보이는 행 범위에서 출발한다. 반대편 도로·램프처럼 화면 아래까지 내려오지 않는
        # 도로를 화면 맨 아래에서 추적하면 근거가 없어 곧바로 멈추고 통째로 비었다(차로 181개).
        rows_c = np.nonzero(np.isin(straight, c["lanes"]).any(1))[0] if straight is not None else np.array([])
        yb = int(rows_c.max()) if len(rows_c) else y_bot
        yt = int(rows_c.min()) if len(rows_c) else y_top
        t, coef, y_stop, lanes = fit_carriageway_curve(geo, c, cands, yt, yb)
        curve_end = max(int(vy) + 8, y_stop - EXTEND_PX // 2)
        g_hold = 0.0
        for y in range(yb, max(yt, int(vy) + 8) - 1, -1):
            if y >= curve_end:
                g = float(np.polyval(coef[::-1], (yb - y) / 100.0))
                if not np.isfinite(g) or abs(g) > 2 * w:
                    curve_end, g = y + 1, g_hold  # 휨 곡선이 폭발하면 그 위로는 마지막 값을 유지
                else:
                    g_hold = g
            else:
                # 추적 근거가 끝난 위쪽은 비우지 않고 마지막 휨 값을 유지해 이어 칠한다.
                # 근거를 전혀 못 찾은 도로는 g = 0 이라 직선 지도와 같아진다.
                g = g_hold
            xb = vx + (y - vy) * t + g
            if not np.all(np.isfinite(xb)):
                break
            for j, lane in enumerate(lanes):
                xl, xr = xb[j], xb[j + 1]
                if xr - xl < 1.5:
                    continue
                m = (xs_row >= xl) & (xs_row < xr) & (road[y] > 0) & (lane_map[y] == 0)
                lane_map[y, m] = lane["id"]
            for j, x in enumerate(xb):
                paths.setdefault((c["id"], j), {})[y] = float(x)
    return lane_map, paths, y_bot


def draw_curved(bg: np.ndarray, geo: dict, lane_map: np.ndarray, paths: dict, row_min: int, title: str) -> np.ndarray:
    over = bg.copy()
    count = {"toward": 0, "away": 0, None: 0}
    for lane in geo["lanes"]:
        m = lane_map == lane["id"]
        if lane["shoulder"]:
            over[m] = (128, 128, 128)
            continue
        d = lane["geo_direction"]
        pal = s2.TOWARD_COLORS if d == "toward" else s2.AWAY_COLORS if d == "away" else s2.NEUTRAL_COLORS
        over[m] = pal[count[d] % len(pal)]
        count[d] += 1
    vis = cv2.addWeighted(bg, 0.45, over, 0.55, 0)
    w = vis.shape[1]
    for p in paths.values():
        line = [[int(round(min(max(x, -2 * w), 3 * w))), y] for y, x in sorted(p.items()) if np.isfinite(x)]
        if len(line) >= 2:
            cv2.polylines(vis, [np.array(line, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(vis, (0, row_min), (vis.shape[1], row_min), (0, 255, 255), 1)
    cv2.rectangle(vis, (0, vis.shape[0] - 30), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, vis.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def demo(scene_ids: list[str]) -> None:
    clips = {c.clip_id: c for c in list_clips()}
    (OUT / "s2d").mkdir(exist_ok=True)
    for sid in scene_ids:
        bg, seg, pts, clip = scene_inputs(sid, clips)
        geo, row_min = near_geometry(seg, pts, bg)
        if not geo["ok"]:
            print(sid, "실패", geo.get("reason"))
            continue
        lane_map, paths = curved_lane_map(geo, seg, row_min)
        ev = s2.evaluate(clip, geo)
        summary = " | ".join(f"C{c['id']}:{c['n_lanes']}" for c in geo["carriageways"])
        vis = draw_curved(bg, geo, lane_map, paths, row_min, f"{clip.lane_token} curved {summary}")
        cv2.imwrite(str(OUT / "s2d" / f"{sid}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(sid, "차로", ev.get("main_lanes"), "정확", ev.get("lane_exact"))


def evaluate_all() -> None:
    clips = {c.clip_id: c for c in list_clips()}
    single = pd.read_csv(OUT / "s2_summary_single.csv").set_index("scene_id")
    rows = []
    for sd in sorted(p.parent for p in (OUT / "s2").glob("*/lanes_single.json")):
        sid = sd.name
        info = json.loads((sd / "lanes_single.json").read_text())
        if not info.get("ok") or sid not in single.index:
            continue
        bg, seg, pts, clip = scene_inputs(sid, clips)
        dev = marking_deviation(seg, info["vp"])
        geo, _ = near_geometry(seg, pts, bg)
        ev = s2.evaluate(clip, geo) if geo["ok"] else {}
        rows.append({"scene_id": sid, "far_dev": dev, "region": clip.camera_id.split("_")[0],
                     "exact_full": bool(single.loc[sid, "lane_exact"]), "w1_full": bool(single.loc[sid, "lane_within1"]),
                     "exact_near": bool(ev.get("lane_exact", False)), "w1_near": bool(ev.get("lane_within1", False)),
                     "lanes_full": single.loc[sid, "main_lanes"], "lanes_near": ev.get("main_lanes", "fail")})
        print(f"[{len(rows)}] {sid[:50]} dev={dev:.1f} full={rows[-1]['lanes_full']} near={rows[-1]['lanes_near']} token={clip.lane_token}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s2d_eval.csv", index=False)
    d["curve"] = pd.cut(d.far_dev, [-1, 2, 4, 8, 180], labels=["곧음(<2°)", "약간(2-4°)", "휨(4-8°)", "많이 휨(>8°)"])
    d["exact_adapt"] = np.where(d.far_dev > 4, d.exact_near, d.exact_full)
    d["w1_adapt"] = np.where(d.far_dev > 4, d.w1_near, d.w1_full)
    agg = dict(n=("scene_id", "size"), exact_full=("exact_full", "mean"), exact_near=("exact_near", "mean"),
               exact_adapt=("exact_adapt", "mean"), w1_full=("w1_full", "mean"), w1_near=("w1_near", "mean"),
               w1_adapt=("w1_adapt", "mean"))
    print("\n휜 정도별 (클립마다 따로 방식 기준)")
    print(d.groupby("curve", observed=True).agg(**agg).round(2).to_string())
    print("\n전체")
    print(d.agg({k: v[1] if v[1] != "size" else "size" for k, v in agg.items() if k != "n"}).round(3).to_string())


# ---------------------------------------------------------------- 차로 모양만 곡선으로 (차로 수는 G 결과 유지)
# 시험 결과: 가까운 구간만으로 차로 수를 세면 근거(점선·차량 점)가 부족해 전체 정확 41% → 28% 로 나빠졌다.
# 그래서 차로 수·방향·갓길은 지금 결과(G)를 그대로 두고, 경계를 화면 아래에서부터 곡선으로 따라가 모양만 바로잡는다.

def load_current(scene_id: str, clips: dict):
    sd = OUT / "s2" / scene_id
    geo = json.loads((sd / "lanes.json").read_text())
    if not geo.get("ok"):
        return None
    # 곡선 지도를 적용한 뒤에도 비교 기준이 직선 지도가 되도록 백업을 먼저 읽는다
    straight = sd / "lanes_straight.npz"
    geo["lane_map"] = np.load(straight if straight.exists() else sd / "lanes.npz")["lane_map"]
    bg, seg, pts, clip = scene_inputs(scene_id, clips)
    return geo, bg, seg, pts, clip


def reshape(geo: dict, seg: np.ndarray) -> tuple[np.ndarray, dict, int]:
    """방향별 도로마다 공통 휨 곡선 하나로 모든 경계를 함께 휘게 한다 (경계별 따로 추적은 엇갈려서 버림)."""
    return joint_lane_map(geo, seg)


def lane_position_stats(lane_map: np.ndarray, pts: np.ndarray, lanes: list[dict]) -> dict:
    """차량 바닥점이 차로 가운데에 얼마나 가까운지. 차는 차로 가운데로 달리므로 차로 모양이 맞을수록 작다."""
    h, w = lane_map.shape
    ys, xs = np.nonzero(lane_map > 0)
    if len(ys) == 0 or len(pts) == 0:
        return {"coverage": np.nan, "off_center": np.nan}
    ids = lane_map[ys, xs].astype(np.int64)
    key = ids * h + ys
    order = np.argsort(key, kind="stable")
    key_s, xs_s = key[order], xs[order]
    uniq, start = np.unique(key_s, return_index=True)
    xmin = dict(zip(uniq.tolist(), np.minimum.reduceat(xs_s, start).tolist()))
    xmax = dict(zip(uniq.tolist(), np.maximum.reduceat(xs_s, start).tolist()))
    shoulder = {l["id"] for l in lanes if l["shoulder"]}
    p = np.clip(np.round(pts).astype(int), [0, 0], [w - 1, h - 1])
    inside, offs = 0, []
    road_pts = 0
    for x, y in p:
        road_pts += 1
        lid = int(lane_map[y, x])
        if lid == 0 or lid in shoulder:
            continue
        k = lid * h + y
        width = xmax[k] - xmin[k] + 1
        if width < 6:
            continue
        inside += 1
        offs.append(abs((x - xmin[k]) / width - 0.5))
    return {"coverage": inside / max(road_pts, 1), "off_center": float(np.mean(offs)) if offs else np.nan}


def pair_stats(map_a: np.ndarray, map_b: np.ndarray, pts: np.ndarray, lanes: list[dict]) -> dict:
    """두 차로 지도를 같은 바닥점으로 공정하게 비교: 두 지도 모두에서 차로(갓길 제외) 안에 든 점만으로 중심 이탈도를 잰다.
    한쪽 지도가 먼 곳을 덜 칠해 점이 줄면 가까운 점만 남아 유리해지는 것을 막는다."""
    h, w = map_a.shape
    shoulder = {l["id"] for l in lanes if l["shoulder"]}
    p = np.clip(np.round(pts).astype(int), [0, 0], [w - 1, h - 1])
    ia, ib = map_a[p[:, 1], p[:, 0]], map_b[p[:, 1], p[:, 0]]
    ok_a = (ia > 0) & ~np.isin(ia, list(shoulder))
    ok_b = (ib > 0) & ~np.isin(ib, list(shoulder))
    both = ok_a & ok_b
    out = {"cov_a": float(ok_a.mean()) if len(p) else np.nan, "cov_b": float(ok_b.mean()) if len(p) else np.nan,
           "n_both": int(both.sum())}
    for name, m in (("a", map_a), ("b", map_b)):
        sa = lane_position_stats(m, pts[both], lanes) if both.any() else {"off_center": np.nan}
        out[f"off_{name}"] = sa["off_center"]
    return out


def reshape_demo(scene_ids: list[str]) -> None:
    clips = {c.clip_id: c for c in list_clips()}
    (OUT / "s2d").mkdir(exist_ok=True)
    for sid in scene_ids:
        cur = load_current(sid, clips)
        if cur is None:
            continue
        geo, bg, seg, pts, clip = cur
        lane_map, paths, row_min = reshape(geo, seg)
        a = lane_position_stats(geo["lane_map"], pts, geo["lanes"])
        b = lane_position_stats(lane_map, pts, geo["lanes"])
        summary = " | ".join(f"C{c['id']}:{c['n_lanes']}" for c in geo["carriageways"])
        vis = draw_curved(bg, geo, lane_map, paths, row_min, f"{clip.lane_token} curved {summary}")
        cv2.imwrite(str(OUT / "s2d" / f"{sid}__reshape.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{sid[:48]} 중심이탈 {a['off_center']:.3f} → {b['off_center']:.3f} | 포함률 {a['coverage']:.2f} → {b['coverage']:.2f}")


def reshape_eval() -> None:
    clips = {c.clip_id: c for c in list_clips()}
    dev = pd.read_csv(OUT / "s2d_eval.csv").set_index("scene_id")["far_dev"]
    rows = []
    for sd in sorted(p.parent for p in (OUT / "s2").glob("*/lanes.json")):
        cur = load_current(sd.name, clips)
        if cur is None:
            continue
        geo, bg, seg, pts, clip = cur
        lane_map, _, _ = reshape(geo, seg)
        s = pair_stats(geo["lane_map"], lane_map, pts, geo["lanes"])
        rows.append({"scene_id": sd.name, "far_dev": dev.get(sd.name, np.nan),
                     "off_straight": s["off_a"], "off_curved": s["off_b"],
                     "cov_straight": s["cov_a"], "cov_curved": s["cov_b"], "n_both": s["n_both"]})
        print(f"[{len(rows)}] {sd.name[:48]} off {s['off_a']:.3f}→{s['off_b']:.3f} cov {s['cov_a']:.2f}→{s['cov_b']:.2f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s2d_reshape_eval.csv", index=False)
    d["curve"] = pd.cut(d.far_dev, [-1, 2, 4, 8, 180], labels=["곧음(<2°)", "약간(2-4°)", "휨(4-8°)", "많이 휨(>8°)"])
    d["better"] = d.off_curved < d.off_straight - 0.005
    d["worse"] = d.off_curved > d.off_straight + 0.005
    print("\n휜 정도별 — 중심 이탈도(낮을수록 차로 모양이 맞음), 포함률")
    print(d.groupby("curve", observed=True).agg(n=("scene_id", "size"), off_straight=("off_straight", "median"),
          off_curved=("off_curved", "median"), cov_straight=("cov_straight", "median"), cov_curved=("cov_curved", "median"),
          better=("better", "sum"), worse=("worse", "sum")).round(3).to_string())


def reshape_apply() -> None:
    """모든 장면의 차로 지도를 공통 휨 곡선 모양으로 저장한다. 차로 수·방향·갓길(lanes.json)은 그대로.
    직선 지도는 lanes_straight.npz / lanes_straight_vis.jpg 로 한 번만 백업한다."""
    clips = {c.clip_id: c for c in list_clips()}
    n = 0
    for sd in sorted(p.parent for p in (OUT / "s2").glob("*/lanes.json")):
        cur = load_current(sd.name, clips)
        if cur is None:
            continue
        geo, bg, seg, pts, clip = cur
        for src, bak in (("lanes.npz", "lanes_straight.npz"), ("lanes_vis.jpg", "lanes_straight_vis.jpg")):
            if not (sd / bak).exists():
                (sd / bak).write_bytes((sd / src).read_bytes())
        straight = np.load(sd / "lanes_straight.npz")
        geo["lane_map"] = straight["lane_map"]
        lane_map, paths, row_min = reshape(geo, seg)
        np.savez_compressed(sd / "lanes.npz", lane_map=lane_map, theta=straight["theta"], vp=straight["vp"])
        summary = " | ".join(f"C{c['id']}:{c['n_lanes']}" for c in geo["carriageways"])
        vis = draw_curved(bg, geo, lane_map, paths, row_min, f"{clip.lane_token} detected {summary} (curved)")
        cv2.imwrite(str(sd / "lanes_vis.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        info = json.loads((sd / "lanes.json").read_text())
        info["lane_shape"] = "curved"
        (sd / "lanes.json").write_text(json.dumps(info, ensure_ascii=False, default=float))
        n += 1
    print(f"곡선 차로 지도 저장: {n}개 장면 (직선 지도는 lanes_straight.*)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "apply":
        reshape_apply()
        sys.exit(0)
    if mode == "eval":
        evaluate_all()
    elif mode == "reshape-demo":
        reshape_demo(sys.argv[2:])
    elif mode == "reshape-eval":
        reshape_eval()
    else:
        demo(sys.argv[2:])
