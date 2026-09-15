"""설명 PDF(docs/pipeline_guide.html → PDF) 에 들어갈 그림을 결과 파일에서 만든다.

실행: cd road_lane && ../.venv/bin/python docs/build_figures.py
출력: docs/fig/
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import s2_road_lanes as s2  # noqa: E402
from common import OUT, frame_index, list_clips  # noqa: E402

FIG = ROOT / "docs" / "fig"
SCENE = "Buan_daemog_20201020_1130_TUE_15m_NH_highway_TW2_sunny_FHD"
EXAMPLE_OCC = "Jincheon_namioverpass_20201021_0900_WED_15m_NH_highway_OW2_sunny_FHD"
FAILURE = "Suwon_CH01_20200720_1830_MON_9m_RH_highway_TW5_sunny_FHD__view0"
PTZ_CLIPS = ["Buan_daemog_20201020_1630_TUE_15m_NH_highway_TW2_sunny_FHD",
             "Suwon_CH10_20201213_0958_SUN_9m_NH_highway_OW5_snow_FHD"]

# 검증된 기본 팔레트 (dataviz reference palette, light)
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"


def setup_matplotlib() -> None:
    family = "AppleGothic"
    try:
        font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
        family = "Apple SD Gothic Neo"
    except Exception:
        font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False, "font.size": 9.5,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save_fig(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_img(img: np.ndarray, name: str) -> None:
    cv2.imwrite(str(FIG / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def hstack(images: list[np.ndarray], height: int, gap: int = 10) -> np.ndarray:
    rs = [cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
          for im in images]
    spacer = np.full((height, gap, 3), 252, np.uint8)
    out = []
    for i, r in enumerate(rs):
        out += [r] + ([spacer] if i < len(rs) - 1 else [])
    return np.concatenate(out, axis=1)


def scene_inputs(scene: str):
    clip_id = scene.split("__view")[0]
    views = json.loads((OUT / "s1" / clip_id / "views.json").read_text())["views"]
    view = next(v for v in views if v["scene_id"] == scene)
    bg = cv2.imread(str(OUT / "s1" / clip_id / view["background"]))
    seg = cv2.imread(str(OUT / "s2" / scene / "seg.png"), cv2.IMREAD_UNCHANGED)
    with open(OUT / "s1" / clip_id / "dets.pkl", "rb") as f:
        dets = [d for d in pickle.load(f)["dets"] if d["frame"] in set(view["frames"])]
    return bg, seg, dets


# ------------------------------------------------------------------ 데이터 특징

def fig_data(clips: dict) -> None:
    c = clips[SCENE]
    frames = [cv2.imread(str(p)) for p in c.frames[:3]]
    save_img(hstack(frames, 300), "f01_consecutive.jpg")
    for i, clip_id in enumerate(PTZ_CLIPS):
        views = json.loads((OUT / "s1" / clip_id / "views.json").read_text())["views"]
        ims = [cv2.imread(str(OUT / "s1" / clip_id / v["background"])) for v in views[:2]]
        save_img(hstack(ims, 300 if ims[0].shape[1] > ims[0].shape[0] else 420), f"f02{'ab'[i]}_views.jpg")


# ------------------------------------------------------------------ 준비: 배경·바닥점

def fig_background(clips: dict) -> None:
    bg, _, dets = scene_inputs(SCENE)
    raw = cv2.imread(str(clips[SCENE].frames[1]))
    save_img(hstack([raw, bg], 300), "f03_background.jpg")
    vis = bg.copy()
    for x, y in s2.foot_points(dets).astype(int):
        cv2.circle(vis, (int(x), int(y)), 4, (252, 252, 251), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(x), int(y)), 2, (214, 120, 42), -1, cv2.LINE_AA)
    save_img(vis, "f04_footpoints.jpg")


# ------------------------------------------------------------------ 2단계

def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def fig_segmentation() -> dict:
    bg, seg, dets = scene_inputs(SCENE)
    over = bg.copy()
    for ids, color in ((s2.ROAD_IDS, AQUA), ([s2.MARKING_ID], YELLOW), (s2.SEPARATOR_IDS, RED)):
        over[np.isin(seg, ids)] = hex_bgr(color)
    save_img(hstack([bg, cv2.addWeighted(bg, 0.45, over, 0.55, 0)], 300), "f05_m2f.jpg")
    proto = Path("/private/tmp/claude-501/-Users-wooseokhan-SNU-KDT-FinTech-----------/"
                 "fd3d369f-5d8d-45bf-a743-648c65b26dea/scratchpad/proto/m2f_res_compare.jpg")
    if proto.exists():
        im = cv2.imread(str(proto))
        h = im.shape[0] // 3
        save_img(hstack([im[0:h], im[2 * h:3 * h]], 260), "f06_m2f_resolution.jpg")

    pts = s2.foot_points(dets)
    geo = s2.lane_geometry(seg, pts, bg)
    h, w = seg.shape
    road = cv2.morphologyEx(np.isin(seg, s2.ROAD_IDS).astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marking = (seg == s2.MARKING_ID).astype(np.uint8)
    sep = np.isin(seg, s2.SEPARATOR_IDS).astype(np.uint8)
    vx, vy = geo["vp"]

    # 소실점: 직선 조각 + 소실점 + 경계 광선
    vis = (bg * 0.6 + 100).astype(np.uint8)
    for group in (s2._hough(marking * 255, 20), s2._hough(cv2.Canny(sep * 255, 50, 150), 40)):
        if group is None:
            continue
        for x1, y1, x2, y2 in group.astype(int):
            cv2.line(vis, (x1, y1), (x2, y2), hex_bgr(ORANGE), 2, cv2.LINE_AA)
    R = 3000
    for lane in geo["lanes"]:
        for t in lane["theta"]:
            e = (int(vx + R * np.sin(np.radians(t))), int(vy + R * np.cos(np.radians(t))))
            cv2.line(vis, (int(vx), int(vy)), e, (252, 252, 251), 1, cv2.LINE_AA)
    cv2.circle(vis, (int(vx), int(vy)), 9, (252, 252, 251), -1, cv2.LINE_AA)
    cv2.circle(vis, (int(vx), int(vy)), 6, hex_bgr(BLUE), -1, cv2.LINE_AA)
    save_img(vis, "f07_vanishing_point.jpg")
    shutil.copy(OUT / "s2" / SCENE / "lanes_vis.jpg", FIG / "f08_lanes.jpg")

    # θ 프로파일: 방향별 도로
    prof = geo["profile"]
    t = np.array(prof["theta"])
    lo = min(c["theta"][0] for c in geo["carriageways"]) - 8
    hi = max(c["theta"][1] for c in geo["carriageways"]) + 8
    sel = (t >= lo) & (t <= hi)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    names = {"toward": "다가오는 도로", "away": "멀어지는 도로", None: "방향 미정"}
    for c in geo["carriageways"]:
        ax.axvspan(*c["theta"], color=BLUE, alpha=0.07, lw=0)
        ax.text(sum(c["theta"]) / 2, 1.04, names[c["geo_direction"]], ha="center", va="bottom",
                fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
    ax.plot(t[sel], np.array(prof["road_cov"])[sel], color=BLUE, label="광선 중 도로 비율")
    ax.plot(t[sel], np.array(prof["sep_cov"])[sel], color=ORANGE, label="광선 중 분리대 비율")
    ax.axhline(s2.ROAD_COVER_MIN, color=MUTED, lw=1)
    away_c = next((c for c in geo["carriageways"] if c["geo_direction"] == "away"), geo["carriageways"][-1])
    ax.text(sum(away_c["theta"]) / 2, s2.ROAD_COVER_MIN + 0.03, f"도로 구간 기준 {s2.ROAD_COVER_MIN}",
            ha="center", fontsize=8, color=MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("소실점에서 본 각도 θ (도)")
    ax.set_ylabel("비율")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    save_fig(fig, "f09_theta_profile.png")

    # 차선 도색 θ 봉우리 vs 차량 바닥점 (멀어지는 도로)
    away = next((c for c in geo["carriageways"] if c["geo_direction"] == "away"), geo["carriageways"][-1])
    theta, radius = geo["theta"], np.hypot(*np.mgrid[0:h, 0:w][::-1].astype(np.float32) - np.array([vx, vy])[:, None, None])
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    zone = (yy > vy) & (radius > 0.12 * max(h, w))
    mk = s2.elongated_markings(marking, theta, radius) & zone & (cv2.dilate(road, np.ones((9, 9), np.uint8)) > 0)
    tmin, tmax = away["theta"]
    ys, xs = np.nonzero(mk)
    mth, mw = theta[ys, xs], 1.0 / np.maximum(yy[ys, xs] - vy, 1.0)
    edges = np.arange(tmin - 1, tmax + 1 + s2.THETA_BIN, s2.THETA_BIN)
    ctr = (edges[:-1] + edges[1:]) / 2
    mh = gaussian_filter1d(np.histogram(mth, bins=edges, weights=mw)[0], 1.5)
    pt_theta = np.degrees(np.arctan2(pts[:, 0] - vx, pts[:, 1] - vy))
    ph = gaussian_filter1d(np.histogram(pt_theta, bins=edges)[0].astype(float), 1.5)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(ctr, mh / mh.max(), color=BLUE, label="차선 도색 (배경 사진)")
    ax.plot(ctr, ph / max(ph.max(), 1e-9), color=ORANGE, label="차량 바닥점 (프레임 전체)")
    lanes = [l for l in geo["lanes"] if l["carriageway"] == away["id"]]
    for b in away.get("boundaries", []):
        ax.axvline(b["theta"], color=INK2, lw=1)
        ax.text(b["theta"], 1.04, "실선" if b["solid"] else "점선", ha="center", va="bottom", fontsize=8.5,
                color=INK2, transform=ax.get_xaxis_transform())
    for l in lanes:
        ax.text(sum(l["theta"]) / 2, 1.04, "갓길" if l["shoulder"] else "차로", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK, transform=ax.get_xaxis_transform())
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("소실점에서 본 각도 θ (도) — 멀어지는 도로")
    ax.set_ylabel("상대 빈도")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    save_fig(fig, "f10_marking_peaks.png")

    # 실선/점선: 경계를 따라 밝은 구간
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    road_zone = (road > 0) & zone
    asphalt = float(np.median(gray[(seg == 13) & zone]))
    rows = []
    for b in away.get("boundaries", []):
        band = road_zone & (np.abs(theta - b["theta"]) <= s2.PAINT_BAND_DEG)
        rb = (radius[band] // 4).astype(np.int64)
        order = np.argsort(rb, kind="stable")
        uniq, start = np.unique(rb[order], return_index=True)
        bright = np.maximum.reduceat(gray[band][order], start) > asphalt * 1.25 + 8
        rows.append((b, bright))
    fig, ax = plt.subplots(figsize=(7.2, 1.0 + 0.45 * len(rows)))
    for i, (b, bright) in enumerate(rows):
        runs, start = [], None
        for k, v in enumerate(np.r_[bright, False]):
            if v and start is None:
                start = k
            elif not v and start is not None:
                runs.append((start, k - start))
                start = None
        ax.broken_barh(runs, (i - 0.3, 0.6), color=BLUE)
    ax.set_yticks(range(len(rows)),
                  [f"{'실선' if b['solid'] else '점선'}  (밝은 비율 {b['paint_cov']:.2f}, {b['runs']}토막)" for b, _ in rows])
    ax.set_xlabel("경계선을 따라 소실점 쪽으로 (4px 구간 번호)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    save_fig(fig, "f11_paint.png")

    return {"scene": SCENE, "vp": [round(vx), round(vy)], "vp_support": geo["vp_support"],
            "carriageways": [{k: c[k] for k in ("id", "geo_direction", "theta", "n_lanes", "boundaries", "lane_source")}
                             for c in geo["carriageways"]],
            "lanes": [{k: l[k] for k in ("id", "carriageway", "n_points", "shoulder")} for l in geo["lanes"]]}


def fig_direction() -> None:
    shutil.copy(OUT / "s2b" / "test_grid.jpg", FIG / "f12_direction_grid.jpg")
    ev = json.loads((OUT / "s2b" / "eval.json").read_text())
    metrics = [("vehicle_acc", "차량 한 대씩 정확도"), ("carriageway_vote_acc", "도로 단위 다수결 정확도"), ("auc", "AUC")]
    models = [("resnet18_finetuned", "ResNet18 파인튜닝", BLUE), ("clip_zero_shot", "CLIP zero-shot", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    y = np.arange(len(metrics))
    for j, (key, label, color) in enumerate(models):
        vals = [ev[key][m] * 100 for m, _ in metrics]
        pos = y + (j - 0.5) * 0.34
        ax.barh(pos, vals, height=0.28, color=color, label=label)
        for p, v in zip(pos, vals):
            ax.text(v + 1, p, f"{v:.0f}%" if v > 1 else "", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [m for _, m in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("%  (학습에 쓰지 않은 카메라 6대, 차량 4,633대)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2)
    save_fig(fig, "f13_direction_bars.png")


def fig_lane_accuracy() -> None:
    by = pd.DataFrame(json.loads((OUT / "report" / "summary.json").read_text())["lane_count"]["by_region"])
    by = by.sort_values("scenes", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(by))
    for j, (col, label, color) in enumerate((("exact", "정확히 맞음", BLUE), ("within1", "±1 이내", ORANGE))):
        pos = y + (j - 0.5) * 0.36
        ax.barh(pos, by[col] * 100, height=0.3, color=color, label=label)
        for p, v in zip(pos, by[col] * 100):
            ax.text(v + 1, p, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"{r} ({n})" for r, n in zip(by["region"], by["scenes"])])
    ax.set_xlim(0, 112)
    ax.set_xlabel("장면 비율 %  (괄호: 장면 수, 정답 = 파일명 차로 수)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    save_fig(fig, "f14_lane_accuracy.png")
    shutil.copy(OUT / "s2" / FAILURE / "lanes_vis.jpg", FIG / "f15_failure.jpg")


# ------------------------------------------------------------------ 3·4단계

def fig_occupancy() -> dict:
    shutil.copy(OUT / "s3" / EXAMPLE_OCC / "peak.jpg", FIG / "f16_peak.jpg")
    lf = pd.read_csv(OUT / "s3" / "lane_frames.csv")
    d = lf[(lf.scene_id == EXAMPLE_OCC) & (~lf.shoulder)].sort_values("t_sec")
    means = d.groupby("lane_id")["occ_ground"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for lid in means.index[1:]:
        s = d[d.lane_id == lid]
        ax.plot(s.t_sec / 60, s.occ_ground.rolling(10, min_periods=1, center=True).mean() * 100,
                color=BASE, lw=1.4)
    s = d[d.lane_id == means.index[0]]
    ax.plot(s.t_sec / 60, s.occ_ground.rolling(10, min_periods=1, center=True).mean() * 100, color=BLUE)
    ax.plot([], [], color=BLUE, label="가장 붐비는 차로")
    ax.plot([], [], color=BASE, lw=1.4, label="같은 화면의 다른 차로")
    ax.set_xlabel("시간 (분, 사진 간격 약 3초로 추정)")
    ax.set_ylabel("점유율 % (30초 평균)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    save_fig(fig, "f17_timeseries.png")

    lanes = pd.read_csv(OUT / "s4" / "lane_delay.csv")
    groups = pd.read_csv(OUT / "s4" / "group_delay.csv")
    road = groups[groups["group"] == "all"][["scene_id", "frame", "occ_smooth", "level"]]
    worst = (lanes.sort_values("v_est_kmh").groupby(["scene_id", "frame"])
             .agg(worst_occ=("occ_smooth", "first"), worst_level=("level", "first")).reset_index())
    m = road.merge(worst, on=["scene_id", "frame"])
    hidden = (m["level"] == "원활") & (m["worst_level"] != "원활")
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(m.loc[~hidden, "occ_smooth"] * 100, m.loc[~hidden, "worst_occ"] * 100, s=5, color=BLUE,
               alpha=0.35, lw=0, label="차로별로 봐도 같은 판정")
    ax.scatter(m.loc[hidden, "occ_smooth"] * 100, m.loc[hidden, "worst_occ"] * 100, s=5, color=ORANGE,
               alpha=0.5, lw=0, label="도로 전체 '원활' / 한 차로 '서행' 이하")
    lim = float(max(m["worst_occ"].max(), m["occ_smooth"].max()) * 100)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1)
    ax.set_xlabel("도로 전체 점유율 %")
    ax.set_ylabel("가장 붐비는 차로 점유율 %")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1, markerscale=3)
    save_fig(fig, "f18_hidden.png")
    ex = means.round(3).to_dict()
    return {"example_lane_means": {int(k): v for k, v in ex.items()}, "hidden_share": round(float(hidden.mean()), 4)}


def fig_delay() -> None:
    occ_jam, v_free = 0.6, 100.0
    o = np.linspace(0, occ_jam, 200)
    v = np.clip(v_free * (1 - o / occ_jam), 5, v_free)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bands = [(80, 100, GOOD, "원활 (80 이상)"), (40, 80, WARNING, "서행 (40~80)"), (0, 40, CRITICAL, "정체 (40 미만)")]
    for lo, hi, color, label in bands:
        ax.axhspan(lo, hi, color=color, alpha=0.08, lw=0)
        ax.text(occ_jam * 100 + 1, (lo + hi) / 2, label, va="center", fontsize=8.5, color=INK2)
    ax.plot(o * 100, v, color=BLUE)
    for occ in (0.12, 0.36):
        ax.plot([occ * 100], [v_free * (1 - occ / occ_jam)], "o", ms=6, color=BLUE, mec=SURF, mew=2)
        ax.text(occ * 100 + 1, v_free * (1 - occ / occ_jam) + 4, f"점유율 {occ * 100:.0f}%", fontsize=8, color=INK2)
    ax.set_xlim(0, occ_jam * 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("차로 점유율 %")
    ax.set_ylabel("추정 속도 km/h")
    save_fig(fig, "f19_greenshields.png")


GROUP_FIXED = ["Busan_busanportbrdg3_20201027_1300_TUE_8m_NH_highway_TW3_sunny_FHD",
               "Busan_busanportbrdg6_20201027_1500_TUE_8m_NH_highway_TW3_sunny_FHD"]
GROUP_BROKE = ["Buan_yeji_20201021_0730_WED_15m_NH_highway_TW2_sunny_FHD"]


def fig_camera_groups() -> dict:
    """카메라 단위 도로 지도(s2c): 적용 전후 사례와 조건별 정확도."""
    def before_after(scene: str) -> np.ndarray:
        sd = OUT / "s2" / scene
        return hstack([cv2.imread(str(sd / "lanes_single_vis.jpg")), cv2.imread(str(sd / "lanes_vis.jpg"))], 260)

    rows = [before_after(s) for s in GROUP_FIXED]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 252)) for r in rows]
    save_img(np.concatenate(rows, axis=0), "f20_group_fixed.jpg")
    save_img(before_after(GROUP_BROKE[0]), "f21_group_broke.jpg")

    c = pd.read_csv(OUT / "s2c_compare.csv")
    subsets = [("전체", c), ("수원 제외", c[c.region != "Suwon"]), ("수원", c[c.region == "Suwon"]),
               ("밤", c[c.night]), ("낮", c[~c.night])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    y = np.arange(len(subsets))
    for ax, (metric, title) in zip(axes, (("exact", "정확히 맞음 (장면 %)"), ("within1", "±1 이내 (장면 %)"))):
        for j, (col, label, color) in enumerate(((f"{metric}_single", "적용 전 (클립마다 따로)", BASE),
                                                 (f"{metric}_G", "적용 후 (카메라 단위 G)", BLUE))):
            vals = [sub[col].mean() * 100 for _, sub in subsets]
            pos = y + (j - 0.5) * 0.36
            ax.barh(pos, vals, height=0.3, color=color, label=label)
            for p, v in zip(pos, vals):
                ax.text(v + 1.2, p, f"{v:.0f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 108)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{n} ({len(s)})" for n, s in subsets])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    save_fig(fig, "f22_group_compare.png")
    return {"fixed": int((~c.exact_single & c.exact_G).sum()), "broke": int((c.exact_single & ~c.exact_G).sum()),
            "groups": int(c.group.nunique()), "multi_groups": int((c.groupby("group").size() > 1).sum()),
            "grouped_scenes": int((c.group_size > 1).sum())}


CURVE_SCENES = ["Cheonan_gangjeong_20201021_0830_WED_15m_NH_highway_TW3_sunny_FHD",
                "Chungju_sangwoo1brdg_20201020_1130_TUE_15m_RH_highway_TW2_sunny_HD"]


def fig_curves() -> None:
    """휜 도로 곡선 보정(s2d) 전후, 그리고 무작위 12장면 점검 모음."""
    rows = []
    for s in CURVE_SCENES:
        sd = OUT / "s2" / s
        rows.append(hstack([cv2.imread(str(sd / "lanes_straight_vis.jpg")), cv2.imread(str(sd / "lanes_vis.jpg"))], 260))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 252)) for r in rows]
    save_img(np.concatenate(rows, axis=0), "f23_curve_before_after.jpg")

    import random
    scenes = sorted(str(p) for p in (OUT / "s2").glob("*/lanes_vis.jpg"))
    random.seed(7)
    pick = random.sample(scenes, 24)[:12]
    W, H = 480, 270
    tiles = []
    for p in pick:
        im = cv2.imread(p)
        s = min(W / im.shape[1], H / im.shape[0])
        r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        t = np.full((H, W, 3), 40, np.uint8)
        y0, x0 = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
        t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        tiles.append(cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(252, 252, 252)))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, 12, 3)], axis=0)
    save_img(sheet, "f24_random_check.jpg")


OVERCOUNT_SCENES = ["Gangneung_wichon2brdg_20201021_1430_WED_15m_NH_highway_TW2_sunny_FHD",
                    "Hongcheon_seomgangbrdg_20201021_0800_WED_15m_NH_highway_TW2_sunny_FHD"]


def fig_overcount() -> None:
    """다음 단계(7장): 수원 밖에서 가장 흔한 오류인 '차로를 많이 센' 사례."""
    save_img(hstack([cv2.imread(str(OUT / "s2" / s / "lanes_vis.jpg")) for s in OVERCOUNT_SCENES], 300),
             "f25_overcount.jpg")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_curves()
    fig_overcount()
    setup_matplotlib()
    clips = {c.clip_id: c for c in list_clips()}
    fig_data(clips)
    fig_background(clips)
    facts = {"segmentation": fig_segmentation()}
    fig_direction()
    fig_lane_accuracy()
    facts["camera_groups"] = fig_camera_groups()
    facts["occupancy"] = fig_occupancy()
    fig_delay()
    (FIG / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(facts, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
