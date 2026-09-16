"""2~4단계 완주 결과를 한 장으로 합친다.

    ../.venv/bin/python final_sheet.py --run ../road_lane/runs/sam3 --base ../road_lane/outputs

결과: <run>/FINAL.jpg
  윗줄  숫자 요약 (2단계 차로 수 · 방향 · 3·4단계 혼잡)
  가운데 한 장면의 전 과정 (배경 → 합친 seg → 차로 지도 → 혼잡 시계열)
  아랫줄 기준 대비 개선 그래프와 숨은 혼잡 산점도
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FG = (235, 232, 228)
MUTED = (150, 150, 150)
BG = (26, 22, 20)
GOOD = (120, 200, 90)
BAD = (80, 110, 240)
F = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, xy, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, xy, F, scale, color, thick, cv2.LINE_AA)


def fit(im, w=None, h=None):
    if im is None:
        return None
    if w:
        return cv2.resize(im, (w, max(1, int(im.shape[0] * w / im.shape[1]))))
    return cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))


def pad_to(im, w):
    if im.shape[1] >= w:
        return im[:, :w]
    return cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=BG)


def label(im, s, sub=""):
    bar = np.full((26 if not sub else 42, im.shape[1], 3), BG, np.uint8)
    text(bar, s, (8, 18), 0.5, FG, 1)
    if sub:
        text(bar, sub, (8, 35), 0.42, MUTED, 1)
    return np.vstack([bar, im])


def summary_block(run: Path, base: Path, W: int) -> np.ndarray:
    d = json.loads((run / "report" / "summary.json").read_text())
    a = pd.read_csv(base / "s2_summary.csv").set_index("scene_id")
    b = pd.read_csv(run / "s2_summary.csv").set_index("scene_id")
    c = a.index.intersection(b.index)
    a, b = a.loc[c], b.loc[c]
    su = a.camera.str.startswith("Suwon")
    ao, bo = a[a.ok == True], b[b.ok == True]
    an, bn = a[~su & (a.ok == True)], b[~su & (b.ok == True)]

    h = 196
    img = np.full((h, W, 3), BG, np.uint8)
    text(img, "road_lane 2-4 단계 완주 결과   SAM3 lane marking + Mask2Former road", (16, 28), 0.62, FG, 1)
    text(img, "2026-09-16 | car_seg_split 8,500 frames | 236 scenes | Apple M5", (16, 48), 0.42, MUTED, 1)
    cv2.line(img, (16, 60), (W - 16, 60), (60, 55, 52), 1)

    col = [16, int(W * 0.30), int(W * 0.57), int(W * 0.79)]

    text(img, "2 LANE COUNT", (col[0], 84), 0.45, MUTED, 1)
    text(img, f"{d['lane_count']['exact']:.3f}", (col[0], 116), 0.95, FG, 2)
    text(img, f"exact   (base {ao.lane_exact.mean():.3f})", (col[0], 136), 0.4, GOOD, 1)
    text(img, f"{d['lane_count']['within1']:.3f}  +-1   (base {ao.lane_within1.mean():.3f})", (col[0], 156), 0.4, FG, 1)
    text(img, f"no-Suwon +-1  {bn.lane_within1.mean():.3f}  (base {an.lane_within1.mean():.3f})", (col[0], 176), 0.4, GOOD, 1)

    text(img, "2b DIRECTION", (col[1], 84), 0.45, MUTED, 1)
    v = d["direction"]["resnet18_finetuned"]["carriageway_vote_acc"]
    text(img, f"{v:.3f}", (col[1], 116), 0.95, FG, 2)
    text(img, "carriageway vote", (col[1], 136), 0.4, MUTED, 1)
    text(img, f"{d['direction']['resnet18_finetuned']['vehicle_acc']:.3f}  per-vehicle", (col[1], 156), 0.4, FG, 1)
    text(img, f"0.982  geometry-CNN agree", (col[1], 176), 0.4, GOOD, 1)

    text(img, "3 OCCUPANCY", (col[2], 84), 0.45, MUTED, 1)
    text(img, "46,202", (col[2], 116), 0.95, FG, 2)
    text(img, "lane-frame rows", (col[2], 136), 0.4, MUTED, 1)
    text(img, f"236 scenes  /  8,494 frames", (col[2], 156), 0.4, FG, 1)
    text(img, f"two-way match {bo.two_way_match.mean():.3f}", (col[2], 176), 0.4, FG, 1)

    hc = d["hidden_congestion"]
    text(img, "4 HIDDEN CONGESTION", (col[3], 84), 0.45, MUTED, 1)
    text(img, f"{hc['road_free_but_some_lane_slow_or_jam']*100:.1f}%", (col[3], 116), 0.95, BAD, 2)
    text(img, "road free, a lane slow", (col[3], 136), 0.4, MUTED, 1)
    text(img, f"{hc['scenes_with_hidden_frames']} scenes affected", (col[3], 156), 0.4, FG, 1)
    text(img, f"median gap {hc['median_gap_pp']}pp", (col[3], 176), 0.4, FG, 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--scene", default="")
    args = ap.parse_args()

    run, base = Path(args.run).resolve(), Path(args.base).resolve()
    W = 1500

    d = json.loads((run / "report" / "summary.json").read_text())
    scene = args.scene or d["examples"][0]

    parts = [summary_block(run, base, W)]

    # --- 가운데: 한 장면의 전 과정
    clip = scene.split("__view")[0]
    bgp = run / "s1" / clip / "background.jpg"
    tiles = []
    src = [
        (bgp, "1  vehicle-free background", "s1_vehicles.py"),
        (run / "vis_seg" / f"{scene}.jpg", "2  merged seg", "blue=road M2F  red=barrier M2F  yellow=lane SAM3"),
        (run / "s2" / scene / "lanes_vis.jpg", "2d  final lane map", "s2 -> s2c -> s2d -> direction"),
        (run / "report" / "examples" / f"{scene}__timeseries.png", "3-4  per-lane occupancy", "s3_occupancy.py + s4_delay.py"),
    ]
    tw = (W - 5 * 8) // 4
    for p, t, sub in src:
        im = cv2.imread(str(p)) if p.exists() else None
        if im is None:
            im = np.full((160, tw, 3), (40, 36, 34), np.uint8)
            text(im, "(no file)", (10, 84), 0.5, MUTED, 1)
        im = fit(im, w=tw)
        tiles.append(label(im, t, sub))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BG) for t in tiles]
    gap = np.full((hmax, 8, 3), BG, np.uint8)
    row = np.hstack([gap] + sum([[t, gap] for t in tiles], []))
    band = np.full((30, W, 3), BG, np.uint8)
    text(band, f"one scene end-to-end:  {scene[:110]}", (16, 20), 0.46, MUTED, 1)
    parts += [band, pad_to(row, W)]

    # --- 아랫줄: 리포트 그래프
    bot = []
    for p, t in ((run / "report" / "lane_accuracy.png", "lane count accuracy by region"),
                 (run / "report" / "hidden_congestion.png", "hidden congestion (red = road free, a lane slow)"),
                 (run / "report" / "examples" / f"{scene}__peak.jpg", "busiest frame")):
        im = cv2.imread(str(p)) if p.exists() else None
        if im is None:
            continue
        bot.append(label(fit(im, h=330), t))
    if bot:
        hm = max(b.shape[0] for b in bot)
        bot = [cv2.copyMakeBorder(b, 0, hm - b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BG) for b in bot]
        g = np.full((hm, 8, 3), BG, np.uint8)
        parts += [np.full((16, W, 3), BG, np.uint8),
                  pad_to(np.hstack([g] + sum([[b, g] for b in bot], [])), W)]

    foot = np.full((54, W, 3), BG, np.uint8)
    cv2.line(foot, (16, 8), (W - 16, 8), (60, 55, 52), 1)
    text(foot, "CAUTION  accuracy is measured against filename lane counts (OW/TW), which a manual check found",
         (16, 28), 0.42, MUTED, 1)
    text(foot, "correct only ~25% of the time -- treat 0.409 vs 0.357 as indicative, not confirmed.",
         (16, 45), 0.42, MUTED, 1)
    parts.append(foot)

    out = run / "FINAL.jpg"
    cv2.imwrite(str(out), np.vstack([pad_to(p, W) for p in parts]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{out}")


if __name__ == "__main__":
    main()
