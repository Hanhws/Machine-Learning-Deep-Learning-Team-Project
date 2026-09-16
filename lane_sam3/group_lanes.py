"""SAM3 가 뽑은 차선 조각을 차선(線)으로 묶고 차로 수를 센다.

SAM3 는 점선을 한 칸씩 따로 준다. 같은 차선의 조각들은 화면에서 소실점 한 점으로 모이므로,
    ① 길쭉한 조각만 남기고
    ② 조각들의 주축에서 소실점을 최소제곱으로 구한 뒤
    ③ 소실점을 향하지 않는 조각을 버리고
    ④ 남은 조각을 '화면 아래쪽 기준선에서의 x 위치'로 묶고
    ⑤ 크게 벌어진 곳(중앙분리대)에서 도로를 나눈다
차로 수 = 도로마다 (차선 수 - 1) 을 더한 값.

    ../.venv/bin/python group_lanes.py
    ../.venv/bin/python group_lanes.py --vis   # 묶은 결과 그림도 저장

결과:
    lanes/<scene_id>.json   차선별 조각 묶음과 기준선 x 위치
    vis_lanes/<scene_id>.jpg
    lane_count.csv          장면별 차선 수·차로 수·파일명 정답 비교
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MASKS = ROOT / "masks"
INPUTS = ROOT / "inputs"
BARRIERS = ROOT / "barriers"
LANES = ROOT / "lanes"
VIS = ROOT / "vis_lanes"

MIN_ELONG = 2.5      # 길쭉함. 노면 글자·패치를 거른다 (road_lane s2 와 같은 기준)
MIN_LENGTH = 8.0     # 너무 짧은 조각은 각도가 못 미더움
VP_ANGLE_TOL = 12.0  # 조각 주축과 '소실점 방향' 의 허용 오차(도)
MERGE_RATIO = 0.5    # 차선 폭의 이 비율보다 가깝게 붙은 묶음은 같은 차선으로 합친다
SPLIT_RATIO = 1.8    # 차선 간격의 이 배보다 벌어지면 중앙분리대로 보고 도로를 나눈다


def fit_vanishing_point(pieces: list[dict]) -> np.ndarray | None:
    """조각 주축 직선들의 최소제곱 교점. 각 직선을 법선형 n·x = c 로 두고 정규방정식을 푼다."""
    if len(pieces) < 3:
        return None
    N, C = [], []
    for g in pieces:
        th = np.deg2rad(g["angle"])
        d = np.array([np.cos(th), np.sin(th)])      # 직선 방향
        n = np.array([-d[1], d[0]])                 # 법선
        p = np.array([g["cx"], g["cy"]])
        w = min(g["length"], 60.0)                  # 긴 조각일수록 각도가 정확
        N.append(n * w)
        C.append(n @ p * w)
    N, C = np.array(N), np.array(C)
    try:
        vp, *_ = np.linalg.lstsq(N, C, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return vp


def points_at_vp(g: dict, vp: np.ndarray) -> float:
    """조각 주축이 소실점을 향하는지 — 방향 오차(도, 0~90)."""
    p = np.array([g["cx"], g["cy"]])
    v = vp - p
    if np.hypot(*v) < 1e-6:
        return 90.0
    a_vp = np.degrees(np.arctan2(v[1], v[0])) % 180
    d = abs(a_vp - g["angle"]) % 180
    return min(d, 180 - d)


def x_at_row(g: dict, vp: np.ndarray, y_ref: float) -> float | None:
    """소실점과 조각을 잇는 직선이 기준 행 y_ref 에서 갖는 x. 여기서 차선들이 잘 벌어진다."""
    dy = g["cy"] - vp[1]
    if abs(dy) < 1e-6:
        return None
    return float(vp[0] + (y_ref - vp[1]) * (g["cx"] - vp[0]) / dy)


def cluster_1d(xs: list[float], gap: float) -> list[list[int]]:
    """정렬 후 간격이 gap 보다 크면 끊는다."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    groups, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if xs[b] - xs[a] > gap:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def barrier_cuts(scene: str, vp: np.ndarray, y_ref: float,
                 cx_list: list[float], W: int) -> list[int] | None:
    """SAM3 가 찾은 중앙분리대로 차선 목록을 끊을 위치를 정한다.

    분리대는 도로를 따라 길게 뻗으므로 차선 조각과 같은 방식으로 기준선 x 로 투영한다.
    화면 양끝의 외곽 방호벽은 차선 바깥에 있어 자연히 걸러진다.
    반환은 '이 인덱스와 다음 인덱스 사이를 끊는다' 목록. 분리대 정보가 없으면 None.
    """
    f = BARRIERS / f"{scene}.json"
    if not f.exists():
        return None
    pieces = json.loads(f.read_text())["pieces"]
    # 길게 뻗은 구조물만. 작은 조각은 표지판·기둥일 수 있다.
    big = [g for g in pieces if g["length"] >= W * 0.06 and g["elong"] >= 2.0]
    if not big:
        return None

    xs = []
    for g in big:
        x = x_at_row(g, vp, y_ref)
        if x is not None and -W < x < 2 * W:
            xs.append(x)
    if not xs:
        return None

    cuts = []
    for i in range(len(cx_list) - 1):
        lo, hi = cx_list[i], cx_list[i + 1]
        # 두 차선 사이에 분리대가 지나가면 거기서 도로가 갈린다.
        if any(lo < x < hi for x in xs):
            cuts.append(i)
    return cuts


def process(scene: str, vis: bool) -> dict:
    d = json.loads((MASKS / f"{scene}.json").read_text())
    H, W = d["shape"]
    pieces = d["pieces"]
    n_raw = len(pieces)

    kept = [g for g in pieces if g["elong"] >= MIN_ELONG and g["length"] >= MIN_LENGTH]
    vp = fit_vanishing_point(kept)
    if vp is None:
        return {"scene_id": scene, "n_raw": n_raw, "n_kept": len(kept),
                "n_lines": 0, "lanes": 0, "vp_x": None, "vp_y": None, "ok": False}

    aligned = [g for g in kept if points_at_vp(g, vp) <= VP_ANGLE_TOL]
    if len(aligned) >= 3:                 # 정렬된 조각으로 소실점을 한 번 더 조인다
        vp2 = fit_vanishing_point(aligned)
        if vp2 is not None:
            vp = vp2
            aligned = [g for g in kept if points_at_vp(g, vp) <= VP_ANGLE_TOL]

    y_ref = H - 1.0
    xs, use = [], []
    for g in aligned:
        x = x_at_row(g, vp, y_ref)
        if x is not None and -2 * W < x < 3 * W:
            xs.append(x)
            use.append(g)

    if len(xs) < 2:
        return {"scene_id": scene, "n_raw": n_raw, "n_kept": len(kept), "n_aligned": len(aligned),
                "n_lines": len(xs), "lanes": max(len(xs) - 1, 0),
                "vp_x": round(float(vp[0]), 1), "vp_y": round(float(vp[1]), 1), "ok": False}

    # 1차로 잘게 묶고(조각 단위 간격은 차선 폭을 대표하지 못한다), 차선 간 간격을 재서 다시 합친다.
    groups = cluster_1d(xs, gap=W * 0.02)
    centers = sorted(float(np.median([xs[i] for i in g])) for g in groups)
    lane_gap = float(np.median(np.diff(centers))) if len(centers) > 1 else W / 8.0

    # 차선 폭의 절반도 안 되게 붙은 묶음은 같은 차선이 쪼개진 것 → 합친다.
    groups = cluster_1d(xs, gap=max(lane_gap * MERGE_RATIO, W * 0.02))

    lines = []
    for gi in groups:
        gx = [xs[i] for i in gi]
        lines.append({
            "x_ref": round(float(np.median(gx)), 1),
            "n_pieces": len(gi),
            "piece_idx": gi,
            # 점선이면 조각이 여러 개로 흩어지고, 실선·가장자리선이면 하나로 길게 나온다.
            "dashed": len(gi) >= 3,
            "span": round(float(max(g["cy"] for g in (use[i] for i in gi))
                                 - min(g["cy"] for g in (use[i] for i in gi))), 1),
        })
    lines.sort(key=lambda L: L["x_ref"])

    # 도로 나누기. SAM3 가 찾은 중앙분리대 위치를 우선 쓰고, 없으면 간격으로 추정한다.
    cx_list = [L["x_ref"] for L in lines]
    gaps = np.diff(cx_list) if len(cx_list) > 1 else np.array([])
    lane_gap2 = float(np.median(gaps)) if len(gaps) else lane_gap

    cuts = barrier_cuts(scene, vp, y_ref, cx_list, W)
    used_barrier = cuts is not None
    if cuts is None:
        cuts = [i for i, g in enumerate(gaps) if g > lane_gap2 * SPLIT_RATIO]
    carriageways, start = [], 0
    for c in cuts:
        carriageways.append(lines[start : c + 1])
        start = c + 1
    carriageways.append(lines[start:])
    carriageways = [c for c in carriageways if c]
    # 도로마다 차선 n 개 → 차로 n-1 개.
    lanes_total = sum(max(len(c) - 1, 0) for c in carriageways)
    # 대안: 차로 N 개 사이에는 점선 구분선이 N-1 개. 가장자리 실선을 안 세므로 갓널에 덜 흔들린다.
    lanes_dashed = sum(sum(L["dashed"] for L in c) + 1 for c in carriageways)

    LANES.mkdir(exist_ok=True)
    (LANES / f"{scene}.json").write_text(json.dumps(
        {"scene_id": scene, "vp": [round(float(vp[0]), 1), round(float(vp[1]), 1)],
         "y_ref": y_ref, "lines": lines,
         "carriageways": [[L["x_ref"] for L in c] for c in carriageways],
         "lanes": lanes_total, "lanes_dashed": lanes_dashed}, ensure_ascii=False))

    if vis:
        VIS.mkdir(exist_ok=True)
        im = cv2.imread(str(INPUTS / f"{scene}.jpg"))
        palette = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
                   (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255)]
        ov = im.copy()
        for li, L in enumerate(lines):
            c = palette[li % len(palette)]
            for i in L["piece_idx"]:
                cv2.fillPoly(ov, [np.array(use[i]["poly"], np.int32)], c)
            cv2.line(ov, (int(vp[0]), int(vp[1])), (int(L["x_ref"]), int(y_ref)), c, 1)
        im = cv2.addWeighted(ov, 0.6, im, 0.4, 0)
        if 0 <= vp[1] < H and -W < vp[0] < 2 * W:
            cv2.circle(im, (int(vp[0]), int(vp[1])), 6, (255, 255, 255), 2)
        cv2.putText(im, f"lines={len(lines)} ways={len(carriageways)} lanes={lanes_total}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(VIS / f"{scene}.jpg"), im)

    return {"scene_id": scene, "n_raw": n_raw, "n_kept": len(kept), "n_aligned": len(aligned),
            "n_lines": len(lines), "n_ways": len(carriageways), "lanes": lanes_total,
            "lanes_dashed": lanes_dashed, "by_barrier": used_barrier,
            "n_dashed": sum(L["dashed"] for L in lines),
            "vp_x": round(float(vp[0]), 1), "vp_y": round(float(vp[1]), 1), "ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(MASKS.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    rows = [process(f.stem, args.vis) for f in files]
    d = pd.DataFrame(rows)

    man = pd.read_csv(ROOT / "manifest.csv").set_index("scene_id")
    d = d.join(man[["camera", "token", "lanes_pred"]], on="scene_id")
    # 파일명 토큰(OW2/TW3...)의 숫자 = 방향당 차로 수. 양방향이면 화면에 그 두 배가 보인다.
    d["token_lanes"] = d.token.str.extract(r"(\d+)").astype(float)
    d["token_two_way"] = d.token.str.startswith("TW")
    d["expect_screen"] = np.where(d.token_two_way, d.token_lanes * 2, d.token_lanes)
    d["diff"] = d.lanes - d.expect_screen

    d.to_csv(ROOT / "lane_count.csv", index=False)
    ok = d[d.ok]
    print(f"장면 {len(d)}개 · 성공 {len(ok)}개")
    print(f"조각 평균 {d.n_raw.mean():.0f} → 길쭉 {d.n_kept.mean():.0f} → 소실점정렬 {d.n_aligned.mean():.0f} → 차선 {ok.n_lines.mean():.1f}")
    print(f"분리대로 도로 나눈 장면: {ok.by_barrier.sum()}/{len(ok)}")
    print(f"\n화면 전체 차로 수(방향 구분 없음) 기준:")
    for col, label in (("lanes", "A 차선수-1"), ("lanes_dashed", "B 점선수+1")):
        e = ok[col] - ok.expect_screen
        print(f"  {label:10s} 정확 {(e == 0).mean():.3f} · ±1 {(e.abs() <= 1).mean():.3f} · "
              f"적게 {(e < 0).mean():.3f} 많이 {(e > 0).mean():.3f}")


if __name__ == "__main__":
    main()
