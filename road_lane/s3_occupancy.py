"""3) 차로별 · 방향별 · 도로 전체 혼잡도(점유율).

프레임마다
  1. 차량 바닥점이 속한 차로에 차량을 배정 (차로 밖 15px 이내면 가장 가까운 차로)
  2. 차로 안에서 그 차로에 배정된 차량이 덮은 픽셀을 센다 (가까운 차가 먼 차를 가림)
  3. 점유율 두 가지
     occ_px     = 차량 픽셀 / 차로 픽셀                         (원근 보정 없음)
     occ_ground = Σ_y f(y)·w(y) / Σ_y w(y),  f = 그 줄의 차량 비율, w = 1/차로폭(y)²
                  평평한 도로를 핀홀 카메라로 보면 한 줄이 담는 실제 도로 길이 ∝ 1/차로폭² 이므로
                  "실제 도로 길이 중 차량이 차지한 비율"에 가깝다.
  차량 높이가 앞쪽 도로를 가리는 만큼은 과대추정된다 (특히 트럭, 카메라 가까운 곳).

출력 outputs/s3/
  lane_frames.csv   장면·프레임·차로별 대수와 점유율
  lane_summary.csv  장면·차로별 평균
  <scene_id>/peak.jpg       가장 붐빈 프레임에 차로별 점유율 표시
  <scene_id>/timeseries.png 차로별 점유율 시계열
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from common import OUT, frame_index, list_clips

FRAME_INTERVAL_SEC = 3.0  # 추정값: 인제 터널 화면 시각(300번 = 클립 시작 +15분 22초)
MIN_LANE_WIDTH_PX = 6  # 이보다 좁은 줄(먼 곳)은 점유율 계산에서 뺀다
NEAR_RANGE_RATIO = 0.3
ASSIGN_RADIUS_PX = 15


def lane_rows(lane_map: np.ndarray, n_lanes: int) -> np.ndarray:
    """[차로 id, 행] → 그 행에서 차로 폭(픽셀)."""
    h = lane_map.shape[0]
    ys = np.nonzero(lane_map)[0]
    return np.bincount(lane_map[lane_map > 0].astype(np.int64) * h + ys, minlength=(n_lanes + 1) * h).reshape(n_lanes + 1, h)


def occupancy(counts: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """counts, width: [차로, 행]. 반환: 차로별 (occ_px, occ_ground)."""
    occ_px = counts.sum(1) / np.maximum(width.sum(1), 1)
    # 먼 곳은 1/폭² 가중치가 수백 배가 되고, 차 높이가 여러 줄을 가려 과대추정된다
    # → 그 차로 최대 폭의 NEAR_RANGE_RATIO 이상인 가까운 줄만 쓴다 (깊이 약 3배 범위)
    near = width >= NEAR_RANGE_RATIO * width.max(1, keepdims=True)
    valid = (width >= MIN_LANE_WIDTH_PX) & near
    wgt = np.where(valid, 1.0 / np.maximum(width, 1) ** 2, 0.0)
    frac = np.where(valid, counts / np.maximum(width, 1), 0.0)
    occ_g = (frac * wgt).sum(1) / np.maximum(wgt.sum(1), 1e-12)
    return occ_px, occ_g


def process_scene(scene_dir, dets: list[dict], frame_paths: dict[int, str]) -> tuple[list[dict], dict] | None:
    info = json.loads((scene_dir / "lanes.json").read_text())
    if not info.get("ok"):
        return None
    lane_map = np.load(scene_dir / "lanes.npz")["lane_map"].astype(np.int16)
    h, w = lane_map.shape
    lanes = {l["id"]: l for l in info["lanes"]}
    # 방향: 기하 규칙이 있으면 그것, 없으면(한쪽 도로만 보임) s2b 차량 앞/뒷모습 CNN 이 확신한 방향
    dir_file = OUT / "s2b" / "directions.json"
    cnn = json.loads(dir_file.read_text()).get(info["scene_id"], {}) if dir_file.exists() else {}
    for lane in lanes.values():
        lane["direction"] = lane["geo_direction"] or cnn.get(str(lane["carriageway"]), {}).get("final")
    n = max(lanes)
    width = lane_rows(lane_map, n)
    dist, (iy, ix) = distance_transform_edt(lane_map == 0, return_indices=True)
    yrow = np.broadcast_to(np.arange(h)[:, None], (h, w))

    by_frame: dict[int, list] = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)

    rows = []
    best = (-1.0, None, None)
    for fid in info["frames"]:
        veh = np.zeros((h, w), np.int16)
        count = np.zeros(n + 1, int)
        items = []
        for d in by_frame.get(fid, []):
            p = d["poly"]
            if len(p) < 3:
                continue
            ymax = p[:, 1].max()
            fx, fy = int(np.clip(p[p[:, 1] >= ymax - 3, 0].mean(), 0, w - 1)), int(np.clip(ymax, 0, h - 1))
            if dist[fy, fx] > ASSIGN_RADIUS_PX:
                continue
            lid = int(lane_map[iy[fy, fx], ix[fy, fx]])
            items.append((fy, lid, p))
            count[lid] += 1
        # 먼 차부터 그려서 가까운 차가 덮어쓰게 한다
        for _, lid, p in sorted(items, key=lambda t: t[0]):
            cv2.fillPoly(veh, [np.round(p).astype(np.int32)], lid)
        match = (veh == lane_map) & (lane_map > 0)
        counts = np.bincount(lane_map[match].astype(np.int64) * h + yrow[match], minlength=(n + 1) * h).reshape(n + 1, h)
        occ_px, occ_g = occupancy(counts, width)

        for lid, lane in lanes.items():
            rows.append({"scene_id": info["scene_id"], "clip_id": info["clip_id"], "frame": fid,
                         "t_sec": fid * FRAME_INTERVAL_SEC, "lane_id": lid, "carriageway": lane["carriageway"],
                         "direction": lane["direction"] or "unknown", "shoulder": lane["shoulder"],
                         "count": int(count[lid]), "occ_px": round(float(occ_px[lid]), 4),
                         "occ_ground": round(float(occ_g[lid]), 4)})
        total = float(occ_g[[l for l in lanes if not lanes[l]["shoulder"]]].mean()) if lanes else 0.0
        if total > best[0]:
            best = (total, fid, occ_g.copy())

    return rows, {"info": info, "lane_map": lane_map, "width": width, "peak": best, "frame_paths": frame_paths}


def aggregate(rows: list[dict], width_total: dict) -> list[dict]:
    """장면·프레임 단위로 방향별·도로 전체 점유율 (차로 면적 가중 평균, 갓길 제외)."""
    out = []
    by_key: dict[tuple, list] = {}
    for r in rows:
        if not r["shoulder"]:
            by_key.setdefault((r["scene_id"], r["frame"]), []).append(r)
    for (scene, fid), rs in by_key.items():
        for group, sel in [("all", rs)] + [(d, [r for r in rs if r["direction"] == d]) for d in ("toward", "away", "unknown")]:
            if not sel:
                continue
            wts = np.array([width_total[(scene, r["lane_id"])] for r in sel], float)
            out.append({"scene_id": scene, "frame": fid, "t_sec": sel[0]["t_sec"], "group": group,
                        "n_lanes": len(sel), "count": sum(r["count"] for r in sel),
                        # 차로 면적이 모두 0 이면(지도에 칠해지지 않은 차로뿐) 가중 평균 대신 단순 평균
                        "occ_ground": round(float(np.average([r["occ_ground"] for r in sel],
                                                             weights=wts if wts.sum() > 0 else None)), 4),
                        "lane_occ_max": round(max(r["occ_ground"] for r in sel), 4),
                        "lane_occ_min": round(min(r["occ_ground"] for r in sel), 4)})
    return out


def draw_peak(scene_dir, ctx: dict) -> None:
    total, fid, occ_g = ctx["peak"]
    if fid is None:
        return
    img = cv2.imread(ctx["frame_paths"][fid])
    h, w = ctx["lane_map"].shape
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    over = img.copy()
    for lane in ctx["info"]["lanes"]:
        m = ctx["lane_map"] == lane["id"]
        if not m.any():
            continue
        if lane["shoulder"]:
            over[m] = (128, 128, 128)
            continue
        o = float(np.clip(occ_g[lane["id"]] / 0.5, 0, 1))  # 50% 이상이면 가장 빨갛게
        over[m] = (0, int(255 * (1 - o)), int(255 * o))
    vis = cv2.addWeighted(img, 0.55, over, 0.45, 0)
    for lane in ctx["info"]["lanes"]:
        ys, xs = np.nonzero(ctx["lane_map"] == lane["id"])
        if len(ys) == 0 or lane["shoulder"]:
            continue
        sel = ys > np.percentile(ys, 70)
        x, y = int(xs[sel].mean()), int(ys[sel].mean())
        arrow = {"toward": "v", "away": "^"}.get(lane["direction"], "")
        label = f"{arrow}{occ_g[lane['id']] * 100:.0f}%"
        cv2.putText(vis, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(scene_dir / "peak.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_timeseries(scene_dir, rows: list[dict], info: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for lane in info["lanes"]:
        if lane["shoulder"]:
            continue
        rs = sorted((r for r in rows if r["lane_id"] == lane["id"]), key=lambda r: r["t_sec"])
        t = np.array([r["t_sec"] for r in rs]) / 60
        y = np.array([r["occ_ground"] for r in rs]) * 100
        k = min(10, len(y))  # 약 30초 이동평균
        ys = np.convolve(y, np.ones(k) / k, mode="same") if k > 1 else y
        ax.plot(t, ys, lw=1.6, ls="-" if lane["direction"] != "away" else "--",
                label=f"L{lane['id']} {lane['geo_direction'] or ''}")
    ax.set_xlabel("minutes (est. 3s/frame)")
    ax.set_ylabel("occupancy % (30s avg)")
    ax.legend(fontsize=7, ncol=4, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(scene_dir / "timeseries.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    clips = {c.clip_id: c for c in list_clips()}
    scene_dirs = sorted(p.parent for p in (OUT / "s2").glob("*/lanes.json"))
    if args.only:
        scene_dirs = [d for d in scene_dirs if any(d.name.startswith(o) for o in args.only)]

    all_rows, width_total, summary = [], {}, []
    dets_cache: dict[str, list] = {}
    for k, sd in enumerate(scene_dirs, 1):
        clip_id = sd.name.split("__view")[0]
        if clip_id not in dets_cache:
            with open(OUT / "s1" / clip_id / "dets.pkl", "rb") as f:
                dets_cache = {clip_id: pickle.load(f)["dets"]}
        frame_paths = {frame_index(p): str(p) for p in clips[clip_id].frames}
        res = process_scene(sd, dets_cache[clip_id], frame_paths)
        if res is None:
            continue
        rows, ctx = res
        out = OUT / "s3" / sd.name
        out.mkdir(parents=True, exist_ok=True)
        draw_peak(out, ctx)
        draw_timeseries(out, rows, ctx["info"])
        for lane in ctx["info"]["lanes"]:
            width_total[(sd.name, lane["id"])] = int(ctx["width"][lane["id"]].sum())
            rs = [r for r in rows if r["lane_id"] == lane["id"]]
            occ = np.array([r["occ_ground"] for r in rs])
            summary.append({"scene_id": sd.name, "clip_id": clip_id, "lane_id": lane["id"],
                            "carriageway": lane["carriageway"], "direction": lane["direction"] or "unknown",
                            "shoulder": lane["shoulder"], "frames": len(rs),
                            "mean_count": round(float(np.mean([r["count"] for r in rs])), 2),
                            "occ_ground_mean": round(float(occ.mean()), 4),
                            "occ_ground_p90": round(float(np.percentile(occ, 90)), 4),
                            "occ_px_mean": round(float(np.mean([r["occ_px"] for r in rs])), 4)})
        all_rows += rows
        print(f"[{k}/{len(scene_dirs)}] {sd.name} lanes={len(ctx['info']['lanes'])} frames={len(ctx['info']['frames'])}",
              flush=True)

    (OUT / "s3").mkdir(exist_ok=True)
    for name, rows in (("lane_frames.csv", all_rows), ("lane_summary.csv", summary),
                       ("group_frames.csv", aggregate(all_rows, width_total))):
        if rows:
            with open(OUT / "s3" / name, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    print(f"장면 {len(scene_dirs)}개, 차로-프레임 행 {len(all_rows)}개")


if __name__ == "__main__":
    main()
