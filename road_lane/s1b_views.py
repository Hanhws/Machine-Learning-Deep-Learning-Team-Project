"""1b) 회전형(PTZ) 카메라의 화면 전환 감지 → 화면(view)별 배경.

수원 카메라는 한 클립 안에서 PRESET 001 ↔ 002 로 방향을 바꾼다. 서로 다른 화면의 차량 위치·배경이 섞이면
차선 추출이 망가지므로, 프레임을 화면별로 묶고 화면마다 배경을 다시 만든다.

화면 특징: 사진 위쪽(하늘·건물·육교 — 차량이 거의 없음) + 전체를 크게 줄인 흑백.
k=2,3 KMeans 의 실루엣 점수가 기준 이상이고 모든 묶음이 MIN_VIEW_FRAMES 장 이상일 때만 여러 화면으로 본다.

출력 outputs/s1/<clip_id>/views.json
  {"n_views", "silhouette", "views": [{"scene_id", "frames", "background"}]}
  여러 화면이면 outputs/s1/<clip_id>/view<k>/background.jpg 를 새로 만든다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import OUT, Clip, frame_index, list_clips
from s1_vehicles import BG_MAX_FRAMES, rasterize

MIN_VIEW_FRAMES = 12
SILHOUETTE_MIN = 0.3
MERGE_NCC = 0.7


def view_feature(path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8).astype(np.float32)
    h = g.shape[0]
    parts = [cv2.resize(g[: int(0.45 * h)], (40, 16), interpolation=cv2.INTER_AREA),
             cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA)]
    feats = [(p.ravel() - p.mean()) / (p.std() + 1e-6) for p in parts]
    return np.concatenate([feats[0], 0.5 * feats[1]])


def split_views(clip: Clip) -> tuple[np.ndarray, float]:
    X = np.stack([view_feature(p) for p in clip.frames])
    labels, best = np.zeros(len(X), int), -1.0
    for k in (2, 3):
        if len(X) < k * MIN_VIEW_FRAMES:
            break
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        if np.bincount(lab).min() < MIN_VIEW_FRAMES:
            continue
        s = float(silhouette_score(X, lab))
        if s > best:
            labels, best = lab, s
    if best < SILHOUETTE_MIN:
        return np.zeros(len(X), int), best
    return labels, best


def view_background(clip: Clip, frame_ids: set[int], dets: dict) -> np.ndarray:
    w, h = dets["size"]
    by_frame: dict[int, list] = {}
    for d in dets["dets"]:
        by_frame.setdefault(d["frame"], []).append(d["poly"])
    paths = [p for p in clip.frames if frame_index(p) in frame_ids]
    pick = np.linspace(0, len(paths) - 1, min(BG_MAX_FRAMES, len(paths))).round().astype(int)
    stack = []
    for i in sorted(set(pick)):
        p = paths[i]
        img = cv2.resize(cv2.imread(str(p)), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        vmask = rasterize(by_frame.get(frame_index(p), []), (h, w))
        img[cv2.dilate(vmask, np.ones((7, 7), np.uint8)) > 0] = np.nan
        stack.append(img)
    bg = np.nanmedian(np.stack(stack), axis=0)
    holes = np.isnan(bg)
    if holes.any():
        bg[holes] = np.median(np.stack([np.nan_to_num(x, nan=0) for x in stack]), axis=0)[holes]
    return bg.clip(0, 255).astype(np.uint8)


def edge_signature(bg: np.ndarray) -> np.ndarray:
    """밝기·색에 둔감한 구도 특징: 윤곽선 세기(log) 를 정규화."""
    g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    m = np.log1p(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)))
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def merge_same_geometry(groups: list[np.ndarray], backgrounds: list[np.ndarray]) -> list[np.ndarray]:
    """구도는 같고 밝기만 다른 화면(가로등 켜짐, 노출, 흑백 전환)은 합친다.

    54개 클립 확인: 같은 구도 쌍은 유사도 0.84 이상, 실제로 다른 화면은 0.48 이하.
    """
    sig = [edge_signature(b) for b in backgrounds]
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if float((sig[i] * sig[j]).sum()) >= MERGE_NCC:
                parent[find(j)] = find(i)
    merged: dict[int, list] = {}
    for i, g in enumerate(groups):
        merged.setdefault(find(i), []).extend(g.tolist())
    return [np.array(sorted(g)) for g in merged.values()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    clips = [c for c in list_clips() if (OUT / "s1" / c.clip_id / "dets.pkl").exists()]
    if args.only:
        clips = [c for c in clips if any(c.clip_id.startswith(o) for o in args.only)]

    multi = 0
    for k, clip in enumerate(clips, 1):
        s1 = OUT / "s1" / clip.clip_id
        if (s1 / "views.json").exists() and not args.redo:
            continue
        for old in s1.glob("view[0-9]*"):
            if old.is_dir():
                shutil.rmtree(old)
        labels, sil = split_views(clip)
        frames = np.array([frame_index(p) for p in clip.frames])
        groups = [frames[labels == v] for v in range(int(labels.max()) + 1)]
        backgrounds = None
        if len(groups) > 1:
            with open(s1 / "dets.pkl", "rb") as f:
                dets = pickle.load(f)
            backgrounds = [view_background(clip, set(g.tolist()), dets) for g in groups]
            merged = merge_same_geometry(groups, backgrounds)
            if len(merged) != len(groups):
                groups = merged
                backgrounds = ([view_background(clip, set(g.tolist()), dets) for g in groups]
                               if len(groups) > 1 else None)

        views = []
        if len(groups) == 1:
            views.append({"scene_id": clip.clip_id, "frames": frames.tolist(), "background": "background.jpg"})
        else:
            multi += 1
            # 프레임 수가 많은 화면부터 view0, view1 …
            for v, i in enumerate(np.argsort([-len(g) for g in groups])):
                out = s1 / f"view{v}"
                out.mkdir(exist_ok=True)
                cv2.imwrite(str(out / "background.jpg"), backgrounds[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
                views.append({"scene_id": f"{clip.clip_id}__view{v}", "frames": groups[i].tolist(),
                              "background": f"view{v}/background.jpg"})
        n_views = len(views)
        with open(s1 / "views.json", "w") as f:
            json.dump({"n_views": n_views, "silhouette": round(sil, 3), "views": views}, f)
        print(f"[{k}/{len(clips)}] {clip.clip_id} views={n_views} silhouette={sil:.2f} "
              f"sizes={[len(v['frames']) for v in views]}", flush=True)
    print(f"여러 화면 클립: {multi}")


if __name__ == "__main__":
    main()
