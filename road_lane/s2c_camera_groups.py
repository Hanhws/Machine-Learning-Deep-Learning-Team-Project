"""2c) 카메라 단위 도로 지도: 같은 카메라·같은 화면의 장면을 묶어 도로 지도를 공유한다.

클립마다 따로 도로를 찾으면 같은 카메라인데도 결과가 흔들린다(고정 화면 카메라 39대 중 30대).
밤·비·흐린 차선 때문에 틀린 장면이, 같은 화면의 좋은 장면 결과를 빌려 쓰게 한다.

같은 화면 판정: Mask2Former 도로 영역 IoU ≥ 0.75, 위치 어긋남 ≤ 40px (윤곽선 유사도는 낮·밤 조명에 흔들려서 안 씀)
그룹: 카메라마다 조건이 가장 좋은 장면(낮 → 소실점 강함)을 기준으로, 기준과 같은 화면인 장면을 묶는다.

  A  대표 공유: 기준 장면의 도로 지도를 위치만 보정해 그룹 전체에 쓴다
  B  근거 합치기: 도로 분할을 투표(밤 가중치 0.4)로 합치고, 모든 장면의 차량 바닥점을 모아 도로 지도를 다시 계산

실행
  python s2c_camera_groups.py            # A·B 를 파일명 차로 수로 채점해 지금 방식과 비교만
  python s2c_camera_groups.py --apply B  # 선택한 방식으로 outputs/s2/<scene>/ 를 덮어씀 (원본은 *_single.* 로 백업)
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import s2_road_lanes as s2
from common import OUT, list_clips

SAME_VIEW_IOU = 0.75
MAX_SHIFT_PX = 40
NIGHT_WEIGHT = 0.4
VOTE_ROAD, VOTE_SEP, VOTE_MARK = 0.5, 0.4, 0.35
# A 대표 분할+대표 바닥점, B 투표 분할+모은 바닥점, C 대표 분할+모은 바닥점, D 투표 분할+대표 바닥점,
# E 투표 도로·분리대 + 대표 차선 도색 + 대표 바닥점, F 투표 분할(차선 도색은 60% 다수결) + 대표 바닥점
# G 대표 장면이 좋으면 E, 약하면(밤이거나 소실점 강도 하위 1/3) D — 정답이 아니라 대표 장면 품질로만 고른다
STRATEGIES = ("A", "B", "C", "D", "E", "F", "G")
STRICT_MARK = 0.6
WEAK_VP = 0.0  # main 에서 전체 장면 소실점 강도의 33번째 백분위로 정한다


def is_night(clip) -> bool:
    return clip.hour >= 19 or clip.hour < 7


def load_scenes() -> list[dict]:
    clips = {c.clip_id: c for c in list_clips()}
    scenes = []
    dets_cache: dict[str, list] = {}
    for vj in sorted((OUT / "s1").glob("*/views.json")):
        clip = clips[vj.parent.name]
        views = json.loads(vj.read_text())["views"]
        if clip.clip_id not in dets_cache:
            with open(vj.parent / "dets.pkl", "rb") as f:
                dets_cache = {clip.clip_id: pickle.load(f)["dets"]}
        for view in views:
            sd = OUT / "s2" / view["scene_id"]
            single_json = sd / "lanes_single.json" if (sd / "lanes_single.json").exists() else sd / "lanes.json"
            info = json.loads(single_json.read_text())
            bg = cv2.imread(str(vj.parent / view["background"]))
            seg = cv2.imread(str(sd / "seg.png"), cv2.IMREAD_UNCHANGED)
            g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
            frames = set(view["frames"])
            scenes.append({
                "scene_id": view["scene_id"], "clip": clip, "camera": clip.camera_id, "night": is_night(clip),
                "bg": bg, "seg": seg, "frames": sorted(frames),
                "grad": np.log1p(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))),
                "road": np.isin(seg, s2.ROAD_IDS),
                "pts": s2.foot_points([d for d in dets_cache[clip.clip_id] if d["frame"] in frames]),
                "ok": bool(info.get("ok")), "vp_support": float(info.get("vp_support", 0.0)),
            })
    return scenes


def translate(img: np.ndarray, dx: float, dy: float, nearest: bool = True) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def match(ref: dict, other: dict) -> tuple[float, float, float] | None:
    """other 를 ref 에 맞추는 이동량(dx, dy)과 도로 IoU. other(x) ≈ ref(x − d) 이면 d 를 돌려준다."""
    if ref["grad"].shape != other["grad"].shape:
        return None
    (dx, dy), _ = cv2.phaseCorrelate(ref["grad"], other["grad"])
    best = None
    for sgn in (1, -1):  # 부호 규약을 가정하지 않고 IoU 가 큰 쪽을 쓴다
        moved = translate(other["road"].astype(np.uint8), -sgn * dx, -sgn * dy) > 0
        valid = translate(np.ones_like(other["road"], np.uint8), -sgn * dx, -sgn * dy) > 0
        inter = (ref["road"] & moved & valid).sum()
        union = ((ref["road"] | moved) & valid).sum()
        iou = float(inter / max(union, 1))
        if best is None or iou > best[2]:
            best = (sgn * dx, sgn * dy, iou)
    return best


def build_groups(scenes: list[dict]) -> list[dict]:
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for s in scenes:
        by_cam[s["camera"]].append(s)
    groups = []
    for cam, ss in by_cam.items():
        # 조건이 좋은 장면부터 기준이 된다: 낮 → 도로 추출 성공 → 소실점 강함
        left = sorted(ss, key=lambda s: (not s["night"], s["ok"], s["vp_support"]), reverse=True)
        while left:
            ref = left.pop(0)
            members, rest = [(ref, 0.0, 0.0, 1.0)], []
            for s in left:
                m = match(ref, s)
                if m and m[2] >= SAME_VIEW_IOU and np.hypot(m[0], m[1]) <= MAX_SHIFT_PX:
                    members.append((s, m[0], m[1], m[2]))
                else:
                    rest.append(s)
            left = rest
            groups.append({"camera": cam, "ref": ref, "members": members})
    return groups


def vote_segmentation(members: list, mark_thr: float = VOTE_MARK, ref_markings: np.ndarray | None = None) -> np.ndarray:
    """장면들의 분할 결과를 기준 좌표로 옮겨 가중 투표. s2 가 쓰는 클래스 id 로만 채운다.
    ref_markings 를 주면 차선 도색은 투표 대신 기준 장면 것을 쓴다 (날짜마다 조금씩 어긋난 도색이 두 겹이 되는 것 방지)."""
    out = _vote(members, mark_thr)
    if ref_markings is not None:
        out[out == s2.MARKING_ID] = 13
        out[ref_markings] = s2.MARKING_ID
    return out


def _vote(members: list, mark_thr: float) -> np.ndarray:
    h, w = members[0][0]["seg"].shape
    f_road, f_sep, f_mark, total = (np.zeros((h, w), np.float32) for _ in range(4))
    for s, dx, dy, _ in members:
        wgt = NIGHT_WEIGHT if s["night"] else 1.0
        seg = translate(s["seg"], -dx, -dy)
        valid = translate(np.ones((h, w), np.uint8), -dx, -dy).astype(np.float32)
        f_road += wgt * valid * np.isin(seg, s2.ROAD_IDS)
        f_sep += wgt * valid * np.isin(seg, s2.SEPARATOR_IDS)
        f_mark += wgt * valid * (seg == s2.MARKING_ID)
        total += wgt * valid
    total = np.maximum(total, 1e-6)
    f_road, f_sep, f_mark = f_road / total, f_sep / total, f_mark / total
    out = np.zeros((h, w), np.uint8)
    out[f_road >= VOTE_ROAD] = 13
    out[(f_sep >= VOTE_SEP) & (f_road < 0.6)] = 5
    out[f_mark >= mark_thr] = s2.MARKING_ID
    return out


def group_geometry(group: dict, strategy: str) -> dict:
    ref = group["ref"]
    if strategy == "G":
        # 대표 장면의 차선 도색을 믿을 만하면 그대로(E), 아니면 여러 장면 투표로 바로잡는다(D)
        strategy = "D" if (ref["night"] or ref["vp_support"] < WEAK_VP) else "E"
    if strategy == "A" or len(group["members"]) == 1:
        return s2.lane_geometry(ref["seg"], ref["pts"], ref["bg"])
    if strategy in ("B", "D"):
        seg = vote_segmentation(group["members"])
    elif strategy == "E":
        seg = vote_segmentation(group["members"], ref_markings=ref["seg"] == s2.MARKING_ID)
    elif strategy == "F":
        seg = vote_segmentation(group["members"], mark_thr=STRICT_MARK)
    else:
        seg = ref["seg"]
    pts = (np.concatenate([s["pts"] - np.array([dx, dy], np.float32) for s, dx, dy, _ in group["members"]])
           if strategy in ("B", "C") else ref["pts"])
    geo = s2.lane_geometry(seg, pts, ref["bg"])
    if not geo["ok"]:  # 합친 근거로 실패하면 대표 공유로 물러난다
        geo = s2.lane_geometry(ref["seg"], ref["pts"], ref["bg"])
        geo["fallback"] = "A"
    return geo


def for_member(geo: dict, dx: float, dy: float) -> dict:
    """그룹(기준 좌표)의 도로 지도를 장면 좌표로 옮긴다. θ 는 소실점과 함께 옮기므로 값이 그대로다."""
    if not geo["ok"]:
        return geo
    out = {k: v for k, v in geo.items() if k not in ("lane_map", "theta", "voted_seg")}
    out["lane_map"] = translate(geo["lane_map"], dx, dy)
    vx, vy = geo["vp"][0] + dx, geo["vp"][1] + dy
    out["vp"] = [vx, vy]
    h, w = geo["lane_map"].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out["theta"] = np.degrees(np.arctan2(xx - vx, yy - vy)).astype(np.float32)
    return out


def evaluate_all(groups: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows, geos = [], {st: {} for st in STRATEGIES}
    single = pd.read_csv(OUT / ("s2_summary_single.csv" if (OUT / "s2_summary_single.csv").exists() else "s2_summary.csv"))
    single = single.set_index("scene_id")
    for gi, g in enumerate(groups):
        per = {st: group_geometry(g, st) for st in STRATEGIES}
        for s, dx, dy, iou in g["members"]:
            row = {"scene_id": s["scene_id"], "group": gi, "group_size": len(g["members"]),
                   "is_ref": s is g["ref"], "night": s["night"], "iou_to_ref": round(iou, 3),
                   "shift": round(float(np.hypot(dx, dy)), 1), "region": s["camera"].split("_")[0],
                   "token": s["clip"].lane_token}
            r0 = single.loc[s["scene_id"]] if s["scene_id"] in single.index else None
            row["exact_single"] = bool(r0["lane_exact"]) if r0 is not None and r0["ok"] else False
            row["within1_single"] = bool(r0["lane_within1"]) if r0 is not None and r0["ok"] else False
            row["lanes_single"] = str(r0["main_lanes"]) if r0 is not None and r0["ok"] else "fail"
            for st in STRATEGIES:
                mg = for_member(per[st], dx, dy)
                ev = s2.evaluate(s["clip"], mg)
                row[f"exact_{st}"] = bool(ev.get("lane_exact", False))
                row[f"within1_{st}"] = bool(ev.get("lane_within1", False))
                row[f"lanes_{st}"] = ev.get("main_lanes", "fail")
                geos[st][s["scene_id"]] = (mg, s)
            rows.append(row)
    return pd.DataFrame(rows), geos


def error_direction(lanes: str, token: str) -> str:
    """파일명 차로 수 대비 방향별 도로마다 많게(+)·적게(−) 틀렸는지."""
    if lanes in ("fail", "nan"):
        return "fail"
    n, want = int(token[2:]), 2 if token.startswith("TW") else 1
    got = [int(x) for x in lanes.split("/")]
    if len(got) < want:
        return "도로누락"
    diffs = [x - n for x in got]
    if all(d == 0 for d in diffs):
        return "정확"
    if all(d >= 0 for d in diffs):
        return "많게"
    if all(d <= 0 for d in diffs):
        return "적게"
    return "섞임"


def report(df: pd.DataFrame) -> None:
    cols = ["single", *STRATEGIES]

    def line(sub: pd.DataFrame, name: str) -> str:
        ex = " · ".join(f"{c[0] if c != 'single' else '지금'} {sub[f'exact_{c}'].mean():.2f}" for c in cols)
        w1 = " · ".join(f"{c[0] if c != 'single' else '지금'} {sub[f'within1_{c}'].mean():.2f}" for c in cols)
        return f"{name:10s} n={len(sub):3d} | 정확 {ex} | ±1 {w1}"
    print(f"그룹 {df.group.nunique()}개 (2장면 이상 {int((df.groupby('group').size() > 1).sum())}개), "
          f"그룹에 속한 장면 {int((df.group_size > 1).sum())}/{len(df)}")
    for sub, name in ((df, "전체"), (df[df.region != "Suwon"], "수원 제외"), (df[df.region == "Suwon"], "수원"),
                      (df[df.night], "밤"), (df[~df.night], "낮")):
        print(line(sub, name))
    ns = df[df.region != "Suwon"]
    print("\n수원 제외 오류 방향 (많게 = 차로를 더 많이 셈)")
    for c in cols:
        print(f"  {c:6s}", ns.apply(lambda r: error_direction(str(r[f'lanes_{c}']), r.token), axis=1).value_counts().to_dict())


def apply(strategy: str, geos: dict, df: pd.DataFrame) -> None:
    rows = []
    for scene_id, (geo, s) in geos[strategy].items():
        sd = OUT / "s2" / scene_id
        for name in ("lanes.json", "lanes.npz", "lanes_vis.jpg"):
            src, bak = sd / name, sd / name.replace("lanes", "lanes_single", 1)
            if src.exists() and not bak.exists():
                shutil.copy(src, bak)
        cv2.imwrite(str(sd / "lanes_vis.jpg"), s2.draw(s["bg"], geo, s["pts"], s["clip"]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if geo["ok"]:
            np.savez_compressed(sd / "lanes.npz", lane_map=geo["lane_map"], theta=geo["theta"], vp=np.array(geo["vp"]))
        payload = {k: v for k, v in geo.items() if k not in ("lane_map", "theta")}
        grp = df[df.scene_id == scene_id].iloc[0]
        payload |= {"clip_id": s["clip"].clip_id, "scene_id": scene_id, "frames": s["frames"],
                    "camera_group": int(grp.group), "camera_group_size": int(grp.group_size), "strategy": strategy}
        (sd / "lanes.json").write_text(json.dumps(payload, ensure_ascii=False, default=float))
        rows.append({"scene_id": scene_id, "n_views": None} | s2.evaluate(s["clip"], geo))
    if not (OUT / "s2_summary_single.csv").exists():
        shutil.copy(OUT / "s2_summary.csv", OUT / "s2_summary_single.csv")
    views = {}
    for vj in (OUT / "s1").glob("*/views.json"):
        n = json.loads(vj.read_text())["n_views"]
        for v in json.loads(vj.read_text())["views"]:
            views[v["scene_id"]] = n
    out = pd.DataFrame(rows)
    out["n_views"] = out["scene_id"].map(views)
    out.to_csv(OUT / "s2_summary.csv", index=False)
    print(f"{strategy} 적용: 장면 {len(out)}개 저장, 원본은 lanes_single.* / s2_summary_single.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", choices=list(STRATEGIES), default=None)
    args = ap.parse_args()
    scenes = load_scenes()
    global WEAK_VP
    WEAK_VP = float(np.percentile([s["vp_support"] for s in scenes if s["ok"]], 33))
    print(f"약한 소실점 기준 vp_support < {WEAK_VP:.3f}")
    groups = build_groups(scenes)
    df, geos = evaluate_all(groups)
    df.to_csv(OUT / "s2c_compare.csv", index=False)
    (OUT / "s2c_groups.json").write_text(json.dumps(
        [{"camera": g["camera"], "ref": g["ref"]["scene_id"],
          "members": [{"scene_id": s["scene_id"], "dx": round(dx, 1), "dy": round(dy, 1), "iou": round(iou, 3)}
                      for s, dx, dy, iou in g["members"]]} for g in groups], ensure_ascii=False, indent=1))
    report(df)
    if args.apply:
        apply(args.apply, geos, df)


if __name__ == "__main__":
    main()
