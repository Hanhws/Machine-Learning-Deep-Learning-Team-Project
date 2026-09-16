"""SAM3 마스크로 s2 가 읽는 seg.png 를 만든다 (Mask2Former 대체).

s2_road_lanes.py 는 장면마다 seg.png(Mapillary Vistas 클래스 지도)를 읽어
  road    = isin(seg, ROAD_IDS)      도로 비율
  marking = (seg == 24)              소실점 추정
  sep     = isin(seg, SEPARATOR_IDS) 방향별 도로 끊기
  asphalt = median(gray[seg == 13])  도색 판정 기준 밝기
를 뽑아 쓴다. 그래서 여기서는 SAM3 결과를 같은 클래스 번호로 그려 준다.
차선(24)을 도로(13) 위에 덮어 그려야 seg==13 이 '도색을 뺀 아스팔트'가 된다.

    ../.venv/bin/python make_sam3_seg.py --out ../road_lane/runs/sam3
    ../.venv/bin/python make_sam3_seg.py --out ... --limit 5   # 맛보기

프롬프트 3개를 set_image 한 번에 물어본다(set_image 가 5초, 질의는 1.7초씩).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RL = PROJECT / "road_lane"
sys.path.insert(0, str(RL))

SRC_OUT = RL / "outputs"          # 배경과 화면 분리 결과는 기준 폴더에서 가져온다

# Mapillary Vistas 클래스 번호 (s2_road_lanes.py 와 맞춰야 한다)
ID_ROAD = 13
ID_MARKING = 24
ID_BARRIER = 5                    # SEPARATOR_IDS 의 Barrier
ID_VOID = 0

# SAM3 는 차선·분리대 같은 '물체'는 잘 잡지만 도로 면 같은 '영역'은 들쭉날쭉하다
# (같은 카메라에서 도로 면적이 38.6% → 12.6% 로 튄다). 그래서 기본은 하이브리드:
# 도로·분리대는 Mask2Former 결과를 그대로 두고 차선만 SAM3 로 덮어쓴다.
PROMPTS_FULL = {
    ID_ROAD: ["road"],
    ID_BARRIER: ["median barrier"],
    ID_MARKING: ["lane marking"],  # 마지막에 그려 도로 위에 덮는다
}
PROMPTS_MARK = {ID_MARKING: ["lane marking"]}
IMGSZ = 1036
CONF = 0.25


def scenes() -> list[tuple[str, Path]]:
    """s2 와 같은 순서로 (scene_id, 배경 경로). 회전형 카메라는 화면마다 배경이 따로 있다."""
    from common import list_clips

    out = []
    for clip in list_clips():
        vf = SRC_OUT / "s1" / clip.clip_id / "views.json"
        if not vf.exists():
            continue
        for v in json.loads(vf.read_text())["views"]:
            bg = SRC_OUT / "s1" / clip.clip_id / v["background"]
            if bg.exists():
                out.append((v["scene_id"], bg))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="실험 결과 폴더 (ROAD_LANE_OUT 로 쓸 곳)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--mode", choices=("hybrid", "full"), default="hybrid",
                    help="hybrid=도로·분리대는 Mask2Former 유지, 차선만 SAM3 (기본) / full=전부 SAM3")
    args = ap.parse_args()
    prompts = PROMPTS_MARK if args.mode == "hybrid" else PROMPTS_FULL

    from ultralytics.models.sam import SAM3SemanticPredictor

    out_root = Path(args.out).resolve()
    todo = scenes()
    if args.limit:
        todo = todo[: args.limit]
    if args.skip_done:
        todo = [(s, b) for s, b in todo if not (out_root / "s2" / s / "seg.png").exists()]
        print(f"건너뛰기 적용 → 남은 {len(todo)}장면")

    pred = SAM3SemanticPredictor(overrides={
        "conf": CONF, "task": "segment", "mode": "predict",
        "model": str(ROOT / "sam3.pt"), "save": False, "verbose": False, "imgsz": args.imgsz,
    })

    t0 = time.time()
    for i, (scene, bg_path) in enumerate(todo, 1):
        bg = cv2.imread(str(bg_path))
        h, w = bg.shape[:2]
        if args.mode == "hybrid":
            base = SRC_OUT / "s2" / scene / "seg.png"
            if not base.exists():
                print(f"[{i}/{len(todo)}] {scene[:52]:52s} 기준 seg.png 없음 → 건너뜀", flush=True)
                continue
            seg = cv2.imread(str(base), cv2.IMREAD_UNCHANGED).copy()
            # 기존 차선 도색을 지우고(도로로 되돌림) SAM3 것으로 다시 그린다
            seg[seg == ID_MARKING] = ID_ROAD
        else:
            seg = np.full((h, w), ID_VOID, np.uint8)

        t = time.time()
        pred.set_image(str(bg_path))
        counts = {}
        for cls_id, prompt in prompts.items():   # dict 는 삽입 순서 유지 → 차선이 마지막
            r = pred(text=prompt)
            res = r[0] if isinstance(r, (list, tuple)) else r
            n = 0
            if res.masks is not None:
                m = res.masks.data.cpu().numpy()
                for k in range(len(m)):
                    mk = m[k] > 0.5
                    if mk.shape != seg.shape:
                        mk = cv2.resize(mk.astype(np.uint8), (w, h),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    seg[mk] = cls_id
                    n += 1
            counts[prompt[0]] = n

        d = out_root / "s2" / scene
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "seg.png"), seg)

        frac = {k: round(float((seg == c).mean()) * 100, 1)
                for c, k in ((ID_ROAD, "road"), (ID_BARRIER, "barrier"), (ID_MARKING, "mark"))}
        print(f"[{i}/{len(todo)}] {scene[:52]:52s} {counts} 면적% {frac} {time.time()-t:.1f}초",
              flush=True)

    print(f"\n{len(todo)}장면 완료 · {(time.time()-t0)/60:.1f}분 → {out_root/'s2'}")


if __name__ == "__main__":
    main()
