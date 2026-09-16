"""합쳐진 seg.png 를 눈으로 볼 수 있게 색칠한다.

seg.png 는 클래스 번호 지도(값이 0~65)라 그냥 열면 까맣게 보인다.
여기서는 s2 가 실제로 쓰는 것만 색을 주고, 어느 모델이 그린 부분인지 구분한다.

    파랑  도로      (Mask2Former)
    빨강  분리대    (Mask2Former)
    노랑  차선 도색 (SAM3)          ← 바뀐 부분
    회색  그 밖

    ../.venv/bin/python vis_seg.py --run ../road_lane/runs/sam3
    ../.venv/bin/python vis_seg.py --run ../road_lane/outputs --out vis_seg_base
    ../.venv/bin/python vis_seg.py --run ... --sheet 6   # 대표 장면 비교 시트도 만든다
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROAD_IDS = [13, 24, 23, 14, 7, 41, 36, 43]
SEPARATOR_IDS = [4, 5, 2, 6, 3]
MARKING_ID = 24

C_ROAD = (150, 90, 30)      # BGR 파랑 계열
C_SEP = (40, 40, 200)       # 빨강
C_MARK = (60, 240, 255)     # 노랑
C_OTHER = (70, 70, 70)


def colorize(seg: np.ndarray, bg: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
    out = np.full((*seg.shape, 3), C_OTHER, np.uint8)
    out[np.isin(seg, ROAD_IDS)] = C_ROAD
    out[np.isin(seg, SEPARATOR_IDS)] = C_SEP
    out[seg == MARKING_ID] = C_MARK          # 도로 위에 덮어 그린다
    if bg is not None:
        if bg.shape[:2] != seg.shape:
            bg = cv2.resize(bg, (seg.shape[1], seg.shape[0]))
        out = cv2.addWeighted(out, alpha, bg, 1 - alpha, 0)
    return out


def legend(img: np.ndarray, title: str) -> np.ndarray:
    bar = np.zeros((30, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    x = img.shape[1] - 330
    for c, t in ((C_ROAD, "road"), (C_SEP, "sep"), (C_MARK, "marking")):
        cv2.rectangle(bar, (x, 9), (x + 16, 23), c, -1)
        cv2.putText(bar, t, (x + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        x += 90
    return np.vstack([bar, img])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="결과 폴더 (runs/sam3 또는 outputs)")
    ap.add_argument("--out", default="vis_seg", help="run 폴더 안에 만들 하위 폴더 이름")
    ap.add_argument("--sheet", type=int, default=0, help="대표 장면 N개로 기준 대비 비교 시트 생성")
    ap.add_argument("--base", default="../road_lane/outputs", help="시트에서 비교할 기준 폴더")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    out = run / args.out
    out.mkdir(parents=True, exist_ok=True)

    scenes = sorted(d.name for d in (run / "s2").iterdir() if (d / "seg.png").exists())
    for s in scenes:
        seg = cv2.imread(str(run / "s2" / s / "seg.png"), cv2.IMREAD_UNCHANGED)
        bgp = run / "s1" / s.split("__view")[0]
        bg = None
        for cand in (bgp / "background.jpg", bgp / f"background_{s.split('__view')[-1]}.jpg"):
            if cand.exists():
                bg = cv2.imread(str(cand))
                break
        cv2.imwrite(str(out / f"{s}.jpg"), colorize(seg, bg))
    print(f"색칠 {len(scenes)}장 → {out}")

    if args.sheet:
        base = Path(args.base).resolve()
        rows = []
        for s in scenes[: args.sheet]:
            pa, pb = base / "s2" / s / "seg.png", run / "s2" / s / "seg.png"
            if not pa.exists():
                continue
            bgp = run / "s1" / s.split("__view")[0] / "background.jpg"
            bg = cv2.imread(str(bgp)) if bgp.exists() else None
            ia = legend(colorize(cv2.imread(str(pa), cv2.IMREAD_UNCHANGED), bg), "BEFORE  Mask2Former only")
            ib = legend(colorize(cv2.imread(str(pb), cv2.IMREAD_UNCHANGED), bg), "AFTER  road=M2F + marking=SAM3")
            h = 280
            rs = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
            ia, ib = rs(ia), rs(ib)
            strip = np.hstack([ia, np.full((h, 4, 3), 255, np.uint8), ib])
            lab = np.zeros((24, strip.shape[1], 3), np.uint8)
            cv2.putText(lab, s[:95], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
            rows.append(np.vstack([lab, strip]))
        if rows:
            W = max(r.shape[1] for r in rows)
            rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    for r in rows]
            p = run / "compare" / "03_seg_전후.jpg"
            p.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(p), np.vstack(rows))
            print(f"비교 시트 → {p}")


if __name__ == "__main__":
    main()
