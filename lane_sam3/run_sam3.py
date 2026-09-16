"""SAM3 로 배경 102장에서 차선 도색을 찾는다.

SAM3 는 점선을 한 칸씩 따로 잡으므로(차선 6줄에 마스크 55개), 여기서는 조각을 그대로 저장하고
묶는 일은 group_lanes.py 에서 한다. 조각마다 무게중심·주축 각도·길이를 미리 재 둔다.

    ../.venv/bin/python run_sam3.py            # 전체 102장
    ../.venv/bin/python run_sam3.py --limit 5  # 5장만

결과:
    masks/<scene_id>.json   조각별 폴리곤과 기하 (무게중심·각도·길이·폭·면적)
    vis/<scene_id>.jpg      확인용 색칠 그림
    run_log.csv             장면별 조각 수·소요 시간
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
MASKS = ROOT / "masks"
VIS = ROOT / "vis"

PROMPT = ["lane marking"]
IMGSZ = 1036  # 배경이 960x540 이라 이 근처가 적정. 1400 은 4배 느린데 조각만 늘어난다.
CONF = 0.25


def piece_geometry(mask: np.ndarray) -> dict | None:
    """조각 하나의 폴리곤과 주축 기하. 차선은 가늘고 길쭉한 조각이다."""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 4:
        return None

    # 최소 외접 회전사각형 → 길이·폭·각도
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    length, width = (h, w) if h >= w else (w, h)
    # 각도를 "긴 축이 x축과 이루는 각(0~180)"으로 통일
    angle = ang + 90 if h >= w else ang
    angle = angle % 180

    poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return {
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "length": round(float(length), 2),
        "width": round(float(width), 2),
        "angle": round(float(angle), 2),
        "area": round(area, 1),
        "elong": round(float(length / max(width, 1e-6)), 2),
        "poly": poly.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (0=전체)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--prompt", nargs="+", default=PROMPT, help='예: --prompt "median barrier"')
    ap.add_argument("--outdir", default="masks", help="결과 폴더 (masks / barriers ...)")
    ap.add_argument("--skip-done", action="store_true", help="이미 있는 장면은 건너뛴다")
    args = ap.parse_args()

    out_masks = ROOT / args.outdir
    out_vis = ROOT / f"vis_{args.outdir}" if args.outdir != "masks" else VIS

    from ultralytics.models.sam import SAM3SemanticPredictor

    imgs = sorted(INPUTS.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    out_masks.mkdir(exist_ok=True)
    out_vis.mkdir(exist_ok=True)
    if args.skip_done:
        imgs = [p for p in imgs if not (out_masks / f"{p.stem}.json").exists()]
        print(f"건너뛰기 적용 → 남은 {len(imgs)}장")

    pred = SAM3SemanticPredictor(
        overrides={
            "conf": CONF, "task": "segment", "mode": "predict",
            "model": str(ROOT / "sam3.pt"), "save": False, "verbose": False,
            "imgsz": args.imgsz,
        }
    )

    rows = []
    rng = np.random.default_rng(0)
    for i, p in enumerate(imgs, 1):
        scene = p.stem
        t = time.time()
        pred.set_image(str(p))
        r = pred(text=args.prompt)
        res = r[0] if isinstance(r, (list, tuple)) else r

        pieces = []
        if res.masks is not None:
            data = res.masks.data.cpu().numpy()
            for k in range(len(data)):
                g = piece_geometry(data[k])
                if g is not None:
                    pieces.append(g)

        took = time.time() - t
        (out_masks / f"{scene}.json").write_text(
            json.dumps({"scene_id": scene, "imgsz": args.imgsz, "prompt": args.prompt,
                        "shape": list(res.orig_shape), "pieces": pieces}, ensure_ascii=False)
        )

        im = cv2.imread(str(p))
        ov = im.copy()
        for g in pieces:
            cv2.fillPoly(ov, [np.array(g["poly"], np.int32)], rng.integers(0, 255, 3).tolist())
        cv2.imwrite(str(out_vis / f"{scene}.jpg"), cv2.addWeighted(ov, 0.55, im, 0.45, 0))

        rows.append({"scene_id": scene, "n_pieces": len(pieces), "sec": round(took, 2)})
        print(f"[{i}/{len(imgs)}] {scene[:58]:58s} 조각 {len(pieces):3d} {took:.1f}초", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(ROOT / f"run_log_{args.outdir}.csv", index=False)
    print(f"\n{len(d)}장 완료 · 조각 평균 {d.n_pieces.mean():.1f}개 "
          f"(최소 {d.n_pieces.min()} 최대 {d.n_pieces.max()}) · 합계 {d.sec.sum()/60:.1f}분")
    if (d.n_pieces == 0).any():
        print("조각 0개 장면:", d[d.n_pieces == 0].scene_id.tolist())


if __name__ == "__main__":
    main()
