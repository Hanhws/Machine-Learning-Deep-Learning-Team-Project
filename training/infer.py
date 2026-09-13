"""추론 — 이미지(또는 폴더)별 차량 대수·픽셀 수·(도로 마스크가 있으면) 점유율.

    python infer.py --run runs/yolo11s_seg --source samples/ --save-vis
    python infer.py --run runs/yolo11s_seg --source frame.jpg --road-mask road.png --save-vis --save-masks
    python infer.py --config configs/yolo11s_seg.yaml --pretrained --source samples/   # 사전학습 모델로

산출물 (--out-dir, 기본 <run>/infer):
    results.csv          이미지별 한 행: num_vehicles, count_car/bus/truck, pixels_*, vehicle_pixels,
                         overlap_pixels, image_ratio, road_pixels, occupancy
    results.json         같은 내용 + 인스턴스별 점수·클래스·상자
    vis/*.jpg            (--save-vis) 마스크·상자·요약 패널 오버레이
    union_masks/*.png    (--save-masks) 차량 합집합 이진 마스크 (255=차량)

차량 픽셀은 인스턴스 마스크의 합집합(masks.any(axis=0))으로 세므로 겹친 픽셀은 한 번만 들어간다.
--road-mask 에 폴더를 주면 이미지와 같은 이름의 .png 를 찾아 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from carseg import CLASS_NAMES
from carseg.config import ExperimentConfig
from carseg.data import load_road_mask
from carseg.inference import VehicleSegmenter
from carseg.models.base import read_rgb
from carseg.utils import save_json
from carseg.viz import draw_prediction, save_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="train.py 가 만든 run 폴더")
    src.add_argument("--config", help="설정 YAML (--weights 또는 --pretrained 와 함께)")
    p.add_argument("--weights", default=None)
    p.add_argument("--pretrained", action="store_true", help="COCO 사전학습 모델 그대로 사용")
    p.add_argument("--source", required=True, help="이미지 파일 또는 폴더")
    p.add_argument("--road-mask", default=None, help="도로 마스크 PNG 파일 또는 폴더 (0=배경)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--score", type=float, default=None, help="차량으로 셀 점수 임계값 (기본 eval.union_conf)")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def list_images(source: Path, limit: int):
    if source.is_file():
        return [source]
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit else images


def find_road_mask(road: Path | None, image_path: Path, shape):
    if road is None:
        return None
    path = road if road.is_file() else road / f"{image_path.stem}.png"
    return load_road_mask(path, shape) if path.exists() else None


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.run and args.set:
        raise SystemExit("--run 과 --set 은 함께 쓸 수 없습니다 (--config 사용)")
    if args.run:
        seg = VehicleSegmenter.from_run(args.run, weights=args.weights, score_thr=args.score)
        default_out = Path(args.run) / "infer"
    else:
        cfg = ExperimentConfig.load(args.config, overrides=args.set)
        if not (args.weights or args.pretrained):
            raise SystemExit("--config 를 쓸 때는 --weights 또는 --pretrained 가 필요합니다")
        seg = VehicleSegmenter(cfg, args.weights or cfg.model.weights,
                               label_mode="coco" if args.pretrained else "finetuned", score_thr=args.score)
        default_out = cfg.run_dir / ("infer_pretrained" if args.pretrained else "infer")

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    road = Path(args.road_mask) if args.road_mask else None
    images = list_images(Path(args.source), args.limit)
    if not images:
        raise SystemExit(f"이미지를 찾지 못했습니다: {args.source}")

    from tqdm.auto import tqdm

    rows, details = [], []
    for path in tqdm(images, desc="infer", unit="img", dynamic_ncols=True):
        image = read_rgb(path)
        road_mask = find_road_mask(road, path, image.shape[:2])
        out = seg.run(image, road_mask=road_mask)
        rows.append({"image": str(path), **out.stats.flat()})

        kept = out.prediction.above(seg.score_thr)
        details.append({"image": str(path), "stats": out.stats.to_dict(), "instances": [
            {"class": CLASS_NAMES[int(l)], "score": round(float(s), 4),
             "box_xyxy": [round(float(v), 1) for v in b], "pixels": int(m.sum())}
            for b, s, l, m in zip(kept.boxes, kept.scores, kept.labels, kept.masks)]})

        if args.save_vis:
            vis = draw_prediction(image, out.prediction, score_thr=seg.score_thr, stats=out.stats,
                                  road_mask=road_mask)
            save_rgb(vis, out_dir / "vis" / f"{path.stem}.jpg")
        if args.save_masks:
            import cv2

            (out_dir / "union_masks").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "union_masks" / f"{path.stem}.png"), out.union_mask.astype(np.uint8) * 255)

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    save_json(details, out_dir / "results.json")

    total = sum(r["num_vehicles"] for r in rows)
    mean_ratio = float(np.mean([r["image_ratio"] for r in rows])) * 100
    print(f"\n[infer] {len(rows)}장  총 차량 {total}대  평균 차량 면적비 {mean_ratio:.2f}%")
    occ = [r["occupancy"] for r in rows if r.get("occupancy") is not None]
    if occ:
        print(f"[infer] 평균 도로 점유율 {np.mean(occ) * 100:.2f}%  ({len(occ)}장)")
    print(f"[infer] 결과: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
