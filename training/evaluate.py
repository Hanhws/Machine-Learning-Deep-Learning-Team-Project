"""통합 평가 — 사전학습 그대로(zero-shot) 또는 학습된 가중치.

1) 사전학습 모델을 바로 val 로 평가 (COCO 80종 중 car/bus/truck 만 추려 채점)
    python evaluate.py --pretrained --config configs/yolo11s_seg.yaml configs/maskrcnn_r50.yaml configs/mask2former_swin_s.yaml

2) 학습된 모델 평가 (run 폴더만 주면 config.yaml·best 가중치를 알아서 찾음)
    python evaluate.py --run runs/yolo11s_seg
    python evaluate.py --run runs/yolo11s_seg --split test
    python evaluate.py --run runs/yolo11s_seg --set eval.conf=0.001 eval.tta=true   # 추론 설정만 바꿔 재채점

어느 쪽이든 결과는 <out_dir>/experiments.csv 에 한 행씩 추가된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from carseg.coco_eval import evaluate
from carseg.config import ExperimentConfig
from carseg.data import CarSegData
from carseg.experiment_log import log_experiment
from carseg.models import build_predictor


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", nargs="+", help="설정 YAML (여러 개 가능, --pretrained 와 함께)")
    src.add_argument("--run", nargs="+", help="train.py 가 만든 run 폴더 (여러 개 가능)")
    p.add_argument("--pretrained", action="store_true", help="model.weights 를 COCO 사전학습 그대로 평가")
    p.add_argument("--weights", default=None, help="평가할 가중치를 직접 지정 (단일 설정일 때)")
    p.add_argument("--split", default=None, help="val(기본) | test | train")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기")
    p.add_argument("--save-predictions", action="store_true", help="COCO 결과 JSON 도 저장")
    p.add_argument("--no-log", action="store_true", help="기록표에 추가하지 않음")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    jobs = []
    if args.run:
        if args.pretrained:
            raise SystemExit("--pretrained 는 --config 와 함께 쓰세요")
        for run in args.run:
            run = Path(run)
            cfg = ExperimentConfig.load(run / "config.yaml", overrides=args.set)
            if not any(item.startswith("out_dir=") for item in args.set):
                cfg.out_dir, cfg.name = str(run.parent), run.name  # run 폴더를 옮겨도 그 자리에 결과를 남긴다
            weights = args.weights or json.loads((run / "train_result.json").read_text())["best_weights"]
            jobs.append((cfg, weights, "finetuned"))
    else:
        for path in args.config:
            cfg = ExperimentConfig.load(path, overrides=args.set)
            if args.pretrained:
                cfg.name = f"{cfg.name}__pretrained"
                cfg.group = "pretrained (zero-shot)"
                jobs.append((cfg, args.weights or cfg.model.weights, "coco"))
            else:
                if not args.weights:
                    raise SystemExit("--config 로 학습된 모델을 평가하려면 --weights 가 필요합니다")
                jobs.append((cfg, args.weights, "finetuned"))

    for cfg, weights, label_mode in jobs:
        split = args.split or cfg.data.val_split
        stage = "pretrained" if label_mode == "coco" else "finetuned"
        out_dir = cfg.run_dir / (f"eval_{stage}" if stage == "pretrained" else f"eval_{split}")
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(out_dir / "config.yaml")
        print(f"\n[eval] {cfg.name}  weights={weights}  split={split}  mode={label_mode}")

        predictor = build_predictor(cfg, weights, label_mode=label_mode)
        metrics = evaluate(predictor, CarSegData(cfg.data.root), split, limit=cfg.data.eval_limit,
                           out_dir=out_dir, save_predictions=args.save_predictions)
        if not args.no_log:
            log_experiment(cfg, metrics, stage=stage, eval_weights=str(weights))
    return 0


if __name__ == "__main__":
    sys.exit(main())
