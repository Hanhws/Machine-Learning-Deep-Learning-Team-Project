"""학습 → val 채점 → 기록표 추가 를 한 번에.

예)
    python train.py --config configs/yolo11s_seg.yaml
    python train.py --config configs/maskrcnn_r50.yaml --set train.epochs=3 data.train_limit=200   # 빠른 점검
    python train.py --config configs/yolo11s_seg.yaml --set name=yolo11s_lr5e-4 train.lr0=5e-4 --test

산출물 (<out_dir>/<name>/):
    config.yaml          실제로 쓴 설정 (덮어쓰기 반영)
    train_result.json    best/last 가중치 경로, 학습 시간
    metrics_val.json     통합 평가 결과 (COCO mAP + 점유율 지표)
    <out_dir>/experiments.csv   ← 한 행 추가
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

from carseg.coco_eval import evaluate
from carseg.config import ExperimentConfig
from carseg.data import CarSegData
from carseg.experiment_log import log_experiment
from carseg.models import build_predictor, build_trainer
from carseg.utils import fmt_duration, save_json


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="실험 설정 YAML")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="설정 덮어쓰기 (예: train.epochs=3 data.root=/content/car_seg_split_upload)")
    p.add_argument("--no-eval", action="store_true", help="학습만 하고 채점·기록은 건너뜀")
    p.add_argument("--test", action="store_true", help="val 채점 후 test 도 채점 (최종 모델에만 권장)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = ExperimentConfig.load(args.config, overrides=args.set)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()
    data = CarSegData(cfg.data.root)
    print(f"[train] 실험 {cfg.name}  ({cfg.model.family} / {cfg.model.weights})")
    print(f"[train] 데이터 {data.root}  → 산출물 {cfg.run_dir}")

    result = build_trainer(cfg).train()
    save_json(dataclasses.asdict(result), cfg.run_dir / "train_result.json")
    print(f"[train] 학습 완료 {fmt_duration(result.train_seconds)} — best: {result.best_weights}")
    if args.no_eval:
        return 0

    predictor = build_predictor(cfg, result.best_weights, label_mode="finetuned")
    splits = [cfg.data.val_split] + ([cfg.data.test_split] if args.test else [])
    for split in splits:
        metrics = evaluate(predictor, data, split, limit=cfg.data.eval_limit, out_dir=cfg.run_dir)
        log_experiment(cfg, metrics, stage="finetuned", eval_weights=result.best_weights,
                       train_seconds=result.train_seconds, epochs_done=result.epochs_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
