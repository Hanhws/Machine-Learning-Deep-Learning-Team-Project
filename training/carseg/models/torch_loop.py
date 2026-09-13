"""PyTorch 모델(Mask R-CNN · Mask2Former)이 공유하는 학습 루프.

모델마다 다른 부분만 콜백으로 받는다.
    step_fn(model, batch, device) -> dict[str, Tensor]   ('loss' 키 필수, None 이면 그 배치 건너뜀)
    make_predictor(model)         -> Predictor           (학습 중 val 채점용)

기능: AMP · 그래디언트 누적 · 워밍업+코사인 LR · last/best 체크포인트 · 이어하기 · 조기 종료 · history.csv
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import ExperimentConfig
from ..data import CarSegData
from ..utils import Timer, fmt_duration, resolve_device, save_json
from .base import Predictor, TrainResult


def build_optimizer(cfg: ExperimentConfig, params):
    import torch

    t = cfg.train
    name = (t.optimizer or "auto").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=t.lr0, momentum=t.momentum, weight_decay=t.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=t.lr0, weight_decay=t.weight_decay)
    return torch.optim.AdamW(params, lr=t.lr0, weight_decay=t.weight_decay)  # auto / adamw


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int, lrf: float):
    import torch

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class TorchLoop:
    def __init__(self, cfg: ExperimentConfig, model, train_loader,
                 step_fn: Callable[[Any, Any, str], Optional[Dict[str, Any]]],
                 make_predictor: Callable[[Any], Predictor],
                 checkpoint_meta: Dict[str, Any]):
        self.cfg = cfg
        self.model = model
        self.loader = train_loader
        self.step_fn = step_fn
        self.make_predictor = make_predictor
        self.meta = checkpoint_meta
        self.device = resolve_device(cfg.train.device)
        self.ckpt_dir = cfg.run_dir / "weights"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 체크포인트
    def _save(self, path: Path, epoch: int, best: float, optimizer, scheduler, scaler) -> None:
        import torch

        torch.save({
            **self.meta,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_metric": best,
            "config": self.cfg.to_dict(),
        }, path)

    # ------------------------------------------------------------------ 실행
    def run(self) -> TrainResult:
        import torch

        cfg, t = self.cfg, self.cfg.train
        model = self.model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(cfg, params)
        steps_per_epoch = math.ceil(len(self.loader) / max(t.accumulate, 1))
        scheduler = warmup_cosine(optimizer, total_steps=steps_per_epoch * t.epochs,
                                  warmup_steps=int(steps_per_epoch * t.warmup_epochs), lrf=t.lrf)
        use_amp = bool(t.amp) and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        start_epoch, best_metric, bad_evals = 0, -1.0, 0
        if t.resume:
            ckpt = torch.load(t.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch, best_metric = ckpt["epoch"] + 1, ckpt.get("best_metric", -1.0)
            print(f"[train] 이어하기: epoch {start_epoch} 부터 (best={best_metric:.4f})")

        data = CarSegData(cfg.data.root)
        history: List[Dict[str, Any]] = []
        last_path, best_path = self.ckpt_dir / "last.pt", self.ckpt_dir / "best.pt"
        grad_clip = float(cfg.model.extra.get("grad_clip", 1.0))

        with Timer() as timer:
            for epoch in range(start_epoch, t.epochs):
                model.train()
                sums: Dict[str, float] = {}
                n_steps, t0 = 0, time.time()
                optimizer.zero_grad(set_to_none=True)
                for i, batch in enumerate(self.loader):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        losses = self.step_fn(model, batch, self.device)
                    if losses is None:
                        continue
                    loss = losses["loss"] / max(t.accumulate, 1)
                    if not torch.isfinite(loss):
                        print(f"[train] 경고: epoch {epoch} step {i} loss={loss.item()} — 배치 건너뜀")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    (scaler.scale(loss) if scaler else loss).backward()

                    if (i + 1) % max(t.accumulate, 1) == 0 or (i + 1) == len(self.loader):
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    for k, v in losses.items():
                        sums[k] = sums.get(k, 0.0) + float(v.detach())
                    n_steps += 1
                    if i % 50 == 0:
                        print(f"  epoch {epoch + 1}/{t.epochs}  step {i}/{len(self.loader)}  "
                              f"loss {float(losses['loss']):.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")

                row: Dict[str, Any] = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                                       "epoch_time": round(time.time() - t0, 1),
                                       **{k: round(v / max(n_steps, 1), 5) for k, v in sums.items()}}

                # ---- val 채점 (eval_every 마다 + 마지막 epoch)
                is_last = epoch + 1 == t.epochs
                every = cfg.eval.eval_every
                if (every and (epoch + 1) % every == 0) or is_last:
                    from ..coco_eval import evaluate

                    model.eval()
                    m = evaluate(self.make_predictor(model), data, cfg.data.val_split,
                                 limit=cfg.eval.train_eval_limit or cfg.data.eval_limit, progress=False)
                    row.update({k: m[k] for k in ("mask_mAP", "mask_mAP50", "box_mAP") if k in m})
                    if m["mask_mAP"] > best_metric:
                        best_metric, bad_evals = m["mask_mAP"], 0
                        self._save(best_path, epoch, best_metric, optimizer, scheduler, scaler)
                        print(f"[train] ★ best 갱신 mask_mAP={best_metric:.4f} → {best_path}")
                    else:
                        bad_evals += 1

                self._save(last_path, epoch, best_metric, optimizer, scheduler, scaler)
                history.append(row)
                self._write_history(history)
                print(f"[train] epoch {epoch + 1} 완료 {row}  (누적 {fmt_duration(time.time() - timer.start)})")

                if t.patience and every and bad_evals * every >= t.patience:
                    print(f"[train] 조기 종료: {t.patience} epoch 동안 개선 없음")
                    break

        if not best_path.exists():  # 채점 없이 끝난 경우
            best_path = last_path
        return TrainResult(best_weights=str(best_path), last_weights=str(last_path),
                           run_dir=str(cfg.run_dir), train_seconds=timer.seconds,
                           epochs_done=len(history) + start_epoch, history=history,
                           extra={"best_val_mask_mAP": best_metric})

    def _write_history(self, history: List[Dict[str, Any]]) -> None:
        keys: List[str] = []
        for row in history:
            keys += [k for k in row if k not in keys]
        path = self.cfg.run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        save_json(history, self.cfg.run_dir / "history.json")
