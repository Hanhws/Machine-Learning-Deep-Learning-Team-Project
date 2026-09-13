"""실험 기록표 — 평가 한 번 = CSV 한 행.

첨부한 AlphaDent 성능기록표처럼 [무엇을 바꿨나(설정) | 결과(지표)] 를 한 줄에 남긴다.
    <out_dir>/experiments.csv   누적 기록 (엑셀·구글시트로 열기)
    <out_dir>/experiments.md    make_report.py 가 만드는 보기 좋은 표

열이 나중에 늘어나도(새 지표·설정) 기존 행은 빈칸으로 두고 합쳐서 다시 쓴다.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import CLASS_NAMES
from .config import ExperimentConfig

# 기록표 열 순서: (CSV 열 이름, 값을 꺼낼 키)  — 키가 'cfg:' 로 시작하면 설정, 'm:' 이면 지표
COLUMNS: List[tuple] = [
    ("no", "auto:no"), ("date", "auto:date"), ("group", "cfg:group"), ("exp", "cfg:name"),
    ("stage", "auto:stage"), ("split", "m:split"),
    # --- 모델
    ("family", "cfg:model.family"), ("weights", "cfg:model.weights"), ("backbone", "cfg:model.backbone"),
    ("arch_mods", "cfg:model.arch_mods"), ("model_extra", "cfg:model.extra"),
    # --- 결과 (앞쪽에 두어 표에서 바로 보이게)
    ("mask_mAP", "m:mask_mAP"), ("mask_mAP50", "m:mask_mAP50"), ("mask_mAP75", "m:mask_mAP75"),
    ("mask_mAP_small", "m:mask_mAP_small"), ("mask_mAP_medium", "m:mask_mAP_medium"),
    ("box_mAP", "m:box_mAP"), ("box_mAP50", "m:box_mAP50"),
    *[(f"AP_{n}", f"m:mask_AP_{n}") for n in CLASS_NAMES],
    ("vehicle_IoU", "m:vehicle_IoU"), ("area_ratio_MAE_pp", "m:area_ratio_MAE_pp"),
    ("count_MAE", "m:count_MAE"), ("fps", "m:fps"),
    # --- 학습 이전
    ("train_limit", "cfg:data.train_limit"),
    # --- 학습 하이퍼파라미터
    ("epochs", "cfg:train.epochs"), ("epochs_done", "auto:epochs_done"), ("imgsz", "cfg:train.imgsz"),
    ("batch", "cfg:train.batch"), ("accumulate", "cfg:train.accumulate"), ("lr0", "cfg:train.lr0"),
    ("optimizer", "cfg:train.optimizer"), ("weight_decay", "cfg:train.weight_decay"),
    ("warmup", "cfg:train.warmup_epochs"), ("freeze", "cfg:train.freeze"), ("amp", "cfg:train.amp"),
    ("hflip", "cfg:train.hflip"), ("vflip", "cfg:train.vflip"), ("color_jitter", "cfg:train.color_jitter"),
    ("mosaic", "cfg:train.mosaic"), ("mixup", "cfg:train.mixup"), ("copy_paste", "cfg:train.copy_paste"),
    ("degrees", "cfg:train.degrees"), ("translate", "cfg:train.translate"), ("scale", "cfg:train.scale"),
    ("box", "cfg:train.box_gain"), ("cls", "cfg:train.cls_gain"), ("dfl", "cfg:train.dfl_gain"),
    # --- 추론
    ("conf", "cfg:eval.conf"), ("iou", "cfg:eval.iou"), ("mask_thr", "cfg:eval.mask_threshold"),
    ("union_conf", "cfg:eval.union_conf"), ("TTA", "cfg:eval.tta"), ("postprocess", "cfg:eval.postprocess"),
    # --- 기타
    ("train_time", "auto:train_time"), ("n_images", "m:n_images"), ("eval_weights", "auto:eval_weights"),
    ("notes", "cfg:notes"),
]


def log_experiment(cfg: ExperimentConfig, metrics: Dict[str, Any], *, stage: str,
                   eval_weights: str = "", train_seconds: Optional[float] = None,
                   epochs_done: Optional[int] = None, csv_path: Optional[str | Path] = None) -> Path:
    """stage: 'pretrained'(사전학습 그대로) | 'finetuned'(학습 후)"""
    from .utils import fmt_duration

    csv_path = Path(csv_path) if csv_path else Path(cfg.out_dir) / "experiments.csv"
    rows, header = read_rows(csv_path)
    flat = cfg.flat()
    auto = {
        "no": len(rows) + 1,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stage": stage,
        "epochs_done": "" if epochs_done is None else epochs_done,
        "train_time": fmt_duration(train_seconds) if train_seconds else "",
        "eval_weights": eval_weights,
    }
    row: Dict[str, Any] = {}
    for col, key in COLUMNS:
        src, name = key.split(":", 1)
        if src == "auto":
            row[col] = auto.get(name, "")
        elif src == "cfg":
            row[col] = flat.get(name, "")
        else:
            row[col] = metrics.get(name, "")
    if stage == "pretrained":  # 학습을 안 했으니 학습 설정 칸은 비운다
        for col, key in COLUMNS:
            if key.startswith("cfg:train.") or key == "cfg:data.train_limit":
                row[col] = ""
    # 기록표에 없는 추가 지표도 버리지 않고 뒤에 붙인다
    for k, v in metrics.items():
        if k not in {key.split(":", 1)[1] for _, key in COLUMNS} and not isinstance(v, (dict, list)):
            row[f"m.{k}"] = v

    rows.append(row)
    columns = [c for c, _ in COLUMNS]
    for r in rows:
        columns += [k for k in r if k not in columns]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀에서 한글 안 깨짐
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[log] 기록표에 추가: {csv_path} (#{auto['no']} {cfg.name} / {stage} / {metrics.get('split')})")
    return csv_path


def read_rows(csv_path: str | Path) -> tuple:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def render_markdown(csv_path: str | Path, *, split: str = "val", sort_by: str = "mask_mAP") -> str:
    """기록표를 그룹별 마크다운 표로. 각 그룹 최고 성능에 ★."""
    rows, _ = read_rows(csv_path)
    rows = [r for r in rows if not split or r.get("split") == split]
    if not rows:
        return f"(split={split} 기록 없음)\n"

    show = ["no", "exp", "stage", "family", "weights", "backbone", "arch_mods", "imgsz", "batch",
            "epochs_done", "lr0", "mask_mAP", "mask_mAP50", "mask_mAP_small", "box_mAP",
            *[f"AP_{n}" for n in CLASS_NAMES], "vehicle_IoU", "area_ratio_MAE_pp", "fps", "conf", "notes"]

    def num(r):
        try:
            return float(r.get(sort_by) or -1)
        except ValueError:
            return -1.0

    best_overall = max(rows, key=num)
    lines = [f"# 차량 세그멘테이션 성능기록표 (split={split}, 정렬={sort_by})", "",
             f"**최고 성능**: #{best_overall['no']} `{best_overall['exp']}` — "
             f"{sort_by} = {best_overall.get(sort_by)}", ""]
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("group") or "-", []).append(r)
    for group, items in groups.items():
        items.sort(key=num, reverse=True)
        lines += [f"## {group}", "", "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
        for i, r in enumerate(items):
            cells = [str(r.get(c, "")) for c in show]
            if i == 0:
                cells[show.index(sort_by)] = f"★ {cells[show.index(sort_by)]}"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
