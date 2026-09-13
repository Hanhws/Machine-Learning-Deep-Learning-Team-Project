"""공통 유틸 — 장치 선택, 시드 고정, 시간 포맷, JSON 저장."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


def resolve_device(device: str = "auto") -> str:
    """'auto' 면 cuda → mps → cpu 순으로 고른다."""
    if device and device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """재현성을 위한 시드 고정 (완전한 결정성은 보장하지 않는다)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    return path


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Timer:
    """with 문으로 감싼 구간의 실행 시간(초)을 잰다."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.time() - self.start

    @property
    def pretty(self) -> str:
        return fmt_duration(getattr(self, "seconds", 0.0))
