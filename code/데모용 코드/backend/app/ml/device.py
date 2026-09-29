"""추론 디바이스 선택 (auto → mps / cuda / cpu)."""

from __future__ import annotations

import torch


def resolve_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
