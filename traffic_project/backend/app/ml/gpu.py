"""GPU(MPS/CUDA) 작업 직렬화 — 단일 전용 스레드.

macOS MPS 는 여러 스레드에서 Metal 커맨드 버퍼를 만지면 프로세스가 통째로 죽는다
(`failed assertion ... CommandBuffer`, SIGSEGV in arange_range_fill_mps 등). 락으로 추론만 감싸도
결과 텐서를 CPU 로 복사하는 `.cpu()` 가 다른 스레드에서 실행되면 같은 문제가 난다.
그래서 **모델 로딩·추론·텐서→numpy 변환 전부**를 하나의 전용 스레드에서만 실행한다.
호출 측은 `run_gpu(fn, *args)` 로 넘기고 결과(numpy 등 CPU 객체)만 돌려받는다.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

import torch

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_gpu_thread_id: int | None = None
INFER_LOCK = threading.RLock()  # 하위 호환 (직접 쓰지 않아도 됨)


def synchronize(device: str) -> None:
    try:
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _wrap(fn: Callable[..., T], device: str, args, kwargs) -> T:
    global _gpu_thread_id
    _gpu_thread_id = threading.get_ident()
    try:
        return fn(*args, **kwargs)
    finally:
        synchronize(device)


def run_gpu(fn: Callable[..., T], *args, device: str = "mps", **kwargs) -> T:
    """fn 을 GPU 전용 스레드에서 실행하고 결과를 기다린다. 이미 그 스레드 안이면 바로 실행(재진입)."""
    if threading.get_ident() == _gpu_thread_id:
        return fn(*args, **kwargs)
    return _executor.submit(_wrap, fn, device, args, kwargs).result()
