"""uvicorn 실행 스크립트: python backend/run.py [--reload]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 미지원 연산은 CPU 로

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload="--reload" in sys.argv, app_dir=str(Path(__file__).parent))
