#!/usr/bin/env python3
"""SAM3 가중치(sam3.pt) 다운로드.

Meta 의 SAM3 는 Hugging Face 'facebook/sam3' 게이트 저장소에 있다.
  1. https://huggingface.co/facebook/sam3 에서 라이선스 동의
  2. https://huggingface.co/settings/tokens 에서 읽기 토큰 발급
  3. python scripts/download_sam3.py --token hf_xxx   (또는 HF_TOKEN 환경변수)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--repo", default="facebook/sam3")
    ap.add_argument("--filename", default="sam3.pt")
    ap.add_argument("--out", default=str(ROOT / "models/weights/sam3.pt"))
    a = ap.parse_args()
    if not a.token:
        raise SystemExit("HF 토큰이 필요합니다: --token 또는 HF_TOKEN")
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=a.repo, filename=a.filename, token=a.token)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, out)
    print(f"저장: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
