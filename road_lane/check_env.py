"""0) 지금 환경이 기준 결과(outputs/)를 만든 환경과 같은지 확인한다. 다르면 경고만 하고 계속한다.

기준 환경: macOS · Apple M5 (mps) · Python 3.13.9 · requirements.txt 의 버전
장치(mps/cuda/cpu)가 다르면 같은 모델도 소수점 계산이 미세하게 달라 결과가 조금 달라진다.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata

from common import IMAGE_ROOT, OUT, ROOT, YOLO_SHA256, list_clips, pick_device

REF_PYTHON = "3.13.9"
REF_DEVICE = "mps"
REF_FRAMES, REF_CLIPS = 20691, 226


def main() -> None:
    warn: list[str] = []
    py = platform.python_version()
    print(f"{'Python':<16}{py:<16}기준 {REF_PYTHON}")
    if py.rsplit(".", 1)[0] != REF_PYTHON.rsplit(".", 1)[0]:
        warn.append(f"Python {py} (기준 {REF_PYTHON})")

    for line in (ROOT / "requirements.txt").read_text().splitlines():
        spec = line.split("#")[0].strip()
        if "==" not in spec:
            continue
        name, want = spec.split("==")
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "없음"
        same = have.split("+")[0] == want  # torch CUDA 빌드는 2.14.0+cu128 처럼 붙는다
        print(f"{name:<16}{have:<16}기준 {want}{'' if same else '  ← 다름'}")
        if not same:
            warn.append(f"{name} {have} (기준 {want})")

    device = pick_device()
    print(f"{'장치':<15}{device:<16}기준 {REF_DEVICE}")
    if device != REF_DEVICE:
        warn.append(f"장치 {device} (기준 {REF_DEVICE}): 결과가 조금 달라질 수 있음")

    clips = list_clips()
    frames = sum(len(c.frames) for c in clips)
    print(f"{'이미지':<14}{f'{frames}장 {len(clips)}클립':<16}기준 {REF_FRAMES}장 {REF_CLIPS}클립  ({IMAGE_ROOT})")
    if frames == 0:
        sys.exit(f"이미지가 없습니다. Google Drive 의 images-*.zip 을 {IMAGE_ROOT.parent} 에 풀어 주세요.")
    if (frames, len(clips)) != (REF_FRAMES, REF_CLIPS):
        warn.append("이미지 수가 기준과 다름 (zip 6개를 모두 풀었는지 확인)")

    w = ROOT / "weights" / "yolo26m-seg.pt"
    if w.exists():
        same = hashlib.sha256(w.read_bytes()).hexdigest() == YOLO_SHA256
        print(f"{'yolo26m-seg.pt':<16}{'기준과 같음' if same else '← 기준과 다른 파일'}")
        if not same:
            warn.append("yolo26m-seg.pt 가 기준과 다른 파일")
    else:
        print(f"{'yolo26m-seg.pt':<16}아직 없음 → s1 에서 자동으로 받음 (ultralytics assets v8.4.0)")
    print(f"{'결과 폴더':<14}{OUT}")

    if warn:
        print("\n경고 (계속 진행):\n  - " + "\n  - ".join(warn))
    else:
        print("\n기준 환경과 같습니다.")


if __name__ == "__main__":
    main()
