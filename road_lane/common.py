"""도로·차선·혼잡도 파이프라인 공통 설정과 유틸."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
IMAGE_ROOT = PROJECT / "images"
# 결과 폴더. 실험은 ROAD_LANE_OUT=runs/내실험 처럼 따로 두면 기준 결과(outputs/)를 덮어쓰지 않는다.
_OUT = Path(os.environ.get("ROAD_LANE_OUT", "outputs"))
OUT = _OUT if _OUT.is_absolute() else ROOT / _OUT

# 작업 해상도: 긴 변을 이 크기로 줄여서 계산한다 (1920 → 960).
WORK_LONG_SIDE = 960

# 차량으로 쓰는 클래스 이름. 모델마다 클래스 번호가 달라서(COCO 2=car, 팀 파인튜닝 모델 0=car) 이름으로 고른다.
VEHICLE_NAMES = ("car", "motorcycle", "bus", "truck")

# 기준 결과(outputs/)를 만든 모델 버전. 같은 파일·같은 커밋을 받아야 결과가 비슷하게 나온다.
YOLO_SHA256 = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"  # yolo26m-seg.pt, ultralytics assets v8.4.0
M2F_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
M2F_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SEED = 0

_LANE_TOKEN = re.compile(r"^(OW|TW)(\d+)$")


def pick_device() -> str:
    """Apple 칩(mps) → NVIDIA(cuda) → cpu. 기준 결과는 mps 로 만들었고, 장치가 다르면 소수점 계산이 미세하게 달라진다."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int = SEED) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class Clip:
    clip_id: str
    frames: tuple[Path, ...]

    @property
    def tokens(self) -> list[str]:
        return self.clip_id.split("_")

    @property
    def camera_id(self) -> str:
        t = self.tokens
        return f"{t[0]}_{t[1]}"

    @property
    def lane_token(self) -> str:
        return self.tokens[8]

    @property
    def two_way(self) -> bool:
        return self.lane_token.startswith("TW")

    @property
    def lanes_per_direction(self) -> int:
        m = _LANE_TOKEN.match(self.lane_token)
        return int(m.group(2)) if m else -1

    @property
    def weather(self) -> str:
        return self.tokens[9]

    @property
    def hour(self) -> int:
        return int(self.tokens[3][:2])


def list_clips(image_root: Path = IMAGE_ROOT) -> list[Clip]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val", "test"):
        d = image_root / split
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            groups[p.stem.rsplit("_", 1)[0]].append(p)
    clips = []
    for cid, paths in groups.items():
        paths.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        clips.append(Clip(cid, tuple(paths)))
    clips.sort(key=lambda c: c.clip_id)
    return clips


def frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def work_scale(width: int, height: int) -> float:
    return WORK_LONG_SIDE / max(width, height)
