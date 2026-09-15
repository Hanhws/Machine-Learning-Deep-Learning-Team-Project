"""도로·차선·혼잡도 파이프라인 공통 설정과 유틸."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
IMAGE_ROOT = PROJECT / "images"
OUT = ROOT / "outputs"

# 작업 해상도: 긴 변을 이 크기로 줄여서 계산한다 (1920 → 960).
WORK_LONG_SIDE = 960

# COCO 클래스 id → 이름 (차량만 사용)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_LANE_TOKEN = re.compile(r"^(OW|TW)(\d+)$")


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
