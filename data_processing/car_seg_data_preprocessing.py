#!/usr/bin/env python3
"""AI-Hub 「교통문제 해결을 위한 CCTV 교통 데이터(고속도로)」 폴리곤 세그멘테이션 데이터를
한 곳으로 모으고, 여러 모델이 바로 쓸 수 있는 라벨 형식(YOLO · COCO · 마스크 · LabelMe)으로 만드는 스크립트.

원본 구조 (AI-Hub 다운로드 그대로)
    traffic_data2/
      100.교통문제_…(고속도로)/01.데이터/2.Validation/라벨링데이터/폴리곤세그멘테이션/3.대전충남.zip.part0
      100.교통문제_…(고속도로) 6/01.데이터/2.Validation/원천데이터/폴리곤세그멘테이션/3.대전충남.zip.part0
                                                                                   3.대전충남.zip.part1073741824
                                                                                   …
    - AI-Hub 에서 받은 download (N).tar 를 풀지 않은 상태도 그대로 읽는다 (tar 안의 조각 위치를 바로 읽음)
      ~/Downloads/download (5).tar  →  100.교통문제_…/01.데이터/…/원천데이터/폴리곤세그멘테이션/3.대전충남.zip.part0 …
    - 분할 zip 조각의 접미사 숫자 = 그 조각의 시작 바이트 오프셋
    - 라벨: CVAT 1.1 XML (클립 1개 = XML 1개, <image> 안에 car/bus/truck <polygon>)
    - 이미지: <클립명>_NNN.png (일부 클립은 '<클립명> NNN.png' 처럼 공백 구분)

처리 과정
    1. 모든 .zip.partN 조각을 zip 단위로 묶고 오프셋 연속성(누락 조각)을 검사
    2. 조각을 디스크에 합치지 않고 가상으로 이어 붙여 바로 읽음 (병합본 60GB를 따로 만들지 않음)
    3. 라벨 XML 파싱 → 정규화한 이미지 파일명 기준으로 원천 이미지와 매칭
    4. 매칭된 이미지만 images/ 에 풀고, 폴리곤을 YOLO seg 라벨로 labels/ 에 저장
    5. YOLO 라벨에서 다른 형식(COCO JSON, semantic 마스크 PNG, LabelMe JSON)을 생성

결과 구조
    car_seg_dataset/
      images/<클립명>_NNN.png
      labels/<클립명>_NNN.txt           [yolo]    "<class_id> x1 y1 x2 y2 …" (0~1 정규화, class 0부터)
      annotations/instances_all.json   [coco]    COCO instance segmentation (픽셀 좌표, category 1부터)
      masks/<클립명>_NNN.png            [mask]    semantic 마스크 (0=배경 1=car 2=bus 3=truck)
      labelme/<클립명>_NNN.json         [labelme] LabelMe 폴리곤 (라벨링 툴에서 열어 수정할 때)
      classes.json                     형식별 클래스 번호표
      manifest.csv                     이미지 1장 = 1행 (파일명 메타데이터 + GT 개수)
      preprocess_report.json           요약 통계, 매칭 실패·제외 목록

    형식별 쓰는 모델
      yolo    Ultralytics YOLO (v8 / 11 / 26 …)
      coco    Mask R-CNN · Detectron2 · MMDetection · Mask2Former(instance) · RT-DETR 등
      mask    SegFormer · DeepLabV3 · U-Net · Mask2Former(semantic) 등
      labelme LabelMe / X-AnyLabeling 등 라벨링 툴

    train / val / test 분할은 car_seg_data_split.py 가 모든 형식에 똑같이 적용한다.

사용 예
    python car_seg_data_preprocessing.py --dry-run          # 조각/매칭만 점검 (아무것도 쓰지 않음)
    python car_seg_data_preprocessing.py                    # 전체 실행 (yolo + coco + mask)
    python car_seg_data_preprocessing.py --raw-root ~/Downloads            # 받은 tar 파일들에서 바로
    python car_seg_data_preprocessing.py --raw-root ~/Downloads ./폴더     # tar + 풀어 둔 폴더 함께 (중복은 자동 제거)
    python car_seg_data_preprocessing.py --formats yolo,coco,mask,labelme
    python car_seg_data_preprocessing.py --formats-only     # 압축 해제 없이 기존 결과에 형식만 (다시) 생성
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date as datetime_date
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# GT 클래스 -> YOLO class id (traffic_project/src/prepare_data.py 와 동일한 매핑)
CLASS_TO_ID = {"car": 0, "bus": 1, "truck": 2}

# 라벨 형식 — yolo(labels/)는 다른 형식의 원본이자 분할의 기준이라 항상 만든다
FORMATS = ("yolo", "coco", "mask", "labelme")
DEFAULT_FORMATS = "yolo,coco,mask"
# 마스크 팔레트 (픽셀값 = YOLO class id + 1, 0 = 배경). 색은 EDA 차트와 동일
MASK_PALETTE = {0: (0, 0, 0), 1: (42, 120, 214), 2: (235, 104, 52), 3: (27, 175, 122)}

# 'xxx.zip.part1073741824' -> ('xxx.zip', 1073741824)
PART_RE = re.compile(r"^(?P<zip>.+\.zip)\.part(?P<offset>\d+)$", re.IGNORECASE)

# macOS 가 같은 이름의 다운로드 폴더에 붙이는 ' 2', ' 3' … 접미사
DOWNLOAD_COPY_SUFFIX_RE = re.compile(r" \d+$")

# 파일명 규약 (230개 클립 전부 '_' 기준 11토큰):
#   Suwon_CH01_20200721_1700_TUE_9m_RH_highway_TW5_sunny_FHD_001.png
#   Inje_injetunnel2_20201210_0956_THU_4.8m_NH_highway_OW2_tunnel_FHD 005.png
META_FIELDS = (
    "site",         # 촬영 지역 (Suwon, Inje, …)
    "channel",      # 카메라 채널 / 촬영 지점 (CH01, injetunnel2, …)
    "date",         # YYYYMMDD
    "time",         # HHMM
    "weekday",      # MON ~ SUN
    "cam_height",   # 카메라 설치 높이 (9m, 4.8m, …)
    "hour_type",    # NH / RH
    "road_type",    # highway
    "lane_config",  # TW5, OW2, … (TW=양방향 / OW=일방향 + 차로 수)
    "weather",      # sunny, rainy, fog, snow, tunnel
    "quality",      # FHD / HD
)

# 촬영 시각(07~21시)을 3시간 단위로 묶은 구간 — 층화 분할/EDA 용
TIME_BANDS = ((7, 10, "07-09"), (10, 13, "10-12"), (13, 16, "13-15"), (16, 19, "16-18"), (19, 22, "19-21"))

LANE_DIRECTION = {"TW": "two_way", "OW": "one_way"}

# 촬영 지역 좌표 (위도, 경도) — 일출/일몰 계산용. 목록에 없는 지역은 한국 중부로 계산
SITE_COORDS = {
    "Suwon": (37.26, 127.03), "Gangneung": (37.75, 128.88), "Jeonju": (35.82, 127.15),
    "Buan": (35.73, 126.73), "Busan": (35.18, 129.08), "Chungju": (36.99, 127.93),
    "Cheonan": (36.81, 127.11), "Hongcheon": (37.70, 127.89), "Jincheon": (36.86, 127.44),
    "Inje": (38.07, 128.17),
}
DEFAULT_COORDS = (36.5, 127.8)
TWILIGHT_MIN = 30  # 일출/일몰 ±30분은 '여명·황혼(twilight)'


def nfc(text: str) -> str:
    """macOS 파일명은 한글이 NFD(자모 분리)로 저장돼 있어 비교 전에 NFC로 통일."""
    return unicodedata.normalize("NFC", text)


# ----------------------------------------------------------------------------
# 분할 zip 조각 → 가상 단일 파일
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """zip 조각 하나의 실제 위치. 일반 파일이면 파일 전체, tar 안에 있으면 tar 파일 속 바이트 구간."""

    path: Path          # 조각 파일, 또는 조각이 들어 있는 tar 파일
    start: int          # path 안에서 조각 데이터가 시작하는 위치
    size: int
    in_tar: bool = False


class MultiPartReader(io.RawIOBase):
    """분할 zip 조각들을 하나의 연속된 파일처럼 읽는 read-only 스트림.

    조각 접미사가 곧 시작 오프셋이므로, 현재 위치가 속한 조각을 이분 탐색해서 읽는다.
    조각이 tar 안에 있으면 tar 파일에서 그 조각의 바이트 구간을 바로 읽는다 (tar 도 풀지 않음).
    zipfile 은 seek/tell/read 만 쓰기 때문에 실제로 합친 파일 없이 바로 압축을 풀 수 있다.
    """

    def __init__(self, parts: list[tuple[int, Part]]):
        super().__init__()
        self._offsets = [offset for offset, _ in parts]
        self._parts = [part for _, part in parts]
        self._size = self._offsets[-1] + self._parts[-1].size
        self._fps: dict[Path, io.BufferedReader] = {}
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"잘못된 whence: {whence}")
        if pos < 0:
            raise ValueError("음수 위치로 seek 할 수 없습니다")
        self._pos = pos
        return pos

    def _file(self, path: Path):
        if path not in self._fps:  # 같은 tar 안의 조각들은 파일 핸들 하나를 같이 쓴다
            self._fps[path] = open(path, "rb")
        return self._fps[path]

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self._pos < self._size:
            idx = bisect.bisect_right(self._offsets, self._pos) - 1
            part = self._parts[idx]
            local = self._pos - self._offsets[idx]
            want = min(len(view) - done, part.size - local)
            fp = self._file(part.path)
            fp.seek(part.start + local)
            n = fp.readinto(view[done : done + want])
            if not n:
                break
            done += n
            self._pos += n
        return done

    def close(self) -> None:
        for fp in self._fps.values():
            fp.close()
        self._fps = {}
        super().close()


@dataclass
class Archive:
    """zip 한 개 (조각 여러 개로 나뉘어 있을 수 있음)."""

    key: str                                  # 다운로드 루트 기준 상대경로 (조각을 묶는 키)
    name: str                                 # zip 파일명, 예: '3.대전충남.zip'
    parts: list[tuple[int, Part]] = field(default_factory=list)
    kind: str = ""                            # 'label' | 'image' | 'unknown'
    error: str = ""                           # 조각 누락/손상 등

    @property
    def stem(self) -> str:
        return self.name[: -len(".zip")] if self.name.lower().endswith(".zip") else self.name

    @property
    def total_bytes(self) -> int:
        return sum(part.size for _, part in self.parts)

    @property
    def source(self) -> str:
        n_tar = sum(1 for _, part in self.parts if part.in_tar)
        return "tar" if n_tar == len(self.parts) else "폴더" if n_tar == 0 else "혼합"


def _archive_entry(rel_dirs: list[str], filename: str) -> tuple[str, str, int] | None:
    """zip 조각이면 (묶음 키, zip 파일명, 시작 오프셋), 아니면 None.

    같은 zip의 조각이 '100.교통…', '100.교통… 2' 처럼 다른 다운로드 폴더(또는 tar)에 흩어져 있어도
    최상위 폴더의 ' N' 접미사를 떼고 상대경로로 묶으므로 하나로 합쳐진다.
    """
    m = PART_RE.match(filename)
    if m:
        zip_name, offset = m["zip"], int(m["offset"])
    elif filename.lower().endswith(".zip"):
        zip_name, offset = filename, 0
    else:
        return None
    dirs = list(rel_dirs)
    if dirs:
        dirs[0] = DOWNLOAD_COPY_SUFFIX_RE.sub("", dirs[0])
    return nfc("/".join([*dirs, zip_name])), nfc(zip_name), offset


def scan_tar(path: Path, add) -> str:
    """AI-Hub 다운로드 tar(download (N).tar) 안의 zip 조각을 등록. tar 는 풀지 않고 위치만 기록한다."""
    try:
        with tarfile.open(path, "r:") as tf:  # 압축 안 된 tar 여야 원하는 위치를 바로 읽을 수 있다
            n = 0
            for member in tf:
                if not member.isfile():
                    continue
                names = [p for p in nfc(member.name).split("/") if p not in ("", ".")]
                if not names or names[-1].startswith(".") or "__MACOSX" in names:  # macOS 의 ._ 메타데이터 파일
                    continue
                entry = _archive_entry(names[:-1], names[-1])
                if entry:
                    add(*entry, Part(path, member.offset_data, member.size, in_tar=True))
                    n += 1
        return f"{path.name}: zip 조각 {n}개"
    except tarfile.ReadError:
        return f"{path.name}: 읽을 수 없음 — 압축된 tar(.tar.gz 등)이면 먼저 풀어주세요"


def discover_archives(raw_roots: list[Path], skip_dirs: list[Path]) -> tuple[list[Archive], list[str]]:
    """raw_roots 아래의 .zip / .zip.partN 과 .tar 안의 조각을 모두 찾아 zip 단위로 묶는다.

    같은 조각이 폴더와 tar 양쪽에 있어도(tar 를 풀어 둔 경우) 하나만 사용한다.
    반환: (zip 목록, tar 별 요약 메시지)
    """
    skip = {path.resolve() for path in skip_dirs}
    archives: dict[str, Archive] = {}
    tar_notes: list[str] = []

    def add(key: str, zip_name: str, offset: int, part: Part) -> None:
        arc = archives.setdefault(key, Archive(key=key, name=zip_name))
        dup = next((p for off, p in arc.parts if off == offset), None)
        if dup is not None:
            if dup.size != part.size:
                arc.error = f"같은 오프셋({offset})의 조각이 크기가 다르게 중복됨: {dup.path} / {part.path}"
            return  # 같은 조각이 두 번 있으면(재다운로드, tar + 풀어 둔 폴더) 하나만 사용
        arc.parts.append((offset, part))

    for root in raw_roots:
        if root.is_file() and root.suffix.lower() == ".tar":  # tar 파일을 직접 지정한 경우
            tar_notes.append(scan_tar(root, add))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not d.startswith("car_seg_") and (here / d).resolve() not in skip
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = here / filename
                if filename.lower().endswith(".tar"):
                    tar_notes.append(scan_tar(path, add))
                    continue
                entry = _archive_entry(list(path.relative_to(root).parts[:-1]), filename)
                if entry:
                    add(*entry, Part(path, 0, path.stat().st_size))

    for arc in archives.values():
        arc.parts.sort(key=lambda op: op[0])
        if not arc.error:
            expected = 0
            for offset, part in arc.parts:
                if offset != expected:
                    arc.error = f"조각 누락: 오프셋 {expected} 조각이 없음 (다음 조각은 {offset})"
                    break
                expected = offset + part.size

    return sorted(archives.values(), key=lambda a: a.key), tar_notes


@contextmanager
def open_archive(arc: Archive):
    reader = io.BufferedReader(MultiPartReader(arc.parts), buffer_size=1 << 20)
    try:
        with zipfile.ZipFile(reader) as zf:
            yield zf
    finally:
        reader.close()


def member_name(info: zipfile.ZipInfo) -> str:
    """zip 내부 파일명 복원. UTF-8 플래그가 없으면 zipfile 이 cp437 로 읽은 것이라 cp949 로 되돌린다."""
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return nfc(name.replace("\\", "/"))


def is_junk_member(name: str) -> bool:
    return "__MACOSX/" in name or Path(name).name.startswith(".")


# ----------------------------------------------------------------------------
# 파일명 / XML 파싱
# ----------------------------------------------------------------------------


def normalize_stem(name: str) -> str:
    """'…/…_FHD 005.png' → '…_FHD_005'. XML 표기가 클립마다 달라 공백을 '_'로 통일."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return re.sub(r"\s+", "_", stem.strip())


def match_key(name: str) -> str:
    return normalize_stem(name).lower()


def time_band_of(hhmm: str) -> str:
    try:
        hour = int(hhmm[:2])
    except ValueError:
        return ""
    for lo, hi, band in TIME_BANDS:
        if lo <= hour < hi:
            return band
    return "other"


def sun_times(yyyymmdd: str, site: str = "") -> tuple[float, float]:
    """(일출, 일몰) 을 한국 시각 자정 기준 '분' 으로 반환 (NOAA 근사식, 오차 수 분)."""
    year, month, day = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    lat, lon = SITE_COORDS.get(site, DEFAULT_COORDS)
    doy = (datetime_date(year, month, day) - datetime_date(year, 1, 1)).days + 1
    g = 2 * math.pi / 365 * (doy - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(math.radians(lat)) * math.cos(decl))
              - math.tan(math.radians(lat)) * math.tan(decl))
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    noon = 720 - 4 * lon - eqtime + 9 * 60  # UTC → KST
    return noon - 4 * ha, noon + 4 * ha


def day_night_of(yyyymmdd: str, hhmm: str, site: str = "") -> tuple[str, int | str]:
    """촬영 날짜·시각·지역으로 ('day' | 'twilight' | 'night', 일몰 기준 분) 계산.

    시각만으로는 부족하다 — 같은 18시라도 7월(일몰 ≈19:45)은 주간, 10월(≈17:50)은 이미 어둡다.
    """
    try:
        t = int(hhmm[:2]) * 60 + int(hhmm[2:4])
        sunrise, sunset = sun_times(yyyymmdd, site)
    except (ValueError, IndexError):
        return "", ""
    if sunrise + TWILIGHT_MIN <= t < sunset - TWILIGHT_MIN:
        phase = "day"
    elif abs(t - sunrise) <= TWILIGHT_MIN or abs(t - sunset) <= TWILIGHT_MIN:
        phase = "twilight"
    else:
        phase = "night"
    return phase, round(t - sunset)


def parse_clip_name(clip: str) -> dict:
    """클립명(11토큰)을 메타데이터 dict 로. 규약에 맞지 않으면 meta_parsed=False."""
    tokens = clip.split("_")
    if len(tokens) != len(META_FIELDS):
        return {"meta_parsed": False}

    meta: dict = {"meta_parsed": True, **dict(zip(META_FIELDS, tokens))}
    meta["time_band"] = time_band_of(meta["time"])
    try:
        meta["cam_height_m"] = float(meta["cam_height"].rstrip("m"))
    except ValueError:
        meta["cam_height_m"] = ""
    lane = meta["lane_config"]
    meta["lane_direction"] = LANE_DIRECTION.get(lane[:2], "")
    meta["lane_count"] = int(lane[2:]) if lane[2:].isdigit() else ""
    meta["day_night"], meta["min_from_sunset"] = day_night_of(meta["date"], meta["time"], meta["site"])
    return meta


@dataclass
class FrameLabel:
    """XML <image> 한 개의 GT."""

    clip: str
    xml_member: str
    label_archive: str
    image_name: str
    width: int
    height: int
    polygons: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)


def parse_cvat_xml(fp, clip: str, xml_member: str, label_archive: str) -> list[FrameLabel]:
    root = ET.parse(fp).getroot()
    frames = []
    for img_el in root.iter("image"):
        name = img_el.get("name")
        if not name:
            continue
        frame = FrameLabel(
            clip=clip,
            xml_member=xml_member,
            label_archive=label_archive,
            image_name=name,
            width=int(float(img_el.get("width", 0) or 0)),
            height=int(float(img_el.get("height", 0) or 0)),
        )
        for poly_el in img_el.iter("polygon"):
            pts = []
            for xy in poly_el.get("points", "").split(";"):
                if "," in xy:
                    x, y = xy.split(",")[:2]
                    pts.append((float(x), float(y)))
            frame.polygons.append((poly_el.get("label", "unknown"), pts))
        frames.append(frame)
    return frames


# ----------------------------------------------------------------------------
# YOLO 라벨 변환
# ----------------------------------------------------------------------------


def polygon_area(pts: list[tuple[float, float]]) -> float:
    """신발끈 공식 (픽셀²)."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clean_polygon(pts, width: int, height: int, min_area: float):
    """이미지 경계로 자르고, 연속 중복점/닫힘점 제거. 점 3개 미만 또는 면적 미달이면 None."""
    out: list[tuple[float, float]] = []
    for x, y in pts:
        p = (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3 or polygon_area(out) < min_area:
        return None
    return out


def to_yolo_lines(frame: FrameLabel, width: int, height: int, min_area: float) -> tuple[list[str], Counter, Counter]:
    """(YOLO 라벨 줄, 클래스별 개수, 제외 사유별 개수)."""
    lines: list[str] = []
    counts: Counter = Counter()
    dropped: Counter = Counter()
    for label, pts in frame.polygons:
        cls_id = CLASS_TO_ID.get(label)
        if cls_id is None:
            dropped[f"unknown_class:{label}"] += 1
            continue
        poly = clean_polygon(pts, width, height, min_area)
        if poly is None:
            dropped["degenerate_polygon"] += 1
            continue
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in poly)
        lines.append(f"{cls_id} {coords}")
        counts[label] += 1
    return lines, counts, dropped


# ----------------------------------------------------------------------------
# 다른 라벨 형식 (YOLO 라벨 → COCO / 마스크 / LabelMe)
# ----------------------------------------------------------------------------


def parse_formats(text: str) -> list[str]:
    formats = [f.strip().lower() for f in text.split(",") if f.strip()]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"알 수 없는 형식: {unknown} (가능: {', '.join(FORMATS)})")
    return ["yolo"] + [f for f in FORMATS[1:] if f in formats]  # yolo 는 항상 포함


def read_yolo_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, list[tuple[float, float]]]]:
    """YOLO seg 라벨 → [(class_id, [(x, y) 픽셀 좌표 …])]."""
    polys = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 7 or len(t) % 2 == 0:  # class + (x, y) 3쌍 이상
            continue
        v = [float(x) for x in t[1:]]
        polys.append((int(t[0]), [(v[i] * width, v[i + 1] * height) for i in range(0, len(v), 2)]))
    return polys


def _render_mask(job: tuple[str, str, int, int]) -> int:
    """YOLO 라벨 → 팔레트 마스크 PNG (픽셀값 = class id + 1).

    차량이 겹치는 곳은 화면 아래쪽(카메라에 가까운) 차량이 위에 오도록 먼 차량부터 칠한다.
    (원본 XML 의 z_order 는 실제 앞뒤 관계와 맞지 않아 쓰지 않음. 겹침은 차량 면적의 약 0.5%)
    """
    label_path, out_path, width, height = job
    polys = read_yolo_polygons(Path(label_path), width, height)
    polys.sort(key=lambda p: max(y for _, y in p[1]))
    mask = Image.new("P", (width, height), 0)
    mask.putpalette([c for i in range(256) for c in MASK_PALETTE.get(i, (0, 0, 0))])
    draw = ImageDraw.Draw(mask)
    for cls, pts in polys:
        draw.polygon(pts, fill=cls + 1)
    mask.save(out_path)
    return len(polys)


def class_table() -> dict:
    """형식마다 다른 클래스 번호 규칙을 한 파일로 정리."""
    names = [name for name, _ in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])]
    return {
        "yolo": {str(i): n for i, n in enumerate(names)},                          # 0부터
        "coco": {str(i + 1): n for i, n in enumerate(names)},                      # 1부터 (COCO 규약)
        "mask": {"0": "background", **{str(i + 1): n for i, n in enumerate(names)}},  # 0 = 배경
        "mask_palette": {str(k): list(v) for k, v in MASK_PALETTE.items() if k <= len(names)},
    }


def build_formats(out_dir: Path, formats: list[str], workers: int) -> dict:
    """manifest.csv + labels/(YOLO) 로부터 나머지 형식을 만든다. 형식별 결과 요약을 반환."""
    rows = list(csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8-sig")))
    lbl_dir = out_dir / "labels"
    names = [name for name, _ in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])]
    (out_dir / "classes.json").write_text(json.dumps(class_table(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict = {"yolo": {"dir": "labels/", "files": sum(1 for _ in lbl_dir.glob("*.txt"))}}

    if "coco" in formats:
        images, annotations = [], []
        for image_id, r in enumerate(tqdm(rows, desc="COCO", unit="img", dynamic_ncols=True), 1):
            w, h = int(r["width"]), int(r["height"])
            images.append({"id": image_id, "file_name": r["file_name"], "width": w, "height": h})
            for cls, pts in read_yolo_polygons(lbl_dir / f"{r['image_id']}.txt", w, h):
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "segmentation": [[round(c, 2) for p in pts for c in p]],
                    "area": round(polygon_area(pts), 2),
                    "bbox": [round(min(xs), 2), round(min(ys), 2), round(max(xs) - min(xs), 2),
                             round(max(ys) - min(ys), 2)],
                    "iscrowd": 0,
                })
        coco = {
            "info": {"description": "AI-Hub 교통문제 해결을 위한 CCTV 교통 데이터(고속도로) — 폴리곤 세그멘테이션",
                     "date_created": datetime.now().strftime("%Y-%m-%d")},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": n, "supercategory": "vehicle"} for i, n in enumerate(names)],
        }
        (out_dir / "annotations").mkdir(exist_ok=True)
        (out_dir / "annotations" / "instances_all.json").write_text(
            json.dumps(coco, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        summary["coco"] = {"file": "annotations/instances_all.json", "images": len(images),
                           "annotations": len(annotations)}

    if "labelme" in formats:
        ld = out_dir / "labelme"
        if ld.exists():
            shutil.rmtree(ld)  # 전부 labels/ 에서 다시 만드는 파생 파일
        ld.mkdir()
        for r in tqdm(rows, desc="LabelMe", unit="img", dynamic_ncols=True):
            w, h = int(r["width"]), int(r["height"])
            shapes = [{"label": names[cls], "points": [[round(x, 2), round(y, 2)] for x, y in pts],
                       "group_id": None, "description": "", "shape_type": "polygon", "flags": {}}
                      for cls, pts in read_yolo_polygons(lbl_dir / f"{r['image_id']}.txt", w, h)]
            (ld / f"{r['image_id']}.json").write_text(json.dumps({
                "version": "5.4.1", "flags": {}, "shapes": shapes,
                "imagePath": f"../images/{r['file_name']}", "imageData": None,
                "imageHeight": h, "imageWidth": w,
            }, ensure_ascii=False), encoding="utf-8")
        summary["labelme"] = {"dir": "labelme/", "files": len(rows)}

    if "mask" in formats:
        md = out_dir / "masks"
        if md.exists():
            shutil.rmtree(md)  # 전부 labels/ 에서 다시 만드는 파생 파일
        md.mkdir()
        jobs = [(str(lbl_dir / f"{r['image_id']}.txt"), str(md / f"{r['image_id']}.png"),
                 int(r["width"]), int(r["height"])) for r in rows]
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            list(tqdm(ex.map(_render_mask, jobs, chunksize=32), total=len(jobs), desc="마스크", unit="img",
                      dynamic_ncols=True))
        summary["mask"] = {"dir": "masks/", "files": len(jobs)}

    return summary


def print_format_summary(summary: dict) -> None:
    labels = {"yolo": "YOLO", "coco": "COCO", "mask": "Mask", "labelme": "LabelMe"}
    for fmt, info in summary.items():
        where = info.get("dir") or info.get("file")
        detail = (f"이미지 {info['images']:,} · 폴리곤 {info['annotations']:,}" if fmt == "coco"
                  else f"파일 {info['files']:,}개")
        print(f"  {labels[fmt]:8s}: {where:32s} {detail}")


def formats_only(args: argparse.Namespace, formats: list[str]) -> int:
    """이미 만들어 둔 car_seg_dataset 에 형식만 (다시) 생성 — 압축 해제는 하지 않는다."""
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not (out_dir / "manifest.csv").exists():
        print(f"[error] {out_dir / 'manifest.csv'} 가 없습니다. 먼저 전체 전처리를 실행하세요.", file=sys.stderr)
        return 1
    print(f"[formats] {out_dir} 에 {', '.join(formats)} 형식 생성")
    summary = build_formats(out_dir, formats, args.workers)
    report_path = out_dir / "preprocess_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["formats"] = summary
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 66)
    print_format_summary(summary)
    print(f"  클래스 번호표: {out_dir / 'classes.json'}")
    print("=" * 66)
    return 0


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------


@dataclass
class ExtractJob:
    archive: Archive
    info: zipfile.ZipInfo
    zip_member: str
    frame: FrameLabel


def classify_and_index(archives: list[Archive]):
    """zip 내용물로 라벨/이미지 zip 을 구분하고, 라벨 인덱스와 이미지 멤버 목록을 만든다."""
    label_index: dict[str, FrameLabel] = {}
    duplicate_labels: list[str] = []
    image_members: dict[str, list[tuple[zipfile.ZipInfo, str]]] = {}

    for arc in archives:
        if arc.error:
            continue
        try:
            with open_archive(arc) as zf:
                infos = [(info, member_name(info)) for info in zf.infolist() if not info.is_dir()]
                infos = [(info, name) for info, name in infos if not is_junk_member(name)]
                n_xml = sum(1 for _, name in infos if name.lower().endswith(".xml"))
                n_img = sum(1 for _, name in infos if Path(name).suffix.lower() in IMAGE_EXTS)

                if n_xml and n_xml >= n_img:
                    arc.kind = "label"
                    for info, name in infos:
                        if not name.lower().endswith(".xml"):
                            continue
                        clip = Path(name).stem
                        try:
                            with zf.open(info) as fp:
                                frames = parse_cvat_xml(fp, clip, name, arc.name)
                        except ET.ParseError as exc:
                            print(f"  [warn] XML 파싱 실패: {arc.name}:{name} ({exc})", file=sys.stderr)
                            continue
                        for frame in frames:
                            key = match_key(frame.image_name)
                            if key in label_index:
                                duplicate_labels.append(frame.image_name)
                                continue
                            label_index[key] = frame
                elif n_img:
                    arc.kind = "image"
                    image_members[arc.key] = [
                        (info, name) for info, name in infos if Path(name).suffix.lower() in IMAGE_EXTS
                    ]
                else:
                    arc.kind = "unknown"
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            arc.error = f"zip 을 열 수 없음 (마지막 조각 누락/손상 가능): {exc}"

    return label_index, duplicate_labels, image_members


def print_archive_table(archives: list[Archive]) -> None:
    print(f"\n{'종류':6s} {'출처':4s} {'조각':>4s} {'크기(GB)':>9s}  zip")
    print("-" * 84)
    for arc in archives:
        status = f"  ❌ {arc.error}" if arc.error else ""
        print(f"{arc.kind or '?':6s} {arc.source:4s} {len(arc.parts):4d} {arc.total_bytes / 1024**3:9.2f}  "
              f"{arc.key}{status}")


def inspect_image(path: Path, verify: bool) -> tuple[int, int, str]:
    with Image.open(path) as im:
        width, height = im.size
        mode = im.mode
        if verify:
            im.verify()  # 청크 CRC 검사 — 잘린/손상된 PNG 를 걸러낸다
    return width, height, mode


def extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dst: Path) -> None:
    """임시 파일에 쓴 뒤 교체 — 중간에 끊겨도 반쯤 쓰인 이미지가 남지 않는다."""
    tmp = dst.with_name(dst.name + ".tmp")
    with zf.open(info) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)
    os.replace(tmp, dst)


def preprocess(args: argparse.Namespace) -> int:
    try:
        formats = parse_formats(args.formats)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.formats_only:
        return formats_only(args, formats)

    raw_roots = [Path(p).expanduser().resolve() for p in args.raw_root]
    out_dir = Path(args.out_dir).expanduser().resolve()
    missing = [p for p in raw_roots if not p.exists()]
    if missing:
        print(f"[error] 원본 경로가 없습니다: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    # 1) zip 조각 수집 (폴더 + tar) ----------------------------------------------
    print("[1/5] zip 조각 검색: " + " · ".join(map(str, raw_roots)))
    archives, tar_notes = discover_archives(raw_roots, skip_dirs=[out_dir])
    if tar_notes:
        print(f"  tar {len(tar_notes)}개 (풀지 않고 안의 조각을 바로 읽음):")
        for note in tar_notes:
            print(f"    {note}")
    if not archives:
        print("[error] .zip / .zip.partN / .tar 에서 zip 조각을 찾지 못했습니다.\n"
              "        AI-Hub 에서 받은 download (N).tar 파일(또는 풀어 둔 '100.교통문제_…' 폴더)이 있는 곳을\n"
              "        --raw-root 로 지정하세요.  예) python car_seg_data_preprocessing.py --raw-root ~/Downloads",
              file=sys.stderr)
        return 1

    # 2) 라벨/이미지 zip 분류 + 라벨 인덱스 ------------------------------------
    print(f"[2/5] zip {len(archives)}개 내용 확인 및 라벨 XML 파싱 중 ...")
    label_index, duplicate_labels, image_members = classify_and_index(archives)
    print_archive_table(archives)

    broken = [arc for arc in archives if arc.error]
    if broken:
        print(f"\n[warn] 문제가 있는 zip {len(broken)}개는 건너뜁니다 (위 ❌ 표시). "
              f"다운로드가 끝났는지 확인하세요.")

    # 3) 이미지 ↔ 라벨 매칭 ----------------------------------------------------
    jobs: list[ExtractJob] = []
    unlabeled_images: list[str] = []
    duplicate_images: list[str] = []
    seen: set[str] = set()
    arc_by_key = {arc.key: arc for arc in archives}

    for arc_key, members in image_members.items():
        # 같은 이름이 두 번 들어있는 경우(예: 5.전북 의 <클립>/<클립>/NNN.png 이중 폴더)는 경로가 짧은 쪽을 사용.
        # 두 사본은 픽셀이 동일하고 PNG 압축만 다름을 확인함.
        for info, name in sorted(members, key=lambda m: (m[1].count("/"), m[1])):
            key = match_key(name)
            frame = label_index.get(key)
            if frame is None:
                unlabeled_images.append(f"{arc_by_key[arc_key].name}:{name}")
                continue
            if key in seen:
                duplicate_images.append(f"{arc_by_key[arc_key].name}:{name}")
                continue
            seen.add(key)
            jobs.append(ExtractJob(arc_by_key[arc_key], info, name, frame))

    labels_without_image = sorted(
        f"{frame.label_archive}:{frame.image_name}" for key, frame in label_index.items() if key not in seen
    )

    print(f"\n[3/5] 매칭 결과")
    print(f"  라벨 프레임(XML <image>) : {len(label_index):,}")
    print(f"  이미지 파일              : {sum(len(m) for m in image_members.values()):,}")
    print(f"  매칭 성공                : {len(jobs):,}")
    print(f"  라벨 없는 이미지 (제외)  : {len(unlabeled_images):,}")
    print(f"  이미지 없는 라벨 (제외)  : {len(labels_without_image):,}")
    if duplicate_labels or duplicate_images:
        print(f"  같은 이름의 사본 (경로가 짧은 쪽 사용): 라벨 {len(duplicate_labels)} / 이미지 {len(duplicate_images)}")

    # 짝이 되는 zip 을 아예 안 받은 경우 알려준다 (예: 이미지 zip 만 받고 라벨 zip 을 안 받음)
    matched_per_image_zip = Counter(job.archive.name for job in jobs)
    matched_per_label_zip = Counter(label_index[k].label_archive for k in seen)
    no_label = [arc for arc in archives if arc.kind == "image" and not matched_per_image_zip[arc.name]]
    no_image = [arc for arc in archives if arc.kind == "label" and not matched_per_label_zip[arc.name]]
    if no_label:
        print("\n  [주의] 라벨이 하나도 없는 이미지 zip — AI-Hub 의 '라벨링데이터' 에서 같은 지역 라벨 파일을 받으세요:")
        for arc in no_label:
            print(f"    {arc.name} (이미지 {len(image_members.get(arc.key, [])):,}장 제외됨)")
    if no_image:
        print("\n  [주의] 이미지가 하나도 없는 라벨 zip — AI-Hub 의 '원천데이터' 에서 같은 지역 이미지 파일을 받으세요:")
        for arc in no_image:
            print(f"    {arc.name}")

    if args.dry_run:
        print("\n--dry-run: 여기서 종료합니다 (아무 파일도 쓰지 않음).")
        return 0
    if not jobs:
        print("[error] 매칭된 이미지가 없습니다.", file=sys.stderr)
        return 1

    # 4) 추출 + YOLO 라벨 변환 -------------------------------------------------
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for tmp in img_dir.glob("*.tmp"):  # 이전 실행이 중간에 끊겨 남은 반쪽 파일
        tmp.unlink()

    rows: list[dict] = []
    skipped: list[dict] = []
    dropped_total: Counter = Counter()
    n_reused = 0

    jobs_by_arc: dict[str, list[ExtractJob]] = defaultdict(list)
    for job in jobs:
        jobs_by_arc[job.archive.key].append(job)

    print(f"\n[4/5] 이미지 {len(jobs):,}장 추출 + 라벨 변환 → {out_dir}")
    progress = tqdm(total=len(jobs), unit="img", dynamic_ncols=True)
    for arc_key, arc_jobs in jobs_by_arc.items():
        arc = arc_by_key[arc_key]
        progress.set_postfix_str(arc.stem[:24])
        with open_archive(arc) as zf:
            # zip 안의 저장 순서대로 읽어야 조각 파일을 순차적으로 훑는다 (훨씬 빠름)
            for job in sorted(arc_jobs, key=lambda j: j.info.header_offset):
                progress.update(1)
                frame = job.frame
                out_stem = normalize_stem(job.zip_member)
                ext = Path(job.zip_member).suffix.lower()
                dst = img_dir / f"{out_stem}{ext}"

                # 4-1) 추출 (이미 같은 크기로 풀려 있으면 재사용 → 중단 후 재실행 시 이어서)
                try:
                    if (not args.overwrite and dst.is_file()
                            and dst.stat().st_size == job.info.file_size):
                        n_reused += 1
                    else:
                        extract_member(zf, job.info, dst)
                    width, height, mode = inspect_image(dst, verify=not args.no_verify)
                except (zipfile.BadZipFile, UnidentifiedImageError, OSError, SyntaxError, EOFError) as exc:
                    dst.unlink(missing_ok=True)
                    skipped.append({"image": job.zip_member, "reason": f"이미지 추출/검증 실패: {exc}"})
                    continue

                # 4-2) XML 크기와 비교 — 같은 종횡비의 단순 리사이즈만 허용
                xml_w, xml_h = frame.width or width, frame.height or height
                size_scaled = (width, height) != (xml_w, xml_h)
                if size_scaled and abs(width * xml_h - height * xml_w) > 0.01 * width * xml_h:
                    dst.unlink(missing_ok=True)
                    skipped.append({
                        "image": job.zip_member,
                        "reason": f"XML 크기({xml_w}x{xml_h})와 이미지 크기({width}x{height})의 종횡비가 다름",
                    })
                    continue

                # 4-3) YOLO 라벨 — 좌표는 XML 좌표계 기준으로 정규화
                lines, counts, dropped = to_yolo_lines(frame, xml_w, xml_h, args.min_area)
                dropped_total.update(dropped)
                (lbl_dir / f"{out_stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )

                row = {
                    "image_id": out_stem,
                    "file_name": dst.name,
                    "clip": frame.clip,
                    "region": arc.stem,
                    "image_archive": arc.name,
                    "label_archive": frame.label_archive,
                    "zip_member": job.zip_member,
                    "xml_member": frame.xml_member,
                    "width": width,
                    "height": height,
                    "xml_width": xml_w,
                    "xml_height": xml_h,
                    "size_scaled": size_scaled,
                    "resolution": f"{width}x{height}",
                    "orientation": "landscape" if width > height else "portrait" if height > width else "square",
                    "pil_mode": mode,
                    "gt_car": counts.get("car", 0),
                    "gt_bus": counts.get("bus", 0),
                    "gt_truck": counts.get("truck", 0),
                    "gt_total": sum(counts.values()),
                    "n_dropped_polygons": sum(dropped.values()),
                }
                meta = parse_clip_name(frame.clip)
                m = re.search(r"(\d+)$", out_stem)
                meta["frame"] = int(m.group(1)) if m else ""
                row.update(meta)
                rows.append(row)
    progress.close()

    if not rows:
        print("[error] 정리된 이미지가 없습니다.", file=sys.stderr)
        return 1

    # manifest.csv ------------------------------------------------------------
    rows.sort(key=lambda r: r["image_id"])
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(k for k in row if k not in fieldnames)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    # 이전 실행에서 남은(이번 manifest 에 없는) 파일 경고
    produced = {r["image_id"] for r in rows}
    stale = [p.name for p in img_dir.iterdir() if p.is_file() and p.stem not in produced]
    stale += [p.name for p in lbl_dir.iterdir() if p.suffix == ".txt" and p.stem not in produced]

    # preprocess_report.json --------------------------------------------------
    def dist(key: str) -> dict:
        return dict(Counter(str(r.get(key, "")) for r in rows).most_common())

    def head(items: list, n: int = 100) -> list:
        return items[:n]

    report = {
        "raw_roots": [str(p) for p in raw_roots],
        "tar_files": tar_notes,
        "out_dir": str(out_dir),
        "classes": {cid: name for name, cid in CLASS_TO_ID.items()},
        "archives": [
            {"key": a.key, "kind": a.kind, "n_parts": len(a.parts), "bytes": a.total_bytes, "error": a.error}
            for a in archives
        ],
        "n_images": len(rows),
        "n_clips": len({r["clip"] for r in rows}),
        "n_reused_existing": n_reused,
        "n_skipped": len(skipped),
        "n_size_scaled": sum(1 for r in rows if r["size_scaled"]),
        "n_meta_unparsed": sum(1 for r in rows if not r.get("meta_parsed")),
        "gt_objects": {name: sum(r[f"gt_{name}"] for r in rows) for name in CLASS_TO_ID},
        "dropped_polygons": dict(dropped_total),
        "distributions": {
            key: dist(key)
            for key in ("region", "site", "hour_type", "weather", "time_band", "day_night", "lane_config",
                        "cam_height", "quality", "resolution", "orientation")
        },
        "unmatched": {
            "images_without_label": {"count": len(unlabeled_images), "examples": head(unlabeled_images)},
            "labels_without_image": {"count": len(labels_without_image), "examples": head(labels_without_image)},
            "duplicate_label_names": {"count": len(duplicate_labels), "examples": head(duplicate_labels)},
            "duplicate_image_names": {"count": len(duplicate_images), "examples": head(duplicate_images)},
        },
        "stale_files": {"count": len(stale), "examples": head(stale)},
        "skipped": skipped,
    }

    # 5) 다른 라벨 형식 --------------------------------------------------------
    print(f"\n[5/5] 라벨 형식 생성: {', '.join(formats)}")
    report["formats"] = build_formats(out_dir, formats, args.workers)

    report_path = out_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 요약 ----------------------------------------------------------------
    gt = report["gt_objects"]
    print("\n" + "=" * 66)
    print(f"완료: 이미지 {len(rows):,}장 / 클립 {report['n_clips']}개 / 제외 {len(skipped)}건"
          f" (기존 파일 재사용 {n_reused:,}장)")
    print(f"  객체: car={gt['car']:,}  bus={gt['bus']:,}  truck={gt['truck']:,}")
    if dropped_total:
        print(f"  제외된 폴리곤: {dict(dropped_total)}")
    if report["n_size_scaled"]:
        print(f"  XML 과 크기가 다른(같은 종횡비) 이미지: {report['n_size_scaled']}장 — XML 좌표계로 정규화함")
    for key in ("region", "weather", "orientation"):
        print(f"  {key:12s}: {report['distributions'][key]}")
    if stale:
        print(f"  [warn] 이번 결과에 없는 이전 파일 {len(stale)}개가 images/labels 에 남아 있습니다.")
    print(f"\n  결과 폴더: {out_dir}")
    print(f"  images  : images/ ({len(rows):,}장)")
    print_format_summary(report["formats"])
    print(f"  classes : classes.json (형식별 클래스 번호표)")
    print(f"  manifest: {manifest_path.name} · report: {report_path.name}")
    print("=" * 66)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-Hub 도로 CCTV 폴리곤 데이터 → images + YOLO / COCO / 마스크 / LabelMe 라벨",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-root", nargs="+", default=[str(BASE_DIR)],
                   help="AI-Hub 다운로드 위치 — 폴더 또는 .tar 파일, 여러 개 가능 (폴더 안의 .tar 도 자동으로 읽음)")
    p.add_argument("--out-dir", default=str(BASE_DIR / "car_seg_dataset"), help="결과 출력 폴더")
    p.add_argument("--dry-run", action="store_true", help="조각 검사와 매칭 통계만 출력하고 종료")
    p.add_argument("--overwrite", action="store_true", help="이미 풀려 있는 이미지도 다시 추출")
    p.add_argument("--no-verify", action="store_true", help="PNG 무결성(CRC) 검사 생략 — 조금 빨라짐")
    p.add_argument("--min-area", type=float, default=1.0, help="이보다 작은(px²) 폴리곤은 제외")
    p.add_argument("--formats", default=DEFAULT_FORMATS,
                   help=f"만들 라벨 형식, 쉼표로 구분 ({', '.join(FORMATS)}). yolo 는 항상 포함")
    p.add_argument("--formats-only", action="store_true",
                   help="압축 해제 없이, 이미 만든 --out-dir 에 라벨 형식만 (다시) 생성")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="마스크 생성 병렬 프로세스 수")
    return p


if __name__ == "__main__":
    sys.exit(preprocess(build_parser().parse_args()))
