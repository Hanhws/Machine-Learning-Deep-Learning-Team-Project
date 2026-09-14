#!/usr/bin/env python3
"""AI-Hub 「교통문제 해결을 위한 CCTV 교통 데이터(고속도로)」 폴리곤 세그멘테이션 데이터를
한 곳으로 모아 **YOLO segmentation 형식 + JPG 이미지**로 만드는 스크립트.

(old_20000_version/car_seg_data_preprocessing.py 를 YOLO-seg 전용으로 다시 정리한 버전)
  - 라벨 형식은 YOLO 하나만 만든다 (COCO · 마스크 · LabelMe 제거)
  - 이미지는 zip 에서 꺼내는 즉시 JPG(기본 품질 95)로 변환해 저장한다 → 원본 PNG 60GB 대비 약 1/4
  - manifest 에 CCTV(카메라) ID · 도로 형태(road_form) · 난이도 조건(difficulty) 컬럼을 추가한다
    → EDA 와 분할(학습용 train 축소)이 같은 기준을 쓴다

원본 구조 (AI-Hub 다운로드 그대로)
    traffic_data2/
      100.교통문제_…(고속도로)/01.데이터/2.Validation/라벨링데이터/폴리곤세그멘테이션/3.대전충남.zip.part0
      100.교통문제_…(고속도로) 6/01.데이터/2.Validation/원천데이터/폴리곤세그멘테이션/3.대전충남.zip.part0
                                                                                   3.대전충남.zip.part1073741824 …
    - AI-Hub 에서 받은 download (N).tar 를 풀지 않은 상태도 그대로 읽는다
    - 분할 zip 조각의 접미사 숫자 = 그 조각의 시작 바이트 오프셋
    - 라벨: CVAT 1.1 XML (클립 1개 = XML 1개, <image> 안에 car/bus/truck <polygon>)
    - 이미지: <클립명>_NNN.png (일부 클립은 '<클립명> NNN.png' 처럼 공백 구분)

처리 과정
    1. 모든 .zip.partN 조각을 zip 단위로 묶고 오프셋 연속성(누락 조각)을 검사
    2. 조각을 디스크에 합치지 않고 가상으로 이어 붙여 바로 읽음
    3. 라벨 XML 파싱 → 정규화한 이미지 파일명 기준으로 원천 이미지와 매칭
    4. 매칭된 이미지를 PNG 디코딩 → JPG 로 저장 (여러 프로세스 병렬), 폴리곤 → YOLO seg 라벨
    5. manifest.csv / preprocess_report.json / classes.json 저장

결과 구조
    car_seg_dataset/
      images/<클립명>_NNN.jpg          JPG (RGB, 품질 95)
      labels/<클립명>_NNN.txt          YOLO seg  "<class_id> x1 y1 x2 y2 …" (0~1 정규화, 0=car 1=bus 2=truck)
      classes.json                    클래스 번호표
      manifest.csv                    이미지 1장 = 1행 (파일명 메타데이터 + CCTV · 도로 형태 · 난이도 + GT 개수)
      preprocess_report.json          요약 통계, 매칭 실패·제외 목록

사용 예
    python car_seg_data_preprocessing.py --dry-run          # 조각/매칭만 점검 (아무것도 쓰지 않음)
    python car_seg_data_preprocessing.py                    # 전체 실행
    python car_seg_data_preprocessing.py --raw-root ~/Downloads
    python car_seg_data_preprocessing.py --jpg-quality 90 --workers 8
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
import sys
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date as datetime_date
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# GT 클래스 -> YOLO class id
CLASS_TO_ID = {"car": 0, "bus": 1, "truck": 2}

# 'xxx.zip.part1073741824' -> ('xxx.zip', 1073741824)
PART_RE = re.compile(r"^(?P<zip>.+\.zip)\.part(?P<offset>\d+)$", re.IGNORECASE)

# macOS 가 같은 이름의 다운로드 폴더에 붙이는 ' 2', ' 3' … 접미사
DOWNLOAD_COPY_SUFFIX_RE = re.compile(r" \d+$")

# 파일명 규약 (클립 전부 '_' 기준 11토큰):
#   Suwon_CH01_20200721_1700_TUE_9m_RH_highway_TW5_sunny_FHD_001.png
#   Inje_injetunnel2_20201210_0956_THU_4.8m_NH_highway_OW2_tunnel_FHD 005.png
META_FIELDS = (
    "site",         # 촬영 지역 (Suwon, Inje, …)
    "channel",      # CCTV 채널 / 설치 지점명 (CH01, injetunnel2, busanportbrdg1 …)
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

# 촬영 시각(07~21시)을 3시간 단위로 묶은 구간
TIME_BANDS = ((7, 10, "07-09"), (10, 13, "10-12"), (13, 16, "13-15"), (16, 19, "16-18"), (19, 22, "19-21"))

LANE_DIRECTION = {"TW": "two_way", "OW": "one_way"}

# 도로 형태 — 파일명의 road_type 은 전부 'highway' 라 구분이 안 되므로, CCTV 설치 지점명(channel)의
# 키워드로 구분한다. 화면 속 텍스트로 확인한 결과 채널명은 'CCTV 지점명'(예: 추점터널, 남이육교, 태인졸음쉼터)이며,
# 카메라가 그 구조물 '근처'를 비추는 경우가 대부분이다 (실제 터널 내부 영상은 injetunnel2 뿐 — weather=tunnel).
# 키워드에 걸리지 않는 지점(CH01, ogsan, jongsin …)은 일반 도로(general).
ROAD_FORM_KEYWORDS = (
    ("tunnel", "tunnel"),       # 터널 (입구 부근 포함)
    ("brdg", "bridge"),         # 교량
    ("bridge", "bridge"),
    ("overpass", "overpass"),   # 육교 / 고가
    ("shelter", "shelter"),     # 졸음쉼터
)
ROAD_FORMS = ("general", "bridge", "tunnel", "overpass", "shelter")
ROAD_FORM_KO = {"general": "일반", "bridge": "교량", "tunnel": "터널", "overpass": "육교·고가", "shelter": "졸음쉼터"}

# 기존 YOLO 가 '잘 잡던' 조건 = 주간 · 맑음 · 일반 도로. 셋을 모두 만족하면 easy, 하나라도 벗어나면 hard.
# car_seg_data_eda.py 와 car_seg_data_split.py 가 이 정의를 그대로 가져다 쓴다.
EASY_CONDITION = {"day_night": "day", "weather": "sunny", "road_form": "general"}

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

    path: Path
    start: int
    size: int
    in_tar: bool = False


class MultiPartReader(io.RawIOBase):
    """분할 zip 조각들을 하나의 연속된 파일처럼 읽는 read-only 스트림.

    조각 접미사가 곧 시작 오프셋이므로, 현재 위치가 속한 조각을 이분 탐색해서 읽는다.
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
        if path not in self._fps:
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

    key: str
    name: str
    parts: list[tuple[int, Part]] = field(default_factory=list)
    kind: str = ""                            # 'label' | 'image' | 'unknown'
    error: str = ""

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
    """zip 조각이면 (묶음 키, zip 파일명, 시작 오프셋), 아니면 None."""
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
    """AI-Hub 다운로드 tar 안의 zip 조각을 등록. tar 는 풀지 않고 위치만 기록한다."""
    try:
        with tarfile.open(path, "r:") as tf:
            n = 0
            for member in tf:
                if not member.isfile():
                    continue
                names = [p for p in nfc(member.name).split("/") if p not in ("", ".")]
                if not names or names[-1].startswith(".") or "__MACOSX" in names:
                    continue
                entry = _archive_entry(names[:-1], names[-1])
                if entry:
                    add(*entry, Part(path, member.offset_data, member.size, in_tar=True))
                    n += 1
        return f"{path.name}: zip 조각 {n}개"
    except tarfile.ReadError:
        return f"{path.name}: 읽을 수 없음 — 압축된 tar(.tar.gz 등)이면 먼저 풀어주세요"


def discover_archives(raw_roots: list[Path], skip_dirs: list[Path]) -> tuple[list[Archive], list[str]]:
    """raw_roots 아래의 .zip / .zip.partN 과 .tar 안의 조각을 모두 찾아 zip 단위로 묶는다."""
    skip = {path.resolve() for path in skip_dirs}
    archives: dict[str, Archive] = {}
    tar_notes: list[str] = []

    def add(key: str, zip_name: str, offset: int, part: Part) -> None:
        arc = archives.setdefault(key, Archive(key=key, name=zip_name))
        dup = next((p for off, p in arc.parts if off == offset), None)
        if dup is not None:
            if dup.size != part.size:
                arc.error = f"같은 오프셋({offset})의 조각이 크기가 다르게 중복됨: {dup.path} / {part.path}"
            return
        arc.parts.append((offset, part))

    for root in raw_roots:
        if root.is_file() and root.suffix.lower() == ".tar":
            tar_notes.append(scan_tar(root, add))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            # 결과 폴더(car_seg_*)와 예전 버전 폴더(old_*)는 뒤지지 않는다
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not d.startswith("car_seg_") and not d.startswith("old_")
                and (here / d).resolve() not in skip
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
# 파일명 / 메타데이터 / XML 파싱
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


def road_form_of(channel: str) -> str:
    """CCTV 지점명 → 도로 형태 (general / bridge / tunnel / overpass / shelter)."""
    c = channel.lower()
    for keyword, form in ROAD_FORM_KEYWORDS:
        if keyword in c:
            return form
    return "general"


def hard_reasons(row) -> list[str]:
    """EASY_CONDITION 에서 벗어난 항목 목록. 빈 목록이면 easy (주간·맑음·일반 도로).

    row 는 dict · pandas Series · namedtuple 어느 것이든 속성/키로 day_night·weather·road_form 을 읽는다.
    """
    get = row.get if hasattr(row, "get") else (lambda k, d="": getattr(row, k, d))
    return [f"{k}={get(k, '')}" for k, v in EASY_CONDITION.items() if str(get(k, "")) != v]


def difficulty_of(row) -> str:
    return "hard" if hard_reasons(row) else "easy"


def parse_clip_name(clip: str) -> dict:
    """클립명(11토큰)을 메타데이터 dict 로. 규약에 맞지 않으면 meta_parsed=False."""
    tokens = clip.split("_")
    if len(tokens) != len(META_FIELDS):
        return {"meta_parsed": False}

    meta: dict = {"meta_parsed": True, **dict(zip(META_FIELDS, tokens))}
    meta["camera"] = f"{meta['site']}_{meta['channel']}"
    meta["time_band"] = time_band_of(meta["time"])
    try:
        meta["cam_height_m"] = float(meta["cam_height"].rstrip("m"))
    except ValueError:
        meta["cam_height_m"] = ""
    lane = meta["lane_config"]
    meta["lane_direction"] = LANE_DIRECTION.get(lane[:2], "")
    meta["lane_count"] = int(lane[2:]) if lane[2:].isdigit() else ""
    meta["day_night"], meta["min_from_sunset"] = day_night_of(meta["date"], meta["time"], meta["site"])
    meta["road_form"] = road_form_of(meta["channel"])
    reasons = hard_reasons(meta)
    meta["difficulty"] = "hard" if reasons else "easy"
    meta["hard_reasons"] = "|".join(reasons)
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


def class_table() -> dict:
    names = [name for name, _ in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])]
    return {"yolo": {str(i): n for i, n in enumerate(names)}}


# ----------------------------------------------------------------------------
# 이미지 → JPG (워커 프로세스)
# ----------------------------------------------------------------------------


def _encode_jpg(job: tuple[bytes, str, int, bool]) -> tuple[str, int, int, str, int, str]:
    """PNG 바이트 → JPG 파일. (dst, 폭, 높이, 원본 모드, jpg 바이트 수, 오류 메시지) 반환.

    임시 파일에 쓴 뒤 교체하므로 중간에 끊겨도 반쯤 쓰인 JPG 가 남지 않는다.
    """
    data, dst, quality, verify = job
    tmp = dst + ".tmp"
    try:
        if verify:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()  # PNG 청크 CRC 검사 — 잘린/손상된 PNG 를 걸러낸다
        with Image.open(io.BytesIO(data)) as im:
            width, height, mode = im.width, im.height, im.mode
            im.convert("RGB").save(tmp, "JPEG", quality=quality)
        os.replace(tmp, dst)
        return dst, width, height, mode, os.path.getsize(dst), ""
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst, 0, 0, "", 0, f"{exc.__class__.__name__}: {exc}"


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


def existing_jpg(dst: Path) -> tuple[int, int] | None:
    """이미 변환된 JPG 가 정상이면 (폭, 높이). 중단 후 재실행 시 이어서 하기 위함."""
    if not dst.is_file() or dst.stat().st_size == 0:
        return None
    try:
        with Image.open(dst) as im:
            if im.format != "JPEG":
                return None
            return im.size
    except (UnidentifiedImageError, OSError):
        return None


def preprocess(args: argparse.Namespace) -> int:
    raw_roots = [Path(p).expanduser().resolve() for p in args.raw_root]
    out_dir = Path(args.out_dir).expanduser().resolve()
    missing = [p for p in raw_roots if not p.exists()]
    if missing:
        print(f"[error] 원본 경로가 없습니다: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    if not (1 <= args.jpg_quality <= 100):
        print("[error] --jpg-quality 는 1~100 사이여야 합니다", file=sys.stderr)
        return 1

    # 1) zip 조각 수집 --------------------------------------------------------
    print("[1/5] zip 조각 검색: " + " · ".join(map(str, raw_roots)))
    archives, tar_notes = discover_archives(raw_roots, skip_dirs=[out_dir])
    for note in tar_notes:
        print(f"    {note}")
    if not archives:
        print("[error] .zip / .zip.partN / .tar 에서 zip 조각을 찾지 못했습니다. --raw-root 를 확인하세요.",
              file=sys.stderr)
        return 1

    # 2) 라벨/이미지 zip 분류 + 라벨 인덱스 ------------------------------------
    print(f"[2/5] zip {len(archives)}개 내용 확인 및 라벨 XML 파싱 중 ...")
    label_index, duplicate_labels, image_members = classify_and_index(archives)
    print_archive_table(archives)
    broken = [arc for arc in archives if arc.error]
    if broken:
        print(f"\n[warn] 문제가 있는 zip {len(broken)}개는 건너뜁니다 (위 ❌ 표시).")

    # 3) 이미지 ↔ 라벨 매칭 ----------------------------------------------------
    jobs: list[ExtractJob] = []
    unlabeled_images: list[str] = []
    duplicate_images: list[str] = []
    seen: set[str] = set()
    arc_by_key = {arc.key: arc for arc in archives}

    for arc_key, members in image_members.items():
        # 같은 이름이 두 번 들어있는 경우(5.전북 의 이중 폴더)는 경로가 짧은 쪽을 사용 (픽셀 동일 확인됨)
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

    print("\n[3/5] 매칭 결과")
    print(f"  라벨 프레임(XML <image>) : {len(label_index):,}")
    print(f"  이미지 파일              : {sum(len(m) for m in image_members.values()):,}")
    print(f"  매칭 성공                : {len(jobs):,}")
    print(f"  라벨 없는 이미지 (제외)  : {len(unlabeled_images):,}")
    print(f"  이미지 없는 라벨 (제외)  : {len(labels_without_image):,}")
    if duplicate_labels or duplicate_images:
        print(f"  같은 이름의 사본 (경로가 짧은 쪽 사용): 라벨 {len(duplicate_labels)} / 이미지 {len(duplicate_images)}")

    if args.dry_run:
        print("\n--dry-run: 여기서 종료합니다 (아무 파일도 쓰지 않음).")
        return 0
    if not jobs:
        print("[error] 매칭된 이미지가 없습니다.", file=sys.stderr)
        return 1

    # 4) JPG 변환 + YOLO 라벨 -------------------------------------------------
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for tmp in img_dir.glob("*.tmp"):
        tmp.unlink()

    rows: dict[str, dict] = {}          # image_id → manifest 행
    skipped: list[dict] = []
    dropped_total: Counter = Counter()
    n_reused = 0
    src_bytes_total = 0
    jpg_bytes_total = 0

    def finish_row(job: ExtractJob, out_stem: str, width: int, height: int, mode: str, jpg_bytes: int) -> None:
        """이미지 크기가 확정된 뒤 라벨을 쓰고 manifest 행을 만든다."""
        nonlocal jpg_bytes_total
        frame = job.frame
        xml_w, xml_h = frame.width or width, frame.height or height
        size_scaled = (width, height) != (xml_w, xml_h)
        if size_scaled and abs(width * xml_h - height * xml_w) > 0.01 * width * xml_h:
            (img_dir / f"{out_stem}.jpg").unlink(missing_ok=True)
            skipped.append({"image": job.zip_member,
                            "reason": f"XML 크기({xml_w}x{xml_h})와 이미지 크기({width}x{height})의 종횡비가 다름"})
            return
        # 좌표는 XML 좌표계 기준으로 정규화 (같은 종횡비 리사이즈면 0~1 좌표가 그대로 맞음)
        lines, counts, dropped = to_yolo_lines(frame, xml_w, xml_h, args.min_area)
        dropped_total.update(dropped)
        (lbl_dir / f"{out_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        jpg_bytes_total += jpg_bytes

        row = {
            "image_id": out_stem,
            "file_name": f"{out_stem}.jpg",
            "clip": frame.clip,
            "region": job.archive.stem,
            "image_archive": job.archive.name,
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
            "src_mode": mode,
            "src_bytes": job.info.file_size,
            "jpg_bytes": jpg_bytes,
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
        rows[out_stem] = row

    jobs_by_arc: dict[str, list[ExtractJob]] = defaultdict(list)
    for job in jobs:
        jobs_by_arc[job.archive.key].append(job)

    print(f"\n[4/5] 이미지 {len(jobs):,}장 → JPG(품질 {args.jpg_quality}) 변환 + YOLO 라벨 → {out_dir}")
    progress = tqdm(total=len(jobs), unit="img", dynamic_ncols=True)
    max_inflight = max(4, args.workers * 4)  # 메모리에 동시에 올려 두는 PNG 수 제한 (장당 ~2-3MB)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending: dict = {}

        def drain(block_until_below: int) -> None:
            while len(pending) >= block_until_below:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    job, out_stem = pending.pop(fut)
                    _, width, height, mode, jpg_bytes, err = fut.result()
                    progress.update(1)
                    if err:
                        skipped.append({"image": job.zip_member, "reason": f"이미지 디코딩/JPG 변환 실패: {err}"})
                        continue
                    finish_row(job, out_stem, width, height, mode, jpg_bytes)

        for arc_key, arc_jobs in jobs_by_arc.items():
            arc = arc_by_key[arc_key]
            progress.set_postfix_str(arc.stem[:24])
            with open_archive(arc) as zf:
                # zip 안의 저장 순서대로 읽어야 조각 파일을 순차적으로 훑는다 (훨씬 빠름)
                for job in sorted(arc_jobs, key=lambda j: j.info.header_offset):
                    out_stem = normalize_stem(job.zip_member)
                    dst = img_dir / f"{out_stem}.jpg"
                    src_bytes_total += job.info.file_size

                    if not args.overwrite:
                        size = existing_jpg(dst)
                        if size is not None:
                            n_reused += 1
                            progress.update(1)
                            # 재사용 시엔 원본 PNG 를 디코딩하지 않으므로 원본 모드는 알 수 없음 → 빈칸
                            finish_row(job, out_stem, size[0], size[1], "", dst.stat().st_size)
                            continue
                    try:
                        data = zf.read(job.info)
                    except (zipfile.BadZipFile, OSError, EOFError) as exc:
                        progress.update(1)
                        skipped.append({"image": job.zip_member, "reason": f"zip 에서 읽기 실패: {exc}"})
                        continue
                    fut = pool.submit(_encode_jpg, (data, str(dst), args.jpg_quality, not args.no_verify))
                    pending[fut] = (job, out_stem)
                    drain(max_inflight)
        drain(1)
    progress.close()

    if not rows:
        print("[error] 정리된 이미지가 없습니다.", file=sys.stderr)
        return 1

    # 5) manifest / report ----------------------------------------------------
    print("\n[5/5] manifest · report 저장")
    row_list = [rows[k] for k in sorted(rows)]
    fieldnames: list[str] = []
    for row in row_list:
        fieldnames.extend(k for k in row if k not in fieldnames)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(row_list)
    (out_dir / "classes.json").write_text(json.dumps(class_table(), ensure_ascii=False, indent=2), encoding="utf-8")

    produced = set(rows)
    stale = [p.name for p in img_dir.iterdir() if p.is_file() and p.stem not in produced]
    stale += [p.name for p in lbl_dir.iterdir() if p.suffix == ".txt" and p.stem not in produced]

    def dist(key: str) -> dict:
        return dict(Counter(str(r.get(key, "")) for r in row_list).most_common())

    def head(items: list, n: int = 100) -> list:
        return items[:n]

    report = {
        "raw_roots": [str(p) for p in raw_roots],
        "tar_files": tar_notes,
        "out_dir": str(out_dir),
        "label_format": "yolo-seg",
        "image_format": {"format": "jpg", "quality": args.jpg_quality,
                         "source_png_bytes": src_bytes_total, "jpg_bytes": jpg_bytes_total},
        "classes": {cid: name for name, cid in CLASS_TO_ID.items()},
        "easy_condition": EASY_CONDITION,
        "archives": [
            {"key": a.key, "kind": a.kind, "n_parts": len(a.parts), "bytes": a.total_bytes, "error": a.error}
            for a in archives
        ],
        "n_images": len(row_list),
        "n_clips": len({r["clip"] for r in row_list}),
        "n_cameras": len({r.get("camera", "") for r in row_list}),
        "n_reused_existing": n_reused,
        "n_skipped": len(skipped),
        "n_size_scaled": sum(1 for r in row_list if r["size_scaled"]),
        "n_meta_unparsed": sum(1 for r in row_list if not r.get("meta_parsed")),
        "gt_objects": {name: sum(r[f"gt_{name}"] for r in row_list) for name in CLASS_TO_ID},
        "dropped_polygons": dict(dropped_total),
        "distributions": {
            key: dist(key)
            for key in ("region", "site", "camera", "hour_type", "weather", "time_band", "day_night", "road_form",
                        "difficulty", "lane_config", "cam_height", "quality", "resolution", "orientation")
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
    report_path = out_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    gt = report["gt_objects"]
    print("\n" + "=" * 70)
    print(f"완료: 이미지 {len(row_list):,}장 / 클립 {report['n_clips']}개 / CCTV {report['n_cameras']}대 / "
          f"제외 {len(skipped)}건 (기존 JPG 재사용 {n_reused:,}장)")
    print(f"  객체: car={gt['car']:,}  bus={gt['bus']:,}  truck={gt['truck']:,}")
    if src_bytes_total:
        print(f"  용량: PNG {src_bytes_total / 1e9:.1f}GB → JPG {jpg_bytes_total / 1e9:.1f}GB "
              f"({jpg_bytes_total * 100 / src_bytes_total:.0f}%)")
    if dropped_total:
        print(f"  제외된 폴리곤: {dict(dropped_total)}")
    for key in ("region", "day_night", "weather", "road_form", "difficulty"):
        print(f"  {key:11s}: {report['distributions'][key]}")
    if stale:
        print(f"  [warn] 이번 결과에 없는 이전 파일 {len(stale)}개가 images/labels 에 남아 있습니다.")
    print(f"\n  결과 폴더: {out_dir}")
    print("  images/ (JPG) · labels/ (YOLO seg) · manifest.csv · preprocess_report.json · classes.json")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-Hub 도로 CCTV 폴리곤 데이터 → JPG 이미지 + YOLO seg 라벨",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-root", nargs="+", default=[str(BASE_DIR)],
                   help="AI-Hub 다운로드 위치 — 폴더 또는 .tar 파일, 여러 개 가능")
    p.add_argument("--out-dir", default=str(BASE_DIR / "car_seg_dataset"), help="결과 출력 폴더")
    p.add_argument("--dry-run", action="store_true", help="조각 검사와 매칭 통계만 출력하고 종료")
    p.add_argument("--overwrite", action="store_true", help="이미 변환된 JPG 도 다시 만든다")
    p.add_argument("--jpg-quality", type=int, default=95, metavar="1-100",
                   help="JPEG 품질 (95: 픽셀 MAE 0.7/255 수준으로 학습 성능 손실 거의 없음)")
    p.add_argument("--no-verify", action="store_true", help="PNG 무결성(CRC) 검사 생략 — 조금 빨라짐")
    p.add_argument("--min-area", type=float, default=1.0, help="이보다 작은(px²) 폴리곤은 제외")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="JPG 변환 병렬 프로세스 수")
    return p


if __name__ == "__main__":
    sys.exit(preprocess(build_parser().parse_args()))
