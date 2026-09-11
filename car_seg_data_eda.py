#!/usr/bin/env python3
"""car_seg_data_preprocessing.py 로 정리한 YOLO seg 데이터셋(car_seg_dataset/) EDA.

입력
    car_seg_dataset/
      images/  labels/  manifest.csv

출력 (기본: car_seg_dataset/eda/)
    00_filename_tokens.md/.csv     파일명 토큰 사전 (토큰별 의미 + 실제 데이터에서 확인한 값·분포)
    01_class_distribution.png      클래스별 객체 수 / 클래스가 등장하는 이미지 비율
    02_metadata_distribution.png   파일명 메타데이터(지역·채널·날짜·시간대·요일·높이·NH/RH·차로·날씨·화질…) 분포
    03_objects_per_image.png       이미지당 객체 수 분포
    04_object_size.png             객체 크기 분포 (학습 해상도 기준 small/medium/large)
    05_vehicle_area_ratio.png      이미지 대비 차량 면적 비 — 속성별 비교
    06_centroid_heatmap.png        객체 중심점 분포 (가로/세로 영상 분리)
    07_vertices_per_polygon.png    폴리곤 꼭짓점 수 분포 (라벨 정밀도)
    08_frames_per_clip.png         클립당 프레임 수 분포
    09_samples.png                 라벨 오버레이 샘플 (좌표 정합성 육안 확인)
    10_nh_rh_check.png             NH / RH 의미 확인 (촬영 시각 분포 + 같은 카메라끼리 차량 수 비교)
    nh_rh_check.csv                10번의 카메라별 비교 표
    11_brightness.png              실제 픽셀 밝기 — 주/야간·촬영 시각·일몰 기준 시각·날씨별
    12_day_night_samples.png       주간 / 여명·황혼 / 야간 대표 이미지
    13_vehicle_overlap.png         차량 폴리곤 겹침 (단순 합 vs 합집합, 차량 수별)
    14_class_by_attribute.png      속성값별 car / bus / truck 비율
    eda_brightness.csv             11번에서 잰 이미지별 밝기·대비
    eda_per_image.csv              이미지별 계산값 (객체 수, 차량 면적 비 …)
    eda_summary.json               위 내용의 수치 요약 + 데이터 무결성 점검 결과

※ 라벨에는 도로 영역이 없으므로 여기서의 '차량 면적 비'는 '이미지 전체 대비' 값입니다.
   (프로젝트 최종 지표인 '도로 대비' 면적 비는 도로 마스크가 생긴 뒤 계산)

사용 예
    python car_seg_data_eda.py
    python car_seg_data_eda.py --dataset-dir car_seg_dataset --n-samples 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from tqdm import tqdm  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

CLASS_NAMES = {0: "car", 1: "bus", 2: "truck"}

# 층화/분포를 볼 파일명 메타데이터 (car_seg_data_preprocessing.py 의 manifest 컬럼)
META_ATTRS = [
    ("region", "권역 (zip)"),
    ("site", "촬영 지역"),
    ("channel", "채널 / 촬영 지점"),
    ("date", "촬영 날짜"),
    ("time_band", "촬영 시간대"),
    ("hour", "촬영 시각 (시)"),
    ("day_night", "주간 / 여명·황혼 / 야간 (일몰 기준)"),
    ("weekday", "요일"),
    ("cam_height", "카메라 높이"),
    ("hour_type", "NH / RH"),
    ("road_type", "도로 종류"),
    ("lane_config", "차로 구성"),
    ("weather", "날씨"),
    ("quality", "화질"),
    ("resolution", "해상도"),
]
WEEKDAY_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# 파일명 토큰 사전: (manifest 컬럼, 파일명 예시 토큰, 의미)
#   Suwon_CH01_20200722_1500_WED_9m_NH_highway_TW5_sunny_FHD_005.png
FILENAME_TOKENS = [
    ("site", "Suwon, Inje", "촬영 지역명 (수원, 인제 등)"),
    ("channel", "CH01~CH10, injetunnel2", "카메라 채널 번호 또는 촬영 지점(터널명 등)"),
    ("date", "20200722", "촬영 날짜 (YYYYMMDD)"),
    ("time", "1500", "촬영 시각 (HHMM, 24시간제)"),
    ("weekday", "WED, THU, SAT", "요일 (영문 약자)"),
    ("cam_height", "9m, 4.8m", "카메라 설치 높이"),
    ("hour_type", "NH / RH", "정의 미확인 코드 — 상행/하행(진행 방향) 구분 또는 혼잡 시간대(Rush Hour)로 추정. "
                             "같은 카메라에서도 촬영건마다 값이 바뀜 (10_nh_rh_check.png 참고)"),
    ("road_type", "highway", "도로 종류 (고속도로)"),
    ("lane_config", "TW5, OW5, OW2", "차로 방향·차로 수로 추정 (TW=양방향 / OW=편도 + 차로 수)"),
    ("weather", "sunny, rainy, snow", "촬영 당시 날씨"),
    ("quality", "FHD", "해상도 등급 (Full HD)"),
    ("frame", "_005.png", "해당 클립에서 추출한 프레임(이미지) 순번 — 이미지 파일명에만 있음"),
]

# 일반적인 출퇴근 시간 (NH/RH 가 혼잡 시간대 구분인지 확인할 때 사용)
RUSH_HOURS = {7, 8, 17, 18}

# 일출/일몰 기준 주·야간 (car_seg_data_preprocessing.day_night_of 가 계산)
SUN_PHASES = ["day", "twilight", "night"]
SUN_PHASE_KO = {"day": "주간", "twilight": "여명·황혼", "night": "야간"}
# 순서가 있는 값이라 한 색상(파랑)의 밝은→어두운 단계로 표현
SUN_COLORS = {"day": "#86b6ef", "twilight": "#2a78d6", "night": "#104281"}

# ----------------------------------------------------------------------------
# 차트 스타일 — 클래스 색은 개체(entity)에 고정: car=파랑, bus=주황, truck=청록
# ----------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
CLASS_COLORS = {"car": "#2a78d6", "bus": "#eb6834", "truck": "#1baf7a"}
BOX_FILL = "#cde2fb"
BOX_MEDIAN = "#184f95"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def setup_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic",
                          "Nanum Gothic", "Noto Sans CJK KR") if f in available]
    plt.rcParams.update({
        "font.family": korean[:1] + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 110,
    })
    if not korean:
        print("[warn] 한글 폰트를 찾지 못해 차트의 한글이 깨질 수 있습니다.", file=sys.stderr)


def header(fig, title: str, subtitle: str = "") -> float:
    """제목/부제를 인치 단위로 배치(그림 높이와 무관하게 겹치지 않음). tight_layout 의 top 값을 반환."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    if subtitle:
        fig.text(0.01, 1 - 0.46 / h, subtitle, ha="left", va="top", fontsize=10, color=TEXT_2)
    return 1 - (0.95 if subtitle else 0.6) / h


def save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def barh(ax, labels: list[str], values: list[float], colors=SERIES, annot: list[str] | None = None) -> None:
    """가로 막대 (값 라벨은 막대 끝, 본문 텍스트 색)."""
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    vmax = max(values) if len(values) else 1
    for yi, v, text in zip(y, values, annot or [f"{v:,.0f}" for v in values]):
        ax.text(v + vmax * 0.01, yi, text, va="center", ha="left", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, vmax * 1.28)


def hist(ax, data, bins, color=SERIES, log_x: bool = False) -> None:
    ax.hist(data, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.6)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 10⁻⁴ 대신 0.0001
    ax.grid(axis="y")
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# 라벨 파싱 / 계산
# ----------------------------------------------------------------------------


def polygon_area_centroid(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """(면적, 중심 x, 중심 y) — 면적 가중 중심. 퇴화 폴리곤은 꼭짓점 평균."""
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = xs * y2 - x2 * ys
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return 0.0, float(xs.mean()), float(ys.mean())
    cx = ((xs + x2) * cross).sum() / (6.0 * a)
    cy = ((ys + y2) * cross).sum() / (6.0 * a)
    return abs(a), float(cx), float(cy)


def analyze(df: pd.DataFrame, dataset_dir: Path, raster_scale: float, imgsz: int):
    """labels/*.txt 를 읽어 폴리곤 단위 표와 이미지 단위 계산값, 무결성 점검 결과를 만든다."""
    lbl_dir = dataset_dir / "labels"
    img_dir = dataset_dir / "images"
    polys: list[tuple] = []
    per_image: list[dict] = []
    issues: dict[str, list] = defaultdict(list)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="라벨 분석", unit="img", dynamic_ncols=True):
        image_id, W, H = row.image_id, int(row.width), int(row.height)
        if not (img_dir / row.file_name).exists():
            issues["missing_image"].append(row.file_name)
        label_path = lbl_dir / f"{image_id}.txt"
        if not label_path.exists():
            issues["missing_label"].append(image_id)
            continue

        # 전처리에서 XML 좌표계로 정규화했으므로, 픽셀 면적은 XML 크기 기준으로 복원
        xw, xh = int(row.xml_width or W), int(row.xml_height or H)
        train_scale = imgsz / max(xw, xh)
        mw, mh = max(1, round(xw * raster_scale)), max(1, round(xh * raster_scale))
        cover = np.zeros((mh, mw), dtype=np.uint16)  # 픽셀마다 몇 대의 차량 폴리곤이 덮고 있는지
        counts = {name: 0 for name in CLASS_NAMES.values()}

        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            tokens = line.split()
            if not tokens:
                continue
            try:
                cls = int(tokens[0])
                coords = np.asarray(tokens[1:], dtype=float)
            except ValueError:
                issues["unparsable_line"].append(f"{image_id}:{line_no}")
                continue
            if cls not in CLASS_NAMES:
                issues["unknown_class"].append(f"{image_id}:{line_no}")
                continue
            if coords.size < 6 or coords.size % 2:
                issues["bad_coord_count"].append(f"{image_id}:{line_no}")
                continue
            if coords.min() < 0 or coords.max() > 1:
                issues["coord_out_of_range"].append(f"{image_id}:{line_no}")

            xs, ys = coords[0::2], coords[1::2]
            a_norm, cx, cy = polygon_area_centroid(xs, ys)
            area_px = a_norm * xw * xh
            name = CLASS_NAMES[cls]
            counts[name] += 1
            polys.append((image_id, name, xs.size, a_norm, area_px, area_px * train_scale**2,
                          cx, cy, row.orientation))

            # 폴리곤의 bounding box 영역만 따로 칠해서 cover 에 더한다 (겹친 곳은 2, 3 …)
            px, py = xs * mw, ys * mh
            x0, x1 = max(0, int(px.min())), min(mw, int(np.ceil(px.max())) + 1)
            y0, y1 = max(0, int(py.min())), min(mh, int(np.ceil(py.max())) + 1)
            if x1 > x0 and y1 > y0:
                piece = Image.new("1", (x1 - x0, y1 - y0), 0)
                ImageDraw.Draw(piece).polygon(list(zip((px - x0).tolist(), (py - y0).tolist())), fill=1)
                cover[y0:y1, x0:x1] += np.asarray(piece, dtype=np.uint16)

        n_px = mw * mh
        union_px = int(np.count_nonzero(cover))
        sum_px = int(cover.sum())
        per_image.append({
            "image_id": image_id,
            "n_car": counts["car"],
            "n_bus": counts["bus"],
            "n_truck": counts["truck"],
            "n_objects": sum(counts.values()),
            "vehicle_area_ratio": union_px / n_px,    # 차량 합집합 면적 / 이미지 면적 (겹침 1번만)
            "vehicle_area_sum": sum_px / n_px,        # 폴리곤 면적 단순 합 / 이미지 면적 (겹침 중복)
            "overlap_ratio": 1 - union_px / sum_px if sum_px else 0.0,  # 단순 합 중 겹쳐서 중복된 비율
        })

    listed = set(df["image_id"])
    issues["orphan_label"] = sorted(p.stem for p in lbl_dir.glob("*.txt") if p.stem not in listed)
    listed_files = set(df["file_name"])
    issues["orphan_image"] = sorted(p.name for p in img_dir.iterdir() if p.is_file() and p.name not in listed_files)

    poly_df = pd.DataFrame(polys, columns=["image_id", "cls", "n_vertices", "area_norm", "area_px",
                                           "area_train_px", "cx", "cy", "orientation"])
    return poly_df, pd.DataFrame(per_image), issues


def display_value(attr: str, value) -> str:
    """차트에 표시할 값 이름 (주/야간은 한글로)."""
    return SUN_PHASE_KO.get(str(value), str(value)) if attr == "day_night" else str(value)


def ordered_counts(series: pd.Series, attr: str) -> pd.Series:
    counts = series.value_counts()
    if attr == "weekday":
        return counts.reindex([d for d in WEEKDAY_ORDER if d in counts.index])
    if attr in ("date", "time_band", "hour", "region"):
        return counts.sort_index()
    if attr == "cam_height":
        return counts.reindex(sorted(counts.index, key=lambda v: float(str(v).rstrip("m") or 0)))
    if attr == "day_night":
        return counts.reindex([p for p in SUN_PHASES if p in counts.index] +
                              [p for p in counts.index if p not in SUN_PHASES])
    return counts


# ----------------------------------------------------------------------------
# 차트
# ----------------------------------------------------------------------------


def plot_class_distribution(poly_df, img_df, n_images, out: Path) -> None:
    names = list(CLASS_NAMES.values())
    inst = [int((poly_df["cls"] == n).sum()) for n in names]
    with_cls = [int((img_df[f"n_{n}"] > 0).sum()) for n in names]
    colors = [CLASS_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    total = sum(inst) or 1
    barh(axes[0], names, inst, colors, [f"{v:,}  ({v / total:.1%})" for v in inst])
    axes[0].set_title("클래스별 객체(폴리곤) 수")
    barh(axes[1], names, with_cls, colors, [f"{v:,}장  ({v / n_images:.1%})" for v in with_cls])
    axes[1].set_title("클래스가 1개 이상 등장하는 이미지 수")
    top = header(fig, "클래스 분포", f"이미지 {n_images:,}장 · 객체 {total:,}개 — 막대 옆 값은 전체 대비 비율")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)


def plot_metadata(df: pd.DataFrame, top_channels: int, out: Path) -> dict:
    summary = {}
    attrs = [(a, t) for a, t in META_ATTRS if a in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(attrs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.3 * nrows))
    axes = axes.ravel()
    clips = df.drop_duplicates("clip")

    for ax, (attr, title) in zip(axes, attrs):
        img_counts = ordered_counts(df[attr].astype(str), attr)
        clip_counts = clips[attr].astype(str).value_counts()
        summary[attr] = {k: {"images": int(v), "clips": int(clip_counts.get(k, 0))} for k, v in img_counts.items()}

        if attr == "channel" and len(img_counts) > top_channels:
            rest = img_counts.iloc[top_channels:]
            img_counts = img_counts.iloc[:top_channels]
            img_counts[f"기타 {len(rest)}개"] = rest.sum()
            clip_counts = clip_counts.copy()
            clip_counts[f"기타 {len(rest)}개"] = clips[attr].isin(rest.index).sum()
            title = f"{title} (상위 {top_channels})"
        labels = [display_value(attr, k) for k in img_counts.index]
        labels = [k if len(k) <= 26 else k[:25] + "…" for k in labels]
        annot = [f"{v:,}장 · {clip_counts.get(k, 0)}클립" for k, v in img_counts.items()]
        barh(ax, labels, img_counts.values.tolist(), SERIES, annot)
        ax.set_title(f"{title} — {len(summary[attr])}종")

    for ax in axes[len(attrs):]:
        ax.set_visible(False)
    top = header(fig, "파일명 메타데이터 분포", "막대 = 이미지 수, 라벨 = 이미지 수 · 클립 수 (같은 클립의 프레임은 메타데이터가 동일)")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


def plot_objects_per_image(img_df: pd.DataFrame, out: Path) -> dict:
    cols = [("n_objects", "전체", SERIES)] + [(f"n_{n}", n, CLASS_COLORS[n]) for n in CLASS_NAMES.values()]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    summary = {}
    for ax, (col, name, color) in zip(axes, cols):
        v = img_df[col]
        hist(ax, v, bins=np.arange(0, v.max() + 2) - 0.5, color=color)
        ax.set_title(f"{name}  (평균 {v.mean():.1f}, 최대 {v.max()})")
        ax.set_xlabel("이미지당 객체 수")
        summary[name] = {"mean": round(float(v.mean()), 3), "median": float(v.median()), "max": int(v.max()),
                         "images_with_zero": int((v == 0).sum())}
    axes[0].set_ylabel("이미지 수")
    top = header(fig, "이미지당 객체 수")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


def plot_object_size(poly_df: pd.DataFrame, imgsz: int, out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.6), sharex=True)
    bins = np.logspace(-3, np.log10(60), 60)  # 이미지 대비 면적 %
    summary = {}
    for ax, name in zip(axes, CLASS_NAMES.values()):
        sub = poly_df[poly_df["cls"] == name]
        if sub.empty:
            ax.set_visible(False)
            continue
        a = sub["area_train_px"]
        sml = {"small(<32²)": float((a < 32**2).mean()), "medium": float(((a >= 32**2) & (a < 96**2)).mean()),
               "large(≥96²)": float((a >= 96**2).mean())}
        hist(ax, (sub["area_norm"] * 100).clip(lower=1e-3), bins=bins, color=CLASS_COLORS[name], log_x=True)
        ax.set_title(f"{name}  S {sml['small(<32²)']:.0%} · M {sml['medium']:.0%} · L {sml['large(≥96²)']:.0%}")
        ax.set_xlabel("이미지 대비 객체 면적 (%, log)")
        summary[name] = {**{k: round(v, 4) for k, v in sml.items()},
                         "median_area_pct": round(float(sub["area_norm"].median() * 100), 4)}
    axes[0].set_ylabel("객체 수")
    top = header(fig, "객체 크기 분포",
             f"S/M/L 은 학습 해상도 imgsz={imgsz} 로 줄였을 때의 픽셀 면적 기준 (COCO 32²/96² 임계값)")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


def plot_area_ratio(merged: pd.DataFrame, out: Path) -> dict:
    ratio = merged["vehicle_area_ratio"] * 100
    groups = [(a, t) for a, t in (("hour_type", "NH / RH"), ("weather", "날씨"), ("time_band", "시간대"),
                                  ("site", "촬영 지역"), ("lane_config", "차로 구성")) if a in merged.columns]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes = axes.ravel()

    hist(axes[0], ratio, bins=60)
    axes[0].axvline(ratio.median(), color=TEXT_2, linewidth=1, linestyle="--")
    axes[0].text(ratio.median(), axes[0].get_ylim()[1] * 0.95, f" 중앙값 {ratio.median():.2f}%",
                 color=TEXT_2, fontsize=8, va="top")
    axes[0].set_title("전체 분포")
    axes[0].set_xlabel("차량 면적 / 이미지 면적 (%)")
    axes[0].set_ylabel("이미지 수")

    summary = {"overall": {"mean": round(float(ratio.mean()), 4), "median": round(float(ratio.median()), 4),
                           "p95": round(float(ratio.quantile(0.95)), 4), "max": round(float(ratio.max()), 4)}}
    for ax, (attr, title) in zip(axes[1:], groups):
        order = merged.groupby(attr)["vehicle_area_ratio"].median().sort_values(ascending=False).index.tolist()
        if attr in ("time_band",):
            order = sorted(order)
        data = [merged.loc[merged[attr] == k, "vehicle_area_ratio"].values * 100 for k in order]
        ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                   boxprops={"facecolor": BOX_FILL, "edgecolor": SERIES},
                   medianprops={"color": BOX_MEDIAN, "linewidth": 2},
                   whiskerprops={"color": SERIES}, capprops={"color": SERIES},
                   flierprops={"marker": "o", "markersize": 2, "markerfacecolor": TEXT_2,
                               "markeredgecolor": "none", "alpha": 0.4})
        ax.set_yticks(range(1, len(order) + 1), [f"{k}  (n={len(d):,})" for k, d in zip(order, data)])
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{title}별")
        ax.set_xlabel("차량 면적 비 (%)")
        summary[attr] = {
            str(k): {"n_images": int(len(d)), "mean": round(float(np.mean(d)), 4),
                     "median": round(float(np.median(d)), 4),
                     "mean_objects": round(float(merged.loc[merged[attr] == k, "n_objects"].mean()), 3)}
            for k, d in zip(order, data)
        }
    for ax in axes[1 + len(groups):]:
        ax.set_visible(False)

    top = header(fig, "이미지 대비 차량 면적 비",
             "차량 폴리곤 합집합 면적 / 이미지 면적 — 라벨에 도로가 없어 '도로 대비'가 아닌 '이미지 대비' 값")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


def plot_centroids(poly_df: pd.DataFrame, out: Path) -> None:
    groups = [(o, poly_df[poly_df["orientation"] == o]) for o in ("landscape", "portrait")]
    groups = [(o, g) for o, g in groups if not g.empty]
    if not groups:
        return
    grids = []
    for orient, g in groups:
        bx, by = (64, 36) if orient == "landscape" else (36, 64)
        h, _, _ = np.histogram2d(g["cy"], g["cx"], bins=(by, bx), range=((0, 1), (0, 1)))
        grids.append(h / h.sum() * 100)  # 셀별 객체 비율(%) — 가로/세로 영상을 같은 척도로 비교
    vmax = max(g.max() for g in grids)

    widths = [16 / 9 if o == "landscape" else 9 / 16 for o, _ in groups]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * sum(widths) + 2.2, 5.6),
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ax, (orient, g), grid in zip(axes[0], groups, grids):
        im = ax.imshow(grid, cmap=SEQ_BLUE, vmin=0, vmax=vmax, extent=(0, 1, 1, 0),
                       aspect=(9 / 16 if orient == "landscape" else 16 / 9), interpolation="nearest")
        ax.set_title(f"{'가로' if orient == 'landscape' else '세로'} 영상 — 객체 {len(g):,}개")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    top = header(fig, "객체 중심점 분포", "정규화 좌표 (0,0)=좌상단 — 진할수록 차량이 자주 지나가는 위치 (도로 영역의 근사)")
    fig.tight_layout(rect=(0, 0, 0.92, top))
    cax = fig.add_axes((0.935, 0.18, 0.015, 0.55))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("셀별 객체 비율 (%)", color=TEXT_2)
    cb.outline.set_visible(False)
    save(fig, out)


def plot_vertices(poly_df: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.4), sharex=True)
    hi = int(poly_df["n_vertices"].quantile(0.995)) + 2
    summary = {}
    for ax, name in zip(axes, CLASS_NAMES.values()):
        v = poly_df.loc[poly_df["cls"] == name, "n_vertices"]
        if v.empty:
            ax.set_visible(False)
            continue
        hist(ax, v.clip(upper=hi), bins=np.arange(3, hi + 2) - 0.5, color=CLASS_COLORS[name])
        ax.set_title(f"{name}  (중앙값 {v.median():.0f}, 최대 {v.max()})")
        ax.set_xlabel("폴리곤 꼭짓점 수")
        summary[name] = {"median": float(v.median()), "mean": round(float(v.mean()), 2), "max": int(v.max())}
    axes[0].set_ylabel("객체 수")
    top = header(fig, "폴리곤 꼭짓점 수", f"상위 0.5%는 {hi}에 합쳐 표시")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


def plot_frames_per_clip(df: pd.DataFrame, out: Path) -> dict:
    per_clip = df.groupby("clip").size()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    hist(ax, per_clip, bins=30)
    ax.set_xlabel("클립당 이미지(프레임) 수")
    ax.set_ylabel("클립 수")
    ax.set_title(f"클립 {len(per_clip)}개 · 평균 {per_clip.mean():.0f}장 · 최소 {per_clip.min()} · 최대 {per_clip.max()}")
    top = header(fig, "클립당 프레임 수", "같은 클립의 프레임은 서로 매우 비슷하므로 분할은 클립 단위로 해야 누수가 없음")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return {"n_clips": int(len(per_clip)), "mean": round(float(per_clip.mean()), 2),
            "min": int(per_clip.min()), "max": int(per_clip.max())}


def plot_samples(df: pd.DataFrame, dataset_dir: Path, n: int, seed: int, out: Path) -> None:
    """지역별로 돌아가며 서로 다른 클립에서 1장씩 뽑아 라벨을 겹쳐 그린다."""
    if n <= 0:
        return
    rng = np.random.default_rng(seed)
    by_site = {s: g.drop_duplicates("clip").sample(frac=1, random_state=seed) for s, g in df.groupby("site")}
    picks = []
    while len(picks) < n and any(len(g) for g in by_site.values()):
        for site in rng.permutation(list(by_site)):
            g = by_site[site]
            if len(g) and len(picks) < n:
                clip = g.iloc[0]["clip"]
                by_site[site] = g.iloc[1:]
                picks.append(df[df["clip"] == clip].sample(1, random_state=seed).iloc[0])

    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), squeeze=False)
    for ax, row in zip(axes.ravel(), picks):
        with Image.open(dataset_dir / "images" / row["file_name"]) as im:
            im = im.convert("RGB")
            W, H = im.size
            im.thumbnail((640, 640))
            w, h = im.size
            ax.imshow(im)
        label_path = dataset_dir / "labels" / f"{row['image_id']}.txt"
        n_obj = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) < 7:
                continue
            color = CLASS_COLORS.get(CLASS_NAMES.get(int(t[0]), ""), TEXT_2)
            xy = np.asarray(t[1:], dtype=float).reshape(-1, 2) * [w, h]
            ax.add_patch(Polygon(xy, closed=True, facecolor=color, alpha=0.35, edgecolor="none"))
            ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.2))
            n_obj += 1
        ax.set_title(f"{row['site']} · {row['weather']} · {row['hour_type']} · {W}x{H} · {n_obj}개",
                     fontsize=9, fontweight="normal")
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    handles = [Patch(facecolor=c, label=n) for n, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=10)
    top = header(fig, "라벨 오버레이 샘플", "지역별로 서로 다른 클립에서 1장씩 — 폴리곤이 차량에 딱 맞으면 좌표 변환이 정상")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)


# ----------------------------------------------------------------------------
# 밝기(픽셀) · 주/야간 · 차량 겹침 · 클래스 × 속성
# ----------------------------------------------------------------------------


def _brightness(path: str) -> tuple[float, float]:
    """(평균 밝기, 대비) — 0~255 흑백 기준. 1/4 로 줄여 계산 (평균값은 거의 같음)."""
    try:
        with Image.open(path) as im:
            gray = np.asarray(im.reduce(4).convert("L"), dtype=np.float32)
        return float(gray.mean()), float(gray.std())
    except (OSError, ValueError):
        return float("nan"), float("nan")


def measure_brightness(df: pd.DataFrame, dataset_dir: Path, per_clip: int, workers: int, seed: int) -> pd.DataFrame:
    """이미지 픽셀 밝기 측정. per_clip>0 이면 클립마다 그 수만큼만 샘플링 (같은 클립은 밝기가 거의 같음)."""
    sample = df if per_clip <= 0 else df.sample(frac=1, random_state=seed).groupby("clip").head(per_clip)
    paths = [str(dataset_dir / "images" / f) for f in sample["file_name"]]
    desc = f"밝기 측정 ({'전체' if per_clip <= 0 else f'클립당 {per_clip}장'})"
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            values = list(tqdm(ex.map(_brightness, paths, chunksize=8), total=len(paths), desc=desc,
                               unit="img", dynamic_ncols=True))
    else:
        values = [_brightness(p) for p in tqdm(paths, desc=desc, unit="img", dynamic_ncols=True)]
    out = sample[["image_id"]].copy()
    out["brightness"] = [v[0] for v in values]
    out["contrast"] = [v[1] for v in values]
    return out.dropna()


def plot_brightness(bdf: pd.DataFrame, out: Path) -> dict:
    """bdf: 밝기 + 메타데이터 (이미지 단위)."""
    phases = [p for p in SUN_PHASES if p in set(bdf["day_night"])]
    clip = bdf.groupby("clip").agg(brightness=("brightness", "mean"), day_night=("day_night", "first"),
                                   time=("time", "first"), min_from_sunset=("min_from_sunset", "first"),
                                   weather=("weather", "first")).reset_index()
    clip["hour_f"] = clip["time"].str[:2].astype(int) + clip["time"].str[2:4].astype(int) / 60
    clip["min_from_sunset"] = pd.to_numeric(clip["min_from_sunset"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    # (1) 전체 분포 — 주/야간 누적
    ax = axes[0, 0]
    ax.hist([bdf.loc[bdf["day_night"] == p, "brightness"] for p in phases], bins=48, range=(0, 255), stacked=True,
            color=[SUN_COLORS[p] for p in phases], edgecolor=SURFACE, linewidth=0.5,
            label=[f"{SUN_PHASE_KO[p]} ({(bdf['day_night'] == p).sum():,}장)" for p in phases])
    ax.set_title("이미지 평균 밝기 분포")
    ax.set_xlabel("평균 밝기 (0 = 검정, 255 = 흰색)")
    ax.set_ylabel("이미지 수")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    # (2)(3) 클립 평균 밝기 vs 시각 / 일몰 기준 시각
    for ax, xcol, title, xlabel in (
        (axes[0, 1], "hour_f", "촬영 시각 vs 밝기 — 같은 시각이라도 날짜에 따라 밝기가 다름", "촬영 시각 (시)"),
        (axes[1, 0], "min_from_sunset", "일몰 기준 시각 vs 밝기 — 0 = 일몰", "일몰까지 남은(−) / 지난(+) 시간 (분)"),
    ):
        if xcol == "min_from_sunset":
            ax.axvspan(-30, 30, color="#f0efec", zorder=0)
            ax.axvline(0, color=TEXT_2, linewidth=1, linestyle="--")
        for p in phases:
            sub = clip[clip["day_night"] == p]
            ax.scatter(sub[xcol], sub["brightness"], s=36, color=SUN_COLORS[p], edgecolor=SURFACE, linewidth=1,
                       label=f"{SUN_PHASE_KO[p]} ({len(sub)}클립)", zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("클립 평균 밝기")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left")

    # (4) 날씨별
    ax = axes[1, 1]
    order = bdf.groupby("weather")["brightness"].median().sort_values(ascending=False).index.tolist()
    data = [bdf.loc[bdf["weather"] == w, "brightness"].values for w in order]
    ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
               boxprops={"facecolor": BOX_FILL, "edgecolor": SERIES}, medianprops={"color": BOX_MEDIAN, "linewidth": 2},
               whiskerprops={"color": SERIES}, capprops={"color": SERIES},
               flierprops={"marker": "o", "markersize": 2, "markerfacecolor": TEXT_2, "markeredgecolor": "none"})
    ax.set_yticks(range(1, len(order) + 1), [f"{w}  (n={len(d):,})" for w, d in zip(order, data)])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("날씨별 밝기")
    ax.set_xlabel("평균 밝기")

    top = header(fig, "이미지 밝기 (실제 픽셀값)",
                 "주간/여명·황혼/야간은 날짜·시각·지역으로 계산한 일출·일몰 기준 — 밝기가 이 구분과 맞는지 확인")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return {
        "n_measured_images": int(len(bdf)),
        "by_day_night": {p: {"n_images": int((bdf["day_night"] == p).sum()),
                             "median": round(float(bdf.loc[bdf["day_night"] == p, "brightness"].median()), 1)}
                         for p in phases},
        "by_weather": {w: round(float(np.median(d)), 1) for w, d in zip(order, data)},
        "darkest_clips": clip.nsmallest(5, "brightness")[["clip", "brightness", "day_night"]]
                             .round(1).to_dict(orient="records"),
    }


def plot_day_night_samples(df: pd.DataFrame, bdf: pd.DataFrame, dataset_dir: Path, per_phase: int, seed: int,
                           out: Path) -> None:
    """주간 / 여명·황혼 / 야간 별로 서로 다른 클립의 대표 이미지."""
    phases = [p for p in SUN_PHASES if p in set(df["day_night"])]
    if not phases or per_phase <= 0:
        return
    bright = dict(zip(bdf["image_id"], bdf["brightness"])) if not bdf.empty else {}
    fig, axes = plt.subplots(len(phases), per_phase, figsize=(4.2 * per_phase, 2.9 * len(phases) + 0.8),
                             squeeze=False)
    for r, phase in enumerate(phases):
        pool = df[df["day_night"] == phase]
        if bright:  # 밝기를 잰 이미지 중에서 고르면 제목에 밝기를 표시할 수 있다
            measured = pool[pool["image_id"].isin(bright)]
            pool = measured if not measured.empty else pool
        picks = pool.drop_duplicates("clip").sample(frac=1, random_state=seed).drop_duplicates("site")
        picks = pd.concat([picks, pool.drop_duplicates("clip").sample(frac=1, random_state=seed)]) \
                  .drop_duplicates("clip").head(per_phase)
        for c in range(per_phase):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks):
                continue
            row = picks.iloc[c]
            with Image.open(dataset_dir / "images" / row["file_name"]) as im:
                im = im.convert("RGB")
                im.thumbnail((480, 480))
                ax.imshow(im)
            b = bright.get(row["image_id"])
            ax.set_title(f"{SUN_PHASE_KO[phase]} · {row['site']} {row['date'][4:]} {row['time']}"
                         + (f" · 밝기 {b:.0f}" if b is not None else ""), fontsize=9, fontweight="normal")
    top = header(fig, "주간 / 여명·황혼 / 야간 샘플", "행마다 서로 다른 클립 — 야간은 헤드라이트·반사광 때문에 분할이 가장 어려운 조건")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)


def plot_overlap(merged: pd.DataFrame, out: Path) -> dict:
    """차량 폴리곤 겹침 — 단순 합 vs 합집합."""
    multi = merged[merged["n_objects"] >= 2]
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 16, "9-16"), (17, 32, "17-32"),
            (33, 10**6, "33+")]
    groups = [(label, merged.loc[merged["n_objects"].between(lo, hi), "overlap_ratio"].values * 100)
              for lo, hi, label in bins]
    groups = [(label, v) for label, v in groups if len(v)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ov = multi["overlap_ratio"] * 100
    hist(ax, ov.clip(upper=ov.quantile(0.995)) if len(ov) else ov, bins=50)  # 극단값 0.5% 는 끝에 합침
    ax.set_title("겹침 비율 분포 (차량 2대 이상 이미지)")
    ax.set_xlabel("겹침 비율 (%) = 1 − 합집합 / 단순 합")
    ax.set_ylabel("이미지 수")

    ax = axes[1]
    ax.boxplot([v for _, v in groups], widths=0.55, patch_artist=True, showfliers=False,
               boxprops={"facecolor": BOX_FILL, "edgecolor": SERIES}, medianprops={"color": BOX_MEDIAN, "linewidth": 2},
               whiskerprops={"color": SERIES}, capprops={"color": SERIES})
    ax.set_xticks(range(1, len(groups) + 1), [f"{label}\n(n={len(v):,})" for label, v in groups])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("차량이 많을수록 겹침이 커지는가")
    ax.set_xlabel("이미지당 차량 수")
    ax.set_ylabel("겹침 비율 (%)")

    ax = axes[2]
    x, y = merged["vehicle_area_sum"] * 100, merged["vehicle_area_ratio"] * 100
    lim = float(max(x.max(), y.max()) * 1.03) if len(x) else 1.0
    ax.plot([0, lim], [0, lim], color=TEXT_2, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(x, y, s=4, color=SERIES, alpha=0.25, edgecolor="none", zorder=2)
    ax.text(lim * 0.97, lim * 0.97, "겹침이 없으면 이 선 위", ha="right", va="top", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("단순 합 vs 합집합 면적 비")
    ax.set_xlabel("폴리곤 면적 단순 합 / 이미지 (%)")
    ax.set_ylabel("합집합 면적 / 이미지 (%)")

    top = header(fig, "차량 폴리곤 겹침",
                 "혼잡할수록 겹침이 커진다면, 혼잡도 지표(차량 면적 비)는 단순 합이 아닌 합집합(겹침 제외)으로 계산해야 함")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return {
        "mean_overlap_pct_multi": round(float(multi["overlap_ratio"].mean() * 100), 3) if len(multi) else 0.0,
        "images_overlap_over_5pct": int((merged["overlap_ratio"] > 0.05).sum()),
        "median_overlap_pct_by_objects": {label: round(float(np.median(v)), 3) for label, v in groups},
    }


def plot_class_by_attribute(merged: pd.DataFrame, out: Path) -> dict:
    """속성값별 car/bus/truck 객체 비율 (100% 누적 막대)."""
    attrs = [(a, t) for a, t in (("site", "촬영 지역"), ("lane_config", "차로 구성"), ("time_band", "시간대"),
                                  ("day_night", "주 / 야간"), ("weather", "날씨"), ("cam_height", "카메라 높이"),
                                  ("hour_type", "NH / RH"), ("orientation", "영상 방향"))
             if a in merged.columns and merged[a].nunique() > 1]
    names = list(CLASS_NAMES.values())
    ncols = 2
    nrows = int(np.ceil(len(attrs) / ncols))
    heights = []
    for r in range(nrows):
        heights.append(max(merged[attrs[i][0]].nunique() for i in range(r * ncols, min(len(attrs), r * ncols + ncols))))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, sum(0.34 * h + 1.0 for h in heights) + 1.2),
                             gridspec_kw={"height_ratios": [0.34 * h + 1.0 for h in heights]}, squeeze=False)
    summary = {}
    for ax, (attr, title) in zip(axes.ravel(), attrs):
        counts = merged.groupby(attr)[[f"n_{n}" for n in names]].sum()
        counts = counts.reindex(ordered_counts(merged[attr], attr).index)
        share = counts.div(counts.sum(1).replace(0, 1), axis=0) * 100
        y = np.arange(len(share))
        left = np.zeros(len(share))
        for n in names:
            vals = share[f"n_{n}"].values
            ax.barh(y, vals, left=left, height=0.64, color=CLASS_COLORS[n], edgecolor=SURFACE, linewidth=2, label=n)
            for yi, (l, v) in enumerate(zip(left, vals)):
                if v >= 7:
                    ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            left += vals
        ax.set_yticks(y, [f"{display_value(attr, k)}  ({int(t):,})" for k, t in zip(share.index, counts.sum(1))])
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel("클래스 비율 (%) — 괄호 = 객체 수")
        summary[attr] = share.round(2).to_dict(orient="index")
    for ax in axes.ravel()[len(attrs):]:
        ax.set_visible(False)
    handles = [Patch(facecolor=CLASS_COLORS[n], label=n) for n in names]
    h = fig.get_figheight()
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1 - 0.1 / h))
    top = header(fig, "속성별 클래스 비율", "특정 지역·조건에 bus/truck 이 몰려 있으면 모델이 '조건 = 차종' 으로 편향되게 배울 수 있음")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return summary


# ----------------------------------------------------------------------------
# 파일명 토큰 사전 / NH·RH 확인
# ----------------------------------------------------------------------------


def _value_list(counts: pd.Series, limit: int = 8) -> str:
    items = [f"{k}({v})" for k, v in counts.items()]
    return ", ".join(items[:limit]) + (f" 외 {len(items) - limit}종" if len(items) > limit else "")


def token_observation(df: pd.DataFrame, clips: pd.DataFrame, attr: str) -> str:
    """토큰별로 실제 데이터에서 확인한 내용 (괄호 안 숫자 = 클립 수)."""
    s = clips[attr].astype(str)
    if attr == "date":
        return f"{s.min()} ~ {s.max()}, {s.nunique()}일"
    if attr == "time":
        bands = clips["time_band"].value_counts().sort_index() if "time_band" in clips else pd.Series(dtype=int)
        return f"{s.min()} ~ {s.max()} · 3시간 구간별: {_value_list(bands)}"
    if attr == "channel":
        per_site = clips.groupby("site")["channel"].nunique()
        ch_sites = sorted(clips.loc[s.str.fullmatch(r"CH\d+"), "site"].unique())
        return (f"지역당 {per_site.min()}~{per_site.max()}개 · 'CHxx' 형식은 {', '.join(ch_sites) or '없음'}, "
                f"나머지 지역은 지점명(교량·터널 등)")
    if attr == "weekday":
        counts = s.value_counts()
        return _value_list(counts.reindex([d for d in WEEKDAY_ORDER if d in counts.index]))
    if attr == "cam_height":
        counts = s.value_counts()
        return _value_list(counts.reindex(sorted(counts.index, key=lambda v: float(v.rstrip("m") or 0))))
    if attr == "hour_type":
        both = sum(1 for _, g in clips.groupby(["site", "channel"]) if g["hour_type"].nunique() > 1)
        return f"{_value_list(s.value_counts())} · 같은 카메라에서 NH·RH 가 모두 나오는 곳 {both}개"
    if attr == "lane_config":
        direction = s.str[:2].value_counts()
        lanes = sorted({int(v[2:]) for v in s if v[2:].isdigit()})
        return (f"{_value_list(s.value_counts())} · TW {direction.get('TW', 0)} / OW {direction.get('OW', 0)}클립, "
                f"차로 수 {lanes}")
    if attr == "quality" and "resolution" in df.columns:
        ct = df.groupby(["quality", "resolution"]).size()
        return " / ".join(f"{q} → " + ", ".join(f"{r} {n:,}장" for r, n in ct.loc[q].items())
                          for q in ct.index.get_level_values(0).unique())
    if attr == "frame":
        frame = pd.to_numeric(df["frame"], errors="coerce")
        per_clip = df.groupby("clip").size()
        return f"{int(frame.min()):03d} ~ {int(frame.max()):03d} · 클립당 {per_clip.min()}~{per_clip.max()}장"
    if s.nunique() == 1:
        return f"전부 {s.iat[0]} (값이 하나뿐 → 분할 층화에서 제외)"
    return _value_list(s.value_counts())


def build_token_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """파일명 토큰 사전을 .csv(엑셀용)와 .md(보고서용)로 저장."""
    clips = df.drop_duplicates("clip")
    rows = []
    for order, (attr, example, meaning) in enumerate(FILENAME_TOKENS, 1):
        if attr not in df.columns:
            continue
        counts = df[attr].astype(str).value_counts()
        clip_counts = clips[attr].astype(str).value_counts()
        rows.append({
            "순서": order,
            "파일명 토큰 예시": example,
            "manifest 컬럼": attr,
            "의미": meaning,
            "고유값 수": int(df[attr].nunique()),
            "데이터 확인 결과": token_observation(df, clips, attr),
            "값 분포 (이미지 수/클립 수)": "" if attr == "frame" else " · ".join(
                f"{k} {v:,}/{clip_counts.get(k, 0)}" for k, v in counts.items()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "00_filename_tokens.csv", index=False, encoding="utf-8-sig")

    cols = ["순서", "파일명 토큰 예시", "의미", "고유값 수", "데이터 확인 결과"]
    lines = [
        "# 파일명 토큰 사전",
        "",
        "`Suwon_CH01_20200722_1500_WED_9m_NH_highway_TW5_sunny_FHD_005.png` 를 `_` 로 나눈 12개 토큰.  ",
        f"데이터: 이미지 {len(df):,}장 / 클립 {df['clip'].nunique()}개 — 괄호 안 숫자는 클립 수.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *["| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |" for r in rows],
        "",
        "값별 전체 분포(이미지 수/클립 수)는 `00_filename_tokens.csv` 의 마지막 열에 있습니다.",
        "",
    ]
    (out_dir / "00_filename_tokens.md").write_text("\n".join(lines), encoding="utf-8")
    print("  saved 00_filename_tokens.md / .csv")
    return table


def plot_nh_rh_check(merged: pd.DataFrame, out: Path) -> dict:
    """NH/RH 가 혼잡 시간대 구분이라면 RH 는 출퇴근 시간에 몰리고, 같은 카메라에서도 차량이 더 많아야 한다."""
    if "hour_type" not in merged.columns or merged["hour_type"].nunique() < 2 or "hour" not in merged.columns:
        return {}
    clips = merged.groupby("clip").agg(
        site=("site", "first"), channel=("channel", "first"), hour_type=("hour_type", "first"),
        hour=("hour", "first"), objects=("n_objects", "mean"), area_pct=("vehicle_area_ratio", "mean"),
    ).reset_index()
    clips["hour"] = pd.to_numeric(clips["hour"], errors="coerce")
    clips["area_pct"] *= 100
    types = sorted(clips["hour_type"].unique())
    colors = dict(zip(types, [CLASS_COLORS["car"], CLASS_COLORS["bus"], CLASS_COLORS["truck"]]))

    # 같은 카메라(지역·채널)에서 NH·RH 가 모두 촬영된 경우만 짝지어 비교 — 지역 차이를 걷어낸다
    pairs = []
    for (site, ch), g in clips.groupby(["site", "channel"]):
        if not {"NH", "RH"} <= set(g["hour_type"]):
            continue
        m = g.groupby("hour_type")[["objects", "area_pct"]].mean()
        hrs = g.groupby("hour_type")["hour"].apply(lambda h: ",".join(f"{int(v):02d}" for v in sorted(h)))
        pairs.append({
            "camera": f"{site} · {ch}",
            "NH_objects": round(float(m.at["NH", "objects"]), 2), "RH_objects": round(float(m.at["RH", "objects"]), 2),
            "NH_area_pct": round(float(m.at["NH", "area_pct"]), 3), "RH_area_pct": round(float(m.at["RH", "area_pct"]), 3),
            "NH_hours": hrs["NH"], "RH_hours": hrs["RH"],
            "NH_clips": int((g["hour_type"] == "NH").sum()), "RH_clips": int((g["hour_type"] == "RH").sum()),
        })
    pair_df = pd.DataFrame(pairs)
    if not pair_df.empty:
        pair_df = pair_df.assign(diff=pair_df["RH_objects"] - pair_df["NH_objects"]).sort_values("diff")
        pair_df = pair_df.drop(columns="diff").reset_index(drop=True)
        pair_df.to_csv(out.with_name("nh_rh_check.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4.8, 0.42 * len(pair_df) + 2.4)),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    # (좌) 촬영 시각별 클립 수 — 출퇴근 시간은 회색 음영
    ax = axes[0]
    hours = list(range(int(clips["hour"].min()), int(clips["hour"].max()) + 1))
    counts = clips.groupby(["hour", "hour_type"]).size().unstack(fill_value=0).reindex(hours, fill_value=0)
    x = np.arange(len(hours))
    for i, h in enumerate(hours):
        if h in RUSH_HOURS:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0)
    width = 0.8 / len(types)
    for k, ht in enumerate(types):
        ax.bar(x + (k - (len(types) - 1) / 2) * width, counts[ht] if ht in counts else 0, width=width * 0.95,
               color=colors[ht], edgecolor=SURFACE, linewidth=1, zorder=2,
               label=f"{ht} ({int((clips['hour_type'] == ht).sum())}클립)")
    ax.set_xticks(x, [f"{h:02d}" for h in hours])
    ax.set_xlabel("촬영 시각 (시)  — 회색 음영 = 출퇴근 시간 07-08시 · 17-18시")
    ax.set_ylabel("클립 수")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    rush_share = {ht: float(clips.loc[clips["hour_type"] == ht, "hour"].isin(RUSH_HOURS).mean()) for ht in types}
    ax.set_title("촬영 시각별 클립 수 — 출퇴근 시간 비율 " + " · ".join(f"{ht} {v:.0%}" for ht, v in rush_share.items()))

    # (우) 같은 카메라의 NH vs RH 이미지당 객체 수
    ax = axes[1]
    n_more = 0
    if pair_df.empty:
        ax.text(0.5, 0.5, "NH 와 RH 가 함께 있는 카메라가 없음", ha="center", va="center", color=TEXT_2)
        ax.axis("off")
    else:
        y = np.arange(len(pair_df))
        lo = pair_df[["NH_objects", "RH_objects"]].min(axis=1)
        hi = pair_df[["NH_objects", "RH_objects"]].max(axis=1)
        ax.hlines(y, lo, hi, color=GRID, linewidth=3, zorder=1)
        for ht in ("NH", "RH"):
            ax.scatter(pair_df[f"{ht}_objects"], y, s=60, color=colors[ht], edgecolor=SURFACE, linewidth=1.5,
                       zorder=3, label=ht)
        ax.set_yticks(y, [f"{c}  (RH {h}시)" for c, h in zip(pair_df["camera"], pair_df["RH_hours"])])
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("이미지당 평균 객체 수")
        ax.legend(loc="upper right")
        n_more = int((pair_df["RH_objects"] > pair_df["NH_objects"]).sum())
        ax.set_title(f"같은 카메라끼리 비교 — RH 쪽 차량이 더 많은 카메라 {n_more}/{len(pair_df)}개")

    top = header(fig, "NH / RH 의미 확인",
                 "RH 가 혼잡 시간대(Rush Hour)라면 출퇴근 시간(음영)에 몰리고, 같은 카메라에서도 차량이 더 많아야 함")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, out)
    return {
        "rush_hours": sorted(RUSH_HOURS),
        "rush_hour_clip_share": {k: round(v, 3) for k, v in rush_share.items()},
        "clip_hours": {ht: sorted(int(h) for h in clips.loc[clips["hour_type"] == ht, "hour"]) for ht in types},
        "cameras_with_both": int(len(pair_df)),
        "cameras_rh_more_objects": n_more,
        "overall_mean_objects": {ht: round(float(clips.loc[clips["hour_type"] == ht, "objects"].mean()), 2)
                                 for ht in types},
        "pairs": pair_df.to_dict(orient="records"),
    }


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dataset_dir / "eda"
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        print(f"[error] manifest.csv 가 없습니다: {manifest}\n"
              f"        먼저 car_seg_data_preprocessing.py 를 실행하세요.", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # 문자열로 읽어야 '0956' 같은 시각/날짜의 앞자리 0 이 보존된다
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in ("width", "height"):
        df[col] = df[col].astype(int)
    if "time" in df.columns:
        df["hour"] = df["time"].str[:2]  # HHMM → HH
    if {"date", "time", "site"} <= set(df.columns) and ("day_night" not in df.columns or (df["day_night"] == "").all()):
        # 주/야간 컬럼이 없는 예전 manifest 면 여기서 계산 (전처리를 다시 돌릴 필요 없음)
        from car_seg_data_preprocessing import day_night_of
        phases = [day_night_of(d, t, s) for d, t, s in zip(df["date"], df["time"], df["site"])]
        df["day_night"] = [p for p, _ in phases]
        df["min_from_sunset"] = [m for _, m in phases]
    print(f"[EDA] {dataset_dir}  — 이미지 {len(df):,}장 / 클립 {df['clip'].nunique()}개")

    poly_df, img_df, issues = analyze(df, dataset_dir, args.raster_scale, args.imgsz)
    merged = df.merge(img_df, on="image_id", how="inner")
    if merged.empty:
        print("[error] 분석할 라벨이 없습니다.", file=sys.stderr)
        return 1

    print(f"\n[EDA] 차트 저장 → {out_dir}")
    token_table = build_token_table(df, out_dir)
    plot_class_distribution(poly_df, img_df, len(merged), out_dir / "01_class_distribution.png")
    meta_summary = plot_metadata(df, args.top_channels, out_dir / "02_metadata_distribution.png")
    per_image_summary = plot_objects_per_image(img_df, out_dir / "03_objects_per_image.png")
    size_summary = plot_object_size(poly_df, args.imgsz, out_dir / "04_object_size.png")
    ratio_summary = plot_area_ratio(merged, out_dir / "05_vehicle_area_ratio.png")
    plot_centroids(poly_df, out_dir / "06_centroid_heatmap.png")
    vertex_summary = plot_vertices(poly_df, out_dir / "07_vertices_per_polygon.png")
    clip_summary = plot_frames_per_clip(df, out_dir / "08_frames_per_clip.png")
    plot_samples(df, dataset_dir, args.n_samples, args.seed, out_dir / "09_samples.png")
    nh_rh_summary = plot_nh_rh_check(merged, out_dir / "10_nh_rh_check.png")

    brightness_summary: dict = {}
    bdf = pd.DataFrame(columns=["image_id", "brightness", "contrast"])
    if not args.no_brightness and "day_night" in df.columns:
        bdf = measure_brightness(df, dataset_dir, args.brightness_per_clip, args.workers, args.seed)
        if not bdf.empty:
            brightness_summary = plot_brightness(bdf.merge(df, on="image_id"), out_dir / "11_brightness.png")
            bdf.to_csv(out_dir / "eda_brightness.csv", index=False, encoding="utf-8-sig")
    if "day_night" in df.columns:
        plot_day_night_samples(df, bdf, dataset_dir, 4, args.seed, out_dir / "12_day_night_samples.png")
    overlap_summary = plot_overlap(merged, out_dir / "13_vehicle_overlap.png")
    class_attr_summary = plot_class_by_attribute(merged, out_dir / "14_class_by_attribute.png")

    img_df.to_csv(out_dir / "eda_per_image.csv", index=False, encoding="utf-8-sig")
    constant = [a for a, _ in META_ATTRS if a in df.columns and df[a].nunique() == 1]
    summary = {
        "dataset_dir": str(dataset_dir),
        "n_images": int(len(df)),
        "n_clips": int(df["clip"].nunique()),
        "n_objects": {name: int((poly_df["cls"] == name).sum()) for name in CLASS_NAMES.values()},
        "images_without_objects": int((img_df["n_objects"] == 0).sum()),
        "constant_attributes": constant,
        "filename_tokens": token_table.to_dict(orient="records"),
        "metadata": meta_summary,
        "objects_per_image": per_image_summary,
        "object_size": size_summary,
        "vehicle_area_ratio_pct": ratio_summary,
        "vertices_per_polygon": vertex_summary,
        "frames_per_clip": clip_summary,
        "nh_rh_check": nh_rh_summary,
        "brightness": brightness_summary,
        "vehicle_overlap": overlap_summary,
        "class_share_by_attribute_pct": class_attr_summary,
        "integrity": {k: {"count": len(v), "examples": v[:50]} for k, v in issues.items()},
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 요약 ----------------------------------------------------------------
    n_obj = summary["n_objects"]
    total = sum(n_obj.values())
    print("\n" + "=" * 66)
    print("파일명 토큰 사전 (00_filename_tokens.md)")
    for r in token_table.to_dict(orient="records"):
        print(f"  {r['순서']:2d}. {r['manifest 컬럼']:12s} {r['고유값 수']:5d}종  {r['데이터 확인 결과']}")
    print("-" * 66)
    print(f"이미지 {len(df):,}장 · 클립 {summary['n_clips']}개 · 객체 {total:,}개")
    print("  클래스      : " + "  ".join(f"{k}={v:,} ({v / total:.1%})" for k, v in n_obj.items()))
    print(f"  이미지당 객체: 평균 {per_image_summary['전체']['mean']:.1f}, 최대 {per_image_summary['전체']['max']}")
    print(f"  차량 면적 비 : 중앙값 {ratio_summary['overall']['median']:.2f}%, "
          f"95%분위 {ratio_summary['overall']['p95']:.2f}% (이미지 대비)")
    if nh_rh_summary:
        share = nh_rh_summary["rush_hour_clip_share"]
        overall = nh_rh_summary["overall_mean_objects"]
        print("  NH / RH 확인 (10_nh_rh_check.png):")
        print("    출퇴근 시간(07-08·17-18시) 촬영 비율: " + " · ".join(f"{k} {v:.0%}" for k, v in share.items()))
        print(f"    같은 카메라 비교: {nh_rh_summary['cameras_rh_more_objects']}/{nh_rh_summary['cameras_with_both']}개 "
              f"카메라에서 RH 쪽 차량이 더 많음")
        print("    전체 평균(" + " vs ".join(f"{k} {v:.1f}대" for k, v in overall.items())
              + ")은 지역 차이가 섞인 값이라 그대로 해석하면 안 됨")
    if "day_night" in df.columns:
        clips = df.drop_duplicates("clip")["day_night"].value_counts()
        print("  주 / 야간    : " + " · ".join(f"{SUN_PHASE_KO.get(p, p)} {clips.get(p, 0)}클립" for p in SUN_PHASES)
              + "  (일출·일몰 기준)")
    if brightness_summary:
        print("  밝기 중앙값  : " + " · ".join(f"{SUN_PHASE_KO[p]} {v['median']:.0f}"
                                           for p, v in brightness_summary["by_day_night"].items())
              + f"  ({brightness_summary['n_measured_images']:,}장 측정)")
    if overlap_summary:
        by_obj = overlap_summary["median_overlap_pct_by_objects"]
        print(f"  차량 겹침    : 2대 이상 이미지 평균 {overlap_summary['mean_overlap_pct_multi']:.1f}% · "
              f"차량 수별 중앙값 " + ", ".join(f"{k}대 {v:.1f}%" for k, v in by_obj.items()))
    if constant:
        print(f"  값이 하나뿐인 속성(분할 층화에서 자동 제외): {constant}")
    problems = {k: len(v) for k, v in issues.items() if v}
    print(f"  무결성 점검  : {'문제 없음' if not problems else problems}")
    print(f"\n  결과 폴더: {out_dir}")
    print("=" * 66)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="car_seg_dataset EDA (차트 + 요약 JSON)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", default=str(BASE_DIR / "car_seg_dataset"), help="전처리 결과 폴더")
    p.add_argument("--out-dir", default=None, help="EDA 결과 폴더 (기본: <dataset-dir>/eda)")
    p.add_argument("--n-samples", type=int, default=12, help="오버레이 샘플 이미지 수 (0 이면 생략)")
    p.add_argument("--imgsz", type=int, default=640, help="객체 크기 S/M/L 판정에 쓸 학습 해상도")
    p.add_argument("--raster-scale", type=float, default=0.25, help="면적 비 계산용 마스크 축소 비율")
    p.add_argument("--top-channels", type=int, default=15, help="채널 분포 차트에 표시할 상위 개수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-brightness", action="store_true", help="픽셀 밝기 측정(11번 차트) 생략")
    p.add_argument("--brightness-per-clip", type=int, default=10,
                   help="밝기를 잴 클립당 이미지 수 (0 이면 전체 — 수 분 걸림)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="밝기 측정 병렬 프로세스 수")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
