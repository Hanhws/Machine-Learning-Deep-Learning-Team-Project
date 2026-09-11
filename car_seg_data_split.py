#!/usr/bin/env python3
"""car_seg_dataset 을 train / val / test (기본 8:1:1) 로 층화 분할 — YOLO · COCO · 마스크 모든 형식에 같은 분할 적용.

분할 원칙
    1. 클립 단위 그룹 분할 — 같은 클립(영상)의 프레임은 서로 거의 같은 그림이라, 한 클립은 반드시
       한 세트에만 들어간다 (train 에서 본 장면이 test 에 나오는 데이터 누수 방지).
    2. 다중 속성 층화 — 파일명 메타데이터(지역·채널·날짜·시간대·요일·카메라 높이·NH/RH·도로 종류·
       차로 구성·날씨·화질) + 영상 방향 + 클래스별 객체 수가 세 세트에 골고루 같은 비율로 들어가도록
       클립 배정을 최적화한다.
         - 목적 함수: 각 속성값의 세트별 이미지 수가 목표(전체 × 비율)에서 벗어난 정도(상대 오차)
                      + 세트 크기 오차 + 클립이 3개 이상인 속성값이 어떤 세트에서 통째로 빠지는 벌점
         - 최적화   : 희귀한 클립부터 탐욕 배정 → 이동/교환 국소 탐색 → 여러 번 재시작해 최선 선택
       ※ 클립이 1~2개뿐인 속성값(예: 터널 1클립, 눈 1클립)은 물리적으로 세 세트에 나눌 수 없어
         한 세트에만 들어가며, 보고서에 따로 표시된다.

입력 (car_seg_data_preprocessing.py 결과)
    car_seg_dataset/  images/  labels/  manifest.csv
                      [annotations/instances_all.json]  [masks/]  [classes.json]   ← 있으면 함께 분할

출력 (기본: car_seg_split/)
    images/{train,val,test}/        모든 형식이 공유하는 이미지 (원본으로의 상대 링크, --link-mode 로 변경)
    labels/{train,val,test}/        [YOLO]  라벨 복사본
    data.yaml                       [YOLO]  Ultralytics 학습 설정
    annotations/instances_{train,val,test}.json   [COCO]
    masks/{train,val,test}/         [마스크] semantic 마스크 (원본으로의 링크)
    classes.json                    형식별 클래스 번호표
    README.md                       형식별(YOLO / Detectron2 / MMDetection / SegFormer) 사용 방법
    format_preview.png              같은 이미지에 형식별 라벨을 그려 비교
    split_manifest.csv              manifest + split 컬럼
    split_clips.csv                 클립 → split 배정표
    split_report.json               세트별 크기·속성 분포·편차·커버리지
    split_distribution.png          속성별 세트 분포 비교 차트

사용 예
    python car_seg_data_split.py
    python car_seg_data_split.py --ratios 8 1 1 --seed 42 --overwrite
    python car_seg_data_split.py --out-dir car_seg_split_share --link-mode hardlink   # 다른 사람에게 줄 때
    yolo segment train data=car_seg_split/data.yaml model=yolo11s-seg.pt imgsz=1280
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

SPLITS = ("train", "val", "test")
OUTPUT_SUBDIRS = ("images", "labels", "masks", "annotations")  # --overwrite 때 지우고 다시 만드는 폴더
DEFAULT_CLASSES = {0: "car", 1: "bus", 2: "truck"}

# 층화 대상 속성과 가중치 — 채널(49종)·날짜(15종)는 값 하나당 클립이 적어 완벽히 맞추기
# 불가능하므로 다른 속성을 해치지 않게 가중치를 낮춘다. 값이 1종뿐인 속성은 자동 제외.
STRATIFY_ATTRS = {
    "site": 1.0,
    "channel": 0.5,
    "date": 0.5,
    "time_band": 1.0,
    "day_night": 1.0,       # 일출·일몰 기준 주간 / 여명·황혼 / 야간 (파일명의 날짜·시각·지역으로 계산)
    "weekday": 1.0,
    "cam_height": 1.0,
    "hour_type": 1.0,
    "road_type": 1.0,
    "lane_config": 1.0,
    "weather": 1.0,
    "quality": 1.0,
    "orientation": 1.0,
}
CLASS_WEIGHT = 1.0      # 클래스별 객체 수 비율
SIZE_WEIGHT = 5.0       # 세트 이미지 수 비율 (8:1:1)
COVERAGE_WEIGHT = 1.0   # 클립 3개 이상인 값이 어떤 세트에 하나도 없을 때의 벌점 (--coverage-weight)

WEEKDAY_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# 차트에 그릴 속성 (값이 너무 많은 channel/date 는 JSON·콘솔로만)
PLOT_ATTRS = [
    ("site", "촬영 지역"), ("time_band", "촬영 시간대"), ("weekday", "요일"), ("cam_height", "카메라 높이"),
    ("hour_type", "NH / RH"), ("lane_config", "차로 구성"), ("weather", "날씨"), ("quality", "화질"),
    ("orientation", "영상 방향"), ("day_night", "주 / 야간"), ("class", "클래스 (객체 수)"),
]


# ----------------------------------------------------------------------------
# 클립 단위 특징 행렬
# ----------------------------------------------------------------------------


def build_clip_table(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """manifest(이미지 단위) → 클립 단위 표 (이미지 수, 속성값, 클래스별 객체 수)."""
    attrs = [a for a in STRATIFY_ATTRS if a in df.columns]
    inconsistent = [a for a in attrs if (df.groupby("clip")[a].nunique() > 1).any()]
    if inconsistent:
        print(f"[warn] 한 클립 안에서 값이 섞인 속성: {inconsistent} → 클립의 최빈값을 사용", file=sys.stderr)

    agg = {a: (a, lambda s: s.mode().iat[0]) for a in attrs}
    agg["n_images"] = ("image_id", "size")
    for name in class_names:
        agg[f"gt_{name}"] = (f"gt_{name}", "sum")
    return df.groupby("clip").agg(**agg).reset_index()


class Problem:
    """클립 × 특징 행렬과 목표값. 특징 = 속성값 one-hot(이미지 수 가중) + 클래스 객체 수 + 세트 크기."""

    def __init__(self, clips: pd.DataFrame, ratios: np.ndarray, class_names: list[str],
                 coverage_weight: float = COVERAGE_WEIGHT):
        n_splits = len(ratios)
        cols, weights, cov_weights, eligible = [], [], [], []
        self.feature_names: list[tuple[str, str]] = []
        n_img = clips["n_images"].to_numpy(float)
        self.used_attrs, self.skipped_attrs = [], []

        for attr, w_attr in STRATIFY_ATTRS.items():
            if attr not in clips.columns:
                continue
            values = sorted(clips[attr].astype(str).unique())
            if len(values) < 2:
                self.skipped_attrs.append(attr)
                continue
            self.used_attrs.append(attr)
            for v in values:
                onehot = (clips[attr].astype(str) == v).to_numpy(float)
                cols.append(onehot * n_img)
                weights.append(w_attr / len(values))
                cov_weights.append(coverage_weight * w_attr / len(values))
                eligible.append(onehot.sum() >= n_splits)  # 세 세트에 하나씩 넣을 수 있는가
                self.feature_names.append((attr, v))

        for name in class_names:
            counts = clips[f"gt_{name}"].to_numpy(float)
            if counts.sum() > 0:
                cols.append(counts)
                weights.append(CLASS_WEIGHT / len(class_names))
                cov_weights.append(0.0)
                eligible.append(False)
                self.feature_names.append(("class", name))

        cols.append(n_img)
        weights.append(SIZE_WEIGHT)
        cov_weights.append(0.0)
        eligible.append(False)
        self.feature_names.append(("size", "n_images"))

        self.X = np.stack(cols, axis=1)                        # (n_clips, F)
        self.P = (self.X > 0).astype(float)                    # 클립이 그 값을 가지는가 (커버리지용)
        self.w = np.asarray(weights)
        self.cov = np.asarray(cov_weights) * np.asarray(eligible)
        self.ratios = ratios
        self.target = ratios[:, None] * self.X.sum(0)[None, :]  # (S, F)
        # 희귀도: 클립이 가진 속성값 중 가장 드문 값의 1/클립수 → 드문 클립부터 배정
        n_clips_per_value = self.P.sum(0)
        self.rarity = (self.P / np.maximum(n_clips_per_value, 1)).max(1)

    def row_cost(self, rows: np.ndarray, cnts: np.ndarray, s: int) -> np.ndarray:
        """rows/cnts: (..., F) — split s 에 대한 비용."""
        dev = np.abs(rows - self.target[s]) / self.target[s]
        return (dev * self.w).sum(-1) + ((cnts < 0.5) * self.cov).sum(-1)

    def state(self, assign: np.ndarray):
        S = len(self.ratios)
        cur = np.stack([self.X[assign == s].sum(0) for s in range(S)])
        cnt = np.stack([self.P[assign == s].sum(0) for s in range(S)])
        return cur, cnt

    def total_cost(self, assign: np.ndarray) -> float:
        cur, cnt = self.state(assign)
        return float(sum(self.row_cost(cur[s], cnt[s], s) for s in range(len(self.ratios))))


# ----------------------------------------------------------------------------
# 최적화: 탐욕 배정 → 이동/교환 국소 탐색
# ----------------------------------------------------------------------------


def greedy(prob: Problem, rng: np.random.Generator) -> np.ndarray:
    n, S = len(prob.X), len(prob.ratios)
    order = np.lexsort((rng.random(n), -prob.X[:, -1], -prob.rarity))  # 희귀 → 큰 클립 → 무작위
    cur = np.zeros((S, prob.X.shape[1]))
    cnt = np.zeros_like(cur)
    assign = np.full(n, -1)
    for i in order:
        deltas = [prob.row_cost(cur[s] + prob.X[i], cnt[s] + prob.P[i], s) - prob.row_cost(cur[s], cnt[s], s)
                  for s in range(S)]
        s = int(np.argmin(deltas))
        assign[i] = s
        cur[s] += prob.X[i]
        cnt[s] += prob.P[i]
    return assign


def local_search(prob: Problem, assign: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """가장 비용을 많이 줄이는 '클립 1개 이동' 또는 '두 세트 간 클립 교환'을 더 이상 개선이 없을 때까지 반복."""
    S = len(prob.ratios)
    X, P = prob.X, prob.P
    assign = assign.copy()
    for _ in range(max_iter):
        cur, cnt = prob.state(assign)
        base = np.array([prob.row_cost(cur[s], cnt[s], s) for s in range(S)])
        best_gain, best_op = 1e-9, None

        for a in range(S):
            ia = np.flatnonzero(assign == a)
            if len(ia) <= 1:
                continue
            for b in range(S):
                if a == b:
                    continue
                # 이동: a → b
                ca = prob.row_cost(cur[a] - X[ia], cnt[a] - P[ia], a)
                cb = prob.row_cost(cur[b] + X[ia], cnt[b] + P[ia], b)
                gain = base[a] + base[b] - (ca + cb)
                k = int(np.argmax(gain))
                if gain[k] > best_gain:
                    best_gain, best_op = gain[k], ("move", ia[k], b)

                # 교환: a 의 i ↔ b 의 j  (a < b 한 번만)
                if a < b:
                    ib = np.flatnonzero(assign == b)
                    if not len(ib):
                        continue
                    dX = X[ib][None, :, :] - X[ia][:, None, :]      # (na, nb, F)
                    dP = P[ib][None, :, :] - P[ia][:, None, :]
                    ca = prob.row_cost(cur[a] + dX, cnt[a] + dP, a)
                    cb = prob.row_cost(cur[b] - dX, cnt[b] - dP, b)
                    gain = base[a] + base[b] - (ca + cb)
                    k = np.unravel_index(int(np.argmax(gain)), gain.shape)
                    if gain[k] > best_gain:
                        best_gain, best_op = gain[k], ("swap", ia[k[0]], ib[k[1]])

        if best_op is None:
            break
        if best_op[0] == "move":
            assign[best_op[1]] = best_op[2]
        else:
            i, j = best_op[1], best_op[2]
            assign[i], assign[j] = assign[j], assign[i]
    return assign


def optimize(prob: Problem, restarts: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in tqdm(range(restarts), desc="층화 분할 최적화", unit="trial", dynamic_ncols=True):
        assign = local_search(prob, greedy(prob, rng))
        cost = prob.total_cost(assign)
        if cost < best_cost:
            best, best_cost = assign, cost
    return best, best_cost


# ----------------------------------------------------------------------------
# 보고서
# ----------------------------------------------------------------------------


def value_order(attr: str, values) -> list[str]:
    """요일·카메라 높이는 문자열 순서가 아닌 자연스러운 순서로."""
    values = [str(v) for v in values]
    if attr == "weekday":
        return sorted(values, key=lambda v: WEEKDAY_ORDER.index(v) if v in WEEKDAY_ORDER else len(WEEKDAY_ORDER))
    if attr == "cam_height":
        def height(v: str) -> float:
            try:
                return float(v.rstrip("m"))
            except ValueError:
                return float("inf")
        return sorted(values, key=height)
    return sorted(values)


def distribution_tables(df: pd.DataFrame, clips: pd.DataFrame, class_names: list[str]) -> dict:
    """속성별 {값: {overall/train/val/test 이미지 비율(%), 세트별 클립 수}} 와 세트별 최대 편차(%p)."""
    out = {}
    attrs = [a for a in STRATIFY_ATTRS if a in df.columns] + ["class"]
    for attr in attrs:
        if attr == "class":
            counts = pd.DataFrame({s: [df.loc[df["split"] == s, f"gt_{c}"].sum() for c in class_names]
                                   for s in SPLITS}, index=class_names)
            counts["overall"] = counts[list(SPLITS)].sum(1)
            clip_counts = None
        else:
            counts = pd.crosstab(df[attr].astype(str), df["split"]).reindex(columns=list(SPLITS), fill_value=0)
            counts = counts.reindex(value_order(attr, counts.index))
            counts["overall"] = counts.sum(1)
            clip_counts = pd.crosstab(clips[attr].astype(str), clips["split"]).reindex(
                index=counts.index, columns=list(SPLITS), fill_value=0)
        share = counts / counts.sum(0).replace(0, 1) * 100
        values = {}
        for v in counts.index:
            values[str(v)] = {
                **{f"{s}_pct": round(float(share.at[v, s]), 2) for s in ("overall", *SPLITS)},
                **({f"{s}_clips": int(clip_counts.at[v, s]) for s in SPLITS} if clip_counts is not None else {}),
            }
        max_dev = {s: round(float((share[s] - share["overall"]).abs().max()), 2) for s in SPLITS}
        out[attr] = {"max_abs_dev_pctpt": max_dev, "values": values}
    return out


def coverage_report(clips: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """(나눌 수 있는데 어떤 세트에서 빠진 값, 클립이 3개 미만이라 나눌 수 없는 값)."""
    missing, uncoverable = [], []
    for attr in STRATIFY_ATTRS:
        if attr not in clips.columns or clips[attr].nunique() < 2:
            continue
        for v, g in clips.groupby(clips[attr].astype(str)):
            present = sorted(set(g["split"]), key=SPLITS.index)
            if len(g) < len(SPLITS):
                uncoverable.append({"attr": attr, "value": v, "n_clips": len(g), "splits": present})
            elif len(present) < len(SPLITS):
                missing.append({"attr": attr, "value": v, "n_clips": len(g),
                                "missing_in": [s for s in SPLITS if s not in present]})
    return missing, uncoverable


def plot_distribution(tables: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    surface, text, text_2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
    available = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic",
                          "Nanum Gothic", "Noto Sans CJK KR") if f in available]
    plt.rcParams.update({
        "font.family": korean[:1] + ["DejaVu Sans"], "axes.unicode_minus": False,
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "axes.edgecolor": grid, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": text_2, "ytick.color": text_2, "axes.labelcolor": text_2,
        "axes.titlecolor": text, "axes.titlelocation": "left", "axes.titleweight": "bold",
    })

    panels = [(a, t) for a, t in PLOT_ATTRS if a in tables]
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    heights = [max(len(tables[panels[r * ncols + c][0]]["values"]) if r * ncols + c < len(panels) else 1
                   for c in range(ncols)) for r in range(nrows)]
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, sum(0.36 * h + 0.9 for h in heights) + 1.2),
                             gridspec_kw={"height_ratios": [0.36 * h + 0.9 for h in heights]}, squeeze=False)
    for ax, (attr, title) in zip(axes.ravel(), panels):
        vals = tables[attr]["values"]
        names = list(vals)
        y = np.arange(len(names))
        overall = [vals[n]["overall_pct"] for n in names]
        ax.barh(y, overall, height=0.7, color="#f0efec", edgecolor="none", zorder=1)  # 전체 비율 = 기준 막대
        for k, s in enumerate(SPLITS):
            ax.scatter([vals[n][f"{s}_pct"] for n in names], y + (k - 1) * 0.2, s=34, color=colors[s],
                       edgecolor=surface, linewidth=1.2, zorder=3)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        dev = tables[attr]["max_abs_dev_pctpt"]
        ax.set_title(f"{title}   최대 편차  val {dev['val']:.1f}%p · test {dev['test']:.1f}%p", fontsize=11)
        ax.set_xlabel("비율 (%)")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, "세트별 속성 분포 비교", fontsize=15, fontweight="bold", color=text, va="top")
    fig.text(0.01, 1 - 0.46 / h, "회색 막대 = 전체 비율, 점 = 각 세트의 비율 — 점이 막대 끝에 모일수록 층화가 잘 된 것",
             fontsize=10, color=text_2, va="top")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=s) for s, c in colors.items()]
    handles.append(Line2D([], [], color="#f0efec", linewidth=8, label="overall"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.99, 1 - 0.1 / h))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / h))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 파일 배치
# ----------------------------------------------------------------------------


_FALLBACK = {"symlink": "hardlink", "hardlink": "copy"}
_effective_mode: dict[str, str] = {}  # 요청한 방식 → 실제로 되는 방식 (한 번 실패하면 이후엔 바로 대체 방식 사용)


def place(src: Path, dst: Path, mode: str) -> None:
    """이미지/마스크 배치. 링크를 만들 수 없는 환경(Windows 일반 권한, 다른 드라이브 등)이면
    symlink → hardlink → copy 순으로 자동으로 바꾼다."""
    actual = _effective_mode.get(mode, mode)
    while True:
        try:
            if actual == "symlink":
                dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))  # 상위 폴더째 옮겨도 유지
            elif actual == "hardlink":
                os.link(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            break
        except (OSError, NotImplementedError) as exc:
            if actual not in _FALLBACK:
                raise
            nxt = _FALLBACK[actual]
            print(f"\n[warn] {actual} 를 만들 수 없어 {nxt} 로 바꿉니다 ({exc.__class__.__name__}: {exc})"
                  + ("\n       Windows 라면 '개발자 모드'를 켜면 symlink 를 쓸 수 있습니다." if actual == "symlink" else ""),
                  file=sys.stderr)
            actual = nxt
    _effective_mode[mode] = actual


# ----------------------------------------------------------------------------
# 다른 라벨 형식 (COCO / 마스크) 분할
# ----------------------------------------------------------------------------


def split_coco(coco_all: Path, out_dir: Path, df: pd.DataFrame) -> dict:
    """instances_all.json → instances_{train,val,test}.json. image/annotation id 는 원본 그대로 유지."""
    coco = json.loads(coco_all.read_text(encoding="utf-8"))
    split_of = dict(zip(df["file_name"], df["split"]))
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    stats = {}
    for s in SPLITS:
        images = [im for im in coco["images"] if split_of.get(im["file_name"]) == s]
        ids = {im["id"] for im in images}
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        part = {**coco, "images": images, "annotations": anns,
                "info": {**coco.get("info", {}), "split": s}}
        (out_dir / "annotations" / f"instances_{s}.json").write_text(
            json.dumps(part, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        stats[s] = {"images": len(images), "annotations": len(anns)}
    return stats


def save_format_preview(out_dir: Path, df: pd.DataFrame, formats: list[str], seed: int) -> None:
    """val 이미지 몇 장에 형식별 라벨을 겹쳐 그려 나란히 저장 — 형식끼리 같은 라벨인지 눈으로 확인."""
    from PIL import Image, ImageDraw

    colors = {0: (42, 120, 214), 1: (235, 104, 52), 2: (27, 175, 122)}  # YOLO class id 기준
    val = df[df["split"] == "val"].drop_duplicates("clip")
    rows = val.sample(min(3, len(val)), random_state=seed)
    coco = None
    if "coco" in formats:
        coco = json.loads((out_dir / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
        coco_img = {im["file_name"]: im["id"] for im in coco["images"]}
        coco_ann: dict[int, list] = {}
        for a in coco["annotations"]:
            coco_ann.setdefault(a["image_id"], []).append(a)

    tw = 560
    tiles = []
    max_ratio = 0.0
    for r in rows.itertuples(index=False):
        base = Image.open(out_dir / "images" / "val" / r.file_name).convert("RGB")
        W, H = base.size
        max_ratio = max(max_ratio, H / W)
        row_tiles = []
        # YOLO (labels/*.txt, 0~1 좌표)
        im = base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for line in (out_dir / "labels" / "val" / f"{Path(r.file_name).stem}.txt").read_text().splitlines():
            t = line.split()
            v = [float(x) for x in t[1:]]
            d.polygon([(v[i] * W, v[i + 1] * H) for i in range(0, len(v), 2)],
                      fill=colors[int(t[0])] + (90,), outline=colors[int(t[0])] + (255,), width=3)
        row_tiles.append((im, "YOLO  labels/"))
        # COCO (polygon + bbox, 픽셀 좌표)
        if coco is not None:
            im = base.copy()
            d = ImageDraw.Draw(im, "RGBA")
            for a in coco_ann.get(coco_img[r.file_name], []):
                c = colors[a["category_id"] - 1]
                d.polygon(a["segmentation"][0], fill=c + (90,), outline=c + (255,), width=3)
                x, y, w, h = a["bbox"]
                d.rectangle((x, y, x + w, y + h), outline=c + (255,), width=2)
            row_tiles.append((im, "COCO  annotations/instances_val.json"))
        # 마스크 (픽셀값 = class id + 1)
        if "mask" in formats:
            m = Image.open(out_dir / "masks" / "val" / f"{Path(r.file_name).stem}.png")
            color = m.convert("RGB")  # 팔레트 색
            alpha = Image.fromarray(((np.asarray(m) > 0) * 150).astype(np.uint8), "L")  # 차량 픽셀만 반투명
            im = base.copy()
            im.paste(color, (0, 0), alpha)
            row_tiles.append((im, "mask  masks/"))
        tiles.append(row_tiles)

    ncol = max(len(t) for t in tiles)
    th = int(tw * max_ratio)
    grid = Image.new("RGB", (tw * ncol, (th + 22) * len(tiles)), (252, 252, 251))
    for ri, row_tiles in enumerate(tiles):
        for ci, (im, title) in enumerate(row_tiles):
            im.thumbnail((tw, th))
            grid.paste(im, (ci * tw, ri * (th + 22) + 22))
            ImageDraw.Draw(grid).text((ci * tw + 6, ri * (th + 22) + 5), title, fill=(11, 11, 11))
    grid.save(out_dir / "format_preview.png")


def write_readme(out_dir: Path, formats: list[str], sizes: dict, classes: dict[int, str],
                 coco_stats: dict) -> None:
    names = [classes[k] for k in sorted(classes)]
    rows = "\n".join(f"| {s} | {v['clips']} | {v['images']:,} |" for s, v in sizes.items())
    parts = [f"""# car_seg_split — 모델별 학습 데이터 (train / val / test 공통)

car_seg_data_split.py 로 생성. **모든 형식이 같은 이미지·같은 분할**을 쓰므로 모델끼리 공정하게 비교할 수 있다.
분할은 클립(영상) 단위 + 파일명 메타데이터 층화 (자세한 편차는 split_report.json / split_distribution.png).

| split | clips | images |
|---|---|---|
{rows}

```
car_seg_split/
├── images/{{train,val,test}}/          ← 모든 형식이 공유하는 이미지 (car_seg_dataset/images 로의 링크)
├── labels/{{train,val,test}}/          ← YOLO
├── data.yaml                          ← YOLO 설정""" + ("""
├── annotations/instances_{train,val,test}.json   ← COCO""" if "coco" in formats else "") + ("""
├── masks/{train,val,test}/            ← semantic 마스크""" if "mask" in formats else "") + """
├── classes.json                       ← 형식별 클래스 번호표
└── format_preview.png                 ← 형식별 라벨을 같은 이미지에 그려 비교
```

## 형식별 클래스 번호
| 형식 | 번호 |
|---|---|
| YOLO | """ + ", ".join(f"{i}={n}" for i, n in enumerate(names)) + """ (0부터) |
| COCO | """ + ", ".join(f"{i + 1}={n}" for i, n in enumerate(names)) + """ (1부터, COCO 규약) |
| 마스크 | 0=background, """ + ", ".join(f"{i + 1}={n}" for i, n in enumerate(names)) + """ |

## YOLO (Ultralytics)
```bash
yolo segment train data=car_seg_split/data.yaml model=yolo11s-seg.pt imgsz=1280
```
"""]
    if "coco" in formats:
        parts.append(f"""
## COCO — Mask R-CNN · Detectron2 · MMDetection · Mask2Former(instance)
annotations: """ + " / ".join(f"{s} {v['annotations']:,}" for s, v in coco_stats.items()) + f"""
```python
# Detectron2
from detectron2.data.datasets import register_coco_instances
for s in ["train", "val", "test"]:
    register_coco_instances(f"car_seg_{{s}}", {{}}, f"car_seg_split/annotations/instances_{{s}}.json",
                            f"car_seg_split/images/{{s}}")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = {len(names)}

# MMDetection (config)
metainfo = dict(classes=({", ".join(repr(n) for n in names)},))
train_dataloader = dict(dataset=dict(data_root="car_seg_split/", metainfo=metainfo,
    ann_file="annotations/instances_train.json", data_prefix=dict(img="images/train/")))
```
""")
    if "mask" in formats:
        parts.append(f"""
## 마스크 — SegFormer · DeepLabV3 · U-Net · Mask2Former(semantic)
- `masks/<split>/<이미지와 같은 이름>.png`, 이미지와 같은 해상도. 팔레트 PNG 라 뷰어에선 색으로 보이고,
  `np.array(Image.open(...))` 로 읽으면 값은 0~{len(names)}
- 프로젝트 지표용 '차량 전체' 영역은 `mask > 0`
- 차량이 겹치는 곳(차량 면적의 약 0.5%)은 화면 아래쪽(카메라에 가까운) 차량의 클래스로 칠함
```python
import json, numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

id2label = {{int(k): v for k, v in json.load(open("car_seg_split/classes.json"))["mask"].items()}}
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=len(id2label), id2label=id2label, label2id={{v: k for k, v in id2label.items()}})
processor = SegformerImageProcessor(do_reduce_labels=False)   # 0 = 배경도 학습 대상

image = Image.open("car_seg_split/images/train/XXX.png").convert("RGB")
mask = np.array(Image.open("car_seg_split/masks/train/XXX.png"))
inputs = processor(images=image, segmentation_maps=mask, return_tensors="pt")
```
""")
    (out_dir / "README.md").write_text("".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        print(f"[error] manifest.csv 가 없습니다: {manifest}\n"
              f"        먼저 car_seg_data_preprocessing.py 를 실행하세요.", file=sys.stderr)
        return 1

    ratios = np.asarray(args.ratios, dtype=float)
    if len(ratios) != len(SPLITS) or (ratios <= 0).any():
        print("[error] --ratios 는 양수 3개여야 합니다 (예: 8 1 1)", file=sys.stderr)
        return 1
    ratios = ratios / ratios.sum()

    for sub in OUTPUT_SUBDIRS:
        d = out_dir / sub
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                print(f"[error] 출력 폴더가 이미 있습니다: {d}\n        다시 나누려면 --overwrite 를 붙이세요.",
                      file=sys.stderr)
                return 1

    classes = DEFAULT_CLASSES
    report_path = dataset_dir / "preprocess_report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8")).get("classes")
        if saved:
            classes = {int(k): v for k, v in saved.items()}
    class_names = [classes[k] for k in sorted(classes)]

    df = pd.read_csv(manifest, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for name in class_names:
        df[f"gt_{name}"] = pd.to_numeric(df.get(f"gt_{name}", 0), errors="coerce").fillna(0).astype(int)
    if {"date", "time", "site"} <= set(df.columns) and ("day_night" not in df.columns or (df["day_night"] == "").all()):
        # 주/야간 컬럼이 없는 예전 manifest 면 여기서 계산 (전처리를 다시 돌릴 필요 없음)
        from car_seg_data_preprocessing import day_night_of
        df["day_night"] = [day_night_of(d, t, s)[0] for d, t, s in zip(df["date"], df["time"], df["site"])]
    missing_meta = [a for a in STRATIFY_ATTRS if a not in df.columns]
    if missing_meta:
        print(f"[warn] manifest 에 없는 층화 속성은 제외: {missing_meta}", file=sys.stderr)

    clips = build_clip_table(df, class_names)
    if len(clips) < len(SPLITS):
        print(f"[error] 클립이 {len(clips)}개뿐이라 3개 세트로 나눌 수 없습니다.", file=sys.stderr)
        return 1
    print(f"[split] 이미지 {len(df):,}장 / 클립 {len(clips)}개 → 비율 "
          + " : ".join(f"{s} {r:.0%}" for s, r in zip(SPLITS, ratios)))

    # 1) 최적화 ---------------------------------------------------------------
    prob = Problem(clips, ratios, class_names, args.coverage_weight)
    print(f"[split] 층화 속성: {prob.used_attrs} + class"
          + (f"  (값이 1종뿐이라 제외: {prob.skipped_attrs})" if prob.skipped_attrs else ""))
    assign, cost = optimize(prob, args.restarts, args.seed)
    clips["split"] = [SPLITS[s] for s in assign]
    df = df.merge(clips[["clip", "split"]], on="clip", how="left")

    # 2) 파일 배치 -------------------------------------------------------------
    # 전처리에서 만든 형식을 확인 (yolo 는 항상, coco·mask 는 있으면 함께 분할)
    formats = ["yolo"]
    if (dataset_dir / "annotations" / "instances_all.json").exists():
        formats.append("coco")
    if (dataset_dir / "masks").is_dir() and any((dataset_dir / "masks").iterdir()):
        formats.append("mask")
    print(f"[split] 분할할 형식: {', '.join(formats)}")

    for sub in OUTPUT_SUBDIRS:
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
    dirs = ["images", "labels"] + (["masks"] if "mask" in formats else [])
    for sub in dirs:
        for s in SPLITS:
            (out_dir / sub / s).mkdir(parents=True, exist_ok=True)

    missing_files = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="파일 배치", unit="img", dynamic_ncols=True):
        src_img = dataset_dir / "images" / row.file_name
        src_lbl = dataset_dir / "labels" / f"{row.image_id}.txt"
        src_mask = dataset_dir / "masks" / f"{row.image_id}.png"
        if not src_img.exists() or not src_lbl.exists() or ("mask" in formats and not src_mask.exists()):
            missing_files.append(row.image_id)
            continue
        place(src_img, out_dir / "images" / row.split / row.file_name, args.link_mode)
        shutil.copy2(src_lbl, out_dir / "labels" / row.split / src_lbl.name)
        if "mask" in formats:
            place(src_mask, out_dir / "masks" / row.split / src_mask.name, args.link_mode)

    # 3) data.yaml ------------------------------------------------------------
    yaml_lines = [
        "# car_seg_data_split.py 가 생성한 Ultralytics YOLO segmentation 데이터 설정",
        f"path: {json.dumps(str(out_dir), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        *[f"  {k}: {v}" for k, v in sorted(classes.items())],
        "",
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    coco_stats = {}
    if "coco" in formats:
        coco_stats = split_coco(dataset_dir / "annotations" / "instances_all.json", out_dir, df)
    if (dataset_dir / "classes.json").exists():
        shutil.copy2(dataset_dir / "classes.json", out_dir / "classes.json")

    # 4) 보고서 ----------------------------------------------------------------
    df.to_csv(out_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    clips.to_csv(out_dir / "split_clips.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    sizes = {s: {"clips": int((clips["split"] == s).sum()),
                 "images": int((df["split"] == s).sum()),
                 "image_pct": round(float((df["split"] == s).mean() * 100), 2),
                 **{name: int(df.loc[df["split"] == s, f"gt_{name}"].sum()) for name in class_names}}
             for s in SPLITS}
    tables = distribution_tables(df, clips, class_names)
    missing, uncoverable = coverage_report(clips)
    report = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "ratios_target": dict(zip(SPLITS, [round(float(r), 4) for r in ratios])),
        "seed": args.seed,
        "coverage_weight": args.coverage_weight,
        "restarts": args.restarts,
        "link_mode": _effective_mode.get(args.link_mode, args.link_mode),
        "objective_cost": round(cost, 6),
        "stratified_attrs": prob.used_attrs + ["class"],
        "skipped_attrs": prob.skipped_attrs,
        "sizes": sizes,
        "coverage_missing": missing,
        "uncoverable_values": uncoverable,
        "missing_files": {"count": len(missing_files), "examples": missing_files[:50]},
        "formats": formats,
        "coco_annotations": coco_stats,
        "distributions": tables,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, formats, sizes, classes, coco_stats)
    if not args.no_plot:
        plot_distribution(tables, out_dir / "split_distribution.png")
        save_format_preview(out_dir, df, formats, args.seed)

    # 콘솔 요약 ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'split':6s} {'clips':>6s} {'images':>8s} {'비율':>7s}  " + "  ".join(f"{n:>7s}" for n in class_names))
    for s in SPLITS:
        z = sizes[s]
        print(f"{s:6s} {z['clips']:6d} {z['images']:8,d} {z['image_pct']:6.1f}%  "
              + "  ".join(f"{z[n]:7,d}" for n in class_names))
    print("\n속성별 최대 편차 (세트 비율 − 전체 비율, %p)")
    for attr, t in tables.items():
        d = t["max_abs_dev_pctpt"]
        print(f"  {attr:12s} train {d['train']:5.2f}   val {d['val']:5.2f}   test {d['test']:5.2f}")
    # 값 종류가 가장 작은 세트의 클립 수보다 많은 속성(channel·date 등)은 구조적으로 모든 값을
    # 모든 세트에 넣을 수 없으므로 개수만 요약하고, 나머지 속성의 누락은 하나씩 경고한다.
    min_clips = min(z["clips"] for z in sizes.values())
    eligible_per_attr = {a: sum(1 for _, g in clips.groupby(a) if len(g) >= len(SPLITS)) for a in prob.used_attrs}
    structural = {a for a, n in eligible_per_attr.items() if n > min_clips}
    real_missing = [m for m in missing if m["attr"] not in structural]
    for attr in sorted(structural):
        n = sum(1 for m in missing if m["attr"] == attr)
        print(f"\n[참고] {attr}: {eligible_per_attr[attr]}종 > 가장 작은 세트의 클립 {min_clips}개 → "
              f"모든 값을 모든 세트에 넣는 건 불가능 ({n}종이 일부 세트에 없음, 편차는 위 표 참고)")
    if real_missing:
        print(f"\n[주의] 클립이 3개 이상인데 일부 세트에 없는 값 {len(real_missing)}개:")
        for m in real_missing:
            print(f"  {m['attr']}={m['value']} ({m['n_clips']}클립) → {m['missing_in']} 에 없음")
        print("  → --coverage-weight 를 올리면(예: 10) 채워지지만, 대신 다른 속성의 비율 편차가 커집니다.")
    else:
        print("\n[확인] 위 속성을 제외하면, 클립이 3개 이상인 모든 값이 train/val/test 에 모두 들어갔습니다.")
    if uncoverable:
        print(f"\n[참고] 클립이 1~2개뿐이라 모든 세트에 넣을 수 없는 값 {len(uncoverable)}개:")
        for u in uncoverable[:15]:
            print(f"  {u['attr']}={u['value']} ({u['n_clips']}클립) → {u['splits']}")
    if missing_files:
        print(f"\n[warn] 이미지/라벨 파일이 없어 건너뛴 항목 {len(missing_files)}개 (split_report.json 참고)")
    print(f"\n  결과 폴더 : {out_dir}")
    print(f"  형식      : YOLO (labels/, data.yaml)"
          + (" · COCO (annotations/)" if "coco" in formats else "")
          + (" · 마스크 (masks/)" if "mask" in formats else ""))
    if coco_stats:
        print("  COCO 폴리곤: " + " / ".join(f"{s} {v['annotations']:,}" for s, v in coco_stats.items()))
    print("  사용 방법 : README.md · 형식 비교: format_preview.png · 분할 보고서: split_report.json")
    print("=" * 72)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="car_seg_dataset → train/val/test 클립 단위 층화 분할 (YOLO · COCO · 마스크)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", default=str(BASE_DIR / "car_seg_dataset"), help="전처리 결과 폴더")
    p.add_argument("--out-dir", default=str(BASE_DIR / "car_seg_split"), help="분할 결과 폴더")
    p.add_argument("--ratios", type=float, nargs=3, default=[8, 1, 1], metavar=("TRAIN", "VAL", "TEST"),
                   help="train val test 비율")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=20, help="최적화 재시작 횟수 (많을수록 균형↑, 느려짐)")
    p.add_argument("--coverage-weight", type=float, default=COVERAGE_WEIGHT,
                   help="'모든 값이 모든 세트에 존재' 우선도. 높이면 커버리지↑ 대신 비율 편차↑")
    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink",
                   help="이미지 배치 방식 (symlink: 추가 용량 거의 없음)")
    p.add_argument("--overwrite", action="store_true", help="기존 분할 결과(images/labels)를 지우고 다시 생성")
    p.add_argument("--no-plot", action="store_true", help="split_distribution.png · format_preview.png 생략")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
