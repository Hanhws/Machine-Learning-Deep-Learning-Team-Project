#!/usr/bin/env python3
"""car_seg_split/split_manifest.csv 하나로 train 축소 결과를 검증하는 발표용 그림 3장.

물음: 19,589장에서 2,500장만 남겼는데, 그게 의도한 대로 줄어든 것인가?

    01_deviation.png   전체 분포 대비 train·val·test 편차 — 어떤 축이 유지되고 어떤 축이 틀어졌나
    02_reduction.png   train 축소 전(19,589) → 후(2,500) — 무엇이 늘고 무엇이 줄었나
    03_distribution.png 주요 조건의 실제 분포 — 전체를 배경으로 세 세트의 위치

    stats_*.csv        위 세 그림의 원본 수치

쓰는 컬럼은 split_manifest.csv 에 다 있다 (이미지 파일은 필요 없다).
    pool   = 프레임이 배정된 세트 (train 후보 19,589 / val 1,000 / test 5,000)
    split  = 최종 세트 (train 은 2,500 만, 나머지는 빈 값)
    gt_*   = 이미지 안의 클래스별 객체 수
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 색은 car_seg_data_split.py 와 같은 것을 쓴다 (발표 자료 사이에서 train=파랑 이 일관되게).
# 검증: CVD 최소 ΔE 9.2(deutan) · 일반 시야 27.6 — 통과. 다만 test 색은 배경 대비 2.74:1 이라
# 색만으로 두지 않고 값 라벨을 반드시 같이 찍는다.
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SETS = ["train", "val", "test"]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAND = "#f0efec"          # 전체 분포를 나타내는 배경 막대
KEEP_LINE = "#b8b7b3"     # '유지' 기준선

CLASSES = ["car", "bus", "truck"]

# (컬럼, 표시 이름) — 분포를 비교할 속성
ATTRS = [
    ("camera", "CCTV (49대)"),
    ("site", "촬영 지역"),
    ("road_form", "도로 형태"),
    ("cam_height", "카메라 높이"),
    ("quality", "화질"),
    ("orientation", "영상 방향"),
    ("lane_config", "차로 구성"),
    ("hour_type", "NH / RH"),
    ("weekday", "요일"),
    ("day_night", "주 / 야간"),
    ("weather", "날씨"),
    ("difficulty", "난이도 조건"),
    ("time_band", "촬영 시간대"),
    ("date", "촬영 날짜"),
]

# 03 번에서 실제 분포를 펼쳐 볼 속성 (값이 적어 한 화면에 들어오는 것들)
DETAIL_ATTRS = [("difficulty", "난이도 조건"), ("day_night", "주 / 야간"),
                ("weather", "날씨"), ("road_form", "도로 형태")]

# 02 번에서 축소 전후를 볼 항목 — (표시 이름, 컬럼, 값)
REDUCTION_ROWS = [
    ("hard (비표준 조건)", "difficulty", "hard"),
    ("야간", "day_night", "night"),
    ("여명·황혼", "day_night", "twilight"),
    ("비", "weather", "rainy"),
    ("안개", "weather", "fog"),
    ("눈", "weather", "snow"),
    ("교량", "road_form", "bridge"),
    ("터널", "road_form", "tunnel"),
]


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic",
                          "NanumGothic", "Nanum Gothic", "Noto Sans CJK KR") if f in available]
    if not korean:
        print("[warn] 한글 폰트를 찾지 못했습니다 — 라벨이 깨질 수 있습니다.")
    plt.rcParams.update({
        "font.family": korean[:1] + ["DejaVu Sans"], "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titleweight": "bold",
        "axes.titlesize": 12, "font.size": 9.5,
    })
    return plt


# ----------------------------------------------------------------------------
# 집계
# ----------------------------------------------------------------------------


def load(manifest: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(manifest, encoding="utf-8-sig", low_memory=False)
    for col in ("pool", "split"):
        if col not in df.columns:
            raise SystemExit(f"[error] {manifest} 에 '{col}' 컬럼이 없습니다 — split_manifest.csv 가 맞습니까?")
    df["split"] = df["split"].fillna("")
    groups = {
        "전체": df,
        "train 후보": df[df["pool"] == "train"],
        "train": df[df["split"] == "train"],
        "val": df[df["split"] == "val"],
        "test": df[df["split"] == "test"],
    }
    print("  " + " · ".join(f"{k} {len(v):,}장" for k, v in groups.items()))
    return groups


def share(g: pd.DataFrame, attr: str) -> pd.Series:
    """속성값별 이미지 비율(%)."""
    if g.empty:
        return pd.Series(dtype=float)
    return g[attr].astype(str).value_counts(normalize=True).mul(100)


def class_share(g: pd.DataFrame) -> pd.Series:
    """클래스별 객체 비율(%)."""
    counts = pd.Series({c: float(g[f"gt_{c}"].sum()) for c in CLASSES})
    return counts / max(1.0, counts.sum()) * 100


def deviation_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """속성마다 '전체 대비 최대 절대 편차(%p)' — 세트별로."""
    overall = groups["전체"]
    rows = []
    for attr, label in ATTRS:
        if attr not in overall.columns:
            continue
        base = share(overall, attr)
        row = {"속성": label, "컬럼": attr, "값 개수": int(base.size)}
        for s in SETS:
            cur = share(groups[s], attr).reindex(base.index).fillna(0)
            row[s] = float((cur - base).abs().max())
        rows.append(row)
    base_cls = class_share(overall)
    row = {"속성": "클래스 (객체)", "컬럼": "class", "값 개수": len(CLASSES)}
    for s in SETS:
        row[s] = float((class_share(groups[s]) - base_cls).abs().max())
    rows.append(row)
    return pd.DataFrame(rows).sort_values("train", ascending=False).reset_index(drop=True)


def reduction_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """train 축소 전(train 후보) → 후(train) 비교."""
    before, after = groups["train 후보"], groups["train"]
    rows = []
    for label, attr, value in REDUCTION_ROWS:
        if attr not in before.columns:
            continue
        b = float((before[attr].astype(str) == value).mean() * 100)
        a = float((after[attr].astype(str) == value).mean() * 100)
        rows.append({"항목": label, "before_pct": b, "after_pct": a, "배수": a / b if b else np.nan,
                     "before_images": int((before[attr].astype(str) == value).sum()),
                     "after_images": int((after[attr].astype(str) == value).sum())})
    cb, ca = class_share(before), class_share(after)
    for c in CLASSES:
        rows.append({"항목": f"{c} (객체 비율)", "before_pct": cb[c], "after_pct": ca[c],
                     "배수": ca[c] / cb[c] if cb[c] else np.nan,
                     "before_images": int(before[f"gt_{c}"].sum()),
                     "after_images": int(after[f"gt_{c}"].sum())})
    return pd.DataFrame(rows)


def detail_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for attr, label in DETAIL_ATTRS:
        if attr not in groups["전체"].columns:
            continue
        base = share(groups["전체"], attr).sort_values(ascending=False)
        for v in base.index:
            row = {"속성": label, "값": v, "전체": base[v]}
            for s in SETS:
                row[s] = float(share(groups[s], attr).get(v, 0.0))
            rows.append(row)
    base_cls = class_share(groups["전체"])
    for c in CLASSES:
        row = {"속성": "클래스 (객체)", "값": c, "전체": base_cls[c]}
        for s in SETS:
            row[s] = float(class_share(groups[s])[c])
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------


def plot_deviation(dev: pd.DataFrame, out: Path, keep_thr: float = 0.5) -> None:
    """가로 그룹 막대 — 전체 대비 편차. 0 에 붙을수록 '전체를 닮았다'."""
    plt = setup_plot()
    n = len(dev)
    fig, ax = plt.subplots(figsize=(10, 0.52 * n + 2.4))
    y = np.arange(n)[::-1]
    h = 0.24

    ax.axvspan(0, keep_thr, color=BAND, zorder=0)
    ax.axvline(keep_thr, color=KEEP_LINE, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(keep_thr, y.max() + 0.75, f"  유지 기준 {keep_thr}%p", color=INK_2, fontsize=8.5, va="center")

    for k, s in enumerate(SETS):
        off = (1 - k) * h
        ax.barh(y + off, dev[s], height=h * 0.86, color=COLORS[s], edgecolor=SURFACE,
                linewidth=0.8, zorder=3, label=s)
        # test 색이 배경 대비 3:1 미만 — 값 라벨을 항상 찍어 색에만 기대지 않는다
        for yy, v in zip(y + off, dev[s]):
            ax.text(v + dev[SETS].to_numpy().max() * 0.012, yy, f"{v:.2f}", va="center",
                    fontsize=7.6, color=INK_2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(dev["속성"])
    ax.set_xlabel("전체 분포 대비 최대 절대 편차 (%p) — 0 에 가까울수록 전체와 같다")
    ax.set_xlim(0, dev[SETS].to_numpy().max() * 1.16)
    ax.set_ylim(-0.7, y.max() + 1.15)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("train 만 의도적으로 틀어져 있다", pad=34)
    ax.annotate("유지해야 할 축(CCTV·지역·화질)은 0 에 붙고, 바꾸려 한 축(난이도·주야간)만 크게 벌어진다",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom")
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_reduction(red: pd.DataFrame, out: Path, n_before: int, n_after: int) -> None:
    """아령(dumbbell) — 축소 전 → 후 비율 이동."""
    plt = setup_plot()
    n = len(red)
    fig, ax = plt.subplots(figsize=(10, 0.46 * n + 2.6))
    y = np.arange(n)[::-1]

    up = red["after_pct"] >= red["before_pct"]
    for yy, b, a, grew in zip(y, red["before_pct"], red["after_pct"], up):
        ax.plot([b, a], [yy, yy], color=COLORS["train"] if grew else KEEP_LINE,
                lw=2, solid_capstyle="round", zorder=2, alpha=0.55 if not grew else 0.9)
    ax.scatter(red["before_pct"], y, s=52, color=BAND, edgecolor=KEEP_LINE, linewidth=1.2,
               zorder=3, label=f"축소 전 · train 후보 {n_before:,}장")
    ax.scatter(red["after_pct"], y, s=58, color=COLORS["train"], edgecolor=SURFACE, linewidth=1.2,
               zorder=4, label=f"축소 후 · train {n_after:,}장")

    span = max(red[["before_pct", "after_pct"]].to_numpy().max(), 1.0)
    for yy, b, a in zip(y, red["before_pct"], red["after_pct"]):
        ax.text(max(a, b) + span * 0.02, yy, f"{b:.1f}% → {a:.1f}%", va="center",
                fontsize=8, color=INK_2, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(red["항목"])
    ax.set_xlabel("train 안에서 차지하는 비율 (%)")
    ax.set_xlim(0, span * 1.30)
    ax.set_ylim(-0.8, y.max() + 0.8)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{n_before:,}장 → {n_after:,}장 : 버린 뒤에 오히려 늘어난 것들", pad=34)
    ax.annotate("어려운 조건과 희귀 클래스만 늘었고, 도로 형태·truck 비율은 그대로다",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_distribution(det: pd.DataFrame, out: Path) -> None:
    """회색 막대 = 전체, 점 3개 = train·val·test 의 위치."""
    plt = setup_plot()
    groups = list(dict.fromkeys(det["속성"]))
    sizes = [int((det["속성"] == g).sum()) for g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(9.6, sum(0.42 * s + 1.0 for s in sizes) + 1.0),
                             gridspec_kw={"height_ratios": [0.42 * s + 1.0 for s in sizes]})
    axes = np.atleast_1d(axes)

    for ax, g in zip(axes, groups):
        sub = det[det["속성"] == g].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub["전체"], height=0.62, color=BAND, edgecolor="none", zorder=1)
        for k, s in enumerate(SETS):
            ax.scatter(sub[s], y + (1 - k) * 0.19, s=40, color=COLORS[s], edgecolor=SURFACE,
                       linewidth=1.1, zorder=3, label=s if ax is axes[0] else None)
        # 라벨은 막대 끝이 아니라 '전체·train·val·test 중 가장 오른쪽' 뒤에 둔다 (점과 겹치지 않게)
        rightmost = sub[["전체", *SETS]].max(axis=1)
        for yy, v, r in zip(y, sub["전체"], rightmost):
            ax.text(r + 1.8, yy, f"전체 {v:.1f}%", va="center", fontsize=7.6, color=INK_2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["값"])
        ax.set_xlim(0, 108)
        ax.set_ylim(-0.7, y.max() + 0.7)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(g, fontsize=10.5, pad=6)

    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=9,
                   bbox_to_anchor=(1.0, 1.42))
    axes[-1].set_xlabel("비율 (%) — 회색 막대가 전체 분포")
    fig.suptitle("val · test 는 전체 위에 겹치고, train 만 떨어져 있다", x=0.007, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).expanduser()
    if not manifest.exists():
        raise SystemExit(f"[error] 파일이 없습니다: {manifest}\n"
                         f"        split 을 먼저 실행하거나 --manifest 로 경로를 지정하세요.")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 읽는 중: {manifest}")
    groups = load(manifest)

    print("[2/3] 집계")
    dev = deviation_table(groups)
    red = reduction_table(groups)
    det = detail_table(groups)
    dev.round(3).to_csv(out_dir / "stats_deviation.csv", index=False, encoding="utf-8-sig")
    red.round(3).to_csv(out_dir / "stats_reduction.csv", index=False, encoding="utf-8-sig")
    det.round(3).to_csv(out_dir / "stats_distribution.csv", index=False, encoding="utf-8-sig")

    print("[3/3] 그림")
    plot_deviation(dev, out_dir / "01_deviation.png", args.keep_threshold)
    plot_reduction(red, out_dir / "02_reduction.png", len(groups["train 후보"]), len(groups["train"]))
    plot_distribution(det, out_dir / "03_distribution.png")

    keep = dev[dev["train"] <= args.keep_threshold]["속성"].tolist()
    moved = dev[dev["train"] > args.keep_threshold].head(4)
    print(f"\n완료 → {out_dir}")
    print(f"  유지된 축 ({len(keep)}개, train 편차 ≤ {args.keep_threshold}%p): {', '.join(keep)}")
    print("  틀어진 축:")
    for _, r in moved.iterrows():
        print(f"    {r['속성']:<14} train {r['train']:5.2f}%p  (val {r['val']:.2f} · test {r['test']:.2f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="split 후 train 축소 결과를 전체·val·test 와 비교하는 그림 3장",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", default=str(BASE_DIR / "car_seg_split" / "split_manifest.csv"),
                   help="split_manifest.csv 경로")
    p.add_argument("--out-dir", default=str(BASE_DIR / "car_seg_train_dist"), help="결과 폴더")
    p.add_argument("--keep-threshold", type=float, default=0.5,
                   help="'분포가 유지됐다'고 볼 편차 상한 (%%p)")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
