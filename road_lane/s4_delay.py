"""4) 점유율 → 추정 속도 → 1km 당 지체 시간 → 소통 등급 → 우회 검토 여부.

Greenshields 속도-밀도 모형:  v = v_free · (1 − k / k_jam)
점유율(occ)은 밀도에 비례한다고 보고  k / k_jam ≈ occ / occ_jam  으로 바꾼다.

  occ_jam   정체 점유율 (기본 0.6 = 차량 길이 4.5m / 정체 차간 7.5m). --occ-jam 으로 바꿀 수 있다.
  v_free    고속도로 자유 주행 속도 (기본 100 km/h)
  지체      1km 통행시간(60/v 분) − 자유 주행 통행시간(60/v_free 분)

⚠ 이 데이터에는 실제 속도가 없어 추정값을 검증하지 못한다.
  한국도로공사 VDS(5분 단위 속도·점유율)와 같은 지점·시각을 맞춰 보면 검증·보정할 수 있다.

입력 outputs/s3/lane_frames.csv, group_frames.csv
출력 outputs/s4/
  lane_delay.csv    차로·프레임별 (30초 이동평균) 추정 속도·지체·등급
  group_delay.csv   방향별·도로 전체
  scene_status.csv  장면·방향별 요약 (중앙값, 정체 비율, 우회 검토)
  model.png         점유율 분포와 점유율→속도 곡선
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT

V_FREE_KMH = 100.0
# 정체 점유율: 평균 차량 길이 약 4.5m / 정체 시 차간(차 앞머리 사이) 약 7.5m ≈ 0.6.
# 데이터 상위 1%(약 0.37)를 쓰면 이 데이터가 대부분 원활이라 너무 낮아, 점유율 7%만 넘어도 '서행'이 된다.
# (수원 진출 차로는 줄지어 서행하는데 40~45% → 정체 점유율은 이보다 커야 한다)
OCC_JAM = 0.6
V_MIN_KMH = 5.0  # 완전 정지로 통행시간이 무한대가 되는 것을 막는다
# 소통 등급 경계 (km/h). 발표 전 한국도로공사 공식 기준으로 확인해서 맞출 것.
LEVELS = [(80.0, "원활"), (40.0, "서행"), (0.0, "정체")]
SMOOTH_FRAMES = 10  # 약 30초
DETOUR_DELAY_MIN_PER_KM = 1.0  # 방향 전체 지체가 1km 당 1분 이상이면 우회 검토


def speed(occ: np.ndarray, occ_jam: float, v_free: float) -> np.ndarray:
    return np.clip(v_free * (1 - occ / occ_jam), V_MIN_KMH, v_free)


def level(v: np.ndarray) -> np.ndarray:
    out = np.full(v.shape, LEVELS[-1][1], dtype=object)
    for bound, name in reversed(LEVELS):
        out[v >= bound] = name
    return out


def add_estimates(df: pd.DataFrame, keys: list[str], occ_jam: float, v_free: float) -> pd.DataFrame:
    df = df.sort_values(keys + ["t_sec"]).copy()
    df["occ_smooth"] = df.groupby(keys)["occ_ground"].transform(
        lambda s: s.rolling(SMOOTH_FRAMES, min_periods=1, center=True).mean())
    df["v_est_kmh"] = speed(df["occ_smooth"].to_numpy(), occ_jam, v_free).round(1)
    df["delay_min_per_km"] = (60 / df["v_est_kmh"] - 60 / v_free).round(3)
    df["level"] = level(df["v_est_kmh"].to_numpy())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-free", type=float, default=V_FREE_KMH)
    ap.add_argument("--occ-jam", type=float, default=OCC_JAM)
    args = ap.parse_args()

    lanes = pd.read_csv(OUT / "s3" / "lane_frames.csv")
    groups = pd.read_csv(OUT / "s3" / "group_frames.csv")
    lanes = lanes[~lanes["shoulder"]]
    p99 = float(np.percentile(lanes["occ_ground"], 99))
    occ_jam = args.occ_jam
    print(f"occ_jam = {occ_jam:.3f} (참고: 데이터 차로 점유율 상위 1% = {p99:.3f}), v_free = {args.v_free} km/h")

    out = OUT / "s4"
    out.mkdir(exist_ok=True)
    lane_delay = add_estimates(lanes, ["scene_id", "lane_id"], occ_jam, args.v_free)
    group_delay = add_estimates(groups, ["scene_id", "group"], occ_jam, args.v_free)
    lane_delay.to_csv(out / "lane_delay.csv", index=False)
    group_delay.to_csv(out / "group_delay.csv", index=False)

    status = (group_delay.groupby(["scene_id", "group"])
              .agg(frames=("frame", "size"), occ_median=("occ_smooth", "median"),
                   v_median=("v_est_kmh", "median"), delay_median=("delay_min_per_km", "median"),
                   jam_share=("level", lambda s: float((s == "정체").mean())),
                   slow_share=("level", lambda s: float((s == "서행").mean())))
              .reset_index())
    status["level_median"] = level(status["v_median"].to_numpy())
    status["detour_check"] = (status["group"] != "all") & (status["delay_median"] >= DETOUR_DELAY_MIN_PER_KM)
    # 같은 장면에서 차로끼리 얼마나 다른지: 차로별로 따로 보는 의미가 있는지 확인용
    spread = (lane_delay.groupby(["scene_id", "direction", "frame"])["occ_smooth"]
              .agg(lambda s: s.max() - s.min()).groupby(["scene_id", "direction"]).median()
              .rename("lane_occ_spread_median").reset_index().rename(columns={"direction": "group"}))
    status = status.merge(spread, on=["scene_id", "group"], how="left").round(4)
    status.to_csv(out / "scene_status.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(lanes["occ_ground"] * 100, bins=60, color="#4C78A8")
    axes[0].axvline(occ_jam * 100, color="#E45756", ls="--", label=f"occ_jam {occ_jam * 100:.0f}%")
    axes[0].set_xlabel("lane occupancy % (per frame)")
    axes[0].set_ylabel("frames × lanes")
    axes[0].legend(frameon=False)
    o = np.linspace(0, occ_jam, 100)
    axes[1].plot(o * 100, speed(o, occ_jam, args.v_free), color="#4C78A8")
    for bound, name in LEVELS[:-1]:
        axes[1].axhline(bound, color="gray", lw=0.8, ls=":")
    axes[1].set_xlabel("occupancy %")
    axes[1].set_ylabel("estimated speed km/h (unvalidated)")
    fig.tight_layout()
    fig.savefig(out / "model.png", dpi=120)

    dirs = status[status["group"] != "all"]
    print(f"장면·방향 {len(dirs)}개 | 등급(중앙값) {dirs['level_median'].value_counts().to_dict()} | "
          f"우회 검토 {int(dirs['detour_check'].sum())}개")


if __name__ == "__main__":
    main()
