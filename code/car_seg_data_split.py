#!/usr/bin/env python3
"""car_seg_dataset → val ≈ 1,000장 · test ≈ 5,000장 · train ≈ 2,500장 (YOLO segmentation 전용).

val · test 에 **CCTV 49대의 구도가 모두** 들어가고, 세 세트 모두 CCTV 비율(전체 데이터에서의 이미지 비율)을 유지한다.

**버리는 프레임 없음** — 모든 프레임이 train 후보 / val / test 중 하나에 배정된다.
(train 후보 중 --train-target 장만 학습 폴더에 들어가는 것은 요청한 train 축소)

1단계 — CCTV 별 val / test 클립 선택
    · 클립 3개 이상인 CCTV : val 클립 1개 · test 클립 1개(클립 4개 이상이면 2개까지) — 서로 다른 클립,
                             train 전용 클립이 최소 1개 남음
    · 클립 1~2개인 CCTV    : 한 클립 안에 val 구간과 test 구간을 둠
    어떤 클립을 쓸지는 val · test 의 촬영 조건(주/야간 · 날씨 · 난이도 · 시간대 · 요일 · 날짜 · NH/RH)과
    클래스 구성이 전체 데이터와 같아지도록 최적화한다 (좌표 하강 + 여러 번 재시작, --seed 고정).
    지역 · 도로 형태 · 차로 · 카메라 높이 · 해상도는 CCTV 에 고정된 속성이라 CCTV 비율만 맞추면 자동으로 맞는다.

2단계 — val / test 연속 구간 배치
    · CCTV 할당량 = 목표 장수 × (그 CCTV 의 전체 데이터 이미지 비율)  (최대 나머지 방식, CCTV 당 최소 1장)
    · 선택된 클립 안에서 할당량만큼 **연속된 구간**을 통째로 가져간다. 구간 위치는 시드 고정 무작위이며,
      한 클립에 val · test 가 같이 있으면 앞 · 뒤로 떨어뜨린다.
      (프레임을 드문드문 뽑으면 사이사이 프레임이 train 과 거의 같은 그림이 되므로 연속 구간으로 가져감)
    · 구간 밖의 프레임은 모두 train 후보가 된다.

3단계 — train 축소 (hard 우선 + 누수 방지)
    a. CCTV 비율 유지 — 위와 같은 CCTV 비율로 --train-target(기본 2500)장을 배분
    b. 선택 순서 — hard·train 전용 클립 → hard·val/test 와 같은 클립 → easy·train 전용 → easy·같은 클립
                   → val/test 구간 바로 옆(±--guard-frames 장) 프레임 (easy 는 --easy-policy fill 일 때만)
       단, 같은 클립 프레임이라도 그 CCTV 의 train 전용 클립에 없는 날씨·주야간(예: 유일한 눈 클립)이면 앞 순위로 쓴다
       → 드문 조건이 val · test 로만 가고 train 에서 사라지는 것을 막음
       easy = 기존 YOLO 가 잘 잡던 조건(주간 · 맑음 · 일반 도로, 셋 다 만족), hard = 그 외
    c. CCTV 안에서는 클립별 비례 → 클립 안 시간 구간마다 1장, 구간 안에서는 bus · truck 이 많은 프레임 우선
       (클래스 가중치 = (car 수 / 그 클래스 수) ** --class-alpha)

셔플
    세트 배정은 무작위 셔플이 아니라 CCTV · 클립 단위로 설계된 배정이다 (프레임을 무작위로 섞으면 거의 같은
    프레임이 train 과 test 에 동시에 들어가는 누수가 생김). 무작위성은 클립 선택 최적화의 재시작과 구간 위치에만
    쓰고 --seed 로 고정한다. 학습 시 배치 순서는 Ultralytics 가 epoch 마다 train 을 섞는다.

입력 (car_seg_data_preprocessing.py 결과)
    car_seg_dataset/  images/(JPG)  labels/  manifest.csv  [preprocess_report.json]

출력 (기본: car_seg_split/)
    images/{train,val,test}/        JPG (car_seg_dataset 으로의 hardlink = 실제 파일, 추가 용량 0 · 업로드 가능)
    labels/{train,val,test}/        YOLO seg 라벨
    data.yaml                       Ultralytics 학습 설정
    README.md                       분할 요약과 학습 명령
    split_manifest.csv              전체 이미지 + pool(배정 세트) + split(학습 폴더 세트, train 축소로 빠진 건 빈칸)
                                    + selection(사유) + leak_risk(train 후보의 누수 위험도)
    split_clips.csv                 클립별 val / test 구간 프레임 범위와 train 후보 장수
    camera_coverage.csv             CCTV × 세트 이미지 수 (val · test 에 49대가 모두 있는지 확인)
    train_selection_cameras.csv     CCTV 별 train 할당 · hard/easy 선택 수
    split_report.json               세트 크기 · 분포 편차 · 커버리지 · 누수 점검 · 축소 전후 비교
    split_distribution.png          세트별 속성 분포 비교
    train_reduction.png             train 축소 전후 비교 (CCTV 비율 · 조건 · 클래스)

사용 예
    python car_seg_data_split.py
    python car_seg_data_split.py --val-size 1000 --test-size 5000 --train-target 2500 --overwrite
    python car_seg_data_split.py --train-target 0          # train 을 줄이지 않음 (train 구간 전체)
    yolo segment train data=car_seg_split/data.yaml model=yolo11s-seg.pt imgsz=1280
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # 같은 폴더의 car_seg_data_preprocessing 을 가져오기 위해

from car_seg_data_preprocessing import EASY_CONDITION, ROAD_FORM_KO, hard_reasons, road_form_of  # noqa: E402

SPLITS = ("train", "val", "test")
OUTPUT_SUBDIRS = ("images", "labels")  # --overwrite 때 지우고 다시 만드는 폴더
DEFAULT_CLASSES = {0: "car", 1: "bus", 2: "truck"}

# val / test 클립 선택 때 맞출 '클립 단위' 속성과 가중치.
# (site · road_form · lane_config · cam_height · quality · orientation 은 CCTV 에 고정 → CCTV 비율로 자동 일치)
CLIP_ATTRS = {
    "day_night": 1.0,
    "weather": 1.0,
    "difficulty": 1.0,
    "time_band": 1.0,
    "weekday": 0.5,
    "hour_type": 0.5,
    "date": 0.3,
}
CLASS_WEIGHT = 1.0
SHORTFALL_WEIGHT = 5.0   # 클립이 짧아 할당량을 못 채울 때의 벌점 (부족 장수 / 세트 목표)
# val · test 클립에 남는 hard 프레임 벌점 (남는 hard 장수 / 전체 hard 장수).
# 남는 프레임은 train 후보로 가지만 val · test 와 같은 클립이라 후순위로 뽑힌다 → hard 가 많은 클립을 val · test 로
# 쓰면 train 이 쓸 수 있는 '깨끗한' hard 가 줄어든다.
LEFTOVER_HARD_WEIGHT = 3.0
DEFAULT_GUARD_FRAMES = 10 # val · test 구간 바로 옆 프레임(±N 장)은 train 에서 가장 마지막에 뽑는다

WEEKDAY_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
SUN_PHASE_KO = {"day": "주간", "twilight": "여명·황혼", "night": "야간"}

DIST_ATTRS = ["site", "camera", "road_form", "day_night", "weather", "difficulty", "time_band", "weekday",
              "hour_type", "date", "lane_config", "cam_height", "quality", "orientation"]
PLOT_ATTRS = [
    ("site", "촬영 지역"), ("road_form", "도로 형태"), ("day_night", "주 / 야간"), ("weather", "날씨"),
    ("difficulty", "난이도 조건"), ("time_band", "촬영 시간대"), ("weekday", "요일"), ("hour_type", "NH / RH"),
    ("lane_config", "차로 구성"), ("cam_height", "카메라 높이"), ("quality", "화질"), ("orientation", "영상 방향"),
    ("class", "클래스 (객체 수)"),
]


# ----------------------------------------------------------------------------
# 공통: 정수 배분 · 프레임 선택
# ----------------------------------------------------------------------------


def largest_remainder(weights: pd.Series, total: int, minimum: int = 0, caps: pd.Series | None = None) -> pd.Series:
    """weights 비율대로 total 을 정수 배분 (최대 나머지 방식). minimum · caps(상한) 를 지키며 남는 몫은 재배분."""
    weights = weights.astype(float)
    alloc = pd.Series(0, index=weights.index, dtype=int)
    if total <= 0 or weights.sum() <= 0:
        return alloc
    caps = caps.reindex(weights.index).fillna(0).astype(int) if caps is not None else \
        pd.Series(np.iinfo(np.int32).max, index=weights.index)
    if minimum and minimum * int((weights > 0).sum()) <= total:
        alloc[:] = np.minimum(minimum, caps).where(weights > 0, 0)
    while True:
        remaining = total - int(alloc.sum())
        open_ = (alloc < caps) & (weights > 0)
        if remaining <= 0 or not open_.any():
            break
        w = weights[open_] / weights[open_].sum()
        raw = w * (remaining + alloc[open_].sum())  # 열린 항목끼리 전체 몫을 다시 나눔
        extra = (np.floor(raw) - alloc[open_]).clip(lower=0).astype(int)
        extra = np.minimum(extra, caps[open_] - alloc[open_])
        if extra.sum() > remaining:  # minimum 때문에 넘치는 경우 — 비율대로 줄임
            extra = np.floor(extra * remaining / extra.sum()).astype(int)
        alloc[open_] += extra
        remaining = total - int(alloc.sum())
        if remaining <= 0:
            break
        progressed = int(extra.sum())
        for idx in (raw - np.floor(raw)).sort_values(ascending=False).index:  # 나머지가 큰 순서로 1장씩
            if remaining <= 0:
                break
            if alloc[idx] < caps[idx]:
                alloc[idx] += 1
                remaining -= 1
                progressed += 1
        if not progressed:
            break
    return alloc


def class_weights(df: pd.DataFrame, class_names: list[str], alpha: float) -> dict[str, float]:
    counts = {n: max(1, int(df[f"gt_{n}"].sum())) for n in class_names}
    ref = counts[class_names[0]]
    return {n: (ref / c) ** alpha for n, c in counts.items()}


def pick_frames(pool: pd.DataFrame, k: int, weights: dict[str, float] | None = None) -> list[str]:
    """pool 에서 k 장 선택 — 클립별 비례 배분 → 클립 안 프레임 순서를 k 구간으로 나눠 구간마다 1장.

    weights 가 있으면 구간 안에서 클래스 점수가 가장 높은 프레임, 없으면 구간 가운데 프레임(편향 없음)."""
    if k <= 0 or pool.empty:
        return []
    if k >= len(pool):
        return pool["image_id"].tolist()
    per_clip = pool.groupby("clip").size()
    alloc = largest_remainder(per_clip, k, caps=per_clip)
    chosen: list[str] = []
    for clip, n in alloc.items():
        if n <= 0:
            continue
        g = pool[pool["clip"] == clip].sort_values("frame_num")
        score = sum(g[f"gt_{c}"].to_numpy(float) * w for c, w in weights.items()) if weights else np.zeros(len(g))
        for b in np.array_split(np.arange(len(g)), n):
            if len(b):
                mid = b[len(b) // 2]
                sub = g.iloc[b].assign(_score=score[b], _d=np.abs(b - mid))
                chosen.append(sub.sort_values(["_score", "_d"], ascending=[False, True])["image_id"].iat[0])
    return chosen


def contiguous_blocks(n_frames: int, quotas: list[int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """프레임 n 개짜리 클립에 연속 구간을 quotas 순서대로 겹치지 않게 놓는다. 반환: [(시작, 끝)] (끝 미포함, 정렬 위치).

    · 구간 1개 : 클립 안 무작위 위치 (시드 고정)
    · 구간 2개 : 한 구간은 클립 앞쪽, 다른 구간은 뒤쪽에 무작위로 놓아 두 구간을 최대한 떨어뜨림 (어느 쪽이 앞인지도 무작위)
    구간 밖의 프레임은 버리지 않고 train 후보가 된다."""
    quotas = [max(0, int(q)) for q in quotas]
    while sum(quotas) > n_frames:  # 할당량이 클립보다 크면 큰 쪽부터 줄임 (caps 로 막지만 안전장치)
        quotas[int(np.argmax(quotas))] -= 1
    if len(quotas) == 1:
        start = int(rng.integers(0, n_frames - quotas[0] + 1))
        return [(start, start + quotas[0])]
    order = [0, 1] if rng.random() < 0.5 else [1, 0]
    q_first, q_last = quotas[order[0]], quotas[order[1]]
    slack = n_frames - q_first - q_last
    s_first = int(rng.integers(0, slack // 2 + 1))
    s_last = n_frames - q_last - int(rng.integers(0, slack // 2 + 1))
    blocks = {order[0]: (s_first, s_first + q_first), order[1]: (s_last, s_last + q_last)}
    return [blocks[0], blocks[1]]


# ----------------------------------------------------------------------------
# 1단계: CCTV 별 클립 역할 배정
# ----------------------------------------------------------------------------


class RoleProblem:
    """CCTV 마다 (val 클립, test 클립들) 선택지 중 하나를 골라 val · test 분포를 전체와 맞추는 문제."""

    def __init__(self, df: pd.DataFrame, class_names: list[str], val_q: pd.Series, test_q: pd.Series,
                 allow_two_test_clips: bool):
        self.df = df
        attrs = [a for a in CLIP_ATTRS if a in df.columns and df[a].nunique() > 1]
        self.attrs = attrs
        clip_first = df.groupby("clip")[["camera", *attrs]].first()
        clip_size = df.groupby("clip").size()
        clip_cls = df.groupby("clip")[[f"gt_{c}" for c in class_names]].mean()
        clip_hard = df.assign(_h=df["difficulty"] == "hard").groupby("clip")["_h"].mean()
        total_hard = max(1, int((df["difficulty"] == "hard").sum()))

        # 특징: 속성값 one-hot (이미지 1장당) + 이미지당 클래스 객체 수
        cols, weights, names = [], [], []
        for a in attrs:
            values = sorted(df[a].astype(str).unique())
            for v in values:
                cols.append((clip_first[a].astype(str) == v).astype(float))
                weights.append(CLIP_ATTRS[a] / len(values))
                names.append((a, v))
        for c in class_names:
            cols.append(clip_cls[f"gt_{c}"])
            weights.append(CLASS_WEIGHT / len(class_names))
            names.append(("class", c))
        self.clip_feat = pd.concat(cols, axis=1).to_numpy(float)                 # (n_clips, F)
        self.clip_index = {c: i for i, c in enumerate(clip_first.index)}
        self.w = np.asarray(weights)
        self.feature_names = names
        overall = np.column_stack([
            *[(df[a].astype(str) == v).to_numpy(float) for a, v in names if a != "class"],
            *[df[f"gt_{c}"].to_numpy(float) for c in class_names],
        ]).mean(0)                                                                 # 이미지 가중 전체 평균
        self.val_total, self.test_total = int(val_q.sum()), int(test_q.sum())
        self.target = {"val": overall * self.val_total, "test": overall * self.test_total}

        # CCTV 별 선택지
        self.cameras = sorted(df["camera"].unique())
        self.options: dict[str, list[dict]] = {}
        for cam in self.cameras:
            clips = sorted(clip_first.index[clip_first["camera"] == cam])
            qv, qt = int(val_q.get(cam, 0)), int(test_q.get(cam, 0))
            opts = []
            if len(clips) >= 3:  # val 클립 · test 클립 · train 전용 클립이 모두 따로
                max_t = 2 if allow_two_test_clips and len(clips) >= 4 else 1
                for v in clips:
                    rest = [c for c in clips if c != v]
                    for k in range(1, max_t + 1):
                        for t in itertools.combinations(rest, k):
                            if len(clips) - 1 - k < 1:
                                continue
                            opts.append({"mode": "separate", "val": v, "test": list(t),
                                         "avail_val": int(clip_size[v]), "avail_test": int(clip_size[list(t)].sum())})
            else:  # 클립 1~2개: 한 클립 안에 val 구간 · test 구간 (클립 2개면 다른 클립은 train 전용)
                for shared in clips:
                    n = int(clip_size[shared])
                    opts.append({"mode": "same_clip" if len(clips) == 2 else "single_clip", "val": shared,
                                 "test": [shared], "avail_val": max(1, n - min(qt, n - 1)),
                                 "avail_test": max(1, n - min(qv, n - 1))})
            for o in opts:  # 이 선택지가 val / test 에 더하는 특징 (할당량 × 클립 특징, 부족분 제외)
                ev, et = min(qv, o["avail_val"]), min(qt, o["avail_test"])
                o["feat_val"] = ev * self.clip_feat[self.clip_index[o["val"]]]
                sizes = np.array([clip_size[c] for c in o["test"]], float)
                mix = (self.clip_feat[[self.clip_index[c] for c in o["test"]]] * sizes[:, None]).sum(0) / sizes.sum()
                o["feat_test"] = et * mix
                short = (qv - ev) / max(1, self.val_total) + (qt - et) / max(1, self.test_total)
                used = {c: 0.0 for c in {o["val"], *o["test"]}}  # val · test 클립에서 쓰는 장수
                used[o["val"]] += ev
                for c, sz in zip(o["test"], sizes):
                    used[c] += et * sz / sizes.sum()
                leftover_hard = sum((clip_size[c] - u) * clip_hard[c] for c, u in used.items())
                if o["mode"] == "single_clip":  # 클립이 하나뿐이면 어차피 남는 프레임이 train → 벌점 없음
                    leftover_hard = 0.0
                o["penalty"] = SHORTFALL_WEIGHT * short + LEFTOVER_HARD_WEIGHT * leftover_hard / total_hard
            self.options[cam] = opts

    def cost(self, cur_val: np.ndarray, cur_test: np.ndarray, penalty: float) -> float:
        """val · test 분포 오차(목표 대비 상대 오차 × 가중치) + 부족분 · 버려지는 hard 벌점."""
        c = 0.0
        for cur, key in ((cur_val, "val"), (cur_test, "test")):
            t = self.target[key]
            ok = t > 1e-9
            c += float((np.abs(cur[ok] - t[ok]) / t[ok] * self.w[ok]).sum())
        return c + penalty

    def solve(self, restarts: int, seed: int) -> tuple[dict[str, dict], float]:
        rng = np.random.default_rng(seed)
        best, best_cost = None, np.inf
        for _ in tqdm(range(restarts), desc="클립 역할 최적화", unit="trial", dynamic_ncols=True):
            choice = {cam: int(rng.integers(len(self.options[cam]))) for cam in self.cameras}
            cur_v = sum(self.options[c][i]["feat_val"] for c, i in choice.items())
            cur_t = sum(self.options[c][i]["feat_test"] for c, i in choice.items())
            pen = sum(self.options[c][i]["penalty"] for c, i in choice.items())
            cost = self.cost(cur_v, cur_t, pen)
            improved = True
            while improved:  # 좌표 하강: CCTV 하나씩 가장 좋은 선택지로 바꿈
                improved = False
                for cam in rng.permutation(self.cameras):
                    old = self.options[cam][choice[cam]]
                    base_v, base_t, base_p = cur_v - old["feat_val"], cur_t - old["feat_test"], pen - old["penalty"]
                    costs = [self.cost(base_v + o["feat_val"], base_t + o["feat_test"], base_p + o["penalty"])
                             for o in self.options[cam]]
                    k = int(np.argmin(costs))
                    if costs[k] < cost - 1e-12:
                        choice[cam] = k
                        new = self.options[cam][k]
                        cur_v, cur_t = base_v + new["feat_val"], base_t + new["feat_test"]
                        pen = base_p + new["penalty"]
                        cost = costs[k]
                        improved = True
            if cost < best_cost:
                best, best_cost = {cam: self.options[cam][i] for cam, i in choice.items()}, cost
        return best, best_cost


# ----------------------------------------------------------------------------
# 3단계: train 축소 (CCTV 비율 유지 + hard 우선 + 클래스 불균형 완화)
# ----------------------------------------------------------------------------


LEAK_RISK_KO = {0: "train 전용 클립", 1: "val·test 와 같은 클립 (구간과 떨어짐)", 2: "val·test 구간 바로 옆"}


def reduce_train(pool: pd.DataFrame, quota: pd.Series, easy_policy: str, class_names: list[str],
                 alpha: float, cam_share: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """반환: (image_id → 'hard' | 'easy_fill' | 'dropped_easy' | 'dropped_hard'), CCTV 별 표.

    CCTV 마다 아래 순서로 할당량을 채운다 (leak_risk: 0 = train 전용 클립, 1 = val·test 와 같은 클립,
    2 = val·test 구간 바로 옆 프레임). hard 를 우선하되, 누수 위험이 가장 큰 프레임은 맨 마지막에 쓴다.
        hard·0 → hard·1 → easy·0 → easy·1 → hard·2 → easy·2   (--easy-policy drop 이면 easy 단계 생략)"""
    weights = class_weights(pool, class_names, alpha)
    tiers = [("hard", 0), ("hard", 1), ("easy", 0), ("easy", 1), ("hard", 2), ("easy", 2)]
    if easy_policy == "drop":
        tiers = [t for t in tiers if t[0] == "hard"]
    selection = pd.Series("dropped", index=pool["image_id"].values, dtype=object)
    rows = []
    for cam, q in quota.items():
        g = pool[pool["camera"] == cam]
        left = int(q)
        picked = {d: [] for d in ("hard", "easy")}
        by_risk = {0: 0, 1: 0, 2: 0}
        for diff, risk in tiers:
            if left <= 0:
                break
            cand = g[(g["difficulty"] == diff) & (g["select_priority"] == risk)]
            got = pick_frames(cand, min(left, len(cand)), weights)
            picked[diff] += got
            for r in g.loc[g["image_id"].isin(got), "leak_risk"]:
                by_risk[int(r)] += 1
            left -= len(got)
        selection[picked["hard"]] = "hard"
        selection[picked["easy"]] = "easy_fill"
        n_sel = len(picked["hard"]) + len(picked["easy"])
        rows.append({
            "camera": cam, "road_form": g["road_form"].iat[0] if len(g) else "",
            "target_share_pct": float(cam_share.get(cam, 0)) * 100,
            "pool_images": len(g), "pool_hard": int((g["difficulty"] == "hard").sum()),
            "pool_easy": int((g["difficulty"] == "easy").sum()),
            "quota": int(q), "selected_hard": len(picked["hard"]), "selected_easy_fill": len(picked["easy"]),
            "selected": n_sel, "shortfall": int(q) - n_sel,
            "selected_risk0_clean_clip": by_risk[0], "selected_risk1_same_clip": by_risk[1],
            "selected_risk2_next_to_block": by_risk[2],
        })
    diff = pool.set_index("image_id")["difficulty"]
    dropped = selection == "dropped"
    selection[dropped] = "dropped_" + diff[selection.index[dropped]].values
    cams = pd.DataFrame(rows).set_index("camera")
    cams["selected_share_pct"] = cams["selected"] / max(1, cams["selected"].sum()) * 100
    cams["share_diff_pctpt"] = cams["selected_share_pct"] - cams["target_share_pct"]
    cams["selected_hard_pct"] = cams["selected_hard"] / cams["selected"].replace(0, np.nan) * 100
    return selection, cams.round(3)


# ----------------------------------------------------------------------------
# 보고서
# ----------------------------------------------------------------------------


def value_order(attr: str, values) -> list[str]:
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
    orders = {"day_night": ["day", "twilight", "night"], "difficulty": ["easy", "hard"],
              "weather": ["sunny", "rainy", "fog", "snow", "tunnel"],
              "road_form": ["general", "bridge", "tunnel", "overpass", "shelter"]}
    if attr in orders:
        o = orders[attr]
        return [v for v in o if v in values] + sorted(v for v in values if v not in o)
    return sorted(values)


def distribution_tables(df: pd.DataFrame, class_names: list[str]) -> dict:
    """속성별 {값: {overall(전체 데이터)/train/val/test 이미지 비율(%), 세트별 클립 수}} 와 세트별 최대 편차(%p).

    세트 값은 최종 선택된 이미지 기준 (train 은 hard 우선이라 편차가 큰 것이 정상)."""
    out = {}
    sel = df[df["split"] != ""]
    for attr in DIST_ATTRS + ["class"]:
        if attr == "class":
            counts = pd.DataFrame({s: [sel.loc[sel["split"] == s, f"gt_{c}"].sum() for c in class_names]
                                   for s in SPLITS}, index=class_names)
            counts["overall"] = [df[f"gt_{c}"].sum() for c in class_names]
            clip_counts = None
        else:
            if attr not in df.columns:
                continue
            counts = pd.crosstab(sel[attr].astype(str), sel["split"]).reindex(columns=list(SPLITS), fill_value=0)
            counts = counts.reindex(value_order(attr, set(counts.index) | set(df[attr].astype(str))), fill_value=0)
            counts["overall"] = df[attr].astype(str).value_counts().reindex(counts.index, fill_value=0)
            clip_counts = sel.groupby([attr, "split"])["clip"].nunique().unstack(fill_value=0) \
                .reindex(index=counts.index, columns=list(SPLITS), fill_value=0)
        share = counts / counts.sum(0).replace(0, 1) * 100
        values = {}
        for v in counts.index:
            values[str(v)] = {
                **{f"{s}_pct": round(float(share.at[v, s]), 2) for s in ("overall", *SPLITS)},
                **{f"{s}_images": int(counts.at[v, s]) for s in SPLITS},
                **({f"{s}_clips": int(clip_counts.at[v, s]) for s in SPLITS} if clip_counts is not None else {}),
            }
        max_dev = {s: round(float((share[s] - share["overall"]).abs().max()), 2) for s in SPLITS}
        out[attr] = {"max_abs_dev_pctpt": max_dev, "values": values}
    return out


def add_leak_risk(df: pd.DataFrame, guard: int) -> pd.Series:
    """train 후보 프레임의 누수 위험도 (0 = val·test 가 없는 클립, 1 = 같은 클립이지만 구간과 guard 장 넘게 떨어짐,
    2 = val·test 구간에서 guard 장 이내). val · test 프레임은 -1."""
    risk = pd.Series(0, index=df.index, dtype=int)
    risk[df["pool"] != "train"] = -1
    touched = df.loc[df["pool"] != "train", "clip"].unique()
    for clip in touched:
        g = df[df["clip"] == clip].sort_values("frame_num")
        pos = np.arange(len(g))
        held = pos[(g["pool"] != "train").to_numpy()]
        is_train = (g["pool"] == "train").to_numpy()
        # 각 프레임에서 가장 가까운 val·test 프레임까지의 거리 (정렬 위치 기준)
        dist = np.abs(pos[:, None] - held[None, :]).min(1)
        r = np.where(dist <= guard, 2, 1)
        risk.loc[g.index[is_train]] = r[is_train]
    return risk


def select_priority(df: pd.DataFrame) -> pd.Series:
    """train 선택 우선순위 = leak_risk 인데, 같은 클립(1) 프레임이라도 그 CCTV 의 train 전용 클립(0)에 없는
    촬영 조건(날씨 · 주야간)이면 0 으로 올린다. 예) 눈 클립이 하나뿐인데 test 클립이 되면, 남은 눈 프레임을
    후순위로 두면 train 에 눈이 한 장도 안 들어가므로 앞 순위로 쓴다. (구간 바로 옆 = 2 는 그대로 마지막)"""
    prio = df["leak_risk"].copy()
    key = df["camera"] + "|" + df["weather"] + "|" + df["day_night"]
    clean_keys = set(key[df["leak_risk"] == 0])
    rare = (df["leak_risk"] == 1) & ~key.isin(clean_keys)
    prio[rare] = 0
    return prio


def leakage_check(df: pd.DataFrame) -> dict:
    """세트 간에 같은 클립을 쓰는 경우와, 그때 세트끼리 가장 가까운 프레임 간격(프레임 번호 차)."""
    sel = df[df["split"] != ""]
    per_clip = sel.groupby("clip")["split"].agg(lambda s: tuple(sorted(set(s), key=SPLITS.index)))
    shared = per_clip[per_clip.map(len) > 1]
    out = {"clips_shared_between_splits": int(len(shared)),
           "val_test_same_clip": int(sum(1 for s in shared if {"val", "test"} <= set(s))),
           "train_with_val_or_test_clip": int(sum(1 for s in shared if "train" in s)),
           "min_frame_gap": {}, "clips": {}}
    gaps: dict[str, list[int]] = {}
    for clip, splits in shared.items():
        g = sel[sel["clip"] == clip]
        frames = {s: np.sort(g.loc[g["split"] == s, "frame_num"].to_numpy()) for s in splits}
        info = {"splits": list(splits), "images": {s: int(len(f)) for s, f in frames.items()}}
        for a, b in itertools.combinations(splits, 2):
            gap = int(np.abs(frames[a][:, None] - frames[b][None, :]).min())
            info[f"min_frame_gap_{a}_{b}"] = gap
            gaps.setdefault(f"{a}_{b}", []).append(gap)
        out["clips"][clip] = info
    out["min_frame_gap"] = {k: {"min": int(min(v)), "median": float(np.median(v))} for k, v in gaps.items()}
    return out


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    surface, text, text_2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
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
    return plt, (surface, text, text_2, grid)


def plot_distribution(tables: dict, out: Path) -> None:
    plt, (surface, text, text_2, grid) = setup_plot()
    from matplotlib.lines import Line2D

    colors = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
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
        ax.barh(y, [vals[n]["overall_pct"] for n in names], height=0.7, color="#f0efec", edgecolor="none", zorder=1)
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
    fig.text(0.01, 1 - 0.12 / h, "세트별 속성 분포 비교 (최종 선택 이미지 기준)", fontsize=15, fontweight="bold",
             color=text, va="top")
    fig.text(0.01, 1 - 0.46 / h, "회색 막대 = 전체 데이터 비율, 점 = 각 세트 비율 — val · test 는 막대 끝에 모일수록 좋음 "
             "(train 은 hard 우선이라 주간·맑음·easy 가 의도적으로 적음)", fontsize=10, color=text_2, va="top")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=s) for s, c in colors.items()]
    handles.append(Line2D([], [], color="#f0efec", linewidth=8, label="overall"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.99, 1 - 0.1 / h))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / h))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def before_after(pool: pd.DataFrame, selected: pd.DataFrame, class_names: list[str]) -> dict:
    """train 구간 전체 vs 축소 train 의 조건 · 클래스 분포 (%)."""
    out = {}
    for attr in ("difficulty", "day_night", "weather", "road_form", "site", "time_band"):
        a = pool[attr].value_counts(normalize=True) * 100
        b = selected[attr].value_counts(normalize=True) * 100
        idx = value_order(attr, set(a.index) | set(b.index))
        out[attr] = {v: {"before_pct": round(float(a.get(v, 0)), 2), "after_pct": round(float(b.get(v, 0)), 2),
                         "before_images": int((pool[attr] == v).sum()),
                         "after_images": int((selected[attr] == v).sum())} for v in idx}
    ob = {n: int(pool[f"gt_{n}"].sum()) for n in class_names}
    oa = {n: int(selected[f"gt_{n}"].sum()) for n in class_names}
    out["class"] = {n: {"before_objects": ob[n], "after_objects": oa[n],
                        "before_pct": round(ob[n] / max(1, sum(ob.values())) * 100, 2),
                        "after_pct": round(oa[n] / max(1, sum(oa.values())) * 100, 2),
                        "before_images_with": int((pool[f"gt_{n}"] > 0).sum()),
                        "after_images_with": int((selected[f"gt_{n}"] > 0).sum())} for n in class_names}
    out["car_to_class_ratio"] = {
        "before": {n: round(ob[class_names[0]] / max(1, ob[n]), 2) for n in class_names[1:]},
        "after": {n: round(oa[class_names[0]] / max(1, oa[n]), 2) for n in class_names[1:]},
    }
    return out


def plot_reduction(cams: pd.DataFrame, ba: dict, class_names: list[str], pool_size: int, target: int,
                   out: Path) -> None:
    plt, (surface, text, text_2, grid) = setup_plot()
    before_c, after_c = "#c3c2bd", "#2a78d6"
    n = len(cams)
    fig = plt.figure(figsize=(17, max(10.0, 0.25 * n + 2.5)))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.7, wspace=0.3)

    ax = fig.add_subplot(gs[:, 0])
    order = cams.sort_values("target_share_pct", ascending=False)
    y = np.arange(n)
    ax.barh(y, order["target_share_pct"], height=0.75, color="#f0efec", label="목표 비율 (전체 데이터 CCTV 비율)")
    ax.scatter(order["selected_share_pct"], y, s=26, color=after_c, zorder=3, label="축소 train 비율")
    fill = (order["selected_easy_fill"] > 0).to_numpy()
    ax.scatter(order.loc[fill, "selected_share_pct"], y[fill], s=60, facecolor="none", edgecolor="#eb6834",
               linewidth=1.3, zorder=4, label="easy 로 채운 CCTV")
    ax.set_yticks(y, [f"{c} [{ROAD_FORM_KO.get(f, f)}]  hard {h}/{s}" for c, f, h, s in
                      zip(order.index, order["road_form"], order["selected_hard"], order["selected"])], fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid)
    ax.set_axisbelow(True)
    ax.set_xlabel("train 안에서 CCTV 이미지 비율 (%)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"CCTV {n}대 비율 유지 — 최대 편차 {cams['share_diff_pctpt'].abs().max():.2f}%p")

    for r, (attr, title) in enumerate((("difficulty", "난이도 조건"), ("day_night", "주 / 야간"),
                                       ("weather", "날씨"), ("road_form", "도로 형태"))):
        ax = fig.add_subplot(gs[r, 1])
        vals = ba[attr]
        names = list(vals)
        yy = np.arange(len(names))
        ax.barh(yy - 0.2, [vals[v]["before_pct"] for v in names], height=0.38, color=before_c, label="train 구간 전체")
        ax.barh(yy + 0.2, [vals[v]["after_pct"] for v in names], height=0.38, color=after_c, label="축소 train")
        for yi, v in zip(yy, names):
            ax.text(vals[v]["after_pct"] + 1, yi + 0.2, f"{vals[v]['after_pct']:.0f}% ({vals[v]['after_images']:,})",
                    va="center", fontsize=7.5, color=text_2)
        if attr == "day_night":
            labels = [SUN_PHASE_KO.get(v, v) for v in names]
        elif attr == "road_form":
            labels = [f"{v} ({ROAD_FORM_KO.get(v, '')})" for v in names]
        else:
            labels = names
        ax.set_yticks(yy, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 115)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        extra = ""
        if r == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
            cls = ba["class"]
            extra = "   클래스(객체%) " + " · ".join(
                f"{c} {cls[c]['before_pct']:.1f}→{cls[c]['after_pct']:.1f}" for c in class_names)
        ax.set_title(f"{title}{extra}", fontsize=10.5)

    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, f"train 축소: {pool_size:,}장 → {int(cams['selected'].sum()):,}장 (목표 {target:,})",
             fontsize=15, fontweight="bold", color=text, va="top")
    cond = " · ".join(f"{k}={v}" for k, v in EASY_CONDITION.items())
    fig.text(0.01, 1 - 0.46 / h, f"CCTV 비율 유지 + hard 우선 (easy = {cond}) + 클래스 가중 프레임 선택",
             fontsize=10, color=text_2, va="top")
    fig.subplots_adjust(top=1 - 1.0 / h, left=0.17, right=0.98, bottom=0.05)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 파일 배치 · README
# ----------------------------------------------------------------------------


_FALLBACK = {"symlink": "hardlink", "hardlink": "copy"}
_effective_mode: dict[str, str] = {}


def place(src: Path, dst: Path, mode: str) -> None:
    """이미지 배치. 링크를 만들 수 없는 환경이면 symlink → hardlink → copy 순으로 자동 전환."""
    actual = _effective_mode.get(mode, mode)
    while True:
        try:
            if actual == "symlink":
                dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
            elif actual == "hardlink":
                os.link(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            break
        except (OSError, NotImplementedError) as exc:
            if actual not in _FALLBACK:
                raise
            nxt = _FALLBACK[actual]
            print(f"\n[warn] {actual} 를 만들 수 없어 {nxt} 로 바꿉니다 ({exc.__class__.__name__}: {exc})",
                  file=sys.stderr)
            actual = nxt
    _effective_mode[mode] = actual


def write_readme(out_dir: Path, sizes: dict, classes: dict[int, str], reduction: dict | None,
                 roles: pd.Series, leak: dict, n_cameras: int) -> None:
    names = [classes[k] for k in sorted(classes)]
    rows = "\n".join(f"| {s} | {v['target']:,} | {v['images']:,} | {v['clips']} | {v['cameras']}/{n_cameras} | "
                     f"{v['hard_pct']:.0f}% | " + " | ".join(f"{v[n]:,}" for n in names) + " |"
                     for s, v in sizes.items())
    mode_counts = roles.value_counts()
    same_cams = sorted(roles.index[roles != "separate"])
    red = ""
    if reduction:
        cls = reduction["before_after"]["class"]
        cond = " · ".join(f"`{k}={v}`" for k, v in EASY_CONDITION.items())
        cls_change = " · ".join(f"{n} {cls[n]['before_pct']:.1f}% → {cls[n]['after_pct']:.1f}%" for n in names)
        red = f"""
## train 축소
- train 구간 {reduction['pool']:,}장 → **{reduction['selected']:,}장** (목표 {reduction['target']:,})
- **CCTV 비율 유지** (전체 데이터 CCTV 비율 기준, 최대 편차 {reduction['max_camera_share_diff_pctpt']:.2f}%p)
- **hard 우선**: easy = {cond} 은 빼고 hard 에서 먼저 선택 — hard {reduction['selected_hard']:,}장 + easy 보충 {reduction['selected_easy_fill']:,}장
  (easy 로 보충한 CCTV {len(reduction['cameras_easy_filled'])}대: {', '.join(reduction['cameras_easy_filled']) or '없음'})
- **클래스 불균형 완화**: bus · truck 이 많은 프레임 우선 — 객체 비율 {cls_change}
- **누수 방지 순서**: train 전용 클립 → val·test 와 같은 클립(구간과 떨어진 프레임) → val·test 구간 바로 옆(±{reduction['guard_frames']}장)
  선택 결과: {reduction['selected_by_leak_risk']}
- 상세: `train_selection_cameras.csv`, `train_reduction.png`
"""
    text = f"""# car_seg_split — YOLO segmentation 학습 데이터

`car_seg_data_split.py` 로 생성. **val · test 에 CCTV {n_cameras}대의 구도가 모두** 들어가고, 세 세트 모두 CCTV 비율을 유지한다.

| split | 목표 | 이미지 | 클립 | CCTV | hard | """ + " | ".join(names) + """ |
|---|---|---|---|---|---|""" + "---|" * len(names) + f"""
{rows}

## 나누는 방법
- **버리는 프레임 없음** — 모든 프레임이 train 후보 / val / test 중 하나에 배정된다 (`split_manifest.csv` 의 `pool`)
- CCTV 마다 val 클립 · test 클립을 골라, 할당량만큼 **연속 구간**을 가져간다 (구간 위치는 시드 고정 무작위)
  - 클립 3개 이상인 CCTV ({mode_counts.get('separate', 0)}대): val · test 가 서로 다른 클립
  - 클립 1~2개인 CCTV ({len(same_cams)}대: {', '.join(same_cams) or '없음'}): 한 클립 안에서 val 구간과 test 구간을 앞·뒤로 떨어뜨림
- val · test 클립의 나머지 프레임은 train 후보가 되며, train 을 뽑을 때 후순위로 쓴다 (위 누수 방지 순서)
- 같은 클립을 쓰는 세트 쌍: train↔val/test {leak['train_with_val_or_test_clip']}클립 · val↔test {leak['val_test_same_clip']}클립
  (가장 가까운 프레임 간격: {', '.join(f"{k} {v['min']}" for k, v in leak['min_frame_gap'].items()) or '없음'})
- val · test 는 hard 편향 없음 → 실제 분포 그대로 평가 (분포 편차: `split_distribution.png`)
{red}
```
car_seg_split/
├── images/{{train,val,test}}/     JPG (car_seg_dataset/images 로의 hardlink = 실제 파일)
├── labels/{{train,val,test}}/     YOLO seg 라벨 (""" + ", ".join(f"{i}={n}" for i, n in enumerate(names)) + """)
├── data.yaml
├── split_manifest.csv            pool(배정 세트) · split(최종 세트) · selection(사유)
├── split_clips.csv               클립 역할
├── camera_coverage.csv           CCTV × 세트 이미지 수
├── train_selection_cameras.csv
├── split_report.json
├── split_distribution.png
└── train_reduction.png
```

## 학습
```bash
yolo segment train data=car_seg_split/data.yaml model=yolo11s-seg.pt imgsz=1280 epochs=100
yolo segment val   model=runs/segment/train/weights/best.pt data=car_seg_split/data.yaml split=test imgsz=1280
```
- Colab 업로드: `tar -cf car_seg_split.tar car_seg_split` (hardlink 라 실제 파일이 묶인다) 후 `data.yaml` 의 `path:` 수정
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        print(f"[error] manifest.csv 가 없습니다: {manifest}\n        먼저 car_seg_data_preprocessing.py 를 실행하세요.",
              file=sys.stderr)
        return 1
    for sub in OUTPUT_SUBDIRS:
        d = out_dir / sub
        if d.exists() and any(d.iterdir()) and not args.overwrite:
            print(f"[error] 출력 폴더가 이미 있습니다: {d}\n        다시 나누려면 --overwrite 를 붙이세요.", file=sys.stderr)
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
    df["frame_num"] = pd.to_numeric(df.get("frame", 0), errors="coerce").fillna(0).astype(int)
    if "camera" not in df.columns:
        df["camera"] = df["site"] + "_" + df["channel"]
    if "road_form" not in df.columns:
        df["road_form"] = df["channel"].map(road_form_of)
    # 난이도는 항상 현재 EASY_CONDITION 으로 다시 계산 (정의를 바꿔도 전처리를 다시 돌릴 필요 없음)
    reasons = ["|".join(hard_reasons(r)) for r in df[["day_night", "weather", "road_form"]].to_dict(orient="records")]
    df["hard_reasons"] = reasons
    df["difficulty"] = ["hard" if r else "easy" for r in reasons]

    n_total = len(df)
    n_cam = df["camera"].nunique()
    train_target = args.train_target if args.train_target else n_total - args.val_size - args.test_size
    if args.val_size < n_cam or args.test_size < n_cam:
        print(f"[error] --val-size / --test-size 는 CCTV 수({n_cam}) 이상이어야 모든 CCTV 를 넣을 수 있습니다.",
              file=sys.stderr)
        return 1
    if args.val_size + args.test_size + min(train_target, n_total) > n_total:
        print("[error] val + test + train 목표가 전체 이미지 수보다 큽니다.", file=sys.stderr)
        return 1
    print(f"[split] 이미지 {n_total:,}장 / 클립 {df['clip'].nunique()}개 / CCTV {n_cam}대 → "
          f"목표 train {args.train_target or '전체'} · val {args.val_size:,} · test {args.test_size:,}")

    # CCTV 비율 = 전체 데이터에서의 이미지 비율 — 세 세트 모두 이 비율로 배분
    cam_count = df.groupby("camera").size()
    cam_share = cam_count / n_total
    val_q = largest_remainder(cam_count, args.val_size, minimum=1)
    test_q = largest_remainder(cam_count, args.test_size, minimum=1)

    # 1) CCTV 별 val / test 클립 선택 ------------------------------------------------
    prob = RoleProblem(df, class_names, val_q, test_q, not args.one_test_clip)
    print(f"[split] val/test 클립 선택 때 맞출 속성: {prob.attrs} + class")
    best, cost = prob.solve(args.restarts, args.seed)
    roles = pd.Series({cam: o["mode"] for cam, o in best.items()})

    # 2) val / test 연속 구간 배치 — 나머지 프레임은 전부 train 후보 (버리는 프레임 없음) ---------
    # 클립이 짧아 할당량을 못 채우는 CCTV 의 부족분은 다른 CCTV 로 재배분
    caps = {"val": largest_remainder(cam_count, args.val_size, minimum=1,
                                     caps=pd.Series({c: o["avail_val"] for c, o in best.items()})),
            "test": largest_remainder(cam_count, args.test_size, minimum=1,
                                      caps=pd.Series({c: o["avail_test"] for c, o in best.items()}))}
    place_rng = np.random.default_rng(args.seed + 1)  # 구간 위치 무작위 (시드 고정 → 재현 가능)
    df["pool"] = "train"   # 배정된 세트 — 모든 프레임이 train / val / test 중 하나
    clip_rows = []
    for cam in sorted(best):
        o = best[cam]
        g = df[df["camera"] == cam]
        qv, qt = int(caps["val"][cam]), int(caps["test"][cam])
        blocks: dict[str, list[tuple[str, int, int]]] = {}  # clip → [(세트, 시작, 끝)]
        if o["val"] in o["test"]:  # 한 클립에 val · test 구간
            n = int((g["clip"] == o["val"]).sum())
            (va, vb), (ta, tb) = contiguous_blocks(n, [qv, qt], place_rng)
            blocks[o["val"]] = [("val", va, vb), ("test", ta, tb)]
        else:
            n = int((g["clip"] == o["val"]).sum())
            (va, vb), = contiguous_blocks(n, [qv], place_rng)
            blocks[o["val"]] = [("val", va, vb)]
            t_sizes = g[g["clip"].isin(o["test"])].groupby("clip").size()
            t_alloc = largest_remainder(t_sizes, qt, caps=t_sizes)
            for c, q in t_alloc.items():
                (ta, tb), = contiguous_blocks(int(t_sizes[c]), [int(q)], place_rng)
                blocks[c] = [("test", ta, tb)]
        for c, parts in blocks.items():
            frames = g[g["clip"] == c].sort_values("frame_num")
            for s, a, b in parts:
                df.loc[frames.index[a:b], "pool"] = s
        for c in sorted(g["clip"].unique()):
            frames = g[g["clip"] == c].sort_values("frame_num")
            row = {"clip": c, "camera": cam, "role": "train"}
            for s, a, b in blocks.get(c, []):
                if b > a:
                    row[f"{s}_frames"] = f"{frames['frame_num'].iat[a]:03d}-{frames['frame_num'].iat[b - 1]:03d}"
                    row[f"{s}_images"] = b - a
            if c in blocks:
                row["role"] = "+".join(s for s, a, b in blocks[c] if b > a) + " 구간 + 나머지 train 후보"
            row["train_candidates"] = int(len(frames)) - sum(b - a for _, a, b in blocks.get(c, []))
            clip_rows.append(row)

    df["split"] = np.where(df["pool"] != "train", df["pool"], "")
    df["selection"] = df["split"]
    df["leak_risk"] = add_leak_risk(df, args.guard_frames)
    df["select_priority"] = select_priority(df)

    # 3) train 축소 -------------------------------------------------------------
    pool = df[df["pool"] == "train"]
    reduction = None
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.train_target and args.train_target < len(pool):
        train_quota = largest_remainder(cam_count, args.train_target, minimum=1,
                                        caps=pool.groupby("camera").size())
        sel, cams = reduce_train(pool, train_quota, args.easy_policy, class_names, args.class_alpha, cam_share)
        df.loc[pool.index, "selection"] = pool["image_id"].map(sel).values
        keep = pool.index[pool["image_id"].map(sel).isin(["hard", "easy_fill"]).to_numpy()]
        df.loc[keep, "split"] = "train"
        cams.to_csv(out_dir / "train_selection_cameras.csv", encoding="utf-8-sig")
        selected_train = df.loc[keep]
        reduction = {
            "target": args.train_target, "easy_policy": args.easy_policy, "class_alpha": args.class_alpha,
            "class_weights": {k: round(v, 3) for k, v in class_weights(pool, class_names, args.class_alpha).items()},
            "easy_condition": EASY_CONDITION,
            "pool": len(pool), "selected": len(selected_train),
            "selected_hard": int((selected_train["selection"] == "hard").sum()),
            "selected_easy_fill": int((selected_train["selection"] == "easy_fill").sum()),
            "n_cameras": int((cams["selected"] > 0).sum()),
            "max_camera_share_diff_pctpt": round(float(cams["share_diff_pctpt"].abs().max()), 3),
            "cameras_easy_filled": cams.index[cams["selected_easy_fill"] > 0].tolist(),
            "cameras_shortfall": cams.loc[cams["shortfall"] > 0, "shortfall"].astype(int).to_dict(),
            "selected_by_leak_risk": {LEAK_RISK_KO[k]: int(v) for k, v in
                                      selected_train["leak_risk"].value_counts().sort_index().items()},
            "guard_frames": args.guard_frames,
            "before_after": before_after(pool, selected_train, class_names),
        }
    else:
        df.loc[pool.index, ["split", "selection"]] = ["train", "train"]
        for stale in ("train_selection_cameras.csv", "train_reduction.png"):
            (out_dir / stale).unlink(missing_ok=True)

    # 4) 파일 배치 -------------------------------------------------------------
    for sub in OUTPUT_SUBDIRS:
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
        for s in SPLITS:
            (out_dir / sub / s).mkdir(parents=True, exist_ok=True)

    missing_files = []
    use = df[df["split"] != ""]
    for row in tqdm(use.itertuples(index=False), total=len(use), desc="파일 배치", unit="img", dynamic_ncols=True):
        src_img = dataset_dir / "images" / row.file_name
        src_lbl = dataset_dir / "labels" / f"{row.image_id}.txt"
        if not src_img.exists() or not src_lbl.exists():
            missing_files.append(row.image_id)
            continue
        place(src_img, out_dir / "images" / row.split / row.file_name, args.link_mode)
        shutil.copy2(src_lbl, out_dir / "labels" / row.split / src_lbl.name)

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
    (out_dir / "classes.json").write_text(json.dumps({"yolo": {str(k): v for k, v in sorted(classes.items())}},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) 보고서 ----------------------------------------------------------------
    df.drop(columns=["frame_num"]).to_csv(out_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clip_rows).sort_values(["camera", "clip"]).to_csv(out_dir / "split_clips.csv", index=False,
                                                                   encoding="utf-8-sig")
    coverage = pd.crosstab(df.loc[df["split"] != "", "camera"], df.loc[df["split"] != "", "split"]) \
        .reindex(index=sorted(cam_count.index), columns=list(SPLITS), fill_value=0)
    coverage.insert(0, "role", roles.reindex(coverage.index))
    coverage.insert(1, "clips", df.groupby("camera")["clip"].nunique().reindex(coverage.index))
    coverage.insert(2, "dataset_share_pct", (cam_share * 100).round(3).reindex(coverage.index))
    coverage.to_csv(out_dir / "camera_coverage.csv", encoding="utf-8-sig")

    sizes = {}
    targets = {"train": args.train_target or len(pool), "val": args.val_size, "test": args.test_size}
    for s in SPLITS:
        sel = df[df["split"] == s]
        sizes[s] = {"target": int(targets[s]), "images": int(len(sel)), "clips": int(sel["clip"].nunique()),
                    "cameras": int(sel["camera"].nunique()),
                    "hard_pct": round(float((sel["difficulty"] == "hard").mean() * 100), 2) if len(sel) else 0,
                    **{name: int(sel[f"gt_{name}"].sum()) for name in class_names}}
    tables = distribution_tables(df, class_names)
    leak = leakage_check(df)
    missing_cams = {s: sorted(set(cam_count.index) - set(df.loc[df["split"] == s, "camera"])) for s in SPLITS}
    pool_counts = df["pool"].value_counts().reindex(list(SPLITS), fill_value=0).to_dict()
    usage = {
        "all_frames": n_total,
        "assigned_pool": pool_counts,                                   # 모든 프레임이 셋 중 하나 — 합계 = 전체
        "unassigned": int(n_total - sum(pool_counts.values())),
        "train_pool_selected": int((df["split"] == "train").sum()),
        "train_pool_not_selected_by_train_target": int(((df["pool"] == "train") & (df["split"] == "")).sum()),
    }
    report = {
        "dataset_dir": str(dataset_dir), "out_dir": str(out_dir), "seed": args.seed, "restarts": args.restarts,
        "link_mode": _effective_mode.get(args.link_mode, args.link_mode),
        "objective_cost": round(cost, 6), "clip_attrs": prob.attrs, "guard_frames": args.guard_frames,
        "sizes": sizes,
        "camera_roles": roles.value_counts().to_dict(),
        "same_clip_val_test_cameras": sorted(roles.index[roles != "separate"]),
        "cameras_missing": missing_cams,
        "leakage_check": leak,
        "image_usage": usage,
        "val_quota_by_camera": caps["val"].to_dict(), "test_quota_by_camera": caps["test"].to_dict(),
        "train_reduction": reduction,
        "missing_files": {"count": len(missing_files), "examples": missing_files[:50]},
        "distributions": tables,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    write_readme(out_dir, sizes, classes, reduction, roles, leak, n_cam)
    if not args.no_plot:
        plot_distribution(tables, out_dir / "split_distribution.png")
        if reduction:
            plot_reduction(pd.read_csv(out_dir / "train_selection_cameras.csv", index_col=0, encoding="utf-8-sig"),
                           reduction["before_after"], class_names, len(pool), args.train_target,
                           out_dir / "train_reduction.png")

    # 콘솔 요약 ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'split':6s} {'목표':>7s} {'이미지':>7s} {'클립':>5s} {'CCTV':>7s} {'hard%':>6s}  "
          + "  ".join(f"{n:>7s}" for n in class_names))
    for s in SPLITS:
        z = sizes[s]
        print(f"{s:6s} {z['target']:7,d} {z['images']:7,d} {z['clips']:5d} {z['cameras']:3d}/{n_cam:<3d} "
              f"{z['hard_pct']:5.1f}%  " + "  ".join(f"{z[n]:7,d}" for n in class_names))
    for s in SPLITS:
        if missing_cams[s]:
            print(f"  [주의] {s} 에 없는 CCTV: {missing_cams[s]}")
    if not any(missing_cams.values()):
        print(f"  [확인] train · val · test 모두 CCTV {n_cam}대 전부 포함")
    same = sorted(roles.index[roles != "separate"])
    print(f"\n프레임 배정: 전체 {n_total:,}장 = train 후보 {pool_counts['train']:,} + val {pool_counts['val']:,} + "
          f"test {pool_counts['test']:,}  (배정 안 된 프레임 {usage['unassigned']}장)")
    print(f"  train 후보 중 --train-target 으로 {usage['train_pool_selected']:,}장 선택 · "
          f"{usage['train_pool_not_selected_by_train_target']:,}장은 학습 폴더에 넣지 않음")
    print(f"  val·test 클립: 서로 다른 클립 {int((roles == 'separate').sum())}대 · 한 클립 안 구간 {len(same)}대 {same}")
    gap_txt = " · ".join(f"{k} 최소 {v['min']}프레임(중앙값 {v['median']:.0f})" for k, v in leak["min_frame_gap"].items())
    print(f"  같은 클립을 쓰는 세트 쌍: train↔val/test {leak['train_with_val_or_test_clip']}클립 · "
          f"val↔test {leak['val_test_same_clip']}클립 — 가장 가까운 프레임 간격: {gap_txt or '없음'}")
    print("\nval · test 분포 최대 편차 (세트 비율 − 전체 데이터 비율, %p)")
    for attr, t in tables.items():
        d = t["max_abs_dev_pctpt"]
        print(f"  {attr:12s} val {d['val']:5.2f}   test {d['test']:5.2f}   (train {d['train']:5.2f})")
    if reduction:
        r = reduction
        ba = r["before_after"]
        print(f"\n[train 축소] train 구간 {r['pool']:,}장 → {r['selected']:,}장 (목표 {r['target']:,}, "
              f"--easy-policy {r['easy_policy']})")
        print(f"  hard {r['selected_hard']:,}장 + easy 보충 {r['selected_easy_fill']:,}장 · "
              f"CCTV 비율 최대 편차 {r['max_camera_share_diff_pctpt']:.2f}%p")
        if r["cameras_easy_filled"]:
            print(f"  easy 로 보충한 CCTV {len(r['cameras_easy_filled'])}대: {', '.join(r['cameras_easy_filled'])}")
        if r["cameras_shortfall"]:
            print(f"  [참고] 할당량을 못 채운 CCTV: {r['cameras_shortfall']}")
        print(f"  누수 위험별 선택: {r['selected_by_leak_risk']}")
        for attr in ("difficulty", "day_night", "weather", "road_form"):
            print(f"  {attr:10s}: " + " · ".join(f"{v} {x['before_pct']:.0f}→{x['after_pct']:.0f}%"
                                                for v, x in ba[attr].items()))
        print("  클래스(객체%): " + " · ".join(f"{n} {x['before_pct']:.1f}→{x['after_pct']:.1f}"
                                             for n, x in ba["class"].items())
              + f"  (car:bus {ba['car_to_class_ratio']['before']['bus']}→{ba['car_to_class_ratio']['after']['bus']}배)")
    if missing_files:
        print(f"\n[warn] 이미지/라벨 파일이 없어 건너뛴 항목 {len(missing_files)}개")
    print(f"\n  결과 폴더: {out_dir}")
    print("  data.yaml · README.md · camera_coverage.csv · split_manifest.csv · split_report.json")
    print("=" * 80)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="car_seg_dataset → val · test (CCTV 전부 포함) + train 축소 (YOLO seg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", default=str(BASE_DIR / "car_seg_dataset"), help="전처리 결과 폴더")
    p.add_argument("--out-dir", default=str(BASE_DIR / "car_seg_split"), help="분할 결과 폴더")
    p.add_argument("--val-size", type=int, default=1000, help="val 이미지 수 (CCTV 비율대로 배분)")
    p.add_argument("--test-size", type=int, default=5000, help="test 이미지 수 (CCTV 비율대로 배분)")
    p.add_argument("--train-target", type=int, default=2500, help="train 이미지 수 (0 = train 구간 전체 사용)")
    p.add_argument("--easy-policy", choices=("fill", "drop"), default="fill",
                   help="train 에서 hard 가 할당량보다 적은 CCTV 처리 — fill: 부족분만 easy 로 채움 / drop: easy 미사용")
    p.add_argument("--class-alpha", type=float, default=0.5,
                   help="train 프레임 선택 시 클래스 가중치 지수 — (car 수/클래스 수)**alpha. 0 이면 클래스 무시")
    p.add_argument("--one-test-clip", action="store_true",
                   help="CCTV 당 test 클립을 1개로 제한 (기본: 클립 4개 이상인 CCTV 는 2개까지 허용)")
    p.add_argument("--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES,
                   help="val · test 구간에서 이 장수 이내의 프레임은 train 에서 가장 마지막에 선택")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=30, help="클립 역할 최적화 재시작 횟수")
    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="hardlink",
                   help="이미지 배치 방식 (hardlink: 실제 파일이면서 추가 용량 0)")
    p.add_argument("--overwrite", action="store_true", help="기존 분할 결과(images/labels)를 지우고 다시 생성")
    p.add_argument("--no-plot", action="store_true", help="차트 생략")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
