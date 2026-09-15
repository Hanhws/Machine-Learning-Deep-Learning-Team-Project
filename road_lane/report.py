"""발표용 핵심 수치와 그림을 한 번에 뽑는다.

출력 outputs/report/
  summary.json          차로 수 정확도, 방향 판정 성능, 숨은 정체 비율 등
  lane_accuracy.png     지역별 차로 수 정확도 (파일명 OW/TW+숫자 기준)
  hidden_congestion.png 도로 전체 점유율 vs 가장 붐비는 차로 점유율
  examples/             차로별 차이가 가장 큰 장면의 peak.jpg · timeseries.png
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT

REP = OUT / "report"


def lane_accuracy(summary: dict) -> None:
    d = pd.read_csv(OUT / "s2_summary.csv")
    d["region"] = d["camera"].str.split("_").str[0]
    ok = d[d["ok"]]
    by = ok.groupby("region").agg(scenes=("scene_id", "size"), exact=("lane_exact", "mean"),
                                  within1=("lane_within1", "mean"))
    summary["lane_count"] = {
        "scenes": int(len(d)), "extracted": int(len(ok)),
        "exact": round(float(ok["lane_exact"].mean()), 3), "within1": round(float(ok["lane_within1"].mean()), 3),
        "by_region": by.round(3).reset_index().to_dict("records"),
        "note": "정답은 파일명 차로 수(OW/TW+숫자). 한방향(OW) 파일인데 반대편 도로가 보이는 화면도 있어 완전한 정답은 아님",
    }
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(by))
    ax.bar(x - 0.2, by["exact"] * 100, 0.4, label="exact", color="#4C78A8")
    ax.bar(x + 0.2, by["within1"] * 100, 0.4, label="within ±1", color="#9ECAE9")
    ax.set_xticks(x, [f"{r}\n(n={n})" for r, n in zip(by.index, by["scenes"])], fontsize=8)
    ax.set_ylabel("% of scenes")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REP / "lane_accuracy.png", dpi=130)
    plt.close(fig)


def direction(summary: dict) -> None:
    ev = OUT / "s2b" / "eval.json"
    if ev.exists():
        summary["direction"] = json.loads(ev.read_text())
    dj = OUT / "s2b" / "directions.json"
    if dj.exists():
        res = json.loads(dj.read_text())
        cws = [v for s in res.values() for v in s.values()]
        both = [v for v in cws if v["geo"] and v["cnn"] and v.get("confident")]
        summary.setdefault("direction", {})["apply"] = {
            "carriageways": len(cws),
            "decided": sum(v["final"] is not None for v in cws),
            "decided_by_cnn_only": sum(v["geo"] is None and v["final"] is not None for v in cws),
            "geo_cnn_agreement": round(float(np.mean([v["geo"] == v["cnn"] for v in both])), 3) if both else None,
        }


def hidden_congestion(summary: dict) -> None:
    lanes = pd.read_csv(OUT / "s4" / "lane_delay.csv")
    groups = pd.read_csv(OUT / "s4" / "group_delay.csv")
    road = groups[groups["group"] == "all"][["scene_id", "frame", "occ_smooth", "level"]]
    worst = (lanes.sort_values("v_est_kmh").groupby(["scene_id", "frame"])
             .agg(worst_occ=("occ_smooth", "first"), worst_level=("level", "first")).reset_index())
    m = road.merge(worst, on=["scene_id", "frame"])
    hidden = (m["level"] == "원활") & (m["worst_level"] != "원활")
    summary["hidden_congestion"] = {
        "frames": int(len(m)),
        "road_free_but_some_lane_slow_or_jam": round(float(hidden.mean()), 4),
        "scenes_with_hidden_frames": int(m.loc[hidden, "scene_id"].nunique()),
        "median_gap_pp": round(float((m["worst_occ"] - m["occ_smooth"]).median() * 100), 2),
        "note": "도로 전체로 평균 내면 '원활'인데 어느 한 차로는 '서행' 이하인 프레임 비율 (속도는 점유율로 추정, 미검증)",
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(m["occ_smooth"] * 100, m["worst_occ"] * 100, s=3, alpha=0.25,
               c=np.where(hidden, "#E45756", "#4C78A8"))
    lim = max(m["worst_occ"].max(), m["occ_smooth"].max()) * 100
    ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("whole-road occupancy % (30s avg)")
    ax.set_ylabel("busiest lane occupancy %")
    ax.set_title("red: road 'free' but a lane is slow/jam", fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "hidden_congestion.png", dpi=130)
    plt.close(fig)

    status = pd.read_csv(OUT / "s4" / "scene_status.csv")
    top = status.dropna(subset=["lane_occ_spread_median"]).sort_values("lane_occ_spread_median", ascending=False)
    (REP / "examples").mkdir(exist_ok=True)
    picked = []
    for sid in top["scene_id"].drop_duplicates().head(3):
        for f in ("peak.jpg", "timeseries.png"):
            src = OUT / "s3" / sid / f
            if src.exists():
                shutil.copy(src, REP / "examples" / f"{sid}__{f}")
        picked.append(sid)
    summary["examples"] = picked


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    lane_accuracy(summary)
    direction(summary)
    hidden_congestion(summary)
    (REP / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
