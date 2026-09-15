"""두 결과 폴더의 핵심 지표를 나란히 놓고 비교한다.

  ../.venv/bin/python compare_runs.py outputs runs/내실험

outputs/ 의 요약 파일(FILES)은 GitHub 에 올라가 있어서 팀원 누구나 같은 기준과 비교할 수 있다.
파인튜닝 효과는 같은 설정으로 다시 돌려도 생기는 차이(README 의 "재실행 차이")보다 커야 의미가 있다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from common import ROOT

FILES = ["s2_summary.csv", "s2b/eval.json", "s2b/directions.json", "s4/scene_status.csv", "report/summary.json"]
LEVELS = ["원활", "서행", "정체"]


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def load(name: str) -> dict:
    out = Path(name) if Path(name).is_absolute() else ROOT / name
    missing = [f for f in FILES if not (out / f).exists()]
    if missing:
        sys.exit(f"{out} 에 없는 파일: {missing} (run_all.sh 를 끝까지 돌렸는지 확인)")
    return {
        "s2": pd.read_csv(out / "s2_summary.csv"),
        "eval": json.loads((out / "s2b/eval.json").read_text()),
        "dirs": json.loads((out / "s2b/directions.json").read_text()),
        "status": pd.read_csv(out / "s4/scene_status.csv"),
        "summary": json.loads((out / "report/summary.json").read_text()),
    }


def metrics(r: dict) -> dict[str, float]:
    ok = r["s2"][truthy(r["s2"]["ok"])]
    cnn = r["eval"]["resnet18_finetuned"]
    cws = [v for scene in r["dirs"].values() for v in scene.values()]
    both = [v for v in cws if v["geo"] and v["cnn"] and v.get("confident")]
    st = r["status"][r["status"]["group"] != "all"]
    m = {
        "차로 수 정확 (파일명 기준)": truthy(ok["lane_exact"]).mean(),
        "차로 수 ±1 이내": truthy(ok["lane_within1"]).mean(),
        "방향 CNN 차량 정확도 (test 카메라)": cnn["vehicle_acc"],
        "방향 CNN 도로 다수결 정확도": cnn["carriageway_vote_acc"],
        "방향 CNN AUC": cnn["auc"],
        "방향이 정해진 도로 비율": sum(v["final"] is not None for v in cws) / len(cws),
        "기하 규칙-CNN 일치율": sum(v["geo"] == v["cnn"] for v in both) / len(both) if both else float("nan"),
        "숨은 정체 프레임 비율": r["summary"]["hidden_congestion"]["road_free_but_some_lane_slow_or_jam"],
    }
    for level in LEVELS:
        m[f"장면·방향 {level} 비율"] = (st["level_median"] == level).mean()
    return m


def agreement(a: dict, b: dict) -> dict[str, float]:
    """장면 하나하나가 같은 결과인지. 평균 지표가 같아도 장면별로는 엇갈릴 수 있다."""
    s = a["s2"].merge(b["s2"], on="scene_id", suffixes=("_a", "_b"))
    da = {(sid, cw): v["final"] for sid, d in a["dirs"].items() for cw, v in d.items()}
    db = {(sid, cw): v["final"] for sid, d in b["dirs"].items() for cw, v in d.items()}
    keys = da.keys() & db.keys()
    t = a["status"].merge(b["status"], on=["scene_id", "group"], suffixes=("_a", "_b"))
    return {
        "공통 장면 수": len(s),
        "차로 수가 같은 장면": (s["main_lanes_a"].astype(str) == s["main_lanes_b"].astype(str)).mean(),
        "방향 판정이 같은 도로": sum(da[k] == db[k] for k in keys) / len(keys) if keys else float("nan"),
        "소통 등급이 같은 장면·그룹": (t["level_median_a"] == t["level_median_b"]).mean(),
        "점유율 중앙값 평균 차이 (%p)": (t["occ_median_a"] - t["occ_median_b"]).abs().mean() * 100,
    }


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name_a, name_b = sys.argv[1], sys.argv[2]
    a, b = load(name_a), load(name_b)
    table = pd.DataFrame({name_a: metrics(a), name_b: metrics(b)})
    table["차이"] = table[name_b] - table[name_a]
    pd.set_option("display.unicode.east_asian_width", True)
    print(table.round(4).to_string())
    print("\n장면별로 같은 결과가 나온 비율")
    for k, v in agreement(a, b).items():
        print(f"  {k}: {v}" if isinstance(v, int) else f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
