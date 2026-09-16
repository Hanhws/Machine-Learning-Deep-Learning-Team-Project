"""SAM3(또는 SAM2) 차선 분할 실험의 입력 세트를 만든다.

화면이 서로 다른 그룹은 102개뿐이라(s2c_groups.json), 236개 장면을 전부 돌릴 필요가 없다.
그룹마다 대표 장면 하나의 '차 없는 배경' 사진만 모아 두면
분할 실험에도, 사람이 세는 채점표(할 일 0)에도 그대로 쓴다.

    ../.venv/bin/python prepare_inputs.py

결과:
    inputs/<scene_id>.jpg   그룹 대표 배경 102장
    manifest.csv            장면별 카메라·파일명 차로수·현재 예측·그룹 크기
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RL = PROJECT / "road_lane"
sys.path.insert(0, str(RL))

OUT = RL / "outputs"
INPUTS = ROOT / "inputs"


def background_path(scene_id: str) -> Path | None:
    """장면의 배경 사진을 찾는다. 회전형 카메라는 한 클립에 화면이 여러 개다."""
    clip_id = scene_id.split("__view")[0]
    views_file = OUT / "s1" / clip_id / "views.json"
    if not views_file.exists():
        return None
    with open(views_file) as f:
        views = json.load(f)["views"]
    for v in views:
        if v["scene_id"] == scene_id:
            p = OUT / "s1" / clip_id / v["background"]
            return p if p.exists() else None
    return None


def main() -> None:
    groups = json.loads((OUT / "s2c_groups.json").read_text())
    summary = pd.read_csv(OUT / "s2_summary.csv").set_index("scene_id")

    INPUTS.mkdir(exist_ok=True)
    for old in INPUTS.glob("*.jpg"):
        old.unlink()

    rows, missing = [], []
    for g in groups:
        scene_id = g["ref"]
        src = background_path(scene_id)
        if src is None:
            missing.append(scene_id)
            continue
        shutil.copy2(src, INPUTS / f"{scene_id}.jpg")
        r = summary.loc[scene_id] if scene_id in summary.index else None
        rows.append(
            {
                "scene_id": scene_id,
                "camera": g["camera"],
                "n_members": len(g["members"]),
                "token": r.token if r is not None else "",
                "lanes_pred": r.main_lanes if r is not None else "",
                "lane_exact": r.lane_exact if r is not None else "",
                "lane_within1": r.lane_within1 if r is not None else "",
                # 사람이 채울 칸 (할 일 0). 방향별로 "2/2" 처럼 적는다.
                "lanes_true": "",
                "note": "",
            }
        )

    pd.DataFrame(rows).to_csv(ROOT / "manifest.csv", index=False)
    print(f"배경 {len(rows)}장 → {INPUTS}")
    print(f"manifest.csv 작성 (사람이 채울 칸: lanes_true, note)")
    if missing:
        print(f"배경을 못 찾은 장면 {len(missing)}개: {missing[:5]}")

    d = pd.DataFrame(rows)
    covered = d.n_members.sum()
    print(f"\n대표 {len(d)}장면이 전체 {covered}장면을 대표함")
    print(f"현재 파이프라인 정확도(대표 장면 기준): 정확 {d.lane_exact.mean():.3f} · ±1 {d.lane_within1.mean():.3f}")


if __name__ == "__main__":
    main()
