# lane_sam3 — SAM3 차선 분할을 road_lane 에 붙이기

`road_lane` 2단계에서 **차선 도색만** Mask2Former → SAM3 로 바꾼다.
도로·분리대·소실점·차로 기하는 기존 코드 그대로다.

자세한 배경·결과·한계는 **[road_lane/docs/sam3_pipeline.html](../road_lane/docs/sam3_pipeline.html)** 을 먼저 볼 것.

## 결과 요약 (2026-09-16, 236장면)

| | 기준 (Mask2Former) | SAM3 차선 |
|---|---|---|
| 차로 수 정확 | 0.357 | **0.409** |
| 차로 수 ±1 | 0.770 | **0.783** |
| ±1, 수원 제외 | 0.893 | **0.964** |
| 양방향 판정 | 0.851 | **0.889** |

수원 밖에서는 나빠진 장면이 0개다. 손해는 전부 수원(68장면)에서 났다.

> **주의** 이 정확도는 파일명 차로 수(OW/TW)를 정답으로 잰 것이고,
> 팀 문서 기준 그 정답은 약 25% 만 맞다. 확정된 개선이 아니라 참고 수치로 읽을 것.

## 먼저 할 일 — SAM3 가중치

자동 다운로드가 안 되는 **게이트 모델**이다.

1. <https://huggingface.co/facebook/sam3> 에서 access 요청 (HF 계정 필요, 수동 승인)
2. 승인 후 토큰 발급 → `./.venv/bin/hf auth login`
3. 받기 — 3.45GB

```bash
cd 팀프로젝트
./.venv/bin/hf download facebook/sam3 sam3.pt --local-dir lane_sam3
```

라이선스는 Meta **SAM License** (Apache/MIT 아님). 실행은 전부 로컬이라 API 요금은 없다.
`sam3.pt` 는 재배포 금지라 `.gitignore` 에 있다 — 각자 받아야 한다.

## 돌리기

`outputs/` 기준 결과가 있다는 전제다. 1단계(차량 검출)와 Mask2Former 는 다시 돌리지 않는다.

```bash
# 1) 실험 폴더 준비
cd road_lane
EXP=runs/sam3
mkdir -p $EXP && cp -R outputs/s1 outputs/s2b $EXP/

# 2) SAM3 차선으로 seg.png 만들기 (236장면, 약 30분)
cd ../lane_sam3
../.venv/bin/python make_sam3_seg.py --out ../road_lane/$EXP

# 3) 2~4단계 (약 40분)
cd ../road_lane
export ROAD_LANE_OUT=runs/sam3  PY=../.venv/bin/python
$PY s2_road_lanes.py
$PY s2c_camera_groups.py --apply G
$PY s2d_curves.py apply
$PY s2b_direction.py apply
$PY s3_occupancy.py && $PY s4_delay.py && $PY report.py

# 4) 결과 보기
cd ../lane_sam3
../.venv/bin/python vis_seg.py --run ../road_lane/runs/sam3 --sheet 6
../.venv/bin/python final_sheet.py --run ../road_lane/runs/sam3 --base ../road_lane/outputs
```

마지막 줄이 `runs/sam3/FINAL.jpg` — 2~4단계 결과를 한 장에 담은 요약 그림을 만든다.

> 실험 폴더는 **매번 새 이름**을 쓸 것. 같은 폴더에서 다시 돌리면 2c·2d 가
> 옛 백업(`lanes_single.*`, `lanes_straight.*`)을 읽어서 고친 결과가 무시된다.

## 파일

| 파일 | 역할 |
|---|---|
| `make_sam3_seg.py` | **실제 채택된 것.** SAM3 차선 → s2 가 읽는 `seg.png`. 기본은 하이브리드(도로·분리대는 Mask2Former 유지) |
| `vis_seg.py` | 합쳐진 `seg.png` 색칠 (클래스 번호 지도라 그냥 열면 까맣다) |
| `final_sheet.py` | 2~4단계 결과를 한 장으로 합친 `FINAL.jpg` |
| `prepare_inputs.py` | 그룹 대표 배경 102장 + `manifest.csv` (사람이 셀 채점표 틀) |
| `run_sam3.py` | SAM3 마스크 추출. `--prompt`, `--outdir` 로 개념을 바꿔 뽑는다 |
| `group_lanes.py` | *참고용.* SAM3 만으로 차로를 세는 독립 시도 — 기존보다 나빠서 채택 안 함 |

## 왜 하이브리드인가

처음엔 SAM3 로 전부 바꾸려 했는데, **SAM3 는 도로 면 같은 영역에 불안정**했다.
같은 카메라에서 도로 면적이 38.6% → 12.6% 로 튀었다. 인스턴스 분할 모델이라
"도로" 같은 stuff 개념에 약하다.

반대로 차선은 SAM3 가 훨씬 잘 찾는다. 어떤 장면에서는 Mask2Former 가 0.5% 밖에
못 찾은 차선을 SAM3 가 5.3% 찾아냈다.

그래서 각자 잘하는 것만 쓴다 — 도로·분리대는 Mask2Former, 차선만 SAM3.
전부 SAM3 로 하려면 `make_sam3_seg.py --mode full` 이지만 권하지 않는다.

`group_lanes.py` 로 차로까지 직접 세 본 결과는 정확 0.303 으로 기존 s2(0.384)보다 나빴다.
**병목은 차선 검출이 아니라 찾은 차선을 차로로 해석하는 기하**였다.

## 다음에 할 일

1. **수원만 기존 차선 유지** — 수원 밖 손해가 0이라 전체가 더 오를 가능성이 크다
2. **채점표** — `prepare_inputs.py` 가 만드는 `manifest.csv` 의 `lanes_true` 칸을 채운다. 102장, 6명이면 1인당 17장
3. SAM3 프롬프트 튜닝 — 지금은 `"lane marking"` 하나뿐
