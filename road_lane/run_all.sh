#!/usr/bin/env bash
# 전체 파이프라인. 기준 결과(outputs/)를 만든 순서 그대로 돌린다.
# 끝난 클립·장면·파일은 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 한다 (s2 이후 기하 계산은 매번 다시 함).
#
#   기준과 같은 설정:   ROAD_LANE_OUT=runs/내이름 ./run_all.sh
#   차량 모델 바꾸기:   ROAD_LANE_OUT=runs/내이름_best YOLO_WEIGHTS=/경로/best.pt ./run_all.sh
#   Windows(Git Bash):  PY=../.venv/Scripts/python.exe ROAD_LANE_OUT=runs/내이름 ./run_all.sh
#   결과 비교:          ../.venv/bin/python compare_runs.py outputs runs/내이름
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"

$PY check_env.py                                 # 0  라이브러리·모델·장치·이미지가 기준과 같은지 (다르면 경고만)
$PY s1_vehicles.py                               # 1  차량 세그멘테이션 + 차 없는 배경 (M5 약 1시간)
$PY s1b_views.py                                 # 1b 회전형 카메라 화면 분리
$PY s2_road_lanes.py                             # 2  도로·방향별 도로·차로 (Mask2Former 장면당 약 5초)
$PY s2b_direction.py build train eval            # 2b 방향 CNN 자동 라벨·학습·평가: 카메라 지도·곡선 보정 "전" 차로 지도로 (기준과 같은 순서)
$PY s2c_camera_groups.py --apply G               # 2c 같은 카메라·같은 화면끼리 도로 지도 공유
$PY s2d_curves.py apply                          # 2d 휜 도로: 방향별 도로마다 공통 휨 곡선으로 차로 모양 보정
$PY s2b_direction.py apply                       # 2b 최종 차로 지도에 방향 적용
$PY s3_occupancy.py                              # 3  차로·방향·도로 전체 점유율
$PY s4_delay.py                                  # 4  추정 속도·지체 시간·소통 등급
$PY report.py                                    # 요약 수치 (compare_runs.py 가 읽음)
