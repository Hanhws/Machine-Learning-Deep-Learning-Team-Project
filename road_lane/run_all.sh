#!/usr/bin/env bash
# 전체 파이프라인. 각 단계는 끝난 클립·장면을 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 한다.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python

$PY s1_vehicles.py                               # 1  차량 세그멘테이션 + 차 없는 배경 (약 1시간)
$PY s1b_views.py                                 # 1b 회전형 카메라 화면 분리
$PY s2_road_lanes.py                             # 2  도로·방향별 도로·차로 (Mask2Former 장면당 약 5초)
$PY s2c_camera_groups.py --apply G               # 2c 같은 카메라·같은 화면끼리 도로 지도 공유
$PY s2d_curves.py apply                          # 2d 휜 도로: 방향별 도로마다 공통 휨 곡선으로 차로 모양 보정
$PY s2b_direction.py build train eval apply      # 2b 차량 앞/뒷모습 CNN 으로 방향
$PY s3_occupancy.py                              # 3  차로·방향·도로 전체 점유율
$PY s4_delay.py                                  # 4  추정 속도·지체 시간·소통 등급
