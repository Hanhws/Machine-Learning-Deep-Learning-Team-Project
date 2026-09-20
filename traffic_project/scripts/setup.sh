#!/usr/bin/env bash
# 최초 1회 환경 구성: Python 가상환경 + 패키지, PostgreSQL DB, 프론트엔드 패키지
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=${VENV:-.venv}

echo "== 1) Python 가상환경 ($VENV)"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo "== 2) .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "   .env 를 생성했습니다. ITS_API_KEY 와 DATABASE_URL 을 채우세요."
fi

echo "== 3) PostgreSQL 데이터베이스"
bash scripts/init_db.sh || echo "   (DB 생성 건너뜀 — DATABASE_URL 을 확인하세요)"

echo "== 4) 프론트엔드 패키지"
(cd frontend && npm install)

echo
echo "완료. 실행: bash scripts/run_dev.sh  (개발)  /  bash scripts/run_prod.sh (배포)"
