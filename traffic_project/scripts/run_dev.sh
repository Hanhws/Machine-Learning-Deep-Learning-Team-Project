#!/usr/bin/env bash
# 개발 실행: FastAPI(8000, --reload) + Vite dev 서버(5173, /api 프록시)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python
  elif [ -x /opt/anaconda3/envs/VITA/bin/python ]; then PY=/opt/anaconda3/envs/VITA/bin/python
  else PY=python3; fi
fi
echo "python: $PY"

# 포트 점유 확인 (.env 의 PORT, 기본 8000)
PORT_=$( [ -f .env ] && grep -E '^PORT=' .env | cut -d= -f2 ); PORT_=${PORT_:-8000}
if lsof -nP -iTCP:"$PORT_" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "PORT_IN_USE: $PORT_ 포트를 다른 프로세스가 쓰고 있습니다:"; lsof -nP -iTCP:"$PORT_" -sTCP:LISTEN | tail -n +2
  echo "종료하려면: lsof -ti :$PORT_ | xargs kill"; exit 1
fi
"$PY" backend/run.py --reload &
BACK=$!
trap 'kill $BACK 2>/dev/null || true' EXIT
(cd frontend && npm run dev)
