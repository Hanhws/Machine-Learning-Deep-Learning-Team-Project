#!/usr/bin/env bash
# 배포 실행: 프론트엔드 빌드 후 FastAPI 하나로 서비스 (http://HOST:PORT)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python
  elif [ -x /opt/anaconda3/envs/VITA/bin/python ]; then PY=/opt/anaconda3/envs/VITA/bin/python
  else PY=python3; fi
fi

# 포트 점유 확인 (.env 의 PORT, 기본 8000)
PORT_=$( [ -f .env ] && grep -E '^PORT=' .env | cut -d= -f2 ); PORT_=${PORT_:-8000}
if lsof -nP -iTCP:"$PORT_" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "PORT_IN_USE: $PORT_ 포트를 다른 프로세스가 쓰고 있습니다:"; lsof -nP -iTCP:"$PORT_" -sTCP:LISTEN | tail -n +2
  echo "종료하려면: lsof -ti :$PORT_ | xargs kill"; exit 1
fi
(cd frontend && npm run build)
exec "$PY" backend/run.py
