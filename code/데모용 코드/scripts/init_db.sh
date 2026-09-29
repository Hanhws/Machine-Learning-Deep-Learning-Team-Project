#!/usr/bin/env bash
# DATABASE_URL 에 적힌 데이터베이스가 없으면 생성한다. (테이블은 서버 시작 시 자동 생성)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
URL=${DATABASE_URL:-postgresql+psycopg://localhost:5433/traffic}
# postgresql+psycopg://user:pass@host:port/db  → 파싱
REST=${URL#*://}
DB=${REST##*/}
HOSTPART=${REST%/*}
USERPASS=""
if [[ "$HOSTPART" == *"@"* ]]; then USERPASS=${HOSTPART%@*}; HOSTPART=${HOSTPART#*@}; fi
HOST=${HOSTPART%%:*}
PORT=${HOSTPART#*:}; [ "$PORT" = "$HOSTPART" ] && PORT=5432
USER_=${USERPASS%%:*}
PASS=${USERPASS#*:}; [ "$PASS" = "$USERPASS" ] && PASS=""
PSQL=$(command -v psql || ls /opt/homebrew/opt/postgresql@*/bin/psql 2>/dev/null | tail -1 || true)
[ -z "$PSQL" ] && { echo "psql 을 찾을 수 없습니다"; exit 1; }
ARGS=(-h "$HOST" -p "$PORT")
[ -n "$USER_" ] && ARGS+=(-U "$USER_")
export PGPASSWORD="$PASS"
if "$PSQL" "${ARGS[@]}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB'" | grep -q 1; then
  echo "DB '$DB' 이미 존재"
else
  "$PSQL" "${ARGS[@]}" -d postgres -c "CREATE DATABASE \"$DB\""
  echo "DB '$DB' 생성"
fi
