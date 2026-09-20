#!/usr/bin/env bash
# traffic_project 를 이 노트북(WSL2 우분투)에 올려서 심사 기간 내내 돌린다.
# 저장소는 한 글자도 고치지 않는다. 받아서 그 저장소의 scripts/setup.sh 로 환경을 만들 뿐이다.
#
#   git clone https://github.com/Hanhws/traffic-deploy.git ~/deploy && bash ~/deploy/install.sh
#
# 필요한 값은 실행하면서 물어본다. 다시 실행해도 안전하다(이미 된 단계는 건너뛴다).

set -euo pipefail

REPO=${REPO:-https://github.com/hantaeho123/traffic_project.git}
APP_DIR=${APP_DIR:-$HOME/traffic}
PORT=${PORT:-8000}
DB_NAME=traffic
DB_USER=traffic
CONF="$HOME/.traffic-deploy.conf"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   ! %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# 지난번에 입력한 값이 있으면 다시 묻지 않는다
[ -f "$CONF" ] && . "$CONF"

ask() { # ask 변수명 "질문" [hidden]
  local var=$1 prompt=$2 hidden=${3:-} cur=${!1:-} ans
  if [ -n "$cur" ]; then ok "$prompt → 저장된 값 사용"; return; fi
  if [ -n "$hidden" ]; then read -rsp "   $prompt: " ans; echo; else read -rp "   $prompt: " ans; fi
  printf -v "$var" '%s' "$ans"
}

cat <<'INTRO'

  ┌─────────────────────────────────────────────┐
  │  도로 CCTV 혼잡도 모니터 — 노트북 설치      │
  └─────────────────────────────────────────────┘

  이 스크립트가 하는 일:
    우분투 패키지 · PostgreSQL · 저장소 클론 · 파이썬 환경 · 프론트 빌드
    · 모델 가중치 · systemd 서비스 · ngrok 터널

  20분쯤 걸립니다. 중간에 비밀번호를 물으면 우분투 계정 비밀번호입니다.

INTRO

# ─────────────────────────────────────────────────────────────
say "0) 환경 점검"
grep -qi microsoft /proc/version || warn "WSL 이 아닌 것 같습니다. 그래도 진행합니다."

if ! pidof systemd >/dev/null 2>&1 && [ ! -d /run/systemd/system ]; then
  warn "systemd 가 꺼져 있습니다. 켜 드립니다."
  sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
  die "윈도우 PowerShell 에서 'wsl --shutdown' 을 실행하고, 우분투를 다시 연 뒤 이 스크립트를 다시 돌리세요."
fi

MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
ok "WSL 메모리 ${MEM_GB}GB"
[ "$MEM_GB" -ge 10 ] || warn "10GB 미만입니다. 윈도우의 .wslconfig 에서 memory 를 올리세요."

if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
  USE_CUDA=1
else
  warn "GPU 가 안 보입니다. CPU 로 돕니다(20~30배 느림)."
  warn "윈도우에 최신 NVIDIA 드라이버를 깔고 'wsl --shutdown' 후 다시 시도해 보세요."
  USE_CUDA=0
fi

# ─────────────────────────────────────────────────────────────
say "1) 필요한 값 입력"
echo "   (붙여넣기: 마우스 우클릭)"
ask ITS_API_KEY     "ITS 인증키" hidden
ask HF_TOKEN        "허깅페이스 토큰 (SAM3 다운로드용, 없으면 엔터)" hidden
ask NGROK_AUTHTOKEN "ngrok Authtoken (없으면 엔터 — 밖에서 못 봅니다)" hidden
ask NGROK_DOMAIN    "ngrok 고정 도메인 (예: brave-fox-123.ngrok-free.app)"
[ -n "${ITS_API_KEY:-}" ] || die "ITS 인증키는 반드시 필요합니다."

umask 077
cat > "$CONF" <<EOF
ITS_API_KEY='$ITS_API_KEY'
HF_TOKEN='${HF_TOKEN:-}'
NGROK_AUTHTOKEN='${NGROK_AUTHTOKEN:-}'
NGROK_DOMAIN='${NGROK_DOMAIN:-}'
EOF
ok "입력값을 $CONF 에 저장했습니다 (다음 실행 때 안 물어봅니다)"

# ─────────────────────────────────────────────────────────────
say "2) 우분투 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-dev \
  postgresql postgresql-client \
  libgl1 libglib2.0-0 ffmpeg lsof

if ! command -v node >/dev/null || [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -lt 20 ]; then
  say "3) Node 22"
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null
  sudo apt-get install -y -qq nodejs
else
  say "3) Node $(node -v) 이미 있음"
fi

# ─────────────────────────────────────────────────────────────
say "4) PostgreSQL"
sudo systemctl enable --now postgresql
DB_PASS_FILE="$HOME/.traffic_dbpass"
if [ -f "$DB_PASS_FILE" ]; then
  DB_PASS=$(cat "$DB_PASS_FILE")
else
  DB_PASS=$(head -c 18 /dev/urandom | base64 | tr -d '/+=' | head -c 24)
  printf '%s' "$DB_PASS" > "$DB_PASS_FILE"; chmod 600 "$DB_PASS_FILE"
fi
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 \
  || sudo -u postgres psql -qc "CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS'"
sudo -u postgres psql -qc "ALTER ROLE $DB_USER PASSWORD '$DB_PASS'"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
  || sudo -u postgres psql -qc "CREATE DATABASE $DB_NAME OWNER $DB_USER"
ok "DB 준비됨"

# ─────────────────────────────────────────────────────────────
say "5) 저장소 받기 (수정하지 않음)"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin
  git -C "$APP_DIR" reset --hard --quiet origin/HEAD
  ok "갱신: $(git -C "$APP_DIR" log -1 --oneline)"
else
  git clone --quiet "$REPO" "$APP_DIR"
  ok "클론: $(git -C "$APP_DIR" log -1 --oneline)"
fi
cd "$APP_DIR"
mkdir -p models/weights data

# ─────────────────────────────────────────────────────────────
say "6) .env"
DEV=cpu; [ "$USE_CUDA" = 1 ] && DEV=cuda
INTERVAL=30; FPS=0.5
[ "$USE_CUDA" = 1 ] && { INTERVAL=10; FPS=2.0; }
cat > .env <<ENV
ITS_API_KEY=$ITS_API_KEY
DATABASE_URL=postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME

YOLO_WEIGHTS=$APP_DIR/models/weights/yolov8s_seg_vehicle.pt
SAM3_WEIGHTS=$APP_DIR/models/weights/sam3.pt
DEVICE=$DEV

YOLO_IMGSZ=960
YOLO_CONF=0.25
INFER_FPS=$FPS
DEFAULT_INFER_INTERVAL_S=$INTERVAL
SAMPLE_WRITE_INTERVAL=5

CONGESTION_THRESHOLDS=0.08,0.15,0.25

HOST=0.0.0.0
PORT=$PORT
CORS_ORIGINS=
ENV
chmod 600 .env
ok "장치 $DEV · 기본 추론 주기 ${INTERVAL}초"

# ─────────────────────────────────────────────────────────────
say "7) 차량 YOLO 가중치 찾기"
TARGET="$APP_DIR/models/weights/yolov8s_seg_vehicle.pt"
if [ -f "$TARGET" ]; then
  ok "이미 있음"
else
  FOUND=$(ls -1 /mnt/c/Users/*/Downloads/*vehicle*.pt \
                /mnt/c/Users/*/Desktop/*vehicle*.pt \
                "$HOME"/*vehicle*.pt 2>/dev/null | head -1 || true)
  if [ -z "$FOUND" ]; then
    echo "   윈도우 다운로드 폴더에서 못 찾았습니다."
    read -rp "   가중치 파일 경로 (건너뛰려면 엔터): " FOUND
  fi
  if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
    cp "$FOUND" "$TARGET"
    ok "복사: $(basename "$FOUND") ($(du -h "$TARGET" | cut -f1))"
  else
    warn "차량 가중치가 없습니다. 나중에 아래처럼 넣고 'sudo systemctl restart traffic':"
    warn "  cp /mnt/c/Users/<사용자>/Downloads/yolo11s960_vehicle_best.pt $TARGET"
  fi
fi

# ─────────────────────────────────────────────────────────────
say "8) 파이썬 환경 + 프론트 패키지 (저장소의 scripts/setup.sh)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  if [ "$USE_CUDA" = 1 ]; then
    echo "   torch (CUDA) 설치 중 — 2.5GB 정도 받습니다"
    ./.venv/bin/pip install --quiet torch torchvision
  else
    echo "   torch (CPU) 설치 중"
    ./.venv/bin/pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi
PYTHON=python3 bash scripts/setup.sh

say "9) 프론트엔드 빌드"
(cd frontend && npm run build --silent)

if [ "$USE_CUDA" = 1 ]; then
  ./.venv/bin/python - <<'PY' || true
import torch
print(f"   torch {torch.__version__} · cuda={torch.cuda.is_available()}"
      + (f" · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY
fi

# ─────────────────────────────────────────────────────────────
say "10) SAM3 가중치 (3.3GB)"
if [ -f models/weights/sam3.pt ]; then
  ok "이미 있음"
elif [ -n "${HF_TOKEN:-}" ]; then
  echo "   내려받는 중... 몇 분 걸립니다"
  ./.venv/bin/python scripts/download_sam3.py --token "$HF_TOKEN" \
    || warn "다운로드 실패. 토큰과 https://huggingface.co/facebook/sam3 라이선스 동의를 확인하세요."
else
  FOUND=$(ls -1 /mnt/c/Users/*/Downloads/sam3.pt "$HOME"/sam3.pt 2>/dev/null | head -1 || true)
  if [ -n "$FOUND" ]; then
    cp "$FOUND" models/weights/sam3.pt; ok "복사: $FOUND"
  else
    warn "SAM3 가 없습니다. 등록 화면의 '도로 자동 제안' 텍스트 프롬프트만 못 쓰고,"
    warn "점·박스·브러시는 그대로 됩니다. 나중에 넣으려면 HF 토큰으로 다시 실행하세요."
  fi
fi

# ─────────────────────────────────────────────────────────────
say "11) 서비스 등록 (항상 켜짐)"
sudo tee /etc/systemd/system/traffic.service >/dev/null <<UNIT
[Unit]
Description=traffic_project (도로 CCTV 혼잡도 모니터)
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/backend/run.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now traffic.service
sudo systemctl restart traffic.service
sleep 3
systemctl is-active --quiet traffic && ok "traffic 서비스 실행 중" \
  || warn "서비스가 안 떴습니다: sudo journalctl -u traffic -n 50"

# ─────────────────────────────────────────────────────────────
say "12) 밖에서 접속 (ngrok)"
PUBLIC="http://localhost:$PORT"
if [ -n "${NGROK_AUTHTOKEN:-}" ] && [ -n "${NGROK_DOMAIN:-}" ]; then
  if ! command -v ngrok >/dev/null; then
    curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
      | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
      | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq ngrok
  fi
  ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
  sudo tee /etc/systemd/system/ngrok.service >/dev/null <<UNIT
[Unit]
Description=ngrok tunnel for traffic_project
After=traffic.service
Requires=traffic.service

[Service]
Type=simple
User=$USER
ExecStart=$(command -v ngrok) http --url=$NGROK_DOMAIN $PORT --log=stdout
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now ngrok.service
  sudo systemctl restart ngrok.service
  sleep 3
  systemctl is-active --quiet ngrok && ok "터널 연결됨" || warn "ngrok 이 안 떴습니다: sudo journalctl -u ngrok -n 30"
  PUBLIC="https://$NGROK_DOMAIN"
else
  warn "ngrok 값이 없어 터널을 건너뜁니다 — 이 노트북에서만 볼 수 있습니다."
fi

# ─────────────────────────────────────────────────────────────
printf '\n\033[1;32m────────────────────────────────────────\033[0m\n'
cat <<DONE

  제출 링크:  $PUBLIC
  내 노트북:  http://localhost:$PORT

  상태 보기:  sudo systemctl status traffic ngrok
  로그 보기:  sudo journalctl -u traffic -f
  재시작:     sudo systemctl restart traffic

  이제 할 일
   1. 위 주소를 열어 /register 로 가서 CCTV 를 5~10대 등록
      (ITS 탭 → 지도 이동 → "현재 지도 영역 검색" → 도로 칠하기 → 추론 주기 ${INTERVAL}초)
   2. 윈도우 설정 → Windows Update → "업데이트 일시 중지" 를 최대로
   3. 어댑터를 꽂아 두고, 하루 한 번 위 주소가 열리는지 확인

DONE
