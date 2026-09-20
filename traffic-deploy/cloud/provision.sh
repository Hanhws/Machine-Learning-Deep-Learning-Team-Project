#!/usr/bin/env bash
# traffic_project 를 우분투 서버 한 대에 올린다.
#
# 저장소는 한 글자도 고치지 않는다. 받아서 그 저장소의 scripts/setup.sh 로 환경을 만들고,
# systemd 로 계속 띄워 둘 뿐이다. 내 PC 는 꺼도 된다.
#
# 쓰는 법 (우분투 22.04 / 24.04 새 서버에서):
#   scp provision.sh ubuntu@<서버IP>:~
#   ssh ubuntu@<서버IP>
#   export ITS_API_KEY='발급받은키'
#   export HF_TOKEN='hf_xxx'          # SAM3 를 서버가 직접 받게 하려면 (선택)
#   export DOMAIN='traffic.example.com'  # 도메인이 있으면. 없으면 비워 두면 http://IP 로 뜬다
#   bash provision.sh
#
# 다시 실행해도 안전하다(이미 된 단계는 건너뛴다).

set -euo pipefail

REPO=${REPO:-https://github.com/hantaeho123/traffic_project.git}
APP_DIR=${APP_DIR:-/opt/traffic}
APP_USER=${APP_USER:-$(id -un)}
PORT=${PORT:-8000}
DOMAIN=${DOMAIN:-}
ITS_API_KEY=${ITS_API_KEY:-}
HF_TOKEN=${HF_TOKEN:-}
DB_NAME=${DB_NAME:-traffic}
DB_USER=${DB_USER:-traffic}
DB_PASS=${DB_PASS:-}

# CPU 로 돌릴 때의 기본값. 카메라가 많으면 주기를 늘린다.
YOLO_IMGSZ=${YOLO_IMGSZ:-960}
DEFAULT_INFER_INTERVAL_S=${DEFAULT_INFER_INTERVAL_S:-30}
INFER_FPS=${INFER_FPS:-0.5}

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$ITS_API_KEY" ] || die "ITS_API_KEY 를 먼저 export 하세요. (https://www.its.go.kr/opendata/opendataList?service=cctv)"
[ "$(id -u)" -ne 0 ] || die "root 말고 일반 사용자(ubuntu 등)로 실행하세요. 필요할 때만 sudo 를 씁니다."

MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
say "메모리 ${MEM_GB}GB 확인"
if [ "$MEM_GB" -lt 8 ]; then
  die "SAM3 를 쓰려면 RAM 이 최소 8GB(권장 16GB) 필요합니다. 현재 ${MEM_GB}GB."
elif [ "$MEM_GB" -lt 16 ]; then
  echo "   경고: 16GB 미만입니다. 등록 화면에서 텍스트 프롬프트와 점/박스를 둘 다 쓰면"
  echo "   sam3 체크포인트가 메모리에 두 번 올라가 죽을 수 있습니다. 스왑을 잡아 둡니다."
fi

# ─────────────────────────────────────────────────────────────
say "1) 시스템 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-dev \
  postgresql postgresql-client \
  libgl1 libglib2.0-0 ffmpeg \
  lsof

# Node 22 (프론트엔드 빌드용). 우분투 기본 패키지는 Vite 가 요구하는 버전보다 낮다.
if ! command -v node >/dev/null || [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -lt 20 ]; then
  say "2) Node 22 설치"
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y -qq nodejs
else
  say "2) Node $(node -v) 이미 있음"
fi

# ─────────────────────────────────────────────────────────────
say "3) 스왑 4GB (메모리 순간 피크 대비)"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "   이미 있음"
fi

# ─────────────────────────────────────────────────────────────
say "4) PostgreSQL 사용자·DB"
if [ -z "$DB_PASS" ]; then
  if [ -f "$APP_DIR/.dbpass" ]; then
    DB_PASS=$(sudo cat "$APP_DIR/.dbpass")
  else
    DB_PASS=$(head -c 18 /dev/urandom | base64 | tr -d '/+=' | head -c 24)
  fi
fi
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 \
  || sudo -u postgres psql -qc "CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS'"
sudo -u postgres psql -qc "ALTER ROLE $DB_USER PASSWORD '$DB_PASS'"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
  || sudo -u postgres psql -qc "CREATE DATABASE $DB_NAME OWNER $DB_USER"
echo "   DB '$DB_NAME' · 사용자 '$DB_USER' 준비됨"

# ─────────────────────────────────────────────────────────────
say "5) 저장소 받기 (수정하지 않음)"
sudo mkdir -p "$APP_DIR"
sudo chown "$APP_USER":"$APP_USER" "$APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin
  git -C "$APP_DIR" reset --hard --quiet origin/HEAD
  echo "   최신으로 갱신: $(git -C "$APP_DIR" log -1 --oneline)"
else
  git clone --quiet "$REPO" "$APP_DIR"
  echo "   클론 완료: $(git -C "$APP_DIR" log -1 --oneline)"
fi
printf '%s' "$DB_PASS" | sudo tee "$APP_DIR/.dbpass" >/dev/null
sudo chmod 600 "$APP_DIR/.dbpass"

# ─────────────────────────────────────────────────────────────
say "6) .env  (저장소의 .env.example 형식 그대로)"
cat > "$APP_DIR/.env" <<ENV
ITS_API_KEY=$ITS_API_KEY
DATABASE_URL=postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME

YOLO_WEIGHTS=$APP_DIR/models/weights/yolov8s_seg_vehicle.pt
SAM3_WEIGHTS=$APP_DIR/models/weights/sam3.pt
DEVICE=cpu

YOLO_IMGSZ=$YOLO_IMGSZ
YOLO_CONF=0.25
INFER_FPS=$INFER_FPS
DEFAULT_INFER_INTERVAL_S=$DEFAULT_INFER_INTERVAL_S
SAMPLE_WRITE_INTERVAL=5

CONGESTION_THRESHOLDS=0.08,0.15,0.25

HOST=127.0.0.1
PORT=$PORT
CORS_ORIGINS=
ENV
chmod 600 "$APP_DIR/.env"
mkdir -p "$APP_DIR/models/weights" "$APP_DIR/data"

# ─────────────────────────────────────────────────────────────
say "7) 파이썬 환경 + 프론트엔드 패키지 (저장소의 scripts/setup.sh)"
cd "$APP_DIR"
# 리눅스에는 CUDA 없는 CPU 휠을 먼저 깔아 둔다(기본 인덱스는 CUDA 포함이라 몇 GB 더 받는다)
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  # x86 는 CPU 전용 인덱스, ARM(오라클 Ampere 등)은 거기에 휠이 없어 기본 인덱스로 되돌린다
  ./.venv/bin/pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    || ./.venv/bin/pip install --quiet torch torchvision
fi
PYTHON=python3 bash scripts/setup.sh

say "8) 프론트엔드 빌드"
(cd frontend && npm run build --silent)

# ─────────────────────────────────────────────────────────────
say "9) 모델 가중치"
if [ ! -f models/weights/sam3.pt ]; then
  if [ -n "$HF_TOKEN" ]; then
    echo "   SAM3 내려받는 중 (3.3GB, 몇 분 걸립니다)"
    ./.venv/bin/python scripts/download_sam3.py --token "$HF_TOKEN"
  else
    echo "   !! models/weights/sam3.pt 가 없습니다."
    echo "      내 PC 에서 올리려면:  scp sam3.pt $APP_USER@<서버IP>:$APP_DIR/models/weights/"
    echo "      또는 HF_TOKEN 을 export 하고 이 스크립트를 다시 실행하세요."
  fi
fi
if ! ls models/weights/*.pt >/dev/null 2>&1 || [ ! -f models/weights/yolov8s_seg_vehicle.pt ]; then
  echo "   !! 차량 YOLO 가중치가 없습니다. 내 PC 에서 올리세요:"
  echo "      scp yolo11s960_vehicle_best.pt $APP_USER@<서버IP>:$APP_DIR/models/weights/yolov8s_seg_vehicle.pt"
fi

# ─────────────────────────────────────────────────────────────
say "10) systemd 서비스 (재부팅해도 자동 시작)"
sudo tee /etc/systemd/system/traffic.service >/dev/null <<UNIT
[Unit]
Description=traffic_project (도로 CCTV 혼잡도 모니터)
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PYTORCH_ENABLE_MPS_FALLBACK=1
Environment=OMP_NUM_THREADS=2
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/backend/run.py
Restart=always
RestartSec=5
# 추론이 메모리를 순간적으로 많이 쓴다. OOM 이 나면 서비스만 재시작되게 둔다.
OOMPolicy=continue

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now traffic.service
sudo systemctl restart traffic.service

# ─────────────────────────────────────────────────────────────
say "11) 앞단 웹서버 (Caddy)"
if ! command -v caddy >/dev/null; then
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy
fi
if [ -n "$DOMAIN" ]; then
  # 도메인이 있으면 Let's Encrypt 인증서를 Caddy 가 알아서 받는다
  sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
$DOMAIN {
	reverse_proxy 127.0.0.1:$PORT {
		flush_interval -1      # MJPEG 실시간 스트림은 버퍼링하면 안 된다
	}
}
CADDY
else
  sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
:80 {
	reverse_proxy 127.0.0.1:$PORT {
		flush_interval -1
	}
}
CADDY
fi
sudo systemctl restart caddy

say "12) 방화벽 (오라클·AWS 이미지는 기본으로 막혀 있다)"
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
command -v netfilter-persistent >/dev/null \
  && sudo netfilter-persistent save >/dev/null 2>&1 || true

# ─────────────────────────────────────────────────────────────
IP=$(curl -fsS --max-time 5 https://api.ipify.org || echo '<서버IP>')
say "끝"
cat <<DONE

  주소:   ${DOMAIN:+https://$DOMAIN}${DOMAIN:-http://$IP}
  상태:   sudo systemctl status traffic
  로그:   sudo journalctl -u traffic -f
  재시작: sudo systemctl restart traffic

  클라우드 콘솔의 보안 목록(Security List / 방화벽 규칙)에서도 80·443 을 열어야 합니다.

  다음: 브라우저로 열어 /register 에서 CCTV 를 5~10대 등록하고
        추론 주기를 30초로 두세요. 그러면 지도와 통계에 값이 쌓입니다.
DONE
