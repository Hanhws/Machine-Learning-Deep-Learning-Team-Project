# 노트북 배포 운영 메모

프로젝트 소개는 [README.md](README.md) 를 보면 된다. 이 문서는 **Windows 노트북(WSL2 + NVIDIA GPU)** 에
올려 심사 기간 내내 무중단으로 돌리기 위한 설정과, 그 과정에서 **실제로 막혔던 것들과 해결책**이다.

상류 저장소(`traffic_project`)를 한 글자도 고치지 않는 것이 원칙이다.

```
공개 주소:  https://<내-도메인>.ngrok-free.dev      (ngrok 고정 도메인)
로컬 주소:  http://localhost:8000                   (WSL 우분투 안에서)
```

배포에 쓰는 파일:

| | |
|---|---|
| `traffic-deploy/install.sh` | 우분투 쪽 설치 전부 |
| `traffic-deploy/windows-setup.ps1` | WSL·전원·자동시작 (관리자 권한) |
| `setup/windows-power-setup.bat` | 위 스크립트의 전원 설정 부분만. 우클릭 → 관리자 권한으로 실행 |
| `setup/wsl-keepalive.vbs` | WSL 유휴 종료 방지. 시작 프로그램에 넣어 둔다 |
| `키입력.example.txt` | 키 입력 템플릿. 복사해서 `키입력.txt` 로 쓴다 (gitignore 됨) |

---

## 처음부터 설치하기

### 1. 윈도우 준비

```powershell
# 관리자 PowerShell
irm https://raw.githubusercontent.com/Hanhws/traffic-deploy/main/windows-setup.ps1 | iex
```

WSL2 + Ubuntu 24.04 설치, `.wslconfig` RAM 할당, 절전 해제, 로그온 자동 시작을 한 번에 한다.
관리자 권한이 막히면 `setup/windows-power-setup.bat` 을 **우클릭 → 관리자 권한으로 실행** 해도 된다.

### 2. 우분투 설치

```bash
git clone https://github.com/Hanhws/traffic-deploy.git ~/deploy && bash ~/deploy/install.sh
```

ITS 키·HF 토큰·ngrok 토큰/도메인 4개를 물어본다. 20분쯤 걸린다.
**터미널에 붙여넣기가 잘 안 되면** (숨김 입력이라 화면에 안 보여서 실수하기 쉽다)
`~/.traffic-deploy.conf` 를 미리 만들어 두면 묻지 않고 넘어간다:

```bash
umask 077
cat > ~/.traffic-deploy.conf <<'EOF'
ITS_API_KEY='...'
HF_TOKEN='...'
NGROK_AUTHTOKEN='...'
NGROK_DOMAIN='...'
EOF
```

### 3. 유휴 종료 방지 (필수 — 아래 "함정 4" 참고)

```powershell
copy setup\wsl-keepalive.vbs "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\"
wscript.exe "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\wsl-keepalive.vbs"
```

그리고 `%USERPROFILE%\.wslconfig` 에 `vmIdleTimeout=-1` 을 넣고 `wsl --shutdown`.

### 4. 수동 작업

- 설정 → Windows Update → **업데이트 일시 중지 최대(5주)**
- 전원 어댑터 연결 유지 (배터리로는 전원 설정이 적용되지 않는다)
- `/register` 에서 CCTV 5~10대 등록, 추론 주기 10초

---

## 운영

실제로 돌아가는 건 이 폴더가 아니라 **WSL 우분투 안의 `~/traffic`** 이다. 이 폴더는 설정과 메모.

| 서비스 | 하는 일 |
|---|---|
| `traffic` | FastAPI 백엔드 + 프론트 서빙 + 추론 워커 |
| `ngrok` | 고정 도메인 터널 |

```bash
sudo systemctl status traffic ngrok     # 상태
sudo journalctl -u traffic -f           # 실시간 로그
sudo systemctl restart traffic ngrok    # 재시작 (둘 다 명시할 것)
nvidia-smi                              # GPU 사용량
```

### 팀이 새 커밋을 올렸을 때

```bash
bash ~/deploy/install.sh
```

`origin/HEAD` 로 하드 리셋하고 다시 빌드한다. `.env`·DB·가중치·등록한 카메라·`.venv` 는 유지된다.

### 마스크 백업 (카메라 재등록을 피하려면)

```bash
tar czf ~/traffic-data.tgz -C ~/traffic data
```

---

## 실제로 막혔던 것 5가지

### 1. torch 가 cu130 으로 깔려 GPU 가 죽는다

`install.sh` 는 `pip install torch torchvision` 을 그냥 부른다. 그러면 **cu130** 빌드가 깔리는데,
드라이버가 CUDA 12.9(576.x)면 못 쓴다:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090)
```

문제는 이게 **조용히** 실패한다는 것이다. `.env` 의 `DEVICE=cuda` 는 `resolve_device()` 에서
`auto` 일 때만 가용성을 검사하고, 그 외에는 그대로 통과시킨다. 그래서 CUDA 가 죽어 있어도
앱은 멀쩡히 뜨고 **CCTV 를 등록하는 순간** 워커가 터진다.

```bash
cd ~/traffic
./.venv/bin/pip uninstall -y torch torchvision triton \
  $(./.venv/bin/pip freeze | grep -oiE '^nvidia-[a-z0-9_-]+')
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 torchvision
./.venv/bin/pip install "nvidia-ml-py>=12.0.0"     # 위에서 같이 지워진다. ultralytics 가 필요로 함
./.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True 여야 함
sudo systemctl restart traffic ngrok
```

드라이버를 580 이상으로 올리면 cu130 도 되지만, cu126 은 양쪽 다에서 돈다.
`requirements.txt` 는 `torch>=2.4` 하한만 걸려 있어서 `install.sh` 를 다시 돌려도 되돌아가지 않는다.

### 2. `start traffic` 만으로는 ngrok 이 안 올라온다

`ngrok.service` 에 `Requires=traffic.service` 가 걸려 있다. traffic 을 멈추면 ngrok 도 연쇄로
내려가지만, **반대 방향은 작동하지 않는다.** 앱은 살아 있는데 외부 링크만 404 가 되는 상황이 된다.

항상 둘 다 명시할 것: `sudo systemctl restart traffic ngrok`

### 3. SAM3 는 토큰보다 라이선스 동의가 먼저다

토큰만 있으면 `401 GatedRepoError` 가 난다.
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) 에서 `Agree and access repository` 를 먼저 눌러야 한다.

`sam3.pt` (3.3GB) 는 **등록할 때만** 쓰인다. `road_seg.py` 첫 줄에 "도로 영역 segmentation (등록 시 1회)"
라고 적혀 있고, 실시간 워커는 `road_seg` 를 import 조차 하지 않는다. 실시간 경로는:

```
프레임 → (디스크에 저장된 도로 마스크) + YOLO 차량 세그멘테이션 → 방향별 점유율 → DB
```

즉 매 프레임에는 YOLO 만 돈다. 그래서 가중치를 한 번 받고 나면 **HF 토큰은 폐기해도 되지만
`sam3.pt` 파일은 남겨둬야 한다** (카메라를 새로 등록하거나 마스크를 다시 그릴 때 필요).

### 4. WSL2 가 유휴 시 VM 을 통째로 종료한다

아무도 안 쓰면 **약 46초** 만에 내려간다. `traffic` 도 `ngrok` 도 같이 죽고 공개 주소는 404 가 된다.
심사 기간 배포에서는 치명적이다.

`windows-setup.ps1` 의 `traffic-wsl` 작업은 로그온 때 `/bin/true` 를 한 번 실행할 뿐이라
부팅 직후에만 살리고 이후 유휴 종료는 못 막는다. 두 가지를 같이 걸어야 한다:

| | |
|---|---|
| `%USERPROFILE%\.wslconfig` | `vmIdleTimeout=-1` |
| 시작 프로그램 폴더 | `setup/wsl-keepalive.vbs` — `wsl.exe -u root -- sleep infinity` 를 숨김으로 상주 |

WSL 을 5분간 건드리지 않고 외부에서만 접속해 검증했다 (1·2·3·4·5분 모두 HTTP 200).

### 5. 설치 중 우분투 창을 닫으면 dpkg 가 깨진다

apt 가 중간에 끊긴 상태가 된다. 복구:

```bash
sudo dpkg --configure -a && sudo apt-get -f install -y
```

---

## 이 환경의 설정값

| | |
|---|---|
| WSL | Ubuntu 24.04 LTS, systemd 활성, RAM 12GB + swap 8GB |
| GPU | RTX 4070 Laptop 8GB · 드라이버 576.52 (CUDA 12.9) |
| torch | 2.14.0+**cu126** (cu130 아님 — 함정 1 참고) |
| 추론 | `DEVICE=cuda` · 기본 주기 10초 · `INFER_FPS=2.0` · `imgsz=960` |
| sudo | 무인 설치를 위해 NOPASSWD. 되돌리려면 `sudo rm /etc/sudoers.d/90-traffic-nopasswd` |

**RTX 4070 기준 여유**: 10초 주기 약 30대 / 5초 주기 약 15대 / 실시간(2 FPS) 약 3대.
