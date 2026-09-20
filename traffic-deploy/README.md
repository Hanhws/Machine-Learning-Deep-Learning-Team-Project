# traffic-deploy

[hantaeho123/traffic_project](https://github.com/hantaeho123/traffic_project) 를 **고치지 않고**
집 노트북(Windows + NVIDIA GPU)에 올려서 심사 기간 내내 돌린다.

내 맥은 꺼도 된다. Vercel 도, 따로 띄워 둘 것도 없다 — 백엔드가 프론트까지 같이 서빙한다.

---

## 먼저 준비할 것 3개

| | 어디서 | 왜 |
|---|---|---|
| **ITS 인증키** | https://www.its.go.kr/opendata/opendataList?service=cctv | 실시간 CCTV 목록·스트림 |
| **허깅페이스 토큰** | https://huggingface.co/facebook/sam3 라이선스 동의 → https://huggingface.co/settings/tokens (read) | SAM3 3.3GB 를 노트북이 직접 받게 |
| **ngrok 고정 도메인 + 토큰** | https://dashboard.ngrok.com → Domains 에서 무료 도메인 1개, Your Authtoken | 집 인터넷은 공유기 뒤라 밖에서 못 들어온다 |

그리고 **차량 YOLO 가중치**(`yolo11s960_vehicle_best.pt`, 20MB)를 노트북 **다운로드 폴더**에 넣어 둔다.
설치 스크립트가 거기서 알아서 찾는다. (AI-Hub 로 학습한 모델이라 이 저장소에는 넣지 않는다)

---

## 1단계 — 윈도우 (관리자 PowerShell)

시작 메뉴 → PowerShell 우클릭 → **관리자 권한으로 실행** → 붙여넣기:

```powershell
irm https://raw.githubusercontent.com/Hanhws/traffic-deploy/main/windows-setup.ps1 | iex
```

이게 알아서 한다:
- WSL2 + Ubuntu 24.04 설치
- RAM 할당 (`.wslconfig`)
- **절전 / 화면 끄기 / 덮개 닫기 전부 해제** ← 노트북이 자면 사이트가 죽는다
- 로그온하면 WSL 자동 시작 (작업 스케줄러)

끝나면 재부팅한다.

## 2단계 — 우분투

재부팅 후 시작 메뉴에서 **Ubuntu 24.04** 실행 → 사용자 이름·비밀번호 정하기 → 붙여넣기:

```bash
git clone https://github.com/Hanhws/traffic-deploy.git ~/deploy && bash ~/deploy/install.sh
```

물어보는 값 4개(ITS 키, HF 토큰, ngrok 토큰·도메인)를 붙여넣으면 나머지는 20분쯤 알아서 돈다.
GPU 를 자동으로 잡아 `DEVICE=cuda` 로 설정한다.

끝나면 **제출 링크**(`https://....ngrok-free.app`)를 출력한다.

## 3단계 — 카메라 등록

링크를 열고 `/register` 에서:

1. ITS 탭 → 지도를 원하는 곳으로 → **현재 지도 영역 검색**
2. 카메라 선택 → 2단계에서 도로 영역 칠하고 방향 나누기
3. 3단계에서 **추론 주기 10초**로 저장 → 워커가 바로 시작
4. **5~10대** 반복

30분쯤 지나면 지도·통계에 값이 쌓이고, 반나절이면 `/apps` 리포트까지 채워진다.

## 4단계 — 딱 하나 남은 수동 작업

**설정 → Windows Update → 업데이트 일시 중지 → 최대(5주)**

자동 재부팅이 심사 기간에 사이트를 죽이는 제일 흔한 원인이다.

---

## 잘 돌고 있나 확인

```bash
sudo systemctl status traffic ngrok    # 둘 다 active (running)
sudo journalctl -u traffic -f          # 워커 로그
nvidia-smi                             # GPU 사용량
```

| 증상 | 해볼 것 |
|---|---|
| 링크가 안 열림 | `sudo systemctl restart traffic ngrok` |
| 지도에 숫자가 안 뜸 | `sudo journalctl -u traffic -n 100` — ITS 키가 맞는지 |
| WSL 이 죽음 | PowerShell 에서 `wsl --shutdown` → Ubuntu 다시 열기 |
| 팀에서 새 커밋을 올림 | `bash ~/deploy/install.sh` 다시 실행 (설정·데이터·가중치는 유지) |

**RTX 4070 기준 여유**: 10초 주기 약 30대 / 5초 주기 약 15대 / 실시간(2 FPS) 약 3대.
눈에 띄는 지점 2~3대만 실시간으로 두고 나머지는 10초가 무난하다.

마스크 백업(카메라 재등록을 피하려면): `tar czf ~/traffic-data.tgz -C ~/traffic data`

---

## 파일

| | |
|---|---|
| `windows-setup.ps1` | 윈도우 쪽 준비 (WSL·전원·자동시작) |
| `install.sh` | 우분투 쪽 설치 전부 |
| `노트북-윈도우.md` | 위 과정의 자세한 설명, Cloudflare Tunnel 대안, 문제 해결 |
| `cloud/` | 노트북 대신 클라우드 CPU 서버에 올릴 때 |

ITS 인증키는 노트북의 `~/traffic/.env`(권한 600)에만 있다. 이 저장소에도, 브라우저에도 나가지 않는다.
