# 심사 기간 동안 계속 켜두기

> 이건 **클라우드 서버(CPU)** 에 올릴 때의 대안이다.
> 집 노트북으로 할 거면 저장소 루트의 `README.md` 를 본다.

`hantaeho123/traffic_project` 를 **고치지 않고** 서버 한 대에 올린다. 내 PC 는 꺼도 된다.
터널(ngrok/cloudflared)도, Vercel 도 필요 없다 — 백엔드가 프론트까지 같이 서빙한다.

## 서버 사양

SAM3 를 그대로 쓰기 때문에 **RAM 이 관건**이다. `sam3.pt` 는 3.3GB 이고,
등록 화면에서 텍스트 프롬프트(`"road"`)와 점/박스를 둘 다 쓰면 코드가 두 예측기를
**따로** 로드해 체크포인트가 메모리에 두 번 올라간다.

| | 최소 | 권장 |
|---|---|---|
| RAM | 8GB (+스왑) | **16GB 이상** |
| vCPU | 2 | 4 |
| 디스크 | 40GB | 60GB |
| GPU | 필요 없음 | — |

### 어디에 만들까

| | 비용 | 사양 | 참고 |
|---|---|---|---|
| **오라클 클라우드 Always Free** | **무료** | Ampere A1 · 4 vCPU · **24GB** · 200GB | ARM. 지역에 따라 "out of capacity" 로 생성이 막힐 때가 있다 |
| **GCP 무료 체험** | 90일 $300 크레딧 | e2-standard-4 · 4 vCPU · 16GB | x86. 심사 기간은 크레딧으로 덮인다 |
| AWS Lightsail | 월 $44 | 4 vCPU · 16GB | 제일 단순하지만 유료 |

Railway·Fly.io·Render 같은 PaaS 는 16GB 상시 가동이 월 $30~60 이라 여기서는 이점이 없다.

둘 중 **오라클 Always Free** 를 먼저 시도하고, 인스턴스 생성이 막히면 GCP 로 간다.
이미지는 **Ubuntu 24.04** 를 고른다.

## 필요한 것 3개

1. **ITS 인증키** — https://www.its.go.kr/opendata/opendataList?service=cctv
2. **차량 YOLO 가중치** — `팀프로젝트/lane_sam3/yolo11s960_vehicle_best.pt` (20MB)
3. **SAM3 가중치** — 둘 중 하나
   - 허깅페이스 토큰(`hf_...`)을 주면 서버가 직접 받는다 (빠름, 권장)
     — https://huggingface.co/facebook/sam3 에서 라이선스 동의 필요
   - 또는 내 PC 의 `팀프로젝트/lane_sam3/sam3.pt` (3.3GB) 를 scp 로 올린다

## 설치

```bash
# 1) 내 PC 에서 스크립트와 YOLO 가중치를 올린다
scp cloud/provision.sh ubuntu@<서버IP>:~
scp "팀프로젝트/lane_sam3/yolo11s960_vehicle_best.pt" ubuntu@<서버IP>:~

# 2) 서버에 접속해서 실행
ssh ubuntu@<서버IP>
export ITS_API_KEY='발급받은키'
export HF_TOKEN='hf_xxx'          # SAM3 를 서버가 직접 받게 함. 없으면 나중에 scp
bash provision.sh                  # 15~30분 (torch·ultralytics 설치가 대부분)

# 3) YOLO 가중치를 제자리에 놓고 재시작
sudo mv ~/yolo11s960_vehicle_best.pt /opt/traffic/models/weights/yolov8s_seg_vehicle.pt
sudo chown $USER /opt/traffic/models/weights/yolov8s_seg_vehicle.pt
sudo systemctl restart traffic
```

HF 토큰을 안 쓴다면 3.3GB 를 직접 올린다 (업로드 속도에 따라 오래 걸린다):

```bash
scp "팀프로젝트/lane_sam3/sam3.pt" ubuntu@<서버IP>:/opt/traffic/models/weights/
ssh ubuntu@<서버IP> 'sudo systemctl restart traffic'
```

**클라우드 콘솔에서 80·443 포트를 여는 것을 잊지 말 것.**
오라클은 VCN → 보안 목록 → 수신 규칙, GCP 는 VPC 네트워크 → 방화벽.

그러면 `http://<서버IP>` 가 열린다. 도메인이 있으면 A 레코드를 서버 IP 로 잡고
`export DOMAIN=traffic.example.com` 을 준 뒤 스크립트를 다시 돌리면 Caddy 가 HTTPS 인증서까지 받아 준다.

## 올린 뒤

```bash
sudo systemctl status traffic      # 떠 있나
sudo journalctl -u traffic -f      # 로그 (워커가 도는지)
free -h                            # 메모리 여유
```

1. 브라우저로 열어 `/register` → ITS 탭 → 지도 이동 → "현재 지도 영역 검색"
2. 카메라 고르고 2단계에서 도로 영역을 칠한다
3. 3단계에서 **추론 주기 30초**로 저장 → 워커가 바로 시작
4. 카메라 **5~10대**를 이렇게 등록한다
5. 30분쯤 지나면 `/` 지도와 `/stats` 에 값이 쌓이고, 반나절이면 `/apps` 리포트도 채워진다

### CPU 로 몇 대까지

YOLO11-S @960 이 CPU 에서 장당 0.3~0.6초다. 워커들이 추론 스레드 하나를 공유하므로
`카메라 수 ÷ 추론주기 × 0.5초 < 1` 이면 밀리지 않는다.

| 추론 주기 | 여유 있는 카메라 수 |
|---|---|
| 30초 | 약 20대 |
| 10초 | 약 6대 |
| 5초 | 약 3대 |

카메라마다 주기는 상세 페이지에서 따로 바꿀 수 있다. 심사용이면 **30초로 통일**하는 편이 안전하다.

## 알아둘 것

- **ITS URL 은 24시간만 유효**하다. 서버를 계속 켜 두면 워커가 만료 전에 자동으로 다시 받아온다.
- 재부팅해도 systemd 가 서비스를 다시 띄우고, 등록된 카메라 워커도 자동으로 시작한다.
- `/opt/traffic/data` 에 도로 마스크가 있다. 지우면 카메라를 다시 등록해야 하니 한 번 백업해 두면 좋다:
  `ssh ubuntu@<서버IP> 'tar czf - -C /opt/traffic data' > data-backup.tgz`
- 저장소를 고치지 않으므로, 팀에서 새 커밋을 올리면 `bash provision.sh` 를 다시 돌리면 최신으로 갱신된다.
  (`.env` 와 `data/`, 가중치는 그대로 둔다)
- ITS 인증키는 서버의 `/opt/traffic/.env`(권한 600)에만 있다. 브라우저로 나가지 않는다.
