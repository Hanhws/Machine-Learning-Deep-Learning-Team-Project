# 도로 CCTV 차량 세그멘테이션 데이터 파이프라인

**Segmentation 기반 도로 CCTV의 도로 대비 차량 면적 비를 활용한 교통 혼잡도 측정 시스템** 프로젝트의 데이터 준비 코드.

AI-Hub **「교통문제 해결을 위한 CCTV 교통 데이터(고속도로)」 폴리곤세그멘테이션** 데이터를
받은 그대로(`download (N).tar`)에서 시작해, 명령어 3줄로 **여러 세그멘테이션 모델이 바로 학습할 수 있는 데이터셋**을 만든다.

- **형식 3가지를 한 번에**: YOLO · COCO · 마스크 PNG → YOLO, Mask R-CNN, Detectron2, MMDetection, SegFormer, DeepLabV3, U-Net …
- **공정한 분할**: train / val / test = 8 : 1 : 1. 같은 영상의 프레임이 여러 세트에 섞이지 않게 **클립 단위**로 나누고,
  지역·시간대·주/야간·날씨·차로·클래스 비율이 세 세트에 고르게 들어가도록 **층화**. 모든 형식이 같은 분할을 쓴다.
- **풀 필요 없음**: tar 와 분할 zip 을 풀지 않고 바로 읽는다. 추가 용량은 추출된 이미지(약 60GB)뿐.
- **EDA 자동화**: 차트 15장 + 파일명 토큰 사전 + 데이터 무결성 점검.

> 데이터는 저장소에 포함되어 있지 않다. AI-Hub 에서 각자 이용 신청 후 받는다.

---

## 파이프라인

```
AI-Hub tar / zip 조각 ──① preprocessing──▶ car_seg_dataset/ ──③ split──▶ car_seg_split/ ──▶ 모델 학습
                                               │
                                               └──② eda──▶ car_seg_dataset/eda/  (차트 · 표)
```

| 파일 | 하는 일 | 시간 |
|---|---|---|
| `car_seg_data_preprocessing.py` | zip 조각 수집(폴더·tar) → 이미지 추출 → YOLO · COCO · 마스크 라벨 생성 | 수십 분 |
| `car_seg_data_eda.py` | 분포·객체 크기·차량 면적 비·밝기·주/야간 등 차트 15장 | 약 30초 |
| `car_seg_data_split.py` | 클립 단위 층화 분할 → 모든 형식을 train / val / test 로 | 약 30초 |

세 파일은 **같은 폴더**에 둔다 (EDA·분할이 전처리 파일의 함수를 가져다 쓴다).

---

## 빠른 시작

### 1) 설치
```bash
git clone <이 저장소 주소>
cd <저장소 폴더>
pip install -r requirements.txt          # Python 3.9 이상
```

### 2) 데이터 받기
AI-Hub 에서 지역별로 **원천데이터(이미지)** 와 **라벨링데이터(XML)** 를 **짝으로** 받는다 (폴리곤세그멘테이션).
받은 `download.tar`, `download (1).tar` … 는 **풀지 말고** 저장소 폴더(또는 다운로드 폴더)에 그대로 둔다.

### 3) 실행
```bash
python car_seg_data_preprocessing.py --dry-run    # 점검만 (몇 초) — 조각 누락·매칭 수 확인
python car_seg_data_preprocessing.py              # 전처리
python car_seg_data_eda.py                        # EDA
python car_seg_data_split.py                      # 분할
```
tar 가 다른 폴더에 있으면 전처리에 `--raw-root ~/Downloads` 를 붙인다 (Windows: `--raw-root C:\Users\<이름>\Downloads`).

---

## 결과

```
car_seg_dataset/                      전체 데이터 (분할 전)
├── images/                           이미지 25,589장 (약 60GB)
├── labels/                           YOLO 라벨
├── annotations/instances_all.json    COCO
├── masks/                            마스크 PNG (0=배경 1=car 2=bus 3=truck)
├── manifest.csv                      이미지별 메타데이터 (지역·날짜·시각·날씨·주/야간 …)
└── eda/                              EDA 차트와 표

car_seg_split/                        ★ 학습에 쓰는 폴더
├── images/{train,val,test}/          모든 형식 공유 (car_seg_dataset 으로의 링크, 용량 거의 0)
├── labels/ + data.yaml               → YOLO
├── annotations/instances_{train,val,test}.json   → Mask R-CNN · Detectron2 · MMDetection
├── masks/{train,val,test}/           → SegFormer · DeepLabV3 · U-Net
├── README.md                         모델별로 불러오는 코드
└── format_preview.png                같은 이미지에 형식별 라벨을 그려 비교
```

| 형식 | 클래스 번호 |
|---|---|
| YOLO | 0=car, 1=bus, 2=truck |
| COCO | 1=car, 2=bus, 3=truck (COCO 규약) |
| 마스크 | 0=background, 1=car, 2=bus, 3=truck |

`car_seg_split/` 은 `car_seg_dataset/` 을 가리키는 링크라서 `car_seg_dataset/` 을 지우면 안 된다.
다른 컴퓨터로 옮길 때: `python car_seg_data_split.py --out-dir car_seg_split_share --link-mode hardlink`

---

## 학습 예시 (YOLO)

```bash
# 짧은 시험 (Mac)
yolo segment train data=car_seg_split/data.yaml model=yolo26s-seg.pt imgsz=640 epochs=3 batch=8 device=mps fraction=0.1

# 본 학습 (NVIDIA GPU)
yolo segment train data=car_seg_split/data.yaml model=yolo26s-seg.pt imgsz=1280 epochs=100 batch=16 device=0 patience=20

# 최종 평가 — test 는 마지막에 한 번만
yolo segment val model=runs/segment/train/weights/best.pt data=car_seg_split/data.yaml split=test imgsz=1280
```
- 차량의 약 83%가 작은 객체(학습 해상도 640 기준 32² px 미만)라 `imgsz=1280` 을 권장.
- 참고 속도 (Apple M5, MPS): imgsz 640·batch 8 → 1 epoch 약 50분 / imgsz 1280·batch 2 → 약 2.6시간.
  본 학습은 NVIDIA GPU 권장.
- COCO·마스크 형식 모델은 `car_seg_split/README.md` 참고.

---

## 데이터 정보 (전체 기준)

| 항목 | 값 |
|---|---|
| 이미지 / 클립 | 25,589장 / 226개 (10개 지역, 49개 카메라) |
| 객체 | car 103,075 · bus 4,849 · truck 49,269 |
| 해상도 | 1920×1080, 1080×1920(세로), 1280×720 |
| 분할 | train 20,463 · val 2,560 · test 2,566 (클립 180 · 22 · 24) |

- 라벨에는 **차량만** 있고 도로 영역은 없다. '도로 대비' 면적 비에는 도로 마스크가 따로 필요하다.
- 파일명의 `NH / RH` 는 정의가 확인되지 않은 코드다 (EDA `10_nh_rh_check.png` 참고).
- 5.전북의 부안 클립 4개는 라벨과 이미지의 촬영 시각이 달라(서로 다른 영상) 자동 제외된다.

---

## 주요 옵션

| 스크립트 | 옵션 | 설명 |
|---|---|---|
| 전처리 | `--raw-root 경로 …` | tar 또는 풀어 둔 폴더 위치 (여러 개 가능, 중복 조각 자동 제거) |
| 전처리 | `--formats yolo,coco,mask,labelme` | 만들 형식 (기본 yolo,coco,mask) |
| 전처리 | `--formats-only` | 이미지 추출 없이 형식만 다시 생성 |
| EDA | `--no-brightness` | 밝기 측정 생략 (빨라짐) |
| 분할 | `--ratios 8 1 1` · `--seed 42` | 비율 · 시드 (같은 데이터 + 같은 시드 = 같은 분할) |
| 분할 | `--overwrite` | 기존 분할 지우고 다시 |
| 공통 | `--help` | 전체 옵션 |

## 문제 해결

| 증상 | 해결 |
|---|---|
| `zip 조각을 찾지 못했습니다` | `--raw-root` 에 tar 가 있는 폴더 지정 |
| `❌ 조각 누락` | 해당 zip 의 tar 를 다시 받기 |
| `라벨이 하나도 없는 이미지 zip` / `이미지가 하나도 없는 라벨 zip` | 그 지역의 라벨링데이터 / 원천데이터를 받기 |
| `압축된 tar 이면 먼저 풀어주세요` | 그 파일만 풀고 다시 실행 |
| 분할 때 `symlink 를 만들 수 없어 …` (Windows) | 자동으로 hardlink / 복사로 바뀌어 진행됨 |
| 전처리가 중간에 끊김 | 같은 명령 다시 실행 → 이어서 진행 |

## 데이터 출처

AI-Hub (https://aihub.or.kr) — 교통문제 해결을 위한 CCTV 교통 데이터(고속도로).
데이터는 이용 신청한 사람만 사용할 수 있으며 재배포가 제한될 수 있으므로 저장소에 포함하지 않는다.
