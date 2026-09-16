# 도로 CCTV 차량 세그멘테이션 데이터 — 수집 · 전처리 · EDA · 분할 기록

AI-Hub 「교통문제 해결을 위한 CCTV 교통 데이터(고속도로)」로 차량 instance segmentation 데이터셋을
만들기까지의 전 과정. 무엇을 했는지와 **왜 그렇게 했는지**, 그리고 재현하는 방법을 담는다.

- 최종 목표: 차량 instance segmentation(YOLO-seg) → **차량 면적 비 기반 혼잡도 측정**
- 클래스 3종: `0=car` `1=bus` `2=truck`
- 모든 난수는 `seed=42` 로 고정 — 아래 명령을 그대로 실행하면 같은 결과가 나온다 (검증함)

---

## 0. 한눈에 보기

```
AI-Hub 전체
   │  Validation · 폴리곤세그멘테이션만 내려받음
   ▼
zip 14개 (조각 70개) · 64.8GB PNG
   │  ① 전처리 — 조각 결합 · 매칭 · JPG 변환 · 라벨 변환
   ▼
car_seg_dataset/  25,589장 · 13.6GB JPG · 폴리곤 157,193개
   │  ② EDA — 전수 분석 (샘플링 없음)
   ▼
car_seg_eda/  그림 18장 · 통계 CSV 27개 · EDA_REPORT.md
   │  ③ 분할 — EDA 결과를 근거로 설계
   ▼
car_seg_split/  train 2,500 · val 1,000 · test 5,000
   │  ④ 검증 — 줄인 결과가 의도대로인지 확인
   ▼
car_seg_train_dist/  그림 3장 · 통계 CSV 3개
```

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| ① 전처리 | `car_seg_data_preprocessing.py` | 원본 zip 14개 | `car_seg_dataset/` |
| ② EDA | `car_seg_data_eda.py` | `car_seg_dataset/` | `car_seg_eda/` |
| ③ 분할 | `car_seg_data_split.py` | `car_seg_dataset/` | `car_seg_split/` |
| ④ 검증 | `car_seg_train_dist.py` | `split_manifest.csv` | `car_seg_train_dist/` |

---

## 1. 데이터 수집 — 무엇을 받고 무엇을 포기했나

AI-Hub 원본 데이터셋은 로컬에서 다루기에 너무 크다. 그래서 **범위를 두 번 좁혔다.**

| 선택 | 내용 |
|---|---|
| 받은 것 | `01.데이터 / 2.Validation / 폴리곤세그멘테이션` 만 |
| 포기한 것 | Train 전체, 바운딩박스(객체 검출) 라벨 |
| 받은 형태 | zip 14개 = 원천데이터 7권역 + 라벨링데이터 7권역 |
| 조각 | AI-Hub 가 1GiB 단위로 자른 `.zip.part*` 파일 70개 |
| 용량 | **64.8GB** (64,823,565,037 바이트) |

### Validation 만으로 충분하다고 판단한 근거

val 안에 **CCTV 49대 전부 · 촬영 지역 10곳 · 권역 7개 · 날씨 5종 · 주야간 3구간**이 모두 들어 있다.
조건 커버리지가 완전하므로, "용량 때문에 어쩔 수 없이"가 아니라 **"val 만으로 조건이 다 나오므로"** 선택했다.

### 권역 구성 (zip 단위)

`1.수도권영동선` · `2.강원` · `3.대전충남` · `4.충북` · `5.전북` · `6.부산` ·
`7.기타지역_1.터널내부(주간,야간,날씨,구분불가지역)`

---

## 2. 전처리 — `car_seg_data_preprocessing.py`

한 번의 실행으로 6가지를 처리한다.

### 2.1 zip 조각 결합

AI-Hub 는 큰 zip 을 `파일명.zip.part0`, `.part1073741824`, … 처럼 **1GiB 오프셋 단위로 잘라서** 배포한다.
스크립트가 이 조각들을 순서대로 이어 붙여 하나의 zip 으로 읽는다. 압축을 미리 풀어둘 필요가 없다.

### 2.2 이미지 ↔ 라벨 매칭

| 항목 | 개수 |
|---|---:|
| 라벨 프레임 (XML `<image>`) | 26,022 |
| 이미지 파일 (PNG) | 26,310 |
| **매칭 성공** | **25,589** |
| 라벨 없는 이미지 → 제외 | 524 |
| 이미지 없는 라벨 → 제외 | 433 |
| 같은 이름의 중복 사본 → 경로가 짧은 쪽 사용 | 197 |

> 중복 사본은 `5.전북` 의 이중 폴더 구조 때문에 생긴다. 픽셀이 동일함을 확인했고, 경로가 짧은 쪽을 남겼다.

### 2.3 PNG → JPG 변환

| | |
|---|---|
| 품질 | 95 |
| 원본 PNG | 64.8GB |
| 변환 후 JPG | **13.6GB** (13,597,911,420 바이트) |
| 압축비 | 약 1/4.8 |
| 해상도 변경 | **없음** (`n_size_scaled = 0`) — 폴리곤 좌표에 영향 없음 |

### 2.4 폴리곤 라벨 변환 (CVAT XML → YOLO-seg)

원본은 CVAT 형식 XML 로 `<polygon points="x1,y1;x2,y2;…">` 형태다. 이를 YOLO segmentation 형식으로 바꾼다.

```
<class_id> x1 y1 x2 y2 x3 y3 …      # 모든 좌표는 이미지 크기로 나눈 0~1 정규화 값
```

- 클래스 매핑: `0=car` `1=bus` `2=truck`
- 면적이 1px² 미만인 폴리곤은 제외 (`--min-area`)

### 2.5 파일명 파싱 — 라벨 없이 얻는 메타데이터

파일명이 `_` 로 구분된 12개 토큰으로 되어 있어, **별도 annotation 없이** 촬영 조건을 알 수 있다.

```
Suwon_CH01_20200722_1500_WED_9m_NH_highway_TW5_sunny_FHD_005.jpg
  1     2      3      4    5   6  7    8     9    10   11  12
```

| # | 컬럼 | 의미 | 고유값 | 실제 분포 |
|---|---|---|---:|---|
| 1 | `site` | 촬영 지역 | 10 | Suwon(49클립) Gangneung(31) Jeonju(28) Busan(27) Chungju(24) Buan(23) Cheonan(18) Hongcheon(17) 외 2 |
| 2 | `channel` | 카메라 채널 / 지점명 | 49 | Suwon 은 `CHxx`, 나머지는 교량·터널 등 지점명 |
| 3 | `date` | 촬영 날짜 | 15 | 20200720 ~ 20201213 |
| 4 | `time` | 촬영 시각 (HHMM) | 46 | 0700 ~ 2149 |
| 5 | `weekday` | 요일 | 6 | MON(35) TUE(85) WED(78) THU(16) SAT(7) SUN(5) |
| 6 | `cam_height` | 카메라 높이 | 5 | 4.8m(1) 8m(27) 9m(49) 10m(12) 15m(137) |
| 7 | `hour_type` | **정의 미확인** — NH/RH | 2 | NH(211) RH(15) |
| 8 | `road_type` | 도로 종류 | 1 | 전부 `highway` (구분 정보 없음) |
| 9 | `lane_config` | 차로 방향·수 (추정: TW=양방향, OW=편도) | 5 | TW2(109) TW3(47) OW5(36) OW2(21) TW5(13) |
| 10 | `weather` | 날씨 | 5 | sunny(205) rainy(13) fog(6) tunnel(1) snow(1) |
| 11 | `quality` | 화질 등급 | 2 | FHD / HD |
| 12 | `frame` | 클립 내 프레임 순번 | 393 | 000 ~ 597 |

### 2.6 파생 컬럼 생성

파일명만으로는 부족한 정보를 계산해서 덧붙인다.

| 컬럼 | 만드는 방법 | 결과 |
|---|---|---|
| `camera` | `site` + `channel` | 49대 |
| `road_form` | 지점명 키워드: `brdg`→bridge, `tunnel`→tunnel, `overpass`→overpass, `shelter`→shelter, 그 외 general | general 13,054 · bridge 8,903 · tunnel 1,654 · overpass 1,346 · shelter 632 |
| `day_night` | 촬영 날짜·시각·지역으로 **일출/일몰 시각을 계산**, ±30분은 twilight | day 20,368 · twilight 2,067 · night 3,154 |
| `time_band` | 촬영 시각 3시간 구간 | 07-09 · 10-12 · 13-15 · 16-18 · 19-21 |
| `difficulty` | `day_night=day` + `weather=sunny` + `road_form=general` 이면 easy | easy 8,319 · hard 17,270 |
| `hard_reasons` | easy 조건에서 벗어난 항목 목록 | 예: `day_night=night\|road_form=bridge` |
| `orientation` | width/height 비교 | 가로 / 세로 |

### 2.7 출력

```
car_seg_dataset/
├── images/                  JPG 25,589장 (13.6GB)
├── labels/                  YOLO-seg 라벨 25,589개 (132MB)
├── manifest.csv             46컬럼 × 25,589행 — 이미지 1장당 모든 메타데이터 (17MB)
├── classes.json             {0: car, 1: bus, 2: truck}
└── preprocess_report.json   변환 통계 · 제외 목록 · 아카이브 정보
```

---

## 3. 데이터 구조

### 3.1 계층

```
권역 7개 → 촬영 지역 10곳 → CCTV 49대 → 클립 226개 → 프레임 25,589장 → 폴리곤 157,193개
```

### 3.2 클립 — 이 데이터를 다루는 핵심 개념

**클립 = 한 CCTV 가 한 번(같은 날짜·시각) 촬영한 영상.** 클립당 29~299장, 평균 113장.

같은 클립의 프레임들은 **배경·조명·날씨가 완전히 같고 차량만 조금씩 움직인다.** 즉 서로 매우 비슷하다.
→ 프레임을 무작위로 섞어 train/val 로 나누면 **거의 같은 사진이 양쪽에 들어가 성능이 부풀려진다.**
→ **분할은 반드시 클립 단위로 해야 한다.** (4장 참고)

또한 속성이 어느 단위로 고정되는지가 중요하다.

- **클립 단위 고정**: 날짜 · 시각 · 날씨 · 주야간 · 요일
- **CCTV 단위 고정**: 지역 · 지점 · 카메라 높이 · 차로 구성 · 해상도 · 도로 형태

### 3.3 기본 수치

| 항목 | 값 |
|---|---|
| 이미지 | 25,589장 |
| 클립 | 226개 |
| CCTV | 49대 |
| 폴리곤(객체) | 157,193개 |
| 촬영 기간 | 2020-07-20 ~ 2020-12-13 (15일) |
| 촬영 시각 | 07:00 ~ 21:49 |
| 해상도 | 1920×1080 19,992장 · 1080×1920 4,964장 · 1280×720 633장 |

---

## 4. EDA — `car_seg_data_eda.py`

### 4.1 방법

**샘플링하지 않았다.** 이미지 25,589장 전부, 폴리곤 157,193개 전부, 픽셀 밝기도 전수 측정했다.
(대표 이미지를 고르는 오버레이 샘플 그림만 예외)

### 4.2 클래스 통계

| class | 객체 | 비율 | 등장 이미지 | 등장 비율 | 장당 평균 | 장당 최대 | small | medium | large | 면적 중앙(%) | 꼭짓점 중앙 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 103,075 | 65.6% | 22,632 | 88.4% | 4.03 | 47 | 83.5% | 16.4% | 0.1% | 0.13 | 22 |
| bus | 4,849 | 3.1% | 3,646 | 14.2% | 0.19 | 19 | 63.2% | 32.7% | 4.0% | 0.28 | 23 |
| truck | 49,269 | 31.3% | 18,809 | 73.5% | 1.93 | 19 | 69.0% | 28.5% | 2.5% | 0.21 | 23 |

- small/medium/large 는 긴 변을 `imgsz=640` 으로 줄였을 때의 픽셀 면적 기준 (COCO 32²/96²)
- **클래스 불균형: car : bus : truck = 1 : 0.047 : 0.48** — car 가 bus 의 21배

### 4.3 주요 조건별 분포

**주간/야간** (일출·일몰 계산 기준)

| 값 | 이미지 | 비율 | 밝기 중앙 |
|---|---:|---:|---:|
| 주간 | 20,368 | 79.6% | 116.7 |
| 여명·황혼 | 2,067 | 8.1% | 113.1 |
| 야간 | 3,154 | 12.3% | **62.4** |

**날씨**

| 값 | 이미지 | 비율 |
|---|---:|---:|
| sunny | 23,324 | 91.1% |
| rainy | 1,179 | 4.6% |
| fog | 610 | 2.4% |
| snow | 299 | 1.2% |
| tunnel | 177 | 0.7% |

**도로 형태** (CCTV 지점명 기준)

| 값 | 이미지 | 비율 |
|---|---:|---:|
| general | 13,054 | 51.0% |
| bridge | 8,903 | 34.8% |
| tunnel | 1,654 | 6.5% |
| overpass | 1,346 | 5.3% |
| shelter | 632 | 2.5% |

### 4.4 easy / hard 정의

```python
EASY_CONDITION = {"day_night": "day", "weather": "sunny", "road_form": "general"}
```

- **easy** = 세 조건을 **모두** 만족 → 8,319장 (32.5%)
- **hard** = 하나라도 어긋남 → 17,270장 (67.5%)

hard 가 된 이유 (중복 포함):

| 사유 | 장수 | hard 중 비중 |
|---|---:|---:|
| road_form=bridge | 8,903 | 51.6% |
| day_night=night | 3,154 | 18.3% |
| day_night=twilight | 2,067 | 12.0% |
| road_form=tunnel | 1,654 | 9.6% |
| road_form=overpass | 1,346 | 7.8% |
| weather=rainy | 1,179 | 6.8% |
| road_form=shelter | 632 | 3.7% |
| weather=fog | 610 | 3.5% |
| weather=snow | 299 | 1.7% |
| weather=tunnel | 177 | 1.0% |

어긋난 항목이 1개인 이미지 14,519장, 2개인 이미지 2,751장.

> **주의**: 이름이 '난이도'지만 **실제 검출 난이도를 측정한 값이 아니다.** 촬영 조건이 표준(주간·맑음·일반도로)에서
> 벗어났는지를 표시한 라벨이며, 용도는 **train 이 쉬운 조건에만 쏠리지 않게 하는 선별 기준**이다.

### 4.5 주요 발견 — 각각이 분할 설계로 이어졌다

| 발견 | → 설계 결정 |
|---|---|
| 클래스 불균형: car 66% / truck 31% / **bus 3.1%**, bus 는 이미지의 14% 에만 등장 | train 선별 시 bus 가중 (×4.532) |
| **작은 객체가 78%** (imgsz=640 기준), bus 63% · car 83% | 학습 해상도를 키우는 것이 유리 |
| **주간 편중** 80%, 야간 12%(29클립), 야간 클립은 6개 지역에만 존재 | train 만 hard 우선 선별 |
| **맑음 편중** 91% | 〃 |
| 같은 클립 프레임은 배경·조명이 동일 | **클립 단위 분할** |
| easy 영상만 있는 CCTV 6대 | hard 만으로 뽑으면 해당 CCTV 소멸 → 부족분만 easy 로 보충 |
| **클립이 1~2개뿐인 CCTV 3대** (Jincheon_jangam, Buan_dongjisan, Inje_injetunnel2) | 클립을 시간 구간으로 나눠 배정 |
| bus 가 특정 지역에 집중 — 상위 3곳(Suwon·Cheonan·Chungju)이 bus 의 74% | — |
| 밝기 중앙값 주간 117 / 여명·황혼 113 / **야간 62** (전수 측정) | day_night 계산이 실제 픽셀과 일치함을 확인 |
| **차량 폴리곤 겹침 평균 0.5%** (2대 이상 이미지) | 면적 비를 단순 합이 아닌 **합집합**으로 계산 |
| NH/RH 정의 미확인 — RH 는 15클립뿐, 같은 카메라 비교에서 RH 쪽 차량이 많은 곳이 6/9 | 분석에서 보조 지표로만 사용 |

### 4.6 무결성 점검

| 항목 | 개수 |
|---|---:|
| orphan_label (이미지 없는 라벨) | **0** |
| orphan_image (라벨 없는 이미지) | **0** |
| images_without_objects (객체 0개 이미지) | **0** |

### 4.7 산출물

```
car_seg_eda/
├── EDA_REPORT.md              ★ 데이터 설명서 (전체 통계표 + 발견 + 무결성)
├── 00_filename_tokens.md/.csv 파일명 토큰 사전
├── 01_class_distribution.png  클래스별 객체 수 · 등장 이미지 비율
├── 02_metadata_distribution.png  메타데이터 전체 분포
├── 03_objects_per_image.png   이미지당 객체 수
├── 04_object_size.png         객체 크기 S/M/L
├── 05_vehicle_area_ratio.png  차량 면적 비 — 조건별
├── 06_centroid_heatmap.png    객체 중심점 분포 (가로/세로 분리)
├── 07_vertices_per_polygon.png 폴리곤 꼭짓점 수 (라벨 정밀도)
├── 08_frames_per_clip.png     클립당 프레임 수
├── 09_samples.png             라벨 오버레이 샘플 (좌표 정합성 육안 확인)
├── 10_nh_rh_check.png         NH/RH 의미 확인
├── 11_brightness.png          픽셀 밝기 (전수 측정)
├── 12_day_night_samples.png   주간/여명·황혼/야간 대표 이미지
├── 13_vehicle_overlap.png     차량 폴리곤 겹침
├── 14_class_by_attribute.png  속성값별 클래스 비율
├── 15_road_form_camera.png    도로 형태 · CCTV별 구성
├── 16_condition_matrix.png    조건 교차 히트맵
├── 17_difficulty.png          easy/hard 분석
├── 18_image_quality.png       화질·해상도
├── stats/                     세부 통계 CSV 27개 (속성별 · CCTV별 · 클립별 · 교차표)
├── eda_per_image.csv          이미지 1장당 집계
├── eda_polygons.csv.gz        폴리곤 1개당 원본 수치
├── eda_brightness.csv         이미지별 밝기
└── eda_summary.json           요약 수치
```

---

## 5. 데이터 분할 — `car_seg_data_split.py`

### 5.1 설계 원칙

| 세트 | 목표 | 이유 |
|---|---|---|
| **val · test** | 전체 분포를 **닮게** | 평가가 실제 데이터를 대표해야 하므로 |
| **train** | 의도적으로 **다르게** | 제한된 장수 안에 어려운 조건과 희귀 클래스를 더 담아야 하므로 |

목적이 반대이므로 **방식도 다르다.**

공통 제약:
- 세 세트 모두 **CCTV 49대가 전부 포함**되어야 한다 (특정 구도가 평가에서 빠지면 안 됨)
- **클립 단위로 자른다** (누수 방지)
- 버리는 프레임이 없다 — 모든 프레임이 train 후보 / val / test 중 하나에 배정된다

### 5.2 1단계 — 클립 역할 배정 (val · test)

**① CCTV별 할당량 배분**
전체 49대의 이미지 비율대로 val 1,000장 · test 5,000장을 정수 배분 (최대 나머지 방식).

**② 각 CCTV의 선택지 생성**

| CCTV 유형 | 대수 | 처리 |
|---|---:|---|
| 클립 3개 이상 | 46 | (val 클립, test 클립, 나머지 train 전용) 조합을 모두 나열 |
| 클립 2개 | 1 | 한 클립 안에서 val·test 구간을 앞뒤로 분리, 다른 클립은 train 전용 |
| 클립 1개 | 2 | 한 클립 안에서 val·test 구간을 앞뒤로 최대한 멀리 배치 |

해당 CCTV: `Buan_dongjisan` · `Inje_injetunnel2` · `Jincheon_jangam`

**③ 목적함수 최소화**

```
비용 = Σ |현재 분포 − 목표 분포| / 목표 분포 × 속성 가중치  +  벌점
목표 분포 = 전체 데이터의 이미지 가중 평균 × 세트 크기
```

속성 가중치:

| 속성 | 가중치 |
|---|---:|
| day_night · weather · difficulty · time_band | 1.0 |
| 클래스 (car/bus/truck 객체 수) | 1.0 |
| weekday · hour_type | 0.5 |
| date | 0.3 |

벌점 2종:

| 벌점 | 계수 | 이유 |
|---|---:|---|
| 할당량 부족 | ×5.0 | 클립이 짧아 목표 장수를 못 채우는 경우 |
| val·test 클립에 남는 hard | ×3.0 | 남은 프레임은 train 후보가 되지만 후순위라, hard 가 많은 클립을 val·test 로 쓰면 train 이 쓸 '깨끗한' hard 가 줄어든다 |

**④ 최적화** — 좌표 하강(CCTV 하나씩 최적 선택지로 교체)을 수렴할 때까지, 무작위 초기값으로 30회 재시작.
최종 비용 **1.431247**.

### 5.3 2단계 — 누수 위험도 산정

train 후보 프레임마다 위험도를 매긴다.

| risk | 정의 |
|---|---|
| **0** | val·test 가 없는 클립 = 깨끗함 |
| **1** | 같은 클립이지만 val·test 구간에서 **10장(guard) 넘게** 떨어짐 |
| **2** | val·test 구간에서 **10장 이내** = 가장 위험 |

**예외 규칙**: risk 1 이라도 그 CCTV 의 risk 0 클립에 **없는 날씨·주야간 조합**이면 0 으로 올린다.
(예: 눈 오는 클립이 하나뿐인데 그게 test 가 되면, 후순위로 두는 순간 train 에 눈이 한 장도 안 들어간다)

### 5.4 3단계 — train 축소 (19,589 → 2,500)

**① CCTV별 할당량** — 전체 CCTV 비율대로 2,500장 배분 (최소 1장 보장)

**② 6단계 우선순위로 채움**

```
hard·risk0 → hard·risk1 → easy·risk0 → easy·risk1 → hard·risk2 → easy·risk2
```

hard 를 먼저, 누수 위험이 큰 것을 나중에. 두 축을 교차시킨 순서다.

**③ 각 단계 안에서 프레임 고르기**

두 가지를 동시에 만족시킨다.

- **시간적으로 고르게**: 클립의 프레임을 k개 구간으로 나눠 **구간마다 1장**씩 뽑는다.
  (점수 순으로만 뽑으면 한 클립의 비슷한 연속 프레임만 선택된다)
- **희귀 클래스 우선**: 구간 안에서는 클래스 점수가 가장 높은 프레임을 선택

```
점수 = Σ (클래스별 객체 수 × 가중치)
가중치 = (car 객체 수 / 해당 클래스 객체 수) ^ alpha,  alpha = 0.5
```

| 클래스 | 가중치 |
|---|---:|
| car | 1.000 |
| bus | **4.532** |
| truck | 1.435 |

### 5.5 분할 결과

| split | 이미지 | 클립 | CCTV | hard 비율 | car | bus | truck |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2,500 | 110 | 49/49 | 88.8% | 11,919 | 946 | 5,977 |
| val | 1,000 | 49 | 49/49 | 67.3% | 3,730 | 143 | 1,984 |
| test | 5,000 | 72 | 49/49 | 67.2% | 21,615 | 922 | 9,538 |

> **test 가 train 보다 큰 것은 의도한 설계다.** 학습은 자원 제약을 받지만, 평가는 신뢰도가 우선이다.

프레임 사용 현황: 전체 25,589장 = train 후보 19,589 + val 1,000 + test 5,000, **미배정 0장**.
train 후보 중 2,500장 선택, 17,089장 미사용(버린 것이 아니라 목표 장수에 따라 남긴 것).

### 5.6 train 축소 상세

| 항목 | 값 |
|---|---|
| 후보 → 선택 | 19,589 → 2,500 |
| hard / easy 보충 | 2,219 / 281 |
| 누수 위험별 선택 | risk 0: **2,199** · risk 1: **301** · risk 2: **0** |
| CCTV 비율 최대 편차 | **0.02%p** |
| easy 로 보충한 CCTV | 8대 (Buan_daemog, Buan_dongjisan, Buan_yulchon, Cheonan_ogsan, Cheonan_sajeong, Cheonan_seongnam2, Gangneung_nodong, Jincheon_jangam) |
| 할당량 미달 CCTV | 없음 |

**risk 2(가장 위험한 프레임)를 한 장도 쓰지 않았다.**

### 5.7 누수 점검

| 항목 | 값 |
|---|---:|
| 세트 간 공유 클립 | 22개 |
| ├ train ↔ val/test | 21 |
| └ val ↔ test | 3 |
| 최소 프레임 간격 — train↔test | 11 (중앙값 11.5) |
| 최소 프레임 간격 — val↔test | 37 (중앙값 49.0) |
| 최소 프레임 간격 — train↔val | 11 (중앙값 11.0) |

### 5.8 산출물

```
car_seg_split/
├── images/{train,val,test}/   JPG (car_seg_dataset/images 로의 hardlink)
├── labels/{train,val,test}/   YOLO-seg 라벨
├── data.yaml                  YOLO 학습 설정
├── split_manifest.csv         51컬럼 — manifest 46컬럼 + pool·split·selection·leak_risk·select_priority
├── split_clips.csv            클립별 역할
├── camera_coverage.csv        CCTV × 세트 이미지 수
├── train_selection_cameras.csv CCTV별 선택 내역
├── split_report.json          전체 통계 · 분포 편차 · 누수 점검
├── split_distribution.png     세트별 분포 비교
├── train_reduction.png        축소 전후
└── README.md                  분할 방식 요약
```

> **이미지는 hardlink 다.** 폴더째 복사하면 깨지거나 용량이 두 배가 된다. 공유할 때는 반드시 `tar` 로 묶을 것.

---

## 6. 분할 검증 — `car_seg_train_dist.py`

줄인 결과가 **의도대로인지** 확인한다. `split_manifest.csv` 하나만 읽으므로 이미지 파일이 없어도 실행된다.

### 6.1 전체 분포 대비 최대 편차 (%p)

| 속성 | train | val | test |
|---|---:|---:|---:|
| 난이도 조건 | **21.27** | 0.19 | 0.29 |
| 주 / 야간 | **18.20** | 0.53 | 0.10 |
| 촬영 시간대 | **10.35** | 1.18 | 0.41 |
| 요일 | 6.51 | 0.79 | 0.69 |
| 날씨 | 6.47 | 1.17 | 0.99 |
| 촬영 날짜 | 4.18 | 3.19 | 1.59 |
| 클래스 (객체) | 2.31 | 2.53 | 1.82 |
| NH / RH | 1.85 | 0.01 | 0.21 |
| 차로 구성 | 0.05 | 1.17 | 0.34 |
| 카메라 높이 | 0.03 | 0.15 | 0.13 |
| 도로 형태 | 0.03 | 0.11 | 0.37 |
| 촬영 지역 | 0.03 | 0.15 | 0.16 |
| **CCTV (49대)** | **0.02** | 0.05 | 0.20 |
| 화질 | 0.01 | 0.03 | 0.01 |
| 영상 방향 | 0.00 | 0.20 | 0.10 |

**읽는 법**: 유지해야 할 축(CCTV·지역·도로 형태·화질·높이·방향)은 train 도 0.05%p 이내로 붙어 있고,
바꾸려 한 축(난이도·주야간·시간대·날씨)만 크게 벌어진다. **편차가 큰 것이 실수가 아니라 설계값이다.**

val·test 는 성능에 직결되는 축이 모두 0.5%p 이내다.

### 6.2 train 축소 전후

| 항목 | 축소 전 (19,589장) | 축소 후 (2,500장) |
|---|---:|---:|
| hard | 67.6% | **88.8%** |
| 주간 | 79.6% | 61.4% |
| 여명·황혼 | 8.1% | **15.5%** |
| 야간 | 12.3% | **23.1%** |
| 맑음 | 91.3% | 84.7% |
| 비 | 4.6% | **9.3%** |
| 안개 | 2.4% | **4.1%** |
| 눈 | 1.1% | 1.2% |
| 교량 | 34.9% | 34.8% |
| 터널 | 6.5% | 6.5% |
| car (객체) | 65.2% | 63.3% |
| **bus (객체)** | 3.2% | **5.0%** |
| truck (객체) | 31.7% | 31.7% |
| car : bus 비 | 20.5 : 1 | **12.6 : 1** |

**88%를 버렸는데 어려운 조건과 희귀 클래스의 비중은 오히려 늘었다.** 반면 도로 형태와 truck 비율은 그대로다.

### 6.3 클래스 비율

| 클래스 | 전체 | train | val | test |
|---|---:|---:|---:|---:|
| car | 65.57% | 63.26% | 63.68% | 67.39% |
| bus | 3.08% | **5.02%** | 2.44% | 2.87% |
| truck | 31.34% | 31.72% | 33.87% | 29.74% |

### 6.4 산출물

```
car_seg_train_dist/
├── 01_deviation.png       전체 대비 편차 — train 만 틀어져 있음
├── 02_reduction.png       축소 전후 (아령 차트)
├── 03_distribution.png    주요 조건의 실제 분포
├── stats_deviation.csv
├── stats_reduction.csv
└── stats_distribution.csv
```

---

## 7. 재현 방법

### 7.1 환경

```bash
pip install pandas numpy pillow matplotlib tqdm
```

외부 패키지는 위 5개뿐이고 나머지는 파이썬 표준 라이브러리다.

> 이 맥에서는 `python3` 가 시스템 파이썬(3.9.6, 패키지 없음)을 가리킨다. **`python`** 또는
> `/opt/anaconda3/bin/python` 을 쓸 것.

### 7.2 실행

```bash
# 0) 사전 점검 — 아무 파일도 쓰지 않고 zip 조각·매칭만 확인 (1~2분)
python car_seg_data_preprocessing.py --dry-run

# 1) 전처리 — zip 결합 · JPG 변환 · 라벨 변환 (30~60분)
caffeinate -i python car_seg_data_preprocessing.py 2>&1 | tee log_1_preprocess.txt

# 2) EDA (10~20분)
python car_seg_data_eda.py 2>&1 | tee log_2_eda.txt

# 3) 분할 (1~2분)
python car_seg_data_split.py 2>&1 | tee log_3_split.txt

# 4) 분할 검증 (10초)
python car_seg_train_dist.py
```

### 7.3 단계별 확인 지점

| 단계 | 확인할 것 | 기대값 |
|---|---|---|
| 0 | 매칭 성공 | 25,589 (제외 524 / 433 / 197) |
| 1 | `preprocess_report.json` | orphan 0 · 빈 라벨 0 |
| 2 | `EDA_REPORT.md` | car 65.6% · bus 3.1% · small 78% |
| 3 | `split_report.json` | CCTV 편차 0.02%p · train hard 88.76% |
| 4 | 콘솔 출력 | 유지된 축 7개 (편차 ≤ 0.5%p) |

`seed=42` 가 고정되어 있어 **위 수치가 소수점까지 동일하게 재현된다** (실제로 두 번 실행해 확인함).

### 7.4 학습

```bash
yolo segment train data=car_seg_split/data.yaml model=yolo11s-seg.pt imgsz=1280 epochs=100
yolo segment val   model=runs/segment/train/weights/best.pt data=car_seg_split/data.yaml split=test imgsz=1280
```

Colab 등에 올릴 때는 `data.yaml` 의 `path:` 를 해당 환경의 경로로 고쳐야 한다.

---

## 8. 데이터 공유

| 패키지 | 내용 | 크기 | 용도 |
|---|---|---:|---|
| 라벨 + 메타 | `labels/` `manifest.csv` `classes.json` `preprocess_report.json` | **33.5MB** | EDA (18개 중 15개 가능) |
| 셋분리 전 전체 | `car_seg_dataset/` | 13.6GB | EDA 전체 + 직접 분할 |
| 셋분리 결과 | `car_seg_split/` | 4.4GB | 바로 학습 |

```bash
# 라벨 + 메타만 (텍스트라 gzip 효과가 크다)
tar -czf car_seg_dataset_meta.tar.gz \
  car_seg_dataset/labels car_seg_dataset/manifest.csv \
  car_seg_dataset/classes.json car_seg_dataset/preprocess_report.json

# 전체 / 분할 결과 (JPG 는 이미 압축이라 -z 는 시간만 든다)
tar -cf car_seg_dataset_full.tar car_seg_dataset
tar -cf car_seg_split.tar car_seg_split
```

라벨+메타만 받은 경우 EDA 는 픽셀이 필요한 항목을 빼고 실행한다.

```bash
python car_seg_data_eda.py --no-brightness --n-samples 0
```

빠지는 것: 11번(밝기) · 09번(오버레이 샘플) · 12번(주야간 샘플). 나머지 15개는 라벨과 파일명 기반이라 그대로 나온다.

---

## 9. 한계와 주의사항

정직하게 기록해 둔다. 이 데이터로 결론을 내릴 때 감안해야 할 것들이다.

**① JPG 변환의 정보 손실을 측정하지 않았다.**
64.8GB → 13.6GB(1/4.8)로 줄였으나 품질 95 라는 **설정값만 있고 PSNR/SSIM 측정치가 없다.**
EDA 항목 중 픽셀에 의존하는 것은 밝기(11번) 하나뿐이고, 나머지는 라벨·파일명 기반이라 영향이 없다.
밝기도 JPEG 압축이 고주파를 깎지 평균 밝기를 옮기지는 않으므로 실질적 영향은 작을 것으로 보이나, **검증되지 않았다.**

**② `road_form` 은 CCTV 지점명 기준이다.**
지점명에 `brdg`/`tunnel` 등이 포함되면 해당 값을 부여했다. 화면 텍스트로 확인한 결과 **대부분 구조물 '근처'를
비추는 카메라**이며(예: 추점터널 CCTV 는 터널 밖 도로), 실제 터널 내부 영상은 `Inje_injetunnel2` 하나뿐이다.
hard 사유의 51.6%가 bridge 이므로, 이 한계는 easy/hard 구분 전체에 영향을 준다.

**③ easy/hard 는 실제 난이도가 아니다.**
촬영 조건 라벨이며, 검출 난이도와의 상관관계는 검증되지 않았다.
예비 실험 두 건에서 bridge vs general 의 mask mAP50-95 가 각각 (0.4034 vs 0.4436), (0.2764 vs 0.2356) 으로
**방향이 반대로 나왔다** (각 45장·43장이라 표본이 작아 판단 근거로 쓸 수 없다).
정식 학습 후 test 5,000장으로 재검증할 것.

**④ val 의 bus 객체가 143개뿐이다.**
전체 3.08% 대비 2.44% 로 비율 편차 자체는 작지만, **절대 개수가 적어 bus mAP 가 크게 출렁인다.**
bus 성능은 val 이 아니라 **test(922개)** 로 판단해야 한다. val 은 car·truck 기준의 조기 종료·모델 선택에 쓴다.

**⑤ `date` 편차가 val 3.19%p 로 다른 축보다 크다.**
의도적이다. 가중치를 0.3 으로 낮게 두었다 — 15일치가 모두 같은 계절·같은 도로라 맞출 실익이 없고,
클립 단위로 자르면 날짜가 통째로 따라오기 때문이다.

**⑥ `hour_type`(NH/RH) 의 정의를 확인하지 못했다.**
상행/하행 또는 러시아워(Rush Hour)로 추정되나 확정하지 못했다. RH 는 15클립뿐이고,
같은 카메라끼리 비교했을 때 RH 쪽 차량이 더 많은 곳이 9개 중 6개로 일관되지 않다. 보조 지표로만 사용했다.

**⑦ 도로 영역 라벨이 없다.**
따라서 '차량 면적 비'는 도로 면적 대비가 아니라 **이미지 전체 면적 대비**다.
혼잡도로 해석할 때 카메라 구도(도로가 화면에서 차지하는 비율)에 따라 값이 달라진다는 점을 감안해야 한다.

**⑧ 원본 데이터셋의 Train 세트를 쓰지 않았다.**
Validation 만으로 CCTV·조건 커버리지는 확보했으나, 절대적인 데이터 양은 원본 전체보다 훨씬 적다.
