# 차량 인스턴스 세그멘테이션 — 학습 · 평가 · 추론

**Segmentation 기반 도로 CCTV 의 도로 대비 차량 면적 비를 활용한 교통 혼잡도 측정 시스템** 프로젝트의 차량 모델 코드.

`data_processing/` 이 만든 분할 데이터(`car_seg_split_upload/`)로 여러 인스턴스 세그멘테이션 모델을
**같은 데이터 · 같은 채점 기준**에서 파인튜닝하고 비교한다.

| 계열 | 모델 | 라이브러리 |
|---|---|---|
| YOLO | YOLOv8-Seg · YOLO11-Seg · YOLO26-Seg | `ultralytics` |
| 2-stage | Mask R-CNN (ResNet-50-FPN v2) | `torchvision` |
| Transformer | Mask2Former (Swin-T/S/B/L) | `transformers` |
| 실시간 DETR | RF-DETR Segmentation (Nano~XLarge) | `rfdetr` |

핵심 설계: 모델마다 출력 형식이 달라도 **예측을 공통 형식(`InstancePrediction`)으로 바꿔 한 채점기(COCOeval)로 평가**한다.
그래서 모든 실험이 한 기록표(`experiments.csv`)에서 바로 비교된다.

---

## 1. 전체 흐름

```
[맥] data_processing/  →  car_seg_split_upload/  →  make_upload_tar.sh  →  .tar (검증 포함)
                                                                              │ 브라우저로 업로드
[Colab] carseg_colab.ipynb                                                     ▼
   ├─ 0 환경: 드라이브 마운트 → tar 를 /content 에 풀기 → pip install
   ├─ 2 evaluate.py --pretrained   사전학습 모델 zero-shot 기준선
   ├─ 3 train.py                   파인튜닝 → val 통합 채점 → 기록표 1행
   ├─ 4 train.py --set ...         하이퍼파라미터 바꿔 반복
   ├─ 5 make_report.py             기록표 → 표
   ├─ 6 evaluate.py --split test   최종 후보만 test
   └─ 7 infer.py                   차량 대수 · 픽셀 수 · 합집합 마스크 · 시각화
```

## 2. 폴더 구조

```
training/
├── carseg_colab.ipynb        ★ Colab 에서 위에서부터 실행
├── train.py                  학습 → val 채점 → 기록
├── evaluate.py               사전학습(zero-shot) / 학습된 모델 평가 → 기록
├── infer.py                  이미지·폴더 추론 → results.csv · 시각화 · 합집합 마스크
├── make_report.py            experiments.csv → 마크다운 표
├── make_upload_tar.sh        업로드용 tar 생성 + 파일 수 검증
├── requirements.txt
├── configs/                  실험 설정 (YAML 하나 = 실험 하나)
│   ├── yolov8s_seg.yaml  yolo11s_seg.yaml  yolo26s_seg.yaml
│   ├── maskrcnn_r50_v2.yaml
│   ├── mask2former_swin_s.yaml
│   └── rfdetr_seg_medium.yaml
└── carseg/                   패키지
    ├── config.py             설정 데이터클래스 · YAML 로드 · --set 덮어쓰기
    ├── data.py               분할 폴더 접근 · COCO 인스턴스 데이터셋 · 증강
    ├── coco_eval.py          ★ 통합 평가기 (COCO mAP + 점유율 지표)
    ├── experiment_log.py     성능 기록표 (CSV 누적, 마크다운 렌더)
    ├── occupancy.py          합집합 마스크 · 클래스별 대수/픽셀 · 도로 점유율
    ├── viz.py                마스크·상자·요약 패널 시각화
    ├── inference.py          VehicleSegmenter — 혼잡도 시스템 연동용 API
    └── models/
        ├── base.py           InstancePrediction · Predictor · Trainer · 클래스 매핑
        ├── __init__.py       family → Trainer/Predictor 레지스트리
        ├── torch_loop.py     공용 학습 루프 (AMP · 누적 · warmup+cosine · best/last · 조기 종료)
        ├── ultralytics_seg.py
        ├── maskrcnn.py
        ├── mask2former.py
        └── rfdetr_seg.py
```

## 3. 빠른 시작

### 3-1. 맥 — 업로드용 tar
```bash
bash training/make_upload_tar.sh ../traffic_data2/car_seg_split_upload
```
tar 를 만든 뒤 목록의 이미지·라벨·마스크 수와 COCO 주석을 원본 폴더와 비교한다. `✅ 검증 통과` 가 나온 파일만 올린다.
(`COPYFILE_DISABLE=1` 로 macOS 의 `._*` 파일이 끼지 않게 한다.)

### 3-2. 구글 드라이브
```
MyDrive/carseg/
├── car_seg_split_upload.tar
└── training/              ← CODE_SOURCE="drive" 일 때만 (git 이면 불필요)
```

### 3-3. Colab
`carseg_colab.ipynb` 를 열고 **0번 셀의 경로만 수정**한 뒤 위에서부터 실행한다.

### 3-4. 명령어 직접 쓰기
```bash
# 공통: 데이터 위치와 결과 저장 위치
COMMON="data.root=/content/car_seg_split_upload out_dir=/content/drive/MyDrive/carseg/runs"

python evaluate.py --pretrained --config configs/yolo11s_seg.yaml configs/maskrcnn_r50_v2.yaml --set $COMMON
python train.py    --config configs/yolo11s_seg.yaml --set $COMMON
python train.py    --config configs/yolo11s_seg.yaml --set $COMMON name=yolo11s_img1536 train.imgsz=1536 train.batch=8
python evaluate.py --run /content/drive/MyDrive/carseg/runs/yolo11s_seg_1280 --split test
python infer.py    --run /content/drive/MyDrive/carseg/runs/yolo11s_seg_1280 --source some_dir/ --save-vis
python make_report.py --csv /content/drive/MyDrive/carseg/runs/experiments.csv
```
> zsh(맥 터미널)에서 리스트 값은 따옴표로 감싼다: `"model.extra.image_size=[768,1376]"`

## 4. 실험 설정

YAML 한 파일에 실험 하나. 명령줄 `--set 키=값` 으로 무엇이든 덮어쓸 수 있고, **실제로 쓴 설정은 `runs/<name>/config.yaml` 에 저장**된다.

| 섹션 | 주요 키 | 설명 |
|---|---|---|
| (최상위) | `name` · `group` · `notes` · `out_dir` | 실험 이름(=결과 폴더) · 기록표 그룹 · 메모 · 결과 루트 |
| `data` | `root` · `train_limit` · `eval_limit` | 데이터 폴더 · 빠른 점검용 장수 제한 |
| `model` | `family` · `weights` · `backbone` · `arch_mods` · `extra` | 계열 · 사전학습 가중치 · 기록용 표기 · 계열별 인자 |
| `train` | `epochs` `batch` `imgsz` `lr0` `optimizer` `weight_decay` `warmup_epochs` `accumulate` `amp` `patience` `freeze` `resume` | 공통 하이퍼파라미터 |
| `train` (증강) | `hflip` `vflip` `color_jitter` `mosaic` `mixup` `copy_paste` `degrees` `translate` `scale` | YOLO 는 전부, 그 외는 `hflip` `vflip` `color_jitter` |
| `train` (손실) | `box_gain` `cls_gain` `dfl_gain` | YOLO 손실 가중치 |
| `eval` | `conf` `iou` `mask_threshold` `max_det` `tta` `union_conf` `eval_every` `train_eval_limit` | 채점·추론 설정 |

계열별로 쓰는 항목:

| 항목 | YOLO | Mask R-CNN | Mask2Former | RF-DETR |
|---|---|---|---|---|
| `imgsz` | 입력 크기 | 짧은 변(min_size) | `extra.image_size` 기본값 계산 | 사용 안 함 (`extra.resolution`) |
| `lr0` / `optimizer` | `optimizer=auto` 면 lr 자동 | ✅ | ✅ | `lr` 로 전달 |
| `eval_every` | 매 epoch 자체 val | ✅ | ✅ | 자체 val |
| `tta` | ✅ | – | – | – |
| `resume` | `ultralytics/weights/last.pt` | `weights/last.pt` | `weights/last.pt` | `rfdetr/last.ckpt` |

## 5. 평가 지표 (`carseg/coco_eval.py`)

모든 계열을 **COCO 주석(`annotations/instances_<split>.json`) + pycocotools COCOeval** 로 채점한다.

| 지표 | 의미 |
|---|---|
| `mask_mAP` | 마스크 AP@[.50:.95] — **주 지표** (정렬·best 선택 기준) |
| `mask_mAP50` · `mask_mAP75` | IoU 0.5 / 0.75 |
| `mask_mAP_small` · `_medium` · `_large` | 객체 크기별 (차량의 약 83% 가 small) |
| `box_mAP` · `box_mAP50` | 상자 기준 |
| `AP_car` · `AP_bus` · `AP_truck` | 클래스별 마스크 AP |
| `vehicle_IoU` | 예측 합집합 마스크 vs 정답 마스크, 데이터셋 전체 IoU — **점유율 계산 정확도에 직결** |
| `area_ratio_MAE_pp` | \|예측 차량 면적비 − 정답 면적비\| 평균 (%p) |
| `count_MAE` | \|예측 대수 − 정답 대수\| 평균 |
| `fps` | 이미지당 추론 속도 (전·후처리 포함) |

- mAP 는 COCO 관례대로 낮은 `conf`(0.01)로 채점하고, 점유율 지표는 실제 사용할 임계값 `union_conf`(0.5) 이상 예측만 쓴다.
- Ultralytics 가 학습 중 출력하는 mAP 는 보간 방식이 달라 수치가 조금 다르다. **모델 간 비교는 이 평가기 값으로 한다.**
- 사전학습 zero-shot 평가(`--pretrained`)는 COCO 80종 중 이름이 `car` / `bus` / `truck` 인 예측만 우리 클래스로 바꿔 채점한다.

## 6. 성능 기록표 (`experiment_log.py`)

`train.py` · `evaluate.py` 가 끝날 때마다 `<out_dir>/experiments.csv` 에 한 행이 추가된다 (엑셀에서 한글이 깨지지 않는 UTF-8-SIG).

| 열 묶음 | 열 |
|---|---|
| 식별 | `no` `date` `group` `exp` `stage`(pretrained/finetuned) `split` |
| 모델 | `family` `weights` `backbone` `arch_mods` `model_extra` |
| 결과 | `mask_mAP` … `AP_truck` `vehicle_IoU` `area_ratio_MAE_pp` `count_MAE` `fps` |
| 학습 | `epochs` `epochs_done` `imgsz` `batch` `accumulate` `lr0` `optimizer` … `box` `cls` `dfl` |
| 추론 | `conf` `iou` `mask_thr` `union_conf` `TTA` `postprocess` |
| 기타 | `train_time` `n_images` `eval_weights` `notes` |

새 설정·지표를 추가해도 기존 행은 빈칸으로 두고 열을 합친다. `make_report.py` 가 그룹별 표(그룹 최고 성능 ★)를 만든다.

## 7. 추론 · 점유율 (`infer.py`, `carseg/inference.py`)

```python
from carseg.inference import VehicleSegmenter

seg = VehicleSegmenter.from_run("runs/yolo11s_seg_1280")   # config.yaml + best 가중치
out = seg.run("frame.jpg", road_mask=road_bool_hw)           # 도로 모델이 없으면 road_mask=None

out.stats.counts            # {'car': 12, 'bus': 1, 'truck': 3}
out.stats.class_pixels      # 클래스별 합집합 픽셀
out.union_mask              # (H, W) bool = masks.any(axis=0)
out.stats.vehicle_pixels    # union_mask.sum() — 겹친 픽셀은 한 번만
out.stats.overlap_pixels    # 겹쳐서 제외된 픽셀 수
out.stats.occupancy         # (차량 ∧ 도로) 픽셀 / 도로 픽셀
```

`infer.py` 결과: `results.csv`(이미지별 한 행) · `results.json`(인스턴스별 점수·클래스·상자·픽셀) ·
`vis/*.jpg`(`--save-vis`) · `union_masks/*.png`(`--save-masks`). 도로 마스크는 `--road-mask` 로 파일 또는 폴더(같은 파일명 `.png`)를 준다.

## 8. 모델별 메모

- **YOLO** — 가중치 이름만 바꾸면 세대·크기 변경 (`yolo26m-seg.pt` 등). `batch=-1` 은 GPU 메모리 자동.
  학습 전 실행 환경 경로로 `data.yaml` 을 새로 쓴다(저장소의 data.yaml 은 맥 절대 경로라 Colab 에서 못 씀).
- **Mask R-CNN** — 추가 설치 없음. 원본 해상도(짧은 변 1080)와 작은 앵커(16~256)를 기본으로 소형 차량에 맞춤.
  zero-shot 평가 때는 사전학습 앵커를 그대로 쓴다.
- **Mask2Former** — HF 기본 후처리는 마스크를 384×384 로 줄였다 키워 작은 차량이 뭉개지므로
  논문 방식(query×class top-k, 점수 = 클래스 확률 × 마스크 평균 확률)을 입력 해상도에서 직접 구현했다.
  입력은 `extra.image_size` 고정 크기(기본 768×1376, 16:9). fp16 에서 매칭 손실이 불안정할 수 있어 `amp: false` 기본.
  학습이 끝나면 `weights/best_hf/` (from_pretrained 가능 폴더)도 만든다.
- **RF-DETR** — `car_seg_split` 을 Roboflow COCO 형식(`train/valid/test/_annotations.coco.json`)으로 심볼릭 링크해 학습한다.
  학습에는 `rfdetr[train,loggers]` 가 필요하다.

## 9. 검증 내역

로컬(맥, CPU)에서 실제 데이터 일부(train 24 / val 8 / test 8장)로 확인했다. **본 학습 성능은 아직 측정 전**이다.

| 확인 항목 | 결과 |
|---|---|
| 평가기 — 정답을 그대로 예측으로 넣기 | mask/box mAP = 1.000, vehicle_IoU = 0.985 (폴리곤↔PNG 래스터 차이) |
| 평가기 — 정답 절반 제거 | mAP 0.71 로 하락 |
| 사전학습 zero-shot (val 8장) | YOLO11n 0.70 · YOLO26n 0.78 · YOLOv8n 0.75 · Mask R-CNN 0.70 · Mask2Former-T 0.82 · RF-DETR-Nano 0.62 |
| 학습 → best 저장 → 재로딩 → val/test 채점 → 기록 | 4개 계열 모두 동작 (Mask R-CNN 2 epoch: 0 → 0.164) |
| 추론 · 시각화 · 합집합 마스크 · CSV | 동작, `vehicle_pixels + overlap_pixels = Σ 인스턴스 픽셀` 확인 |

GPU·AMP·멀티 워커 경로와 전체 데이터 학습은 Colab 에서 처음 돌게 되므로, 노트북 **1번(빠른 점검)** 을 먼저 실행하길 권한다.

## 10. 문제 해결

| 증상 | 해결 |
|---|---|
| Colab 에서 이미지 수가 기대값(20463/2560/2566)보다 적음 | tar 가 불완전. 맥에서 `make_upload_tar.sh` 로 다시 만들어 검증 후 업로드 |
| `CUDA out of memory` | `train.batch` 절반 + `train.accumulate=2`, 또는 `train.imgsz` 축소 |
| Mask2Former loss 가 nan | `train.amp=false` 유지, `train.lr0` 낮추기 |
| `rfdetr 에 RFDETRSegMedium 이 없습니다` | `pip install -U "rfdetr[train,loggers]"` — 오류 메시지에 설치된 변형 목록이 나온다 |
| 세션이 끊김 | 같은 명령에 `train.resume=<체크포인트>` (4장 표) 추가 |
| `... 에 없는 설정 키` | 설정 키 오타 — 메시지에 가능한 키 목록이 나온다 |
