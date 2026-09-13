"""실험 설정 — YAML 파일 하나가 실험 하나.

    cfg = ExperimentConfig.load("configs/yolo11s_seg_1280.yaml",
                                overrides=["train.epochs=3", "data.root=/content/car_seg_split_upload"])

설정 파일에 적힌 값이 그대로 실험 기록표의 한 행이 된다(`carseg.experiment_log`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """car_seg_data_split.py 가 만든 분할 폴더."""

    root: str = "car_seg_split_upload"  # images/ labels/ annotations/ masks/ 가 들어 있는 폴더
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # 빠른 점검용 — 학습/평가에 쓸 이미지 수 제한 (0 이면 전체)
    train_limit: int = 0
    eval_limit: int = 0


@dataclass
class ModelConfig:
    """모델 계열과 사전학습 가중치."""

    family: str = "ultralytics"          # ultralytics | maskrcnn | mask2former | rfdetr
    weights: str = "yolo11s-seg.pt"      # 사전학습 체크포인트(파일 또는 허브 ID)
    backbone: str = ""                   # 기록용 표기 (CSPDarknet, ResNet-50, Swin-S ...)
    arch_mods: str = "X"                 # 기록용 — 아키텍처 수정 여부 (P2-Head 등)
    extra: Dict[str, Any] = field(default_factory=dict)  # 계열별 추가 인자


@dataclass
class TrainConfig:
    """학습 하이퍼파라미터. 계열마다 쓰는 항목이 조금씩 다르다(README 표 참고)."""

    epochs: int = 50
    batch: int = 8
    imgsz: int = 1280
    lr0: float = 1e-4
    lrf: float = 0.01                 # 최종 LR 배율 (cosine/linear 스케줄)
    optimizer: str = "auto"           # auto | SGD | Adam | AdamW
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    workers: int = 4
    device: str = "auto"
    amp: bool = True
    seed: int = 42
    patience: int = 20                # 조기 종료 (0 이면 사용 안 함)
    freeze: int = 0                   # 동결할 백본 단계 수 (계열별 해석)
    resume: str = ""                  # 이어서 학습할 체크포인트
    accumulate: int = 1               # 그래디언트 누적 (작은 GPU 에서 배치 키우기)
    # 증강 — ultralytics 는 그대로 전달, 그 외 계열은 hflip/색상만 사용
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: float = 0.0
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    # ultralytics 손실 가중치
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5


@dataclass
class EvalConfig:
    """추론·채점 설정. mAP 를 볼 때는 conf 를 아주 낮게 둔다(COCO 관례)."""

    conf: float = 0.01          # 점수 임계값
    iou: float = 0.7            # NMS IoU (계열이 지원할 때)
    mask_threshold: float = 0.5  # 마스크 확률 → 이진화
    max_det: int = 100          # COCOeval maxDets 와 같게 (EDA: 이미지당 최대 61대)
    union_conf: float = 0.5     # 점유율(합집합 픽셀) 지표·추론에 쓸 점수 임계값
    tta: bool = False           # test-time augmentation (ultralytics 만 지원)
    postprocess: str = "X"      # 기록용 — WBF 등 후처리 표기
    eval_every: int = 0         # 학습 중 N epoch 마다 val 채점 (0 이면 마지막에만) — Mask R-CNN·Mask2Former
    train_eval_limit: int = 500  # 학습 중 채점에 쓸 val 이미지 수 (속도용, 최종 채점은 전체)


@dataclass
class ExperimentConfig:
    name: str = "exp"
    group: str = "baseline"     # 기록표의 Group 열
    notes: str = ""
    out_dir: str = "runs"       # 실험 산출물 루트 (Colab 에서는 드라이브 경로 권장)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---------------------------------------------------------------- 불러오기
    @classmethod
    def load(cls, path: str | Path | None = None,
             overrides: Optional[List[str]] = None) -> "ExperimentConfig":
        raw: Dict[str, Any] = {}
        if path:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        cfg = _from_dict(cls, raw)
        if path:
            cfg._source = str(Path(path).resolve())  # type: ignore[attr-defined]
        return cfg

    # ---------------------------------------------------------------- 경로
    @property
    def run_dir(self) -> Path:
        """이 실험의 산출물 폴더 (가중치·지표·예측 결과)."""
        return Path(self.out_dir) / self.name

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.run_dir / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        return path

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def flat(self) -> Dict[str, Any]:
        """기록표(CSV) 한 행으로 쓰기 위한 평탄화: {'train.epochs': 50, ...}"""
        out: Dict[str, Any] = {}
        for section, value in self.to_dict().items():
            if not isinstance(value, dict):
                out[section] = value
                continue
            for key, item in value.items():
                out[f"{section}.{key}"] = _compact(item) if isinstance(item, dict) else item
        return out


def _compact(d: Dict[str, Any]) -> str:
    """extra 처럼 중첩된 dict 는 한 칸에 요약해 넣는다."""
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else ""


# -------------------------------------------------------------------- 내부 헬퍼
def _from_dict(cls, data: Dict[str, Any]):
    """데이터클래스 정의에 없는 키는 오타로 보고 즉시 알려 준다."""
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: 매핑이 필요한데 {type(data).__name__} 을 받았습니다")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__} 에 없는 설정 키: {sorted(unknown)} "
                       f"(가능한 키: {sorted(known)})")
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _from_dict(f.type, value)
        elif isinstance(f.type, str) and f.type in _NESTED and isinstance(value, dict):
            kwargs[name] = _from_dict(_NESTED[f.type], value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_override(raw: Dict[str, Any], item: str) -> None:
    """--set train.epochs=3 형태의 덮어쓰기."""
    if "=" not in item:
        raise ValueError(f"덮어쓰기는 key=value 형태여야 합니다: {item!r}")
    key, value = item.split("=", 1)
    node = raw
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"{key}: 중간 경로 {p} 가 매핑이 아닙니다")
    node[parts[-1]] = yaml.safe_load(value)


_NESTED = {"DataConfig": DataConfig, "ModelConfig": ModelConfig,
           "TrainConfig": TrainConfig, "EvalConfig": EvalConfig}
