"""데이터 접근 — car_seg_split 폴더 하나를 모든 모델 계열이 공유한다.

    car_seg_split_upload/
    ├── images/{train,val,test}/*.jpg
    ├── labels/{train,val,test}/*.txt              ← YOLO  (0=car 1=bus 2=truck)
    ├── annotations/instances_{train,val,test}.json ← COCO (1=car 2=bus 3=truck)
    ├── masks/{train,val,test}/*.png                ← 시맨틱 마스크 (0=배경 1/2/3)
    └── data.yaml

COCO 주석(annotations/)이 이 저장소의 '정답 원본'이다. 평가는 모든 계열이 COCO 로 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import CLASS_NAMES

# COCO 카테고리 id(1-based) ↔ 학습용 클래스 번호(0-based)
CAT_IDS: Tuple[int, ...] = tuple(i + 1 for i in range(len(CLASS_NAMES)))
CATID_TO_CLASS = {cid: i for i, cid in enumerate(CAT_IDS)}
CLASS_TO_CATID = {i: cid for cid, i in CATID_TO_CLASS.items()}


class CarSegData:
    """분할 폴더의 경로·주석을 다루는 얇은 래퍼."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"데이터 폴더가 없습니다: {self.root}")
        if not (self.root / "images").exists():
            raise FileNotFoundError(
                f"{self.root} 안에 images/ 가 없습니다. car_seg_data_split.py 결과 폴더를 지정하세요.")

    # ------------------------------------------------------------------ 경로
    def images_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def labels_dir(self, split: str) -> Path:
        return self.root / "labels" / split

    def masks_dir(self, split: str) -> Path:
        return self.root / "masks" / split

    def ann_file(self, split: str) -> Path:
        path = self.root / "annotations" / f"instances_{split}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"COCO 주석이 없습니다: {path}\n"
                "car_seg_data_preprocessing.py 를 --formats yolo,coco,mask 로 실행했는지 확인하세요.")
        return path

    def image_paths(self, split: str, limit: int = 0) -> List[Path]:
        paths = sorted(p for p in self.images_dir(split).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                       and not p.name.startswith("._"))  # macOS tar 가 끼워 넣는 AppleDouble 파일 제외
        return paths[:limit] if limit else paths

    # ------------------------------------------------------------------ 주석
    def coco(self, split: str):
        """pycocotools COCO 객체 (평가용 정답)."""
        from pycocotools.coco import COCO

        return COCO(str(self.ann_file(split)))

    def class_names(self) -> List[str]:
        """classes.json 이 있으면 그걸, 없으면 패키지 기본값을 쓴다."""
        path = self.root / "classes.json"
        if path.exists():
            table = json.loads(path.read_text(encoding="utf-8"))
            yolo = table.get("yolo")  # {"0": "car", "1": "bus", "2": "truck"}
            if yolo:
                return [yolo[k] for k in sorted(yolo, key=int)]
        return list(CLASS_NAMES)

    # ------------------------------------------------------------------ YOLO
    def write_data_yaml(self, dest: str | Path | None = None) -> Path:
        """현재 위치를 가리키는 data.yaml 을 새로 쓴다.

        저장소에 딸려 온 data.yaml 은 만들었던 PC 의 절대 경로를 담고 있어서
        Colab 에 풀면 그대로는 못 쓴다. 학습 직전에 항상 다시 쓴다.
        """
        dest = Path(dest) if dest else self.root / "data.yaml"
        names = self.class_names()
        lines = [
            "# carseg.data.write_data_yaml() 이 실행 환경에 맞춰 생성",
            f"path: {json.dumps(str(self.root), ensure_ascii=False)}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(names)],
            "",
        ]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
        return dest

    # ------------------------------------------------------------------ 요약
    def summary(self) -> Dict[str, Any]:
        report = self.root / "split_report.json"
        if report.exists():
            return json.loads(report.read_text(encoding="utf-8"))
        return {s: {"images": len(self.image_paths(s))} for s in ("train", "val", "test")}


# ====================================================================== 토치 데이터셋
@dataclass
class Sample:
    """한 장의 이미지와 그 인스턴스 정답."""

    image: np.ndarray          # (H, W, 3) uint8 RGB
    masks: np.ndarray          # (N, H, W) uint8 {0,1}
    boxes: np.ndarray          # (N, 4) float32 xyxy
    labels: np.ndarray         # (N,) int64 — 0-based (0=car 1=bus 2=truck)
    image_id: int
    file_name: str


class CocoInstanceDataset:
    """COCO 주석을 읽어 인스턴스 마스크를 돌려주는 데이터셋.

    Mask R-CNN · Mask2Former 가 함께 쓴다. 계열별 텐서 변환은 각 모델 모듈이 맡는다.
    """

    def __init__(self, data: CarSegData, split: str, *, limit: int = 0,
                 augment: bool = False, hflip: float = 0.0, vflip: float = 0.0,
                 color_jitter: float = 0.0, seed: int = 0):
        from pycocotools.coco import COCO

        self.data = data
        self.split = split
        self.images_dir = data.images_dir(split)
        self.coco = COCO(str(data.ann_file(split)))
        ids = sorted(self.coco.imgs)
        self.ids = ids[:limit] if limit else ids
        self.augment = augment
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        import cv2

        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        path = self.images_dir / info["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=False))
        masks, boxes, labels = [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0) or not ann.get("segmentation"):
                continue
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue
            x, y, bw, bh = ann["bbox"]
            if bw <= 1 or bh <= 1:          # 1px 이하 상자는 학습을 망친다
                continue
            masks.append(mask)
            boxes.append([x, y, x + bw, y + bh])
            labels.append(CATID_TO_CLASS[ann["category_id"]])

        sample = Sample(
            image=image,
            masks=np.stack(masks).astype(np.uint8) if masks else np.zeros((0, h, w), np.uint8),
            boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
            labels=np.asarray(labels, np.int64),
            image_id=int(image_id),
            file_name=info["file_name"],
        )
        return self._augment(sample) if self.augment else sample

    # ------------------------------------------------------------------ 증강
    def _augment(self, s: Sample) -> Sample:
        h, w = s.image.shape[:2]
        if self.hflip and self.rng.random() < self.hflip:
            s.image = np.ascontiguousarray(s.image[:, ::-1])
            s.masks = np.ascontiguousarray(s.masks[:, :, ::-1])
            if len(s.boxes):
                s.boxes = np.stack([w - s.boxes[:, 2], s.boxes[:, 1],
                                    w - s.boxes[:, 0], s.boxes[:, 3]], axis=1)
        if self.vflip and self.rng.random() < self.vflip:
            s.image = np.ascontiguousarray(s.image[::-1])
            s.masks = np.ascontiguousarray(s.masks[:, ::-1])
            if len(s.boxes):
                s.boxes = np.stack([s.boxes[:, 0], h - s.boxes[:, 3],
                                    s.boxes[:, 2], h - s.boxes[:, 1]], axis=1)
        if self.color_jitter:
            k = float(self.color_jitter)
            gain = 1.0 + self.rng.uniform(-k, k, size=3).astype(np.float32)
            s.image = np.clip(s.image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return s


def letterbox_free_resize(image: np.ndarray, masks: np.ndarray,
                          size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """이미지·마스크를 같은 (H, W) 로 리사이즈. 마스크는 최근접(라벨 보존)."""
    import cv2

    th, tw = int(size[0]), int(size[1])
    out_img = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
    if len(masks) == 0:
        return out_img, np.zeros((0, th, tw), np.uint8)
    out_masks = np.stack([cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
                          for m in masks]).astype(np.uint8)
    return out_img, out_masks


def load_road_mask(path: str | Path, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """도로 세그멘테이션 마스크 PNG 를 불러온다 (0=배경, 그 외=도로)."""
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"도로 마스크를 열 수 없습니다: {path}")
    if shape is not None and mask.shape[:2] != tuple(shape):
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0
