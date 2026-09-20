"""파인튜닝된 YOLO-seg 로 차량(car/bus/truck) 인스턴스 마스크를 뽑는다.

출력은 프레임 크기의 라벨맵(uint8): 0=배경, 1=car, 2=bus, 3=truck.
여러 워커 스레드가 하나의 모델을 공유하므로 추론은 락으로 직렬화한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml.device import resolve_device
from app.ml.gpu import run_gpu

log = logging.getLogger(__name__)

VEHICLE_CLASSES = ["car", "bus", "truck"]
CLASS_TO_LABEL = {name: i + 1 for i, name in enumerate(VEHICLE_CLASSES)}  # car=1 bus=2 truck=3
_BOX_PAD = 2  # 마스크가 박스를 살짝 넘칠 때를 대비한 여유 (px)


@dataclass
class VehicleInstance:
    cls: str
    conf: float
    area: int
    cx: float
    cy: float
    box: tuple[float, float, float, float]


@dataclass
class VehicleResult:
    label_map: np.ndarray  # (H, W) uint8
    instances: list[VehicleInstance] = field(default_factory=list)
    infer_ms: float = 0.0

    @property
    def vehicle_mask(self) -> np.ndarray:
        return self.label_map > 0


class VehicleSegmenter:
    def __init__(self, weights: Path, device: str = "auto", imgsz: int = 960, conf: float = 0.25):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {weights}")
        self.weights = Path(weights)
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(str(weights))
        self.names = {int(k): v for k, v in self.model.names.items()}
        log.info("YOLO 로드 완료: %s (device=%s, classes=%s)", weights, self.device, self.names)
        # 워밍업
        try:
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            self.infer(dummy)  # run_gpu 를 통해 GPU 스레드에서 워밍업
        except Exception as e:  # pragma: no cover
            log.warning("YOLO 워밍업 실패(무시): %s", e)

    def infer(self, frame_bgr: np.ndarray) -> VehicleResult:
        import time

        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        def _run():
            results = self.model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                retina_masks=True,
                verbose=False,
            )
            r = results[0]
            if r.masks is None or len(r.masks) == 0:
                return None
            # 텐서 → numpy 변환까지 GPU 스레드 안에서 끝낸다
            return (
                r.masks.data.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy(),
            )

        out = run_gpu(_run, device=self.device)
        label_map = np.zeros((h, w), dtype=np.uint8)
        instances: list[VehicleInstance] = []
        if out is not None:
            masks, clss, confs, boxes = out
            if masks.shape[1:] != (h, w):
                import cv2

                masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks])
            # 마스크는 자기 박스 안에만 존재한다(ultralytics 가 박스로 자른다).
            # 그래서 인스턴스마다 프레임 전체(H×W)를 훑지 않고 박스 영역만 다룬다.
            crops = []  # (area, x0, y0, sub_mask, label, conf, box)
            for i in range(len(masks)):
                bx = boxes[i]
                x0, y0 = max(0, int(bx[0]) - _BOX_PAD), max(0, int(bx[1]) - _BOX_PAD)
                x1, y1 = min(w, int(bx[2]) + 1 + _BOX_PAD), min(h, int(bx[3]) + 1 + _BOX_PAD)
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = masks[i][y0:y1, x0:x1] > 0.5
                area = int(sub.sum())
                if area == 0:
                    continue
                name = self.names.get(int(clss[i]), "car")
                crops.append((area, x0, y0, sub, name, float(confs[i]), tuple(float(v) for v in bx)))
            # 큰 객체부터 칠하고 작은 객체를 위에 덮어 겹침 시 작은 객체 우선
            for area, x0, y0, sub, name, conf, box in sorted(crops, key=lambda c: -c[0]):
                view = label_map[y0 : y0 + sub.shape[0], x0 : x0 + sub.shape[1]]
                view[sub] = CLASS_TO_LABEL.get(name, 1)
                ys, xs = np.nonzero(sub)
                instances.append(
                    VehicleInstance(
                        cls=name,
                        conf=conf,
                        area=area,
                        cx=float(xs.mean()) + x0,
                        cy=float(ys.mean()) + y0,
                        box=box,
                    )
                )
        return VehicleResult(label_map=label_map, instances=instances, infer_ms=(time.perf_counter() - t0) * 1000)
