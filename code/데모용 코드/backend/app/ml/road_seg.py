"""도로 영역 segmentation (등록 시 1회).

우선순위:
  1. SAM3 (models/weights/sam3.pt) — 텍스트 프롬프트("road") + 점/박스 프롬프트
  2. SAM2.1 (자동 다운로드) — 점/박스 프롬프트만 (텍스트 불가)

반환은 항상 프레임 크기의 bool 마스크. 사람이 편집기에서 다듬는다는 전제이므로
정밀도보다 "일단 그럴듯한 초기 마스크" 를 빠르게 주는 것이 목적이다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.ml.device import resolve_device
from app.ml.gpu import run_gpu

log = logging.getLogger(__name__)


class RoadSegmenter:
    def __init__(self, sam3_weights: Path, fallback_weights: str = "sam2.1_b.pt", device: str = "auto", conf: float = 0.3):
        self.device = resolve_device(device)
        self.conf = conf
        self.sam3_weights = Path(sam3_weights)
        self.fallback_weights = fallback_weights
        self.has_sam3 = self.sam3_weights.exists()
        self._interactive = None  # SAM (sam3 or sam2.1) 점/박스용
        self._semantic = None  # SAM3SemanticPredictor 텍스트용
        log.info("RoadSegmenter: sam3=%s device=%s", self.has_sam3, self.device)

    # ---------- 상태 ----------
    @property
    def backend_name(self) -> str:
        return "sam3" if self.has_sam3 else "sam2.1"

    def status(self) -> dict:
        return {
            "backend": self.backend_name,
            "text_prompt": self.has_sam3,
            "point_prompt": True,
            "box_prompt": True,
            "sam3_weights": str(self.sam3_weights),
            "device": self.device,
        }

    # ---------- 로딩 ----------
    def _load_interactive(self):
        if self._interactive is None:
            from ultralytics import SAM

            w = str(self.sam3_weights) if self.has_sam3 else str(self.fallback_weights)
            log.info("SAM 인터랙티브 모델 로드: %s", w)
            self._interactive = SAM(w)
        return self._interactive

    def _load_semantic(self):
        if not self.has_sam3:
            raise RuntimeError("텍스트 프롬프트는 SAM3 가중치(models/weights/sam3.pt)가 있어야 합니다.")
        if self._semantic is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            log.info("SAM3 semantic predictor 로드: %s", self.sam3_weights)
            self._semantic = SAM3SemanticPredictor(
                overrides=dict(
                    model=str(self.sam3_weights),
                    conf=self.conf,
                    imgsz=1008,
                    device=self.device,
                    save=False,
                    verbose=False,
                )
            )
        return self._semantic

    # ---------- 추론 ----------
    def _run(self, fn, *args, **kwargs):
        """MPS 에서 실패하면 CPU 로 한 번 더 시도."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if self.device != "cpu":
                log.warning("SAM %s 실패 → cpu 재시도: %s", self.device, e)
                self.device = "cpu"
                self._interactive = None
                self._semantic = None
                return fn(*args, **kwargs)
            raise

    def segment_text(self, image_bgr: np.ndarray, text: str = "road", conf: float | None = None) -> np.ndarray:
        """텍스트 프롬프트 → 해당 개념의 모든 인스턴스 합집합 마스크."""
        h, w = image_bgr.shape[:2]

        def _go():
            pred = self._load_semantic()
            pred.args.conf = conf if conf is not None else self.conf
            results = pred(source=image_bgr, text=[t.strip() for t in text.split(",") if t.strip()])
            return _union(results, h, w)  # 텐서 변환까지 GPU 스레드에서

        return self._run(lambda: run_gpu(_go, device=self.device))

    def segment_concepts(self, image_bgr: np.ndarray, concepts: list[str], conf: float | None = None) -> dict[str, np.ndarray]:
        """SAM3 한 번 호출로 여러 개념(예: road, car, truck, bus)을 찾아 개념별 합집합 마스크로 돌려준다."""
        h, w = image_bgr.shape[:2]
        concepts = [c.strip() for c in concepts if c.strip()]

        def _go():
            pred = self._load_semantic()
            pred.args.conf = conf if conf is not None else self.conf
            results = pred(source=image_bgr, text=concepts)
            return _by_class(results, h, w, len(concepts))

        per = self._run(lambda: run_gpu(_go, device=self.device))
        return {c: per[i] for i, c in enumerate(concepts)}

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        boxes: list[list[float]] | None = None,
    ) -> np.ndarray:
        """점(+/−)·박스 프롬프트 → 단일 객체 마스크(여러 프롬프트면 합집합)."""
        h, w = image_bgr.shape[:2]
        if not points and not boxes:
            return np.zeros((h, w), dtype=bool)

        def _go():
            sam = self._load_interactive()
            kwargs = dict(device=self.device, verbose=False, save=False, imgsz=1024)
            masks_out = np.zeros((h, w), dtype=bool)
            if points:
                # 모든 점을 하나의 객체 프롬프트로 묶어 전달 (positive/negative 혼합)
                res = sam.predict(image_bgr, points=[points], labels=[labels or [1] * len(points)], **kwargs)
                masks_out |= _union(res, h, w)
            if boxes:
                res = sam.predict(image_bgr, bboxes=boxes, **kwargs)
                masks_out |= _union(res, h, w)
            return masks_out

        return self._run(lambda: run_gpu(_go, device=self.device))


def _union(results, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        return out
    m = r.masks.data.cpu().numpy() > 0.5
    if m.shape[1:] != (h, w):
        import cv2

        m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for x in m])
    return m.any(0)


def _by_class(results, h: int, w: int, n: int) -> list[np.ndarray]:
    """SAM3 결과를 텍스트 프롬프트(클래스) 별 합집합 마스크로."""
    out = [np.zeros((h, w), dtype=bool) for _ in range(n)]
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        return out
    m = r.masks.data.cpu().numpy() > 0.5
    if m.shape[1:] != (h, w):
        import cv2

        m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for x in m])
    cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(len(m), int)
    for mi, ci in zip(m, cls):
        if 0 <= ci < n:
            out[ci] |= mi
    return out
