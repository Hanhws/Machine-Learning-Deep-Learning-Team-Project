"""모델 싱글턴. 첫 사용 시 로드하며 워커/요청 스레드가 공유한다."""

from __future__ import annotations

import logging
import threading

from app.config import get_settings
from app.ml.road_seg import RoadSegmenter
from app.ml.vehicle_seg import VehicleSegmenter

log = logging.getLogger(__name__)
_lock = threading.Lock()
_vehicle: VehicleSegmenter | None = None
_road: RoadSegmenter | None = None


def get_vehicle_segmenter() -> VehicleSegmenter:
    global _vehicle
    if _vehicle is None:
        with _lock:
            if _vehicle is None:
                s = get_settings()
                weights = s.yolo_weights
                if not weights.exists():
                    # .env 의 파일이 없으면 models/weights 의 다른 YOLO 가중치를 찾아 쓴다 (sam* 제외)
                    cands = sorted(p for p in weights.parent.glob("*.pt") if not p.name.lower().startswith("sam"))
                    if cands:
                        log.warning("YOLO_WEIGHTS(%s) 가 없어 %s 를 대신 사용합니다. .env 를 갱신하세요.", weights.name, cands[0].name)
                        weights = cands[0]
                _vehicle = VehicleSegmenter(weights, device=s.device, imgsz=s.yolo_imgsz, conf=s.yolo_conf)
    return _vehicle


def get_road_segmenter() -> RoadSegmenter:
    global _road
    if _road is None:
        with _lock:
            if _road is None:
                s = get_settings()
                _road = RoadSegmenter(s.sam3_weights, str(s.sam_fallback_weights), device=s.device, conf=s.sam_conf)
    return _road


def model_status() -> dict:
    s = get_settings()
    return {
        "yolo_weights": str(_vehicle.weights) if _vehicle else str(s.yolo_weights),
        "yolo_weights_exists": s.yolo_weights.exists(),
        "yolo_loaded": _vehicle is not None,
        "yolo_device": _vehicle.device if _vehicle else None,
        "yolo_classes": list(_vehicle.names.values()) if _vehicle else None,
        "road": get_road_segmenter().status(),
    }
