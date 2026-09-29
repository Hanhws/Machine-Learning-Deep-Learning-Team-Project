"""업로드 영상 전체 분석 (오프라인 배치).

프레임을 stride 간격으로 추론하고 영상 내 1초 단위로 평균을 내 occupancy_samples 에
source='file', video_time 을 채워 저장한다. 진행률은 analysis_jobs.progress 로 노출.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

import cv2
import numpy as np

from app.db.models import AnalysisJob, Camera, OccupancySample
from app.db.session import SessionLocal
from app.ml.occupancy import compute_occupancy
from app.ml.registry import get_vehicle_segmenter
from app.ml.roi import infer_in_roi, roi_from_mask
from app.ml.vehicle_seg import VEHICLE_CLASSES
from app.services import media

log = logging.getLogger(__name__)
_running: set[int] = set()


def start_job(job_id: int) -> None:
    if job_id in _running:
        return
    _running.add(job_id)
    threading.Thread(target=_run, args=(job_id,), name=f"job-{job_id}", daemon=True).start()


def _run(job_id: int) -> None:
    try:
        with SessionLocal() as db:
            job = db.get(AnalysisJob, job_id)
            cam = db.get(Camera, job.camera_id)
            if cam is None or not cam.video_path or not cam.mask_path:
                raise RuntimeError("영상 또는 도로 마스크가 없습니다")
            video_path, mask_path, camera_id, stride = cam.video_path, cam.mask_path, cam.id, job.frame_stride
            n_dir = max([d.index for d in cam.directions] + [0])
            job.status, job.started_at, job.progress = "running", datetime.now(timezone.utc), 0.0
            # 이전 파일 분석 결과 삭제 (재분석)
            db.query(OccupancySample).filter(
                OccupancySample.camera_id == camera_id, OccupancySample.source == "file"
            ).delete()
            db.commit()

        road = media.load_mask(mask_path)
        seg = get_vehicle_segmenter()
        roi = None
        cap = media.open_capture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        bucket: dict[int, dict[int, dict]] = {}  # sec -> dir -> acc
        n_in_bucket: dict[int, int] = {}
        per_dir_series: dict[int, list[float]] = {}
        last_progress_write = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    if road.shape != frame.shape[:2]:
                        road = cv2.resize(road, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        roi = None
                    if roi is None:
                        roi = roi_from_mask(road)
                    res = infer_in_roi(seg, frame, road, roi)
                    metrics = compute_occupancy(res, road, n_dir, roi)
                    sec = int(idx / fps)
                    b = bucket.setdefault(sec, {})
                    n_in_bucket[sec] = n_in_bucket.get(sec, 0) + 1
                    for d, m in metrics.items():
                        a = b.setdefault(d, {})
                        a["occupancy"] = a.get("occupancy", 0) + m.occupancy
                        a["vehicle_px"] = a.get("vehicle_px", 0) + m.vehicle_px
                        a["road_px"] = a.get("road_px", 0) + m.road_px
                        for c in VEHICLE_CLASSES:
                            a[f"n_{c}"] = a.get(f"n_{c}", 0) + m.counts.get(c, 0)
                            a[f"{c}_px"] = a.get(f"{c}_px", 0) + m.class_px.get(c, 0)
                        per_dir_series.setdefault(d, []).append(m.occupancy)
            idx += 1
            if total and idx - last_progress_write >= max(total // 50, 1):
                last_progress_write = idx
                with SessionLocal() as db:
                    j = db.get(AnalysisJob, job_id)
                    j.progress = min(idx / total, 0.99)
                    db.commit()
        cap.release()

        rows = []
        for sec, dirs in sorted(bucket.items()):
            n = n_in_bucket[sec]
            for d, a in dirs.items():
                rows.append(
                    OccupancySample(
                        camera_id=camera_id,
                        direction_index=d,
                        source="file",
                        video_time=float(sec),
                        occupancy=a["occupancy"] / n,
                        vehicle_px=a["vehicle_px"] / n,
                        road_px=a["road_px"] / n,
                        n_car=a["n_car"] / n,
                        n_bus=a["n_bus"] / n,
                        n_truck=a["n_truck"] / n,
                        car_px=a["car_px"] / n,
                        bus_px=a["bus_px"] / n,
                        truck_px=a["truck_px"] / n,
                        n_frames=n,
                    )
                )
        summary = {
            "duration_s": round(idx / fps, 1),
            "frames": idx,
            "processed": sum(n_in_bucket.values()),
            "directions": {
                str(d): {
                    "mean": float(np.mean(v)) if v else 0.0,
                    "max": float(np.max(v)) if v else 0.0,
                    "p90": float(np.percentile(v, 90)) if v else 0.0,
                }
                for d, v in per_dir_series.items()
            },
        }
        with SessionLocal() as db:
            db.add_all(rows)
            j = db.get(AnalysisJob, job_id)
            j.status, j.progress, j.finished_at, j.summary = "done", 1.0, datetime.now(timezone.utc), summary
            db.commit()
        log.info("분석 완료 job=%s samples=%d", job_id, len(rows))
    except Exception as e:
        log.exception("분석 실패 job=%s", job_id)
        with SessionLocal() as db:
            j = db.get(AnalysisJob, job_id)
            if j:
                j.status, j.error, j.finished_at = "error", str(e), datetime.now(timezone.utc)
                db.commit()
    finally:
        _running.discard(job_id)
