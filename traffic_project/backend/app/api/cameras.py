"""카메라 등록/조회/마스크/워커 제어."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import get_camera_or_404
from app.api.snapshots import snapshot_path
from app.config import get_settings
from app.db.models import AnalysisJob, Camera, Direction, OccupancySample
from app.db.session import get_db
from app.ml.render import DIRECTION_COLORS, encode_jpeg, hex_to_bgr, render_overlay
from app.schemas import CameraCreate, CameraOut, CameraUpdate, DirectionIn, JobOut, MaskUpdate
from app.services import its_client, media, video_jobs
from app.services.worker_manager import manager
from app.utils.images import b64_to_array

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _to_out(cam: Camera) -> CameraOut:
    out = CameraOut.model_validate(cam)
    w = manager.get(cam.id)
    out.running = manager.is_running(cam.id)
    out.live = w.latest_json() if w and w.is_alive() else None
    return out


@router.get("", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    cams = db.query(Camera).order_by(Camera.id).all()
    return [_to_out(c) for c in cams]


@router.post("", response_model=CameraOut, status_code=201)
def create_camera(body: CameraCreate, db: Session = Depends(get_db)):
    if body.source_type == "its" and not body.stream_url:
        raise HTTPException(400, "ITS 카메라는 stream_url 이 필요합니다")
    if body.source_type == "upload" and not body.video_path:
        raise HTTPException(400, "업로드 카메라는 video_path 가 필요합니다")
    if body.source_type == "url" and not body.stream_url:
        raise HTTPException(400, "stream_url 이 필요합니다")

    cam = Camera(
        name=body.name,
        source_type=body.source_type,
        its_cctv_name=body.its_cctv_name,
        its_road_type=body.its_road_type,
        its_cctv_type=body.its_cctv_type,
        stream_url=body.stream_url,
        stream_url_fetched_at=datetime.now(timezone.utc) if body.stream_url else None,
        video_path=body.video_path,
        lon=body.lon,
        lat=body.lat,
        route=body.route or (its_client.parse_route(body.its_cctv_name) if body.its_cctv_name else None),
        region=body.region,
        section=body.section,
        infer_interval_s=body.infer_interval_s if body.infer_interval_s is not None else get_settings().default_infer_interval_s,
        meta=body.meta or {},
        enabled=True,
    )
    db.add(cam)
    db.flush()

    # 스냅샷: 임시 스냅샷 복사 또는 소스에서 직접 캡처
    frame = None
    if body.snapshot_id:
        frame = media.load_image(snapshot_path(body.snapshot_id))
    else:
        src = body.video_path if body.source_type == "upload" else body.stream_url
        try:
            frame = media.grab_snapshot(src)
        except Exception as e:
            db.rollback()
            raise HTTPException(400, f"스냅샷을 가져오지 못했습니다: {e}")
    p = media.save_snapshot(cam.id, frame)
    cam.snapshot_path = str(p)
    cam.frame_height, cam.frame_width = frame.shape[:2]
    db.commit()
    db.refresh(cam)
    return _to_out(cam)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    return _to_out(get_camera_or_404(db, camera_id))


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, body: CameraUpdate, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    data = body.model_dump(exclude_unset=True)
    if "stream_url" in data and data["stream_url"]:
        cam.stream_url_fetched_at = datetime.now(timezone.utc)
    if "infer_interval_s" in data and data["infer_interval_s"] is None:
        data["infer_interval_s"] = get_settings().default_infer_interval_s
    interval_changed = "infer_interval_s" in data and data["infer_interval_s"] != cam.infer_interval_s
    for k, v in data.items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)
    if cam.enabled is False:
        manager.stop(cam.id)
    elif interval_changed and manager.is_running(cam.id):
        manager.restart(cam.id)
    return _to_out(cam)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    manager.stop(camera_id, join=True)
    db.delete(cam)
    db.commit()
    shutil.rmtree(get_settings().cameras_dir / str(camera_id), ignore_errors=True)
    return Response(status_code=204)


# ---------- 스냅샷 ----------
@router.post("/{camera_id}/snapshot", response_model=CameraOut)
def refresh_snapshot(camera_id: int, db: Session = Depends(get_db)):
    """소스에서 새 스냅샷을 캡처 (ITS 는 필요 시 URL 갱신)."""
    cam = get_camera_or_404(db, camera_id)
    src = cam.video_path if cam.source_type == "upload" else cam.stream_url
    try:
        frame = media.grab_snapshot(src)
    except Exception as e:
        if cam.source_type == "its" and cam.its_cctv_name:
            new_url = its_client.refresh_url(cam.its_cctv_name, cam.lon, cam.lat, cam.its_road_type or "ex", cam.its_cctv_type or "1")
            if not new_url:
                raise HTTPException(400, f"스냅샷 실패: {e}")
            cam.stream_url, cam.stream_url_fetched_at = new_url, datetime.now(timezone.utc)
            try:
                frame = media.grab_snapshot(new_url)
            except Exception as e2:
                raise HTTPException(400, f"스냅샷 실패(URL 갱신 후): {e2}")
        else:
            raise HTTPException(400, f"스냅샷 실패: {e}")
    cam.snapshot_path = str(media.save_snapshot(cam.id, frame))
    cam.frame_height, cam.frame_width = frame.shape[:2]
    db.commit()
    db.refresh(cam)
    return _to_out(cam)


@router.get("/{camera_id}/snapshot.jpg")
def get_snapshot(camera_id: int, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    if not cam.snapshot_path:
        raise HTTPException(404, "스냅샷 없음")
    return FileResponse(cam.snapshot_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/{camera_id}/preview.jpg")
def get_preview(camera_id: int, db: Session = Depends(get_db)):
    """스냅샷 + 도로 마스크 오버레이 (카드/지도 팝업용)."""
    cam = get_camera_or_404(db, camera_id)
    if not cam.snapshot_path:
        raise HTTPException(404, "스냅샷 없음")
    img = media.load_image(cam.snapshot_path)
    road = media.load_mask(cam.mask_path) if cam.mask_path else None
    colors = [hex_to_bgr(d.color, DIRECTION_COLORS[(d.index - 1) % len(DIRECTION_COLORS)]) for d in cam.directions] or None
    out = render_overlay(img, None, road, mode="road", direction_colors=colors)
    return Response(encode_jpeg(out, 80), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


# ---------- 마스크 ----------
@router.put("/{camera_id}/mask", response_model=CameraOut)
def put_mask(camera_id: int, body: MaskUpdate, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    arr = b64_to_array(body.mask_png_base64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    label = arr.astype(np.uint8)
    if cam.frame_width and cam.frame_height and label.shape != (cam.frame_height, cam.frame_width):
        label = cv2.resize(label, (cam.frame_width, cam.frame_height), interpolation=cv2.INTER_NEAREST)
    if not body.directions:
        raise HTTPException(400, "방향을 1개 이상 정의하세요")
    idx = sorted({d.index for d in body.directions})
    if idx != list(range(1, len(idx) + 1)):
        raise HTTPException(400, "방향 index 는 1부터 연속이어야 합니다")
    label[label > len(idx)] = 0
    if int((label > 0).sum()) == 0:
        raise HTTPException(400, "도로 영역이 비어 있습니다")
    cam.mask_path = str(media.save_mask(cam.id, label))
    cam.directions.clear()
    db.flush()
    for d in body.directions:
        cam.directions.append(Direction(**_dir_fields(d)))
    stats = {str(i): int((label == i).sum()) for i in idx}
    cam.meta = dict(cam.meta or {}, road_px=stats, road_coverage=float((label > 0).mean()))
    db.commit()
    db.refresh(cam)
    if manager.is_running(cam.id):
        manager.restart(cam.id)
    return _to_out(cam)


def _dir_fields(d: DirectionIn) -> dict:
    return dict(index=d.index, name=d.name, color=d.color, road=(d.road or None), heading_deg=d.heading_deg, lat=d.lat, lon=d.lon)


@router.put("/{camera_id}/directions", response_model=CameraOut)
def put_directions(camera_id: int, body: list[DirectionIn], db: Session = Depends(get_db)):
    """방향 메타데이터(이름·색·도로·진행 각도·지도 위치)만 수정. 마스크 라벨 번호는 그대로여야 한다."""
    cam = get_camera_or_404(db, camera_id)
    existing = {d.index: d for d in cam.directions}
    if sorted(d.index for d in body) != sorted(existing):
        raise HTTPException(400, "방향 번호가 마스크와 다릅니다. 방향 추가/삭제는 마스크 편집에서 하세요.")
    for d in body:
        row = existing[d.index]
        for k, v in _dir_fields(d).items():
            setattr(row, k, v)
    # 카메라 좌표가 없고 화살표 위치가 있으면 그 평균을 카메라 좌표로
    pts = [(d.lat, d.lon) for d in body if d.lat is not None and d.lon is not None]
    if (cam.lat is None or cam.lon is None) and pts:
        cam.lat = sum(p[0] for p in pts) / len(pts)
        cam.lon = sum(p[1] for p in pts) / len(pts)
    db.commit()
    db.refresh(cam)
    if manager.is_running(cam.id):
        manager.restart(cam.id)  # HUD/방향 이름 반영
    return _to_out(cam)


@router.get("/{camera_id}/mask.png")
def get_mask(camera_id: int, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    if not cam.mask_path:
        raise HTTPException(404, "마스크 없음")
    return FileResponse(cam.mask_path, media_type="image/png", headers={"Cache-Control": "no-cache"})


# ---------- 워커 ----------
@router.post("/{camera_id}/start", response_model=CameraOut)
def start_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    if not cam.mask_path:
        raise HTTPException(400, "도로 마스크를 먼저 등록하세요")
    cam.enabled = True
    db.commit()
    manager.restart(camera_id)
    db.refresh(cam)
    return _to_out(cam)


@router.post("/{camera_id}/stop", response_model=CameraOut)
def stop_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    manager.stop(camera_id)
    db.refresh(cam)
    return _to_out(cam)


# ---------- 업로드 영상 전체 분석 ----------
@router.post("/{camera_id}/analyze", response_model=JobOut, status_code=202)
def analyze_video(camera_id: int, frame_stride: int = Query(5, ge=1, le=120), db: Session = Depends(get_db)):
    cam = get_camera_or_404(db, camera_id)
    if cam.source_type != "upload" or not cam.video_path:
        raise HTTPException(400, "업로드 영상 카메라만 전체 분석이 가능합니다")
    if not cam.mask_path:
        raise HTTPException(400, "도로 마스크를 먼저 등록하세요")
    running = db.query(AnalysisJob).filter(AnalysisJob.camera_id == camera_id, AnalysisJob.status == "running").first()
    if running:
        return running
    job = AnalysisJob(camera_id=camera_id, frame_stride=frame_stride)
    db.add(job)
    db.commit()
    db.refresh(job)
    video_jobs.start_job(job.id)
    return job


@router.get("/{camera_id}/jobs", response_model=list[JobOut])
def list_jobs(camera_id: int, db: Session = Depends(get_db)):
    get_camera_or_404(db, camera_id)
    return db.query(AnalysisJob).filter(AnalysisJob.camera_id == camera_id).order_by(AnalysisJob.id.desc()).limit(10).all()


@router.delete("/{camera_id}/samples", status_code=204)
def clear_samples(camera_id: int, source: str | None = Query(None, pattern="^(live|file)$"), db: Session = Depends(get_db)):
    get_camera_or_404(db, camera_id)
    q = db.query(OccupancySample).filter(OccupancySample.camera_id == camera_id)
    if source:
        q = q.filter(OccupancySample.source == source)
    q.delete()
    db.commit()
    return Response(status_code=204)
