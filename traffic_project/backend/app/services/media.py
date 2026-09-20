"""영상 소스 열기, 스냅샷, 마스크/스냅샷 파일 입출력."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)

# HLS/네트워크 소스 타임아웃 (마이크로초)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rw_timeout;15000000|timeout;15000000")


def is_file_source(source: str) -> bool:
    return not source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://"))


class StreamBlocked(RuntimeError):
    """서버가 접속을 거부했다(HTTP 4xx). 바로 다시 시도해도 소용없는 상태.

    ITS CCTV(cctvsec.ktict.co.kr)는 재접속이 잦은 IP 를 한동안 403 으로 막는다.
    이때 빠르게 재시도하면 차단이 더 길어지므로 호출 측에서 길게 쉬어야 한다.
    """

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _http_status(url: str, timeout: float = 5.0) -> int | None:
    """열기에 실패한 이유가 HTTP 에러인지 확인한다 (네트워크 문제면 None)."""
    import httpx

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            return client.get(url, headers={"Range": "bytes=0-0"}).status_code
    except Exception:
        return None


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        if source.lower().startswith(("http://", "https://")):
            code = _http_status(source)
            if code is not None and 400 <= code < 500:
                raise StreamBlocked(f"서버가 접속을 거부했습니다 (HTTP {code})", code)
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source[:80]}")
    return cap


def grab_snapshot(source: str, skip: int = 5, timeout_s: float = 20.0) -> np.ndarray:
    """소스에서 프레임 하나를 읽는다 (앞쪽 몇 프레임은 버려 안정된 프레임을 얻음)."""
    cap = open_capture(source)
    try:
        t0 = time.time()
        frame = None
        got = 0
        while time.time() - t0 < timeout_s and got <= skip:
            ok, f = cap.read()
            if not ok:
                if frame is not None:
                    break
                time.sleep(0.1)
                continue
            frame, got = f, got + 1
        if frame is None:
            raise RuntimeError("프레임을 읽지 못했습니다.")
        return frame
    finally:
        cap.release()


def video_info(path: str) -> dict:
    cap = open_capture(path)
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


# ---------- 카메라별 파일 ----------
def camera_dir(camera_id: int) -> Path:
    d = get_settings().cameras_dir / str(camera_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_snapshot(camera_id: int, frame: np.ndarray) -> Path:
    p = camera_dir(camera_id) / "snapshot.jpg"
    cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return p


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def save_mask(camera_id: int, label: np.ndarray) -> Path:
    """도로 라벨맵(uint8, 0=도로아님, 1..N=방향) 저장."""
    p = camera_dir(camera_id) / "road_mask.png"
    cv2.imwrite(str(p), label.astype(np.uint8))
    return p


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def decode_png_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("PNG 디코딩 실패")
    return img
