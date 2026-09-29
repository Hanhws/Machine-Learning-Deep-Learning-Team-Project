"""카메라 1대 = 워커 스레드 1개.

영상(HLS/파일)을 읽어 INFER_FPS 주기로 YOLO 추론 → 방향별 점유율 계산 →
최신 상태를 메모리에 두고(SAMPLE_WRITE_INTERVAL 마다 평균을 DB 에 기록).
MJPEG 스트림은 최신 프레임에 오버레이를 그려 내보낸다.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from app.config import congestion_level, get_settings
from app.db.models import Camera, OccupancySample
from app.db.session import SessionLocal
from app.ml.occupancy import DirectionMetrics, compute_occupancy
from app.ml.registry import get_vehicle_segmenter
from app.ml.roi import Roi, infer_in_roi, roi_from_mask
from app.ml.render import DIRECTION_COLORS, RoadLayer, encode_jpeg, hex_to_bgr, render_overlay
from app.ml.vehicle_seg import VEHICLE_CLASSES
from app.services import its_client, media

log = logging.getLogger(__name__)

RECONNECT_BASE = 2.0  # 재접속 대기 시작값(초)
RECONNECT_MAX = 60.0
BLOCKED_WAIT = 300.0  # 서버가 접속을 거부(HTTP 4xx)할 때 쉬는 시간
REFRESH_MIN_INTERVAL = 60.0  # ITS URL 재조회 최소 간격


@dataclass
class LiveState:
    camera_id: int
    status: str = "starting"  # starting | running | reconnecting | error | stopped
    error: str | None = None
    ts: float = 0.0
    seq: int = 0
    frame: np.ndarray | None = None
    vehicle_label: np.ndarray | None = None
    metrics: dict[int, DirectionMetrics] = field(default_factory=dict)
    infer_ms: float = 0.0
    fps: float = 0.0
    video_time: float | None = None

    def metrics_json(self, thresholds: list[float], directions: dict[int, str]) -> dict:
        dirs = []
        for d, m in sorted(self.metrics.items()):
            j = m.to_dict()
            j["name"] = "전체" if d == 0 else directions.get(d, f"방향 {d}")
            j["level"] = congestion_level(m.occupancy, thresholds)
            dirs.append(j)
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "error": self.error,
            "ts": datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat() if self.ts else None,
            "seq": self.seq,
            "infer_ms": round(self.infer_ms, 1),
            "fps": round(self.fps, 2),
            "video_time": self.video_time,
            "directions": dirs,
        }


class StreamWorker(threading.Thread):
    def __init__(self, camera_id: int):
        super().__init__(name=f"cam-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.settings = get_settings()
        self.state = LiveState(camera_id=camera_id)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._render_cache: dict[str, tuple[int, bytes]] = {}
        # 카메라 설정 (DB 에서 로드)
        self.source: str = ""
        self.source_type: str = ""
        self.road_label: np.ndarray | None = None
        self.roi: Roi | None = None
        self._road_layer: RoadLayer | None = None  # 렌더용 도로 레이어 (마스크가 바뀔 때만 재생성)
        self.interval: float = 0.0  # 초. 0 = 실시간(INFER_FPS)
        self.n_directions: int = 0
        self.direction_names: dict[int, str] = {}
        self.direction_roads: dict[int, str | None] = {}
        self.direction_dests: dict[int, str | None] = {}
        self.direction_colors: list[tuple[int, int, int]] = list(DIRECTION_COLORS)
        self.name = ""
        self._cam_meta: dict = {}
        # DB 집계 버퍼
        self._acc: dict[int, dict[str, float]] = {}
        self._acc_n = 0
        self._last_flush = time.time()
        self._last_refresh = 0.0

    # ---------- 설정 ----------
    def load_config(self) -> None:
        with SessionLocal() as db:
            cam = db.get(Camera, self.camera_id)
            if cam is None:
                raise RuntimeError("카메라가 없습니다")
            self.name = cam.name
            self.source_type = cam.source_type
            self.source = cam.video_path if cam.source_type == "upload" else (cam.stream_url or "")
            self._cam_meta = {
                "its_cctv_name": cam.its_cctv_name,
                "its_road_type": cam.its_road_type,
                "its_cctv_type": cam.its_cctv_type,
                "lon": cam.lon,
                "lat": cam.lat,
                "fetched_at": cam.stream_url_fetched_at,
            }
            iv = cam.infer_interval_s
            self.interval = float(self.settings.default_infer_interval_s if iv is None else iv)  # 0 = 실시간
            if cam.mask_path:
                self.road_label = media.load_mask(cam.mask_path)
                self.roi = roi_from_mask(self.road_label, self.settings.roi_pad)
            self.direction_names = {d.index: d.name for d in cam.directions}
            self.direction_roads = {d.index: d.road for d in cam.directions}
            self.direction_dests = {d.index: d.destination for d in cam.directions}
            self.direction_colors = [
                hex_to_bgr(d.color, DIRECTION_COLORS[(d.index - 1) % len(DIRECTION_COLORS)]) for d in cam.directions
            ] or list(DIRECTION_COLORS)
            self.n_directions = max([d.index for d in cam.directions] + [0])
        if not self.source:
            raise RuntimeError("영상 소스가 없습니다")
        if self.road_label is None:
            raise RuntimeError("도로 마스크가 등록되지 않았습니다")

    def _maybe_refresh_its_url(self, force: bool = False) -> None:
        if self.source_type != "its":
            return
        fetched = self._cam_meta.get("fetched_at")
        age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600 if fetched else 1e9
        if not force and age_h < self.settings.its_url_ttl_hours:
            return
        # 열기에 실패할 때마다 ITS 를 다시 부르면 초당 몇 번씩 호출하게 된다.
        if time.time() - self._last_refresh < REFRESH_MIN_INTERVAL:
            return
        self._last_refresh = time.time()
        new_url = its_client.refresh_url(
            self._cam_meta["its_cctv_name"] or self.name,
            self._cam_meta["lon"],
            self._cam_meta["lat"],
            self._cam_meta["its_road_type"] or "ex",
            self._cam_meta["its_cctv_type"] or "1",
        )
        if new_url:
            self.source = new_url
            self._cam_meta["fetched_at"] = datetime.now(timezone.utc)
            with SessionLocal() as db:
                cam = db.get(Camera, self.camera_id)
                if cam:
                    cam.stream_url = new_url
                    cam.stream_url_fetched_at = self._cam_meta["fetched_at"]
                    db.commit()
            log.info("[%s] ITS URL 갱신", self.name)

    # ---------- 실행 ----------
    def stop(self) -> None:
        self._stop.set()

    def _wait_jitter(self, seconds: float) -> None:
        """카메라들이 같은 순간에 몰려서 재접속하지 않도록 대기 시간을 조금 흩뜨린다."""
        self._stop.wait(seconds * (0.75 + 0.5 * random.random()))

    def run(self) -> None:
        try:
            self.load_config()
        except Exception as e:
            self.state.status, self.state.error = "error", str(e)
            log.error("[cam %s] 시작 실패: %s", self.camera_id, e)
            return
        try:
            seg = get_vehicle_segmenter()
        except Exception as e:
            self.state.status, self.state.error = "error", f"차량 모델 로드 실패: {e}"
            log.error("[cam %s] %s", self.camera_id, self.state.error)
            return
        backoff = RECONNECT_BASE
        fails = 0
        while not self._stop.is_set():
            try:
                self._maybe_refresh_its_url(force=fails >= 2)
                cap = media.open_capture(self.source)
            except media.StreamBlocked as e:
                # 서버가 막은 상태라 빠른 재시도는 차단만 길어지게 한다.
                fails += 1
                self.state.status, self.state.error = "reconnecting", str(e)
                log.warning("[%s] %s — %.0f초 후 재시도", self.name, e, BLOCKED_WAIT)
                self._wait_jitter(BLOCKED_WAIT)
                backoff = RECONNECT_BASE
                continue
            except Exception as e:
                fails += 1
                self.state.status, self.state.error = "reconnecting", str(e)
                log.warning("[%s] 열기 실패(%d): %s", self.name, fails, e)
                self._wait_jitter(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)
                continue
            self.state.status, self.state.error = "running", None
            seq0 = self.state.seq
            blocked = False
            try:
                if self.interval >= 1.0:
                    self._interval_loop(cap, seg)
                else:
                    self._loop(cap, seg)
            except media.StreamBlocked as e:  # 주기 모드에서 다시 열 때 막힌 경우
                blocked = True
                log.warning("[%s] %s — %.0f초 후 재시도", self.name, e, BLOCKED_WAIT)
                self.state.status, self.state.error = "reconnecting", str(e)
            except Exception as e:
                log.warning("[%s] 루프 오류: %s", self.name, e)
                self.state.status, self.state.error = "reconnecting", str(e)
            finally:
                cap.release()
            if blocked:
                self._wait_jitter(BLOCKED_WAIT)
                backoff = RECONNECT_BASE
            elif self.state.seq > seq0:
                # 프레임을 실제로 받아 본 뒤에만 대기 시간을 되돌린다.
                # (열기만 성공하고 바로 끊기는 경우 2초 간격으로 계속 두드리게 된다)
                backoff, fails = RECONNECT_BASE, 0
            else:
                self._wait_jitter(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)
            if not self._stop.is_set():
                self._stop.wait(1.0)
        self._flush(force=True)
        self.state.status = "stopped"

    def _loop(self, cap: cv2.VideoCapture, seg) -> None:
        """프레임을 읽어 stride 프레임마다 추론한다.

        HLS 는 세그먼트(보통 2초) 단위로 프레임이 한꺼번에 도착하므로 시간 기준으로 거르면
        세그먼트당 1번밖에 추론하지 못한다. 그래서 프레임 개수 기준(src_fps / INFER_FPS)으로 거르고,
        과부하 방지용 최소 간격만 시간으로 둔다.
        """
        is_file = media.is_file_source(self.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (1.0 <= src_fps <= 120.0):
            src_fps = 15.0
        stride = max(1, int(round(src_fps / max(self.settings.infer_fps, 0.1))))
        min_interval = 0.5 / max(self.settings.infer_fps, 0.1)  # 설정 주기의 절반보다 빠르게는 안 돔
        last_infer = 0.0
        t_wall0 = time.time()
        frame_idx = 0
        consecutive_fail = 0
        self.state.status = "running"
        while not self._stop.is_set():
            if is_file:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 반복 재생
                    frame_idx = 0
                    t_wall0 = time.time()
                    continue
                frame_idx += 1
                # 실시간 속도 재현
                target = t_wall0 + frame_idx / src_fps
                delay = target - time.time()
                if delay > 0:
                    self._stop.wait(min(delay, 0.5))
                video_time = frame_idx / src_fps
                if frame_idx % stride != 0:
                    continue
            else:
                ok = cap.grab()
                if not ok:
                    consecutive_fail += 1
                    if consecutive_fail > 50:
                        raise RuntimeError("스트림 읽기 실패가 반복됩니다")
                    self._stop.wait(0.05)
                    continue
                consecutive_fail = 0
                frame_idx += 1
                video_time = None
                if frame_idx % stride != 0 or time.time() - last_infer < min_interval:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
            now = time.time()
            self._process(frame, seg, now, video_time)
            self.state.fps = 1.0 / max(now - last_infer, 1e-3) if last_infer else 0.0
            last_infer = now

    def _interval_loop(self, cap: cv2.VideoCapture, seg) -> None:
        """주기 모드: interval 초마다 프레임 1장만 추론한다 (GPU 오버헤드 최소화).

        - 파일: 캡처를 열어 둔 채 위치를 건너뛰며 1장씩 읽는다.
        - 실시간 스트림, 주기 < 60초: 스트림은 열어 둔 채 프레임을 계속 흘려보내고(grab, 디코딩만) 주기마다 1장만 추론.
          (ITS CDN 은 동시 세션·재접속에 제한이 있어 자주 여닫으면 URL 이 막힌다)
        - 실시간 스트림, 주기 ≥ 60초: 매 주기마다 새로 열어 최신 프레임 1장을 읽고 닫는다(네트워크·디코딩 상시 부담 없음).
        """
        is_file = media.is_file_source(self.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (1.0 <= src_fps <= 120.0):
            src_fps = 15.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0
        reopen = (not is_file) and self.interval >= 60.0
        pos = 0
        first = True
        last_infer = 0.0
        consecutive_fail = 0
        while not self._stop.is_set():
            t0 = time.time()
            frame = None
            video_time = None
            if is_file:
                if total > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos % total)
                ok, frame = cap.read()
                if not ok:
                    pos = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                video_time = (pos % max(total, 1)) / src_fps
                pos += int(self.interval * src_fps)
            elif reopen:
                if not first:
                    cap.release()
                    self._maybe_refresh_its_url()
                    # 열기 실패는 그대로 올려 보낸다 (StreamBlocked 여야 바깥에서 길게 쉰다)
                    cap = media.open_capture(self.source)
                for _ in range(4):  # 앞쪽 프레임은 버리고 안정된 프레임
                    ok, f = cap.read()
                    if ok:
                        frame = f
                if frame is None:
                    raise RuntimeError("프레임을 읽지 못했습니다")
            else:
                # 스트림을 흘려보내며(grab) 주기가 되면 1장 retrieve
                ok = cap.grab()
                if not ok:
                    consecutive_fail += 1
                    if consecutive_fail > 50:
                        raise RuntimeError("스트림 읽기 실패가 반복됩니다")
                    self._stop.wait(0.05)
                    continue
                consecutive_fail = 0
                if time.time() - last_infer < self.interval:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
            first = False
            if frame is not None:
                now = time.time()
                self._process(frame, seg, now, video_time, flush_now=True)
                self.state.fps = 1.0 / self.interval
                last_infer = now
            if is_file or reopen:
                # 다음 주기까지 대기 (처리 시간 제외)
                self._stop.wait(max(0.5, self.interval - (time.time() - t0)))

    def _process(self, frame: np.ndarray, seg, now: float, video_time: float | None, flush_now: bool = False) -> None:
        if self.road_label.shape != frame.shape[:2]:
            self.road_label = cv2.resize(self.road_label, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.roi = roi_from_mask(self.road_label, self.settings.roi_pad)
            self._road_layer = None
        # 도로 마스크 영역 안에서만 차량 세그멘테이션
        res = infer_in_roi(seg, frame, self.road_label, self.roi, crop=self.settings.roi_crop)
        metrics = compute_occupancy(res, self.road_label, self.n_directions, self.roi)
        with self._lock:
            self.state.frame = frame
            self.state.vehicle_label = res.label_map
            self.state.metrics = metrics
            self.state.ts = now
            self.state.seq += 1
            self.state.infer_ms = res.infer_ms
            self.state.video_time = video_time
        self._accumulate(metrics)
        if flush_now or now - self._last_flush >= self.settings.sample_write_interval:
            self._flush()

    # ---------- DB 집계 ----------
    def _accumulate(self, metrics: dict[int, DirectionMetrics]) -> None:
        for d, m in metrics.items():
            a = self._acc.setdefault(d, {})
            a["occupancy"] = a.get("occupancy", 0.0) + m.occupancy
            a["vehicle_px"] = a.get("vehicle_px", 0.0) + m.vehicle_px
            a["road_px"] = a.get("road_px", 0.0) + m.road_px
            for c in VEHICLE_CLASSES:
                a[f"n_{c}"] = a.get(f"n_{c}", 0.0) + m.counts.get(c, 0)
                a[f"{c}_px"] = a.get(f"{c}_px", 0.0) + m.class_px.get(c, 0)
        self._acc_n += 1

    def _flush(self, force: bool = False) -> None:
        if self._acc_n == 0:
            self._last_flush = time.time()
            return
        n = self._acc_n
        rows = []
        for d, a in self._acc.items():
            rows.append(
                OccupancySample(
                    camera_id=self.camera_id,
                    direction_index=d,
                    source="live",  # 워커는 항상 벽시계 시계열 (업로드 영상 반복 재생 포함)
                    video_time=self.state.video_time,
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
        try:
            with SessionLocal() as db:
                db.add_all(rows)
                db.commit()
        except Exception as e:
            log.warning("[%s] 샘플 저장 실패: %s", self.name, e)
        self._acc, self._acc_n, self._last_flush = {}, 0, time.time()

    # ---------- 출력 ----------
    def latest_json(self) -> dict:
        with self._lock:
            j = self.state.metrics_json(self.settings.congestion_thresholds, self.direction_names)
        for d in j["directions"]:
            d["road"] = self.direction_roads.get(d["direction_index"])
            d["destination"] = self.direction_dests.get(d["direction_index"])
        j["name"] = self.name
        j["interval_s"] = self.interval or None
        return j

    def render_jpeg(self, mode: str = "class", with_hud: bool = True) -> bytes | None:
        with self._lock:
            if self.state.frame is None:
                return None
            key = f"{mode}:{int(with_hud)}"
            cached = self._render_cache.get(key)
            if cached and cached[0] == self.state.seq:
                return cached[1]
            frame, vlabel, metrics, seq = self.state.frame, self.state.vehicle_label, self.state.metrics, self.state.seq
        hud = None
        if with_hud:
            hud = []
            for d, m in sorted(metrics.items()):
                nm = "ALL" if d == 0 else f"D{d}"
                lvl = congestion_level(m.occupancy, self.settings.congestion_thresholds) or ""
                hud.append(f"{nm} {m.occupancy * 100:5.1f}%  n={sum(m.counts.values())}  {_ascii_level(lvl)}")
        if self._road_layer is None or self._road_layer.shape != frame.shape[:2]:
            self._road_layer = RoadLayer(self.road_label, self.direction_colors)
        img = render_overlay(frame, vlabel, self.road_label, mode=mode, direction_colors=self.direction_colors, hud=hud, road_layer=self._road_layer)
        data = encode_jpeg(img, self.settings.stream_jpeg_quality)
        with self._lock:
            self._render_cache[key] = (seq, data)
        return data


def _ascii_level(level: str) -> str:
    return {"원활": "FREE", "서행": "SLOW", "지체": "DELAY", "정체": "JAM"}.get(level, "")
