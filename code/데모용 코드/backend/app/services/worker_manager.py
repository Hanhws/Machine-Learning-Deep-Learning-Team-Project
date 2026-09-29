"""카메라 워커 생명주기 관리 (프로세스 내 싱글턴)."""

from __future__ import annotations

import logging
import threading

from app.db.models import Camera
from app.db.session import SessionLocal
from app.services.stream_worker import StreamWorker

log = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self):
        self._workers: dict[int, StreamWorker] = {}
        self._lock = threading.Lock()

    def start(self, camera_id: int) -> StreamWorker:
        with self._lock:
            w = self._workers.get(camera_id)
            if w and w.is_alive():
                return w
            w = StreamWorker(camera_id)
            self._workers[camera_id] = w
            w.start()
            log.info("워커 시작: camera %s", camera_id)
            return w

    def stop(self, camera_id: int, join: bool = False) -> None:
        with self._lock:
            w = self._workers.pop(camera_id, None)
        if w:
            w.stop()
            if join:
                w.join(timeout=5)
            log.info("워커 중지: camera %s", camera_id)

    def restart(self, camera_id: int) -> StreamWorker:
        self.stop(camera_id, join=True)
        return self.start(camera_id)

    def get(self, camera_id: int) -> StreamWorker | None:
        return self._workers.get(camera_id)

    def is_running(self, camera_id: int) -> bool:
        w = self._workers.get(camera_id)
        return bool(w and w.is_alive() and w.state.status != "stopped")

    def all_latest(self) -> dict[int, dict]:
        return {cid: w.latest_json() for cid, w in list(self._workers.items()) if w.is_alive()}

    def start_all_enabled(self) -> int:
        n = 0
        with SessionLocal() as db:
            cams = db.query(Camera).filter(Camera.enabled.is_(True), Camera.mask_path.isnot(None)).all()
            ids = [c.id for c in cams]
        for cid in ids:
            self.start(cid)
            n += 1
        return n

    def shutdown(self) -> None:
        for cid in list(self._workers):
            self.stop(cid)


manager = WorkerManager()
