"""애플리케이션 설정. `.env` 파일(프로젝트 루트) 또는 환경변수에서 읽는다."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Traffic Occupancy Monitor"

    # ---- 외부 연동 ----
    its_api_key: str = ""
    its_base_url: str = "https://openapi.its.go.kr:9443/cctvInfo"
    its_verify_ssl: bool = False  # ITS 서버 인증서 체인이 불완전한 경우가 있어 기본 비활성

    # ---- DB ----
    database_url: str = "postgresql+psycopg://localhost:5433/traffic"

    # ---- 모델 ----
    yolo_weights: Path = PROJECT_ROOT / "models/weights/yolov8s_seg_vehicle.pt"
    sam3_weights: Path = PROJECT_ROOT / "models/weights/sam3.pt"
    sam_fallback_weights: Path = PROJECT_ROOT / "models/weights/sam2.1_b.pt"  # 없으면 ultralytics 가 이 경로로 자동 다운로드
    device: str = "auto"
    yolo_imgsz: int = 960
    yolo_conf: float = 0.25
    sam_conf: float = 0.3

    # ---- 실시간 처리 ----
    infer_fps: float = 2.0  # '실시간' 모드의 카메라당 초당 추론 횟수
    default_infer_interval_s: float = 5.0  # 새 카메라의 기본 추론 주기(초). 0 = 실시간
    roi_crop: bool = True  # 도로 마스크 바운딩 박스만 잘라 추론
    roi_pad: int = 32
    sample_write_interval: float = 5.0
    stream_jpeg_quality: int = 80
    its_url_ttl_hours: float = 23.0  # ITS URL 은 24h 유효 → 그 전에 갱신

    # ---- 혼잡 단계 ----
    congestion_thresholds: Annotated[list[float], NoDecode] = [0.08, 0.15, 0.25]

    # ---- 경로 ----
    data_dir: Path = PROJECT_ROOT / "data"
    frontend_dist: Path = PROJECT_ROOT / "frontend/dist"

    # ---- 서버 ----
    host: str = "0.0.0.0"
    port: int = 8000
    # 프론트를 다른 도메인(Vercel 등)에 올리면 그 주소를 추가: CORS_ORIGINS=http://localhost:5173,https://xxx.vercel.app
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_origin_regex: str | None = r"https://.*\.vercel\.app"  # 프리뷰 배포 도메인까지 허용

    @field_validator("congestion_thresholds", mode="before")
    @classmethod
    def _parse_thresholds(cls, v):
        if isinstance(v, str):
            return [float(x) for x in v.split(",") if x.strip()]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("yolo_weights", "sam3_weights", "sam_fallback_weights", "data_dir", "frontend_dist", mode="before")
    @classmethod
    def _abs_path(cls, v):
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # 파생 경로
    @property
    def cameras_dir(self) -> Path:
        return self.data_dir / "cameras"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    def ensure_dirs(self) -> None:
        for d in (self.cameras_dir, self.uploads_dir, self.snapshots_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


CONGESTION_LEVELS = ["원활", "서행", "지체", "정체"]


def congestion_level(occupancy: float | None, thresholds: list[float]) -> str | None:
    """면적비 → 혼잡 단계 문자열."""
    if occupancy is None:
        return None
    for i, t in enumerate(thresholds):
        if occupancy < t:
            return CONGESTION_LEVELS[i]
    return CONGESTION_LEVELS[-1]
