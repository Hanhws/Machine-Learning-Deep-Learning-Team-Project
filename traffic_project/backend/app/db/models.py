"""DB 스키마.

cameras            등록된 CCTV (ITS 실시간 / 업로드 영상 / 임의 스트림 URL)
directions         카메라별 도로 방향 (마스크 라벨값 1..N 과 대응)
occupancy_samples  시계열 점유율 (direction_index 0 = 카메라 전체)
app_groups         응용 페이지용 카메라 묶음 (예: 한강 대교)
analysis_jobs      업로드 영상 전체 분석 작업
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(16))  # its | upload | url

    # ITS 메타
    its_cctv_name: Mapped[str | None] = mapped_column(String(200))
    its_road_type: Mapped[str | None] = mapped_column(String(8))  # ex | its
    its_cctv_type: Mapped[str | None] = mapped_column(String(4))
    stream_url: Mapped[str | None] = mapped_column(Text)
    stream_url_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 업로드 영상
    video_path: Mapped[str | None] = mapped_column(Text)

    # 위치/분류
    lon: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float | None] = mapped_column(Float)
    route: Mapped[str | None] = mapped_column(String(100))    # 노선 (예: 경부선)
    region: Mapped[str | None] = mapped_column(String(100))   # 지역 (예: 경기 수원)
    section: Mapped[str | None] = mapped_column(String(200))  # 구간 (예: 신갈JC~수원IC)

    # 스냅샷/마스크
    snapshot_path: Mapped[str | None] = mapped_column(Text)
    mask_path: Mapped[str | None] = mapped_column(Text)
    frame_width: Mapped[int | None] = mapped_column(Integer)
    frame_height: Mapped[int | None] = mapped_column(Integer)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # 추론 주기(초). None/0 = 실시간(INFER_FPS), 60 = 1분마다 프레임 1장
    infer_interval_s: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    directions: Mapped[list["Direction"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", order_by="Direction.index"
    )


class Direction(Base):
    __tablename__ = "directions"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)  # 1..N (마스크 라벨값)
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(16), default="#2a78d6")
    # 한 CCTV 에 도로가 여러 개일 수 있다 (본선/램프, 교차 도로). 같은 road 끼리 묶어 보여준다.
    road: Mapped[str | None] = mapped_column(String(100))
    # 지도 표시: 진행 방향 각도(북=0, 시계방향, 도) 와 화살표 위치(없으면 카메라 좌표)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)

    camera: Mapped[Camera] = relationship(back_populates="directions")


class OccupancySample(Base):
    __tablename__ = "occupancy_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    direction_index: Mapped[int] = mapped_column(Integer, default=0)  # 0 = 전체
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    video_time: Mapped[float | None] = mapped_column(Float)  # 업로드 영상 분석 시 영상 내 시각(초)
    source: Mapped[str] = mapped_column(String(8), default="live")  # live | file

    occupancy: Mapped[float] = mapped_column(Float)
    vehicle_px: Mapped[float] = mapped_column(Float, default=0)
    road_px: Mapped[float] = mapped_column(Float, default=0)
    n_car: Mapped[float] = mapped_column(Float, default=0)
    n_bus: Mapped[float] = mapped_column(Float, default=0)
    n_truck: Mapped[float] = mapped_column(Float, default=0)
    car_px: Mapped[float] = mapped_column(Float, default=0)
    bus_px: Mapped[float] = mapped_column(Float, default=0)
    truck_px: Mapped[float] = mapped_column(Float, default=0)
    n_frames: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (Index("ix_samples_cam_dir_ts", "camera_id", "direction_index", "ts"),)


class AppGroup(Base):
    __tablename__ = "app_groups"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(32), default="custom")  # han_river | custom
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    members: Mapped[list["AppGroupCamera"]] = relationship(
        back_populates="group", cascade="all, delete-orphan", order_by="AppGroupCamera.order"
    )


class AppGroupCamera(Base):
    __tablename__ = "app_group_cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("app_groups.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    label: Mapped[str | None] = mapped_column(String(200))  # 예: 한남대교
    order: Mapped[int] = mapped_column(Integer, default=0)

    group: Mapped[AppGroup] = relationship(back_populates="members")
    camera: Mapped[Camera] = relationship()


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    frame_stride: Mapped[int] = mapped_column(Integer, default=5)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
