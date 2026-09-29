"""Pydantic 요청/응답 스키마."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------- 카메라 ----------
class DirectionIn(BaseModel):
    index: int = Field(ge=1, le=8)
    name: str
    color: str = "#2a78d6"
    road: str | None = None  # 노선 (예: 경부선, 강동IC 진출램프)
    destination: str | None = None  # 방면 = 표지판 목적지 (예: 서울)
    heading_source: str | None = Field(default=None, pattern="^(auto|manual)$")
    heading_deg: float | None = Field(default=None, ge=0, lt=360)  # 진행 방향 (북=0, 시계방향)
    lat: float | None = None  # 지도 화살표 위치 (없으면 카메라 좌표)
    lon: float | None = None


class DirectionOut(DirectionIn):
    id: int

    model_config = {"from_attributes": True}


class CameraCreate(BaseModel):
    name: str
    source_type: str = Field(pattern="^(its|upload|url)$")
    # its
    its_cctv_name: str | None = None
    its_road_type: str | None = None
    its_cctv_type: str | None = "1"
    stream_url: str | None = None
    # upload
    video_path: str | None = None
    # 등록 전 임시 스냅샷 id (POST /api/snapshots 결과)
    snapshot_id: str | None = None
    # 공통
    lon: float | None = None
    lat: float | None = None
    route: str | None = None
    region: str | None = None
    section: str | None = None
    infer_interval_s: float | None = Field(default=None, ge=0, le=3600)
    meta: dict[str, Any] | None = None


class CameraUpdate(BaseModel):
    name: str | None = None
    lon: float | None = None
    lat: float | None = None
    route: str | None = None
    region: str | None = None
    section: str | None = None
    enabled: bool | None = None
    stream_url: str | None = None
    infer_interval_s: float | None = Field(default=None, ge=0, le=3600)
    meta: dict[str, Any] | None = None


class CameraOut(BaseModel):
    id: int
    name: str
    source_type: str
    its_cctv_name: str | None
    its_road_type: str | None
    stream_url: str | None
    stream_url_fetched_at: datetime | None
    video_path: str | None
    lon: float | None
    lat: float | None
    route: str | None
    region: str | None
    section: str | None
    snapshot_path: str | None
    mask_path: str | None
    frame_width: int | None
    frame_height: int | None
    enabled: bool
    infer_interval_s: float | None = None
    meta: dict[str, Any] | None
    created_at: datetime
    directions: list[DirectionOut] = []
    # 런타임
    running: bool = False
    live: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class MaskUpdate(BaseModel):
    mask_png_base64: str  # uint8 라벨맵 PNG (0=도로아님, 1..N=방향)
    directions: list[DirectionIn]


# ---------- 세그멘테이션 ----------
class SegmentTextIn(BaseModel):
    camera_id: int | None = None
    snapshot_id: str | None = None  # 등록 전 임시 스냅샷
    text: str = "road"
    conf: float | None = None
    fill_vehicles: bool = True  # 스냅샷의 차량 자리도 도로로 (분모 = 노면 전체). 화면에서는 항상 켬


class SegmentPromptIn(BaseModel):
    camera_id: int | None = None
    snapshot_id: str | None = None
    points: list[list[float]] | None = None  # [[x,y],...]
    labels: list[int] | None = None  # 1=포함 0=제외
    boxes: list[list[float]] | None = None  # [[x1,y1,x2,y2],...]
    fill_vehicles: bool = True


class MaskOut(BaseModel):
    mask_png_base64: str
    coverage: float  # 마스크가 프레임에서 차지하는 비율
    width: int
    height: int
    backend: str
    vehicles_added: float = 0.0  # 차량 자리로 채운 픽셀 비율 (프레임 대비)


# ---------- 응용 그룹 ----------
class GroupMemberIn(BaseModel):
    camera_id: int
    label: str | None = None
    order: int = 0


class GroupCreate(BaseModel):
    name: str
    kind: str = "custom"
    description: str | None = None
    members: list[GroupMemberIn] = []


class GroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    members: list[GroupMemberIn] | None = None


class GroupMemberOut(BaseModel):
    id: int
    camera_id: int
    label: str | None
    order: int
    camera_name: str | None = None

    model_config = {"from_attributes": True}


class GroupOut(BaseModel):
    id: int
    name: str
    kind: str
    description: str | None
    created_at: datetime
    members: list[GroupMemberOut] = []

    model_config = {"from_attributes": True}


# ---------- 분석 작업 ----------
class JobOut(BaseModel):
    id: int
    camera_id: int
    status: str
    progress: float
    frame_stride: int
    started_at: datetime | None
    finished_at: datetime | None
    summary: dict[str, Any] | None
    error: str | None

    model_config = {"from_attributes": True}
