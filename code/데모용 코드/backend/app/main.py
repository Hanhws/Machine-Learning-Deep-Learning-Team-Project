"""FastAPI 진입점.

- /api/...          REST API
- /                 빌드된 프론트엔드(frontend/dist) — 없으면 안내 메시지
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import get_settings
from app.db.session import init_db
from app.services.worker_manager import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if os.environ.get("AUTO_START_WORKERS", "1") == "1":
        n = manager.start_all_enabled()
        log.info("등록된 카메라 워커 %d개 시작", n)
    yield
    manager.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)

dist: Path = settings.frontend_dist
if dist.exists() and (dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str, request: Request):
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # 화면 껍데기는 캐시하지 않는다 (빌드가 바뀌면 바로 새 JS 를 받도록). JS/CSS 는 파일명에 해시가 있어 캐시돼도 안전
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

else:

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            {"message": "프론트엔드가 빌드되지 않았습니다. `cd frontend && npm run build` 후 다시 실행하거나 dev 서버(npm run dev)를 쓰세요.", "docs": "/docs"}
        )
