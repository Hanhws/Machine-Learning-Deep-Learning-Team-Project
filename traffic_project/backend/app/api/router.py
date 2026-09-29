from fastapi import APIRouter

from app.api import apps, cameras, captures, its, metrics, metrics_extra, routes, segment, snapshots, stream, system

api_router = APIRouter(prefix="/api")
for r in (system.router, its.router, snapshots.router, segment.router, cameras.router, captures.router, stream.router, metrics.router, metrics_extra.router, routes.router, apps.router):
    api_router.include_router(r)
