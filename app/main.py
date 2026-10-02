"""FastAPI HTTP layer for the hillshade tile service."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import config
from .cache import TileCache
from .reproject import validate_tile
from .storage import DatasetError, Storage
from .tiles import render_tile

app = FastAPI(title="Hillshade Tile Service")
storage = Storage()
cache = TileCache(config.TILE_CACHE_CAPACITY)
semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TILES)


class PublishRequest(BaseModel):
    datasets: list[str]


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".tif", dir=config.DATA_DIR)
    size = 0
    try:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, "upload exceeds size limit")
            tmp.write(chunk)
        tmp.close()
        meta = storage.save_dataset(Path(tmp.name))
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return {"id": meta["id"], "meta": meta}


@app.get("/datasets")
async def list_datasets():
    return {"datasets": storage.list_datasets()}


@app.post("/lists")
async def publish_list(req: PublishRequest):
    try:
        version = storage.publish_list(req.datasets)
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"version": version}


@app.get("/lists/{version}")
async def get_list(version: int):
    datasets = storage.get_list(version)
    if datasets is None:
        raise HTTPException(404, f"unknown list version {version}")
    return {"version": version, "datasets": datasets}


@app.get("/tiles/{version}/{z}/{x}/{y}.png")
async def get_tile(
    version: int, z: int, x: int, y: int,
    azimuth: float = Query(315.0, ge=0.0, lt=360.0),
    altitude: float = Query(45.0, gt=0.0, le=90.0),
):
    if not validate_tile(z, x, y, config.MIN_ZOOM, config.MAX_ZOOM):
        raise HTTPException(400, "tile coordinates out of range")
    dataset_ids = storage.get_list(version)
    if dataset_ids is None:
        raise HTTPException(404, f"unknown list version {version}")
    key = (version, z, x, y, round(azimuth, 4), round(altitude, 4))
    cached = cache.get(key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")
    metas = [storage.get_dataset(d) for d in dataset_ids]
    metas = [m for m in metas if m is not None]
    async with semaphore:
        png = await asyncio.to_thread(
            render_tile, metas, z, x, y, azimuth, altitude)
    cache.put(key, png)
    return Response(content=png, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "latest_version": storage.latest_version()}
