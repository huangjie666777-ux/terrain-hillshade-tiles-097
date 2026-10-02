from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import terrain.config as config
import terrain.storage as storage
from terrain.tiles import TileCache, TileRenderer


@pytest.fixture()
def temp_service(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    registry = storage.Registry(root=root)
    import terrain.tiles as tiles_mod
    renderer = TileRenderer(
        dataset_dir=str(registry.dataset_dir),
        cache=TileCache(capacity=3),
        slots=__import__("threading").BoundedSemaphore(1),
    )
    yield registry, renderer, tmp_path
    registry.registry_path.unlink(missing_ok=True)


def make_geotiff(
    path: Path,
    values: np.ndarray,
    crs: str = "EPSG:4326",
    bounds=(99.8, 26.8, 99.95, 26.95),
    nodata=-9999.0,
    count: int = 1,
    transform=None,
    write_mask: bool = False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape[:2] if values.ndim == 3 else values.shape
    transform = transform or from_bounds(*bounds, width, height)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": values.dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None and count == 1:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if count == 1:
            dst.write(values, 1)
        else:
            dst.write(values)
        if write_mask and count == 1:
            dst.write_mask((values != nodata) & np.isfinite(values))


@pytest.fixture()
def make_tif():
    return make_geotiff


@pytest.fixture()
def client(temp_service):
    registry, _renderer, _tmp = temp_service
    from fastapi.testclient import TestClient

    monkeypatch_client = TestClient(app_module.app)
    # Replace module-level objects with fixtures after reload.
    app_module.registry = registry
    app_module.renderer = _renderer
    return monkeypatch_client
