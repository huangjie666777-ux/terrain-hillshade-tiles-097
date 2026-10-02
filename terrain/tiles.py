"""XYZ tile geometry, bounded rendering concurrency, and tile cache."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from affine import Affine
from PIL import Image

from .config import settings
from .hillshade import EARTH_RADIUS_M, horn_shade
from .mosaic import build_mosaic
from .storage import DatasetMeta

ORIGIN_SHIFT = math.pi * EARTH_RADIUS_M


class TileError(ValueError):
    """Raised for invalid tile coordinates or lighting parameters."""


def validate_light(azimuth: float, altitude: float) -> tuple[float, float]:
    if not math.isfinite(azimuth) or not 0.0 <= azimuth < 360.0:
        raise TileError("azimuth must be in [0, 360) degrees clockwise from north")
    if not math.isfinite(altitude) or not 0.0 < altitude <= 90.0:
        raise TileError("altitude must be in (0, 90] degrees above horizon")
    return float(azimuth), float(altitude)


def validate_xyz(z: int, x: int, y: int) -> None:
    if not 0 <= z <= settings.max_zoom:
        raise TileError(f"zoom z must be in [0, {settings.max_zoom}]")
    limit = 1 << z
    if not 0 <= x < limit or not 0 <= y < limit:
        raise TileError(f"x and y must be in [0, {limit - 1}] at z={z}")


def expanded_geometry(z: int, x: int, y: int):
    """Return 258x258 transform and bounds for a one-pixel-rendered tile."""
    size = settings.tile_size
    world_tile = (2.0 * ORIGIN_SHIFT) / float(1 << z)
    pixel = world_tile / float(size)
    left = -ORIGIN_SHIFT + x * world_tile - pixel
    top = ORIGIN_SHIFT - y * world_tile + pixel
    shape = (size + 2, size + 2)
    transform = Affine(pixel, 0.0, left, 0.0, -pixel, top)
    tile_left = left + pixel
    tile_top = top - pixel
    tile_right = tile_left + world_tile
    tile_bottom = tile_top - world_tile
    bounds = (tile_left, tile_bottom, tile_right, tile_top)
    return shape, transform, bounds


class TileCache:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = capacity if capacity is not None else settings.cache_max_tiles
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            try:
                value = self._items.pop(key)
                self._items[key] = value
                return value
            except KeyError:
                return None

    def put(self, key, value: bytes) -> None:
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = value
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


@dataclass
class TileRenderer:
    dataset_dir: str
    cache: TileCache
    slots: threading.BoundedSemaphore

    def render(
        self,
        version: str,
        datasets: list[DatasetMeta],
        z: int,
        x: int,
        y: int,
        azimuth: float,
        altitude: float,
    ) -> bytes:
        azimuth, altitude = validate_light(azimuth, altitude)
        validate_xyz(z, x, y)
        cache_key = (version, z, x, y, azimuth, altitude)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        with self.slots:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
            shape, transform, bounds = expanded_geometry(z, x, y)
            elevation = build_mosaic(datasets, shape, transform, bounds, self.dataset_dir)
            gray, alpha = horn_shade(elevation, transform, azimuth, altitude)
            rgba = np.zeros((settings.tile_size, settings.tile_size, 4), dtype=np.uint8)
            rgba[..., 0] = gray
            rgba[..., 1] = gray
            rgba[..., 2] = gray
            rgba[..., 3] = alpha
            image = Image.fromarray(rgba, mode="RGBA")
            from io import BytesIO

            fh = BytesIO()
            image.save(fh, format="PNG", optimize=False)
            png = fh.getvalue()
        self.cache.put(cache_key, png)
        return png
