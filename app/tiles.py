"""Tile rendering pipeline: mosaic -> pad -> hillshade -> crop -> RGBA PNG."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from . import config
from .hillshade import ORIGIN, hillshade
from .reproject import mosaic_tile, tile_bounds_3857


def _pixel_centers(minx, miny, maxx, maxy, n):
    res = (maxx - minx) / n
    cols = minx + (np.arange(n) + 0.5) * res
    rows = maxy - (np.arange(n) + 0.5) * res
    xs, ys = np.meshgrid(cols, rows)
    return xs, ys, res


def _mercator_y_to_lat(y: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(np.sinh(y / ORIGIN * np.pi)))


def render_tile(metas: list[dict], z: int, x: int, y: int,
                azimuth: float, altitude: float) -> bytes:
    """Render a 256x256 grayscale RGBA PNG hillshade tile."""
    size = config.TILE_SIZE
    pad = 1
    n = size + 2 * pad
    minx, miny, maxx, maxy = tile_bounds_3857(z, x, y)
    res = (maxx - minx) / size
    # expand bounds by one pixel so gradients are computed on a unified grid
    xs, ys, _ = _pixel_centers(minx - pad * res, miny - pad * res,
                               maxx + pad * res, maxy + pad * res, n)
    elev = mosaic_tile(metas, xs, ys)
    lats = _mercator_y_to_lat(ys[:, 0])
    shade = hillshade(elev, res, lats, azimuth, altitude)
    # the 3x3 gradient consumes the 1px padding; output is already 256x256
    gray = np.where(np.isnan(shade), 0, shade).astype(np.uint8)
    alpha = np.where(np.isnan(shade), 0, 255).astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
