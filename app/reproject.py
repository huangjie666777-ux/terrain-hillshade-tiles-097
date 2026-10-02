"""Windowed reprojection + priority mosaicking of DEM datasets onto XYZ tiles.

All sampling is done by reading only the raster window that intersects the
requested tile; the full dataset is never loaded into memory.
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

# Web-mercator world extent in EPSG:3857 meters.
ORIGIN = 20037508.342789244


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) of an XYZ tile in EPSG:3857."""
    n = 2 ** z
    size = 2 * ORIGIN / n
    minx = -ORIGIN + x * size
    maxx = minx + size
    maxy = ORIGIN - y * size
    miny = maxy - size
    return minx, miny, maxx, maxy


def validate_tile(z: int, x: int, y: int, min_z: int, max_z: int) -> bool:
    if not (min_z <= z <= max_z):
        return False
    n = 2 ** z
    return 0 <= x < n and 0 <= y < n


def _read_window(ds, minx, miny, maxx, maxy):
    """Read the window of ds intersecting the given bounds in ds's own CRS.

    Returns (array2d float64, mask2d bool valid, window_transform) or None.
    """
    inv = ~ds.transform
    c0, r0 = inv @ (minx, maxy)
    c1, r1 = inv @ (maxx, miny)
    col_off = max(0, int(math.floor(min(c0, c1))))
    row_off = max(0, int(math.floor(min(r0, r1))))
    col_stop = min(ds.width, int(math.ceil(max(c0, c1))))
    row_stop = min(ds.height, int(math.ceil(max(r0, r1))))
    if col_stop <= col_off or row_stop <= row_off:
        return None
    win = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)
    arr = ds.read(1, window=win, masked=True).astype("float64")
    valid = ~np.ma.getmaskarray(arr)
    data = np.asarray(arr.filled(np.nan), dtype="float64")
    if ds.nodata is not None:
        nodata_mask = data == ds.nodata
        valid &= ~nodata_mask
        data[nodata_mask] = np.nan
    data[~valid] = np.nan
    win_transform = ds.transform @ Affine.translation(col_off, row_off)
    return data, valid, win_transform


def _bilinear(data, valid, rows, cols):
    """Bilinear sample of data at fractional rows/cols arrays.

    Any output whose 2x2 stencil touches an invalid cell is NaN so that
    NoData never leaks into interpolation.
    """
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    h, w = data.shape
    inside = (r0 >= 0) & (c0 >= 0) & (r1 < h) & (c1 < w)
    out = np.full(rows.shape, np.nan)
    if not inside.any():
        return out
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r1, 0, h - 1)
    c0c = np.clip(c0, 0, w - 1)
    c1c = np.clip(c1, 0, w - 1)
    ok = inside & valid[r0c, c0c] & valid[r0c, c1c] & valid[r1c, c0c] & valid[r1c, c1c]
    if not ok.any():
        return out
    fr = rows - r0
    fc = cols - c0
    v00 = data[r0c, c0c]
    v01 = data[r0c, c1c]
    v10 = data[r1c, c0c]
    v11 = data[r1c, c1c]
    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    val = top * (1 - fr) + bot * fr
    out[ok] = val[ok]
    return out


def sample_dataset(meta: dict, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample one dataset at EPSG:3857 coordinate grids (pixel centers).

    xs/ys are 2D arrays of shape (n, n). Returns float64 elevations with NaN
    where the dataset has no valid coverage.
    """
    out = np.full(xs.shape, np.nan)
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    with rasterio.open(meta["path"]) as ds:
        if ds.crs.to_epsg() == 3857:
            bounds = (minx, miny, maxx, maxy)
        else:
            xs_t, ys_t = warp_transform(
                "EPSG:3857", ds.crs,
                [minx, maxx, minx, maxx], [miny, miny, maxy, maxy],
            )
            bounds = (min(xs_t), min(ys_t), max(xs_t), max(ys_t))
        got = _read_window(ds, *bounds)
        if got is None:
            return out
        data, valid, win_transform = got
        if ds.crs.to_epsg() == 3857:
            src_xs, src_ys = xs, ys
        else:
            flat_x, flat_y = warp_transform(
                "EPSG:3857", ds.crs, xs.ravel().tolist(), ys.ravel().tolist()
            )
            src_xs = np.asarray(flat_x).reshape(xs.shape)
            src_ys = np.asarray(flat_y).reshape(ys.shape)
        inv = ~win_transform
        cols, rows = inv @ (src_xs, src_ys)
        cols = np.asarray(cols)
        rows = np.asarray(rows)
        return _bilinear(data, valid, rows, cols)


def mosaic_tile(metas: list[dict], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Ordered mosaic: first dataset wins; holes may be filled by later ones."""
    result = np.full(xs.shape, np.nan)
    remaining = np.ones(xs.shape, dtype=bool)
    for meta in metas:
        if not remaining.any():
            break
        sample = sample_dataset(meta, xs, ys)
        fill = remaining & ~np.isnan(sample)
        result[fill] = sample[fill]
        remaining &= np.isnan(result)
    return result
