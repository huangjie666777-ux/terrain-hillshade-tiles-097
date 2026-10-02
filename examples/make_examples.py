"""Generate example DEMs: a sloped plane and an overlapping DEM with holes.

Usage: .venv/bin/python examples/make_examples.py [outdir]
Writes slope_4326.tif and holes_3857.tif into the output directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


def make_slope(path: Path) -> None:
    """EPSG:4326 ramp rising to the east, 600x400 px around (116E, 40N)."""
    w, h = 600, 400
    res = 0.001
    transform = Affine(res, 0, 116.0, 0, -res, 40.0)
    cols = np.arange(w)[None, :]
    rows = np.arange(h)[:, None]
    dem = (100.0 + 0.5 * cols + 0.25 * rows).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as ds:
        ds.write(dem, 1)


def make_holes(path: Path) -> None:
    """EPSG:3857 hill with NoData holes, overlapping the slope's area."""
    w, h = 500, 500
    res = 40.0
    # roughly covers lon 116.0-116.2, lat 39.8-40.0 in mercator meters
    cx, cy = 12932244.0, 4852834.0
    transform = Affine(res, 0, cx - w * res / 2, 0, -res, cy + h * res / 2)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2)
    dem = (800.0 * np.exp(-(r / 150.0) ** 2)).astype("float32")
    dem[(yy % 97 < 8) & (xx % 83 < 8)] = -9999.0  # scattered holes
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1,
        dtype="float32", crs="EPSG:3857", transform=transform, nodata=-9999.0,
    ) as ds:
        ds.write(dem, 1)


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/out")
    outdir.mkdir(parents=True, exist_ok=True)
    make_slope(outdir / "slope_4326.tif")
    make_holes(outdir / "holes_3857.tif")
    print(f"wrote {outdir}/slope_4326.tif and {outdir}/holes_3857.tif")
