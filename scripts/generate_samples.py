#!/usr/bin/env python3
"""Generate a slope raster with a hole and an overlapping lower-priority fill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


# A compact patch near Lijiang, Yunnan. Values and geometry stay in degrees
# for CRS EPSG:4326; elevation values are metres.
WEST, SOUTH, EAST, NORTH = 99.80, 26.85, 99.90, 26.95
WIDTH = HEIGHT = 120
NODATA = -9999.0


def elevation_grid() -> np.ndarray:
    x = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[::-1]
    gx, gy = np.meshgrid(x, y)
    ridge = 900.0 * np.exp(-((gx - 0.52) ** 2 / 0.010 + (gy - 0.48) ** 2 / 0.025))
    return (1800.0 + 650.0 * gx - 450.0 * gy + ridge).astype(np.float32)


def write_raster(path: Path, values: np.ndarray, with_mask: bool = False) -> None:
    transform = from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)
    profile = {
        "driver": "GTiff",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        if with_mask:
            # Also persist an explicit band mask. This exercises masked holes
            # in addition to the sentinel NoData value.
            dst.write_mask(np.isfinite(values) & (values != NODATA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples"))
    args = parser.parse_args()

    primary = elevation_grid()
    # Deliberate coverage hole through the ridge. Upload this first: it has
    # priority everywhere except this transparent/NoData rectangle.
    primary[45:70, 38:72] = NODATA
    write_raster(args.out / "01_slope_with_hole.tif", primary, with_mask=True)

    # Slightly lower synthetic elevations make priority observable. The only
    # visible contribution in the requested example is the primary's hole.
    fill = elevation_grid() - 35.0
    write_raster(args.out / "02_overlapping_fill.tif", fill, with_mask=False)
    print(f"wrote {args.out / '01_slope_with_hole.tif'}")
    print(f"wrote {args.out / '02_overlapping_fill.tif'}")
    print("publish in this order so the second dataset fills the first's hole")


if __name__ == "__main__":
    main()
