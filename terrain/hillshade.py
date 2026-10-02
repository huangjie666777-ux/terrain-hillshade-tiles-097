"""Horn gradient hillshading with latitude-corrected ground spacing."""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_378_137.0
MAX_LATITUDE = 85.05112878


def _row_latitudes(shape: tuple[int, int], transform) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64)
    y = transform.f + transform.e * (rows + 0.5)
    lat = np.degrees(np.arctan(np.sinh(y / EARTH_RADIUS_M)))
    return np.clip(lat, -MAX_LATITUDE, MAX_LATITUDE)


def horn_shade(
    elevation: np.ma.MaskedArray,
    transform,
    azimuth: float,
    altitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RGBA hillshade for the interior pixels of a one-cell padding.

    Gradients are calculated on the complete stitched padded array. An output
    pixel is transparent unless all nine Horn neighbours have valid elevation.
    """
    if elevation.ndim != 2 or elevation.shape[0] < 3 or elevation.shape[1] < 3:
        raise ValueError("at least a 3x3 padded elevation array is required")

    values = elevation.astype(np.float64, copy=False).filled(np.nan)
    valid = np.isfinite(values)
    a = values[:-2, :-2]
    b = values[:-2, 1:-1]
    c = values[:-2, 2:]
    d = values[1:-1, :-2]
    f = values[1:-1, 2:]
    g = values[2:, :-2]
    h = values[2:, 1:-1]
    i = values[2:, 2:]

    interior_valid = (
        valid[:-2, :-2]
        & valid[:-2, 1:-1]
        & valid[:-2, 2:]
        & valid[1:-1, :-2]
        & valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[2:, :-2]
        & valid[2:, 1:-1]
        & valid[2:, 2:]
    )

    # Web Mercator is conformal: at latitude phi, one projected metre equals
    # cos(phi) ground metres in both local axes. Avoid treating projected
    # horizontal metres directly as ground metres.
    pixel_m = abs(float(transform.a))
    if abs(float(transform.e)) > 0:
        pixel_m = (pixel_m + abs(float(transform.e))) / 2.0
    lat = _row_latitudes(elevation.shape, transform)
    ground_m = (pixel_m * np.cos(np.radians(lat)))[:, None]
    ground_m = np.maximum(ground_m[1:-1, :], 1e-9)

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * ground_m)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * ground_m)

    azimuth_rad = math.radians(azimuth)
    altitude_rad = math.radians(altitude)
    # Local axes are east, north, up. Screen-row dz/dy is positive when
    # elevation increases southward, so the upward normal is (-dx, +dy, 1).
    light_x = math.cos(altitude_rad) * math.sin(azimuth_rad)
    light_y = math.cos(altitude_rad) * math.cos(azimuth_rad)
    light_z = math.sin(altitude_rad)
    numerator = light_z - light_x * dz_dx + light_y * dz_dy
    denominator = np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
    shade = numerator / denominator
    shade = np.where(interior_valid, shade, 0.0)
    shade = np.clip(shade, 0.0, 1.0)
    gray = np.rint(shade * 255.0).astype(np.uint8)
    alpha = np.where(interior_valid, 255, 0).astype(np.uint8)
    return gray, alpha
