"""Horn 3x3 gradient + Lambert diffuse hillshading on a padded tile grid."""
from __future__ import annotations

import numpy as np

ORIGIN = 20037508.342789244


def horn_gradients(elev: np.ndarray, dx, dy: float):
    """Horn's method gradients. dx may be scalar or per-row column vector."""
    a = elev[:-2, :-2]
    b = elev[:-2, 1:-1]
    c = elev[:-2, 2:]
    d = elev[1:-1, :-2]
    f = elev[1:-1, 2:]
    g = elev[2:, :-2]
    h = elev[2:, 1:-1]
    i = elev[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    return dzdx, dzdy


def hillshade(elev: np.ndarray, res_3857: float, lats_deg: np.ndarray,
              azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """Compute a grayscale hillshade for a (n, n) elevation grid.

    elev: NaN where no data. res_3857: pixel size in mercator meters.
    lats_deg: per-row latitude (degrees) of pixel centers, shape (n,).
    Returns float array 0..255 with NaN where the 3x3 neighborhood is
    incomplete (those pixels become transparent).
    """
    valid = ~np.isnan(elev)
    full = (valid[:-2, :-2] & valid[:-2, 1:-1] & valid[:-2, 2:]
            & valid[1:-1, :-2] & valid[1:-1, 1:-1] & valid[1:-1, 2:]
            & valid[2:, :-2] & valid[2:, 1:-1] & valid[2:, 2:])
    safe = np.where(valid, elev, 0.0)

    # Mercator distance correction: ground meters = mercator meters / cos(lat)
    cos_lat = np.cos(np.radians(lats_deg))
    ground = (res_3857 / cos_lat)[1:-1, None]  # per-row, shape (n-2, 1)
    dzdx, dzdy = horn_gradients(safe, ground, 1.0)
    dzdy = dzdy / ground

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdx, -dzdy)

    zenith = np.radians(90.0 - altitude_deg)
    azim = np.radians(azimuth_deg)
    shade = (np.cos(zenith) * np.cos(slope)
             + np.sin(zenith) * np.sin(slope) * np.cos(azim - aspect))
    shade = np.clip(shade, 0.0, 1.0) * 255.0
    shade[~full] = np.nan
    return shade
