"""Solar-disk geometry for the limb filter.

MAGFiLO annotators were told to ignore filaments "roughly beyond ±70° from the central meridian"
(Ahmadzadeh et al. 2024), so predictions there are mostly scored as false positives. The disk is
found from the image itself (GONG JPEGs are remapped to a near-constant radius, but not exactly
centred); the solar B0 tilt is ignored, which is fine for a coarse longitude cut.
"""

from __future__ import annotations

import numpy as np


def fit_disk(img: np.ndarray, frac: float = 0.5) -> tuple[float, float, float] | None:
    """(cx, cy, r) of the solar disk: pixels brighter than frac * 99th percentile, area-equivalent radius."""
    m = img > frac * float(np.percentile(img, 99))
    ys, xs = np.nonzero(m)
    if len(ys) < 1000:
        return None
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(len(ys) / np.pi))


def central_meridian_deg(x: float, y: float, disk: tuple[float, float, float]) -> float:
    """|heliographic longitude| in degrees of image point (x, y); 90 at or beyond the limb."""
    cx, cy, r = disk
    lat = np.arcsin(np.clip((cy - y) / r, -1.0, 1.0))
    s = (x - cx) / max(r * np.cos(lat), 1e-6)
    return float(np.degrees(np.arcsin(np.clip(abs(s), 0.0, 1.0))))
