"""COCO RLE helpers for Kaggle filament submission (counts only, 2048x2048)."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util


def mask_to_rle_counts(mask: np.ndarray) -> str:
    """Encode a binary HxW mask to COCO compressed RLE counts string.

    Competition requires counts only (size is fixed 2048x2048).
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape {mask.shape}")
    binary = (mask > 0).astype(np.uint8)
    # pycocotools expects Fortran order (column-major)
    rle = mask_util.encode(np.asfortranarray(binary))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def rle_counts_to_mask(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode COCO RLE counts back to uint8 HxW mask."""
    rle = {"size": [height, width], "counts": counts}
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    decoded = mask_util.decode(rle)
    return (decoded > 0).astype(np.uint8)


def roundtrip_ok(mask: np.ndarray) -> bool:
    """True if encode→decode preserves the binary mask."""
    counts = mask_to_rle_counts(mask)
    h, w = mask.shape
    back = rle_counts_to_mask(counts, height=h, width=w)
    return bool(np.array_equal((mask > 0).astype(np.uint8), back))
