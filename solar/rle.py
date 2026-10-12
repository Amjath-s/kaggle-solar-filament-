from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


MASK_SIZE = (2048, 2048)


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode a binary mask as COCO compressed RLE counts."""
    if mask.shape != MASK_SIZE:
        raise ValueError(f"Expected mask shape {MASK_SIZE}, got {mask.shape}")

    binary = np.asfortranarray((mask > 0).astype(np.uint8))
    encoded = mask_utils.encode(binary)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        return counts.decode("utf-8")
    return str(counts)


def decode_rle_counts(counts: str, size: tuple[int, int] = MASK_SIZE) -> np.ndarray:
    rle = {"size": list(size), "counts": counts.encode("utf-8")}
    return mask_utils.decode(rle).astype(bool)

