"""Post-process probability maps into instance masks for submission."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def probs_to_instances(
    prob: np.ndarray,
    *,
    out_size: int = 2048,
    threshold: float = 0.5,
    min_area: int = 64,
) -> list[np.ndarray]:
    """Threshold, resize to out_size, split connected components.

    Args:
        prob: HxW float probability map (model resolution, e.g. 768).
        out_size: Output spatial size (competition: 2048).
        threshold: Probability cutoff.
        min_area: Drop components smaller than this many pixels at out_size.

    Returns:
        List of uint8 HxW masks (0/1), one per instance, sorted by area desc.
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob must be HxW, got {prob.shape}")

    binary_small = (prob >= threshold).astype(np.uint8) * 255
    img = Image.fromarray(binary_small, mode="L")
    img = img.resize((out_size, out_size), resample=Image.NEAREST)
    binary = (np.asarray(img) > 0).astype(np.uint8)

    labeled, n = ndimage.label(binary)
    if n == 0:
        return []

    instances: list[tuple[int, np.ndarray]] = []
    for idx in range(1, n + 1):
        comp = (labeled == idx).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        instances.append((area, comp))

    instances.sort(key=lambda t: t[0], reverse=True)
    return [comp for _, comp in instances]
