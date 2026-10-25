"""YOLO box proposals + U-Net mask refine (+ orphan CCs)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def upsample_prob(prob: np.ndarray, out_size: int = 2048) -> np.ndarray:
    """Bilinear upsample HxW float prob map to out_size."""
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob must be HxW, got {prob.shape}")
    if prob.shape[0] == out_size and prob.shape[1] == out_size:
        return prob
    img = Image.fromarray(prob, mode="F")
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def scale_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    *,
    from_size: tuple[int, int],
    to_size: int = 2048,
    pad: int = 16,
) -> np.ndarray:
    """Scale boxes from (w,h) image space to square to_size, then pad and clip.

    Args:
        boxes_xyxy: (N, 4) in absolute xyxy for an image of size from_size=(w,h).
        from_size: (width, height) of the coordinate space boxes are in.
        to_size: output square size.
        pad: pixels to expand each side after scaling.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    w0, h0 = float(from_size[0]), float(from_size[1])
    sx = to_size / w0
    sy = to_size / h0
    boxes[:, 0] *= sx
    boxes[:, 2] *= sx
    boxes[:, 1] *= sy
    boxes[:, 3] *= sy
    boxes[:, 0] -= pad
    boxes[:, 1] -= pad
    boxes[:, 2] += pad
    boxes[:, 3] += pad
    boxes[:, 0] = np.clip(boxes[:, 0], 0, to_size - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, to_size - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, to_size - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, to_size - 1)
    return boxes


def _largest_cc(binary: np.ndarray) -> np.ndarray | None:
    labeled, n = ndimage.label(binary)
    if n == 0:
        return None
    best_idx = 0
    best_area = 0
    for idx in range(1, n + 1):
        area = int((labeled == idx).sum())
        if area > best_area:
            best_area = area
            best_idx = idx
    return (labeled == best_idx).astype(np.uint8)


def refine_boxes_with_unet(
    prob_2048: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area: int = 64,
) -> list[np.ndarray]:
    """One instance per box: largest CC of (prob >= thr) inside the padded box."""
    h, w = prob_2048.shape
    binary = (prob_2048 >= threshold).astype(np.uint8)
    instances: list[tuple[int, np.ndarray]] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = np.zeros_like(binary)
        roi[y1 : y2 + 1, x1 : x2 + 1] = binary[y1 : y2 + 1, x1 : x2 + 1]
        cc = _largest_cc(roi)
        if cc is None:
            continue
        area = int(cc.sum())
        if area < min_area:
            continue
        instances.append((area, cc))
    instances.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in instances]


def orphan_components(
    prob_2048: np.ndarray,
    claimed: list[np.ndarray],
    *,
    threshold: float = 0.5,
    orphan_min_area: int = 256,
    max_overlap: float = 0.10,
) -> list[np.ndarray]:
    """U-Net CCs with little overlap vs YOLO-refined masks (recover misses).

    Claimed pixels are subtracted so orphans never share pixels with refined masks.
    """
    binary = (prob_2048 >= threshold).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    if n == 0:
        return []

    claimed_union = np.zeros_like(binary)
    for m in claimed:
        claimed_union = np.maximum(claimed_union, (m > 0).astype(np.uint8))

    orphans: list[tuple[int, np.ndarray]] = []
    for idx in range(1, n + 1):
        comp = (labeled == idx).astype(np.uint8)
        # Drop any pixels already claimed by YOLO-refined instances
        exclusive = np.logical_and(comp, claimed_union == 0).astype(np.uint8)
        area = int(exclusive.sum())
        if area < orphan_min_area:
            continue
        orig_area = int(comp.sum())
        inter = orig_area - area
        overlap = inter / max(orig_area, 1)
        if overlap >= max_overlap:
            continue
        orphans.append((area, exclusive))
    orphans.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in orphans]


def make_masks_exclusive(
    masks: list[np.ndarray],
    *,
    min_area: int = 64,
) -> list[np.ndarray]:
    """Assign each pixel to at most one instance (greedy by area desc).

    Required for Kaggle: predictions of the same image must not overlap.
    """
    if not masks:
        return []
    ordered = sorted(masks, key=lambda m: int((m > 0).sum()), reverse=True)
    h, w = ordered[0].shape
    claimed = np.zeros((h, w), dtype=np.uint8)
    exclusive_list: list[np.ndarray] = []
    for m in ordered:
        binary = (m > 0).astype(np.uint8)
        kept = np.logical_and(binary, claimed == 0).astype(np.uint8)
        area = int(kept.sum())
        if area < min_area:
            continue
        exclusive_list.append(kept)
        claimed = np.maximum(claimed, kept)
    exclusive_list.sort(key=lambda m: int(m.sum()), reverse=True)
    return exclusive_list


def hybrid_instances(
    prob: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    out_size: int = 2048,
    box_from_size: tuple[int, int] = (2048, 2048),
    pad: int = 16,
    threshold: float = 0.5,
    min_area: int = 64,
    orphan_min_area: int = 256,
    max_orphan_overlap: float = 0.10,
    use_orphans: bool = True,
) -> list[np.ndarray]:
    """Full hybrid: U-Net prob + YOLO boxes → instance masks at out_size.

    Args:
        prob: U-Net probability at model resolution (e.g. 768) or already out_size.
        boxes_xyxy: detector boxes in box_from_size coordinates (usually native image).
        box_from_size: (width, height) for boxes_xyxy.

    Returns:
        Non-overlapping instance masks (pixel-exclusive).
    """
    prob_2048 = upsample_prob(prob, out_size=out_size)
    boxes = scale_boxes_xyxy(
        boxes_xyxy, from_size=box_from_size, to_size=out_size, pad=pad
    )
    refined = refine_boxes_with_unet(
        prob_2048, boxes, threshold=threshold, min_area=min_area
    )
    if use_orphans:
        orphans = orphan_components(
            prob_2048,
            refined,
            threshold=threshold,
            orphan_min_area=orphan_min_area,
            max_overlap=max_orphan_overlap,
        )
        all_inst = refined + orphans
    else:
        all_inst = refined
    return make_masks_exclusive(all_inst, min_area=min_area)
