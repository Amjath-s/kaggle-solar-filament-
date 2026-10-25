"""Detector-guided local crop geometry for V6-A."""

from __future__ import annotations

import numpy as np
from PIL import Image


def expand_xyxy(
    box_xyxy: np.ndarray | list[float],
    pad_frac: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Expand xyxy by pad_frac on each side, clamp to image bounds.

    Returns int xyxy [x1, y1, x2, y2] inclusive-ish for slicing (x2,y2 exclusive
    when used as crop_arr[y1:y2, x1:x2] with y2/x2 as exclusive ends — we return
    exclusive-style ends as ceil of expanded box, clamped to [0,W]/[0,H].
    """
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box_xyxy, dtype=np.float64).reshape(4)]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    px = bw * float(pad_frac)
    py = bh * float(pad_frac)
    nx1 = int(np.floor(x1 - px))
    ny1 = int(np.floor(y1 - py))
    nx2 = int(np.ceil(x2 + px))
    ny2 = int(np.ceil(y2 + py))
    nx1 = max(0, min(nx1, width - 1))
    ny1 = max(0, min(ny1, height - 1))
    nx2 = max(nx1 + 1, min(nx2, width))
    ny2 = max(ny1 + 1, min(ny2, height))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def mask_to_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    # exclusive-style max+1 for expand_xyxy width/height
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], dtype=np.int32)


def crop_region(
    image: np.ndarray | Image.Image,
    box_xyxy: np.ndarray,
) -> np.ndarray:
    """Crop HxW (or HxWxC) array / PIL image with exclusive x2,y2."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = image
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    return arr[y1:y2, x1:x2].copy()


def resize_image_mask(
    image_hw: np.ndarray,
    mask_hw: np.ndarray | None,
    size: int = 512,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize crop image (bilinear) and optional mask (nearest) to size×size."""
    img = Image.fromarray(image_hw if image_hw.dtype == np.uint8 else (image_hw * 255).astype(np.uint8))
    if img.mode != "L":
        img = img.convert("L")
    img_r = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    mask_r = None
    if mask_hw is not None:
        m = Image.fromarray((mask_hw > 0).astype(np.uint8) * 255, mode="L")
        mask_r = (np.asarray(m.resize((size, size), Image.NEAREST)) > 127).astype(np.float32)
    return img_r, mask_r


def paste_mask_to_canvas(
    local_mask: np.ndarray,
    box_xyxy: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Resize local binary/float mask to crop box size and paste onto out canvas."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    ch, cw = y2 - y1, x2 - x1
    if ch <= 0 or cw <= 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)
    m = (local_mask > 0.5).astype(np.uint8) if local_mask.dtype != np.uint8 else (local_mask > 0).astype(np.uint8)
    m_img = Image.fromarray(m * 255, mode="L").resize((cw, ch), Image.NEAREST)
    m_resized = (np.asarray(m_img) > 127).astype(np.uint8)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    canvas[y1:y2, x1:x2] = m_resized
    return canvas


def crop_pair_to_tensorsize(
    image_hw: np.ndarray,
    mask_hw: np.ndarray,
    box_xyxy: np.ndarray,
    size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop + resize; returns float image [0,1], float mask {0,1}, used box."""
    crop_img = crop_region(image_hw, box_xyxy)
    crop_mask = crop_region(mask_hw, box_xyxy)
    img_r, mask_r = resize_image_mask(crop_img, crop_mask, size=size)
    assert mask_r is not None
    return img_r, mask_r, box_xyxy
