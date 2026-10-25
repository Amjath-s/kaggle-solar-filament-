"""Tile extraction and center-weighted stitch for V8-A."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data.v8_tile_dataset import soft_tile_weights, _tile_origins


@torch.no_grad()
def predict_stitched(
    model: nn.Module,
    image_1xhw: torch.Tensor,
    device: torch.device,
    tile_size: int = 1152,
    overlap: int = 128,
    use_amp: bool = True,
) -> np.ndarray:
    """image: float tensor [1,H,W] or [1,1,H,W] in [0,1] → probability map HxW."""
    if image_1xhw.dim() == 3:
        image_1xhw = image_1xhw.unsqueeze(0)
    assert image_1xhw.dim() == 4 and image_1xhw.size(1) == 1
    _, _, h, w = image_1xhw.shape
    origins = _tile_origins(h, w, tile_size, overlap)
    weights = soft_tile_weights(tile_size, overlap)
    acc = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)

    model.eval()
    for y0, x0 in origins:
        tile = image_1xhw[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            out = model(tile)
            logits = out[0] if isinstance(out, tuple) else out
            prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        acc[y0 : y0 + tile_size, x0 : x0 + tile_size] += prob * weights
        wsum[y0 : y0 + tile_size, x0 : x0 + tile_size] += weights

    wsum = np.maximum(wsum, 1e-6)
    return acc / wsum
