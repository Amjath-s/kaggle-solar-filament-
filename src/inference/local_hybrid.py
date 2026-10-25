"""V6-A/B: YOLO boxes + local U-Net crop refine + V2 orphans + exclusive."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.inference.hybrid import make_masks_exclusive, orphan_components, upsample_prob
from src.inference.local_crops import expand_xyxy, paste_mask_to_canvas


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


def _largest_cc_area(binary: np.ndarray) -> int:
    cc = _largest_cc(binary)
    return 0 if cc is None else int(cc.sum())


@torch.no_grad()
def predict_local_prob(
    model: torch.nn.Module,
    crop_u8: np.ndarray,
    *,
    size: int = 512,
    device: torch.device,
) -> np.ndarray:
    """Run local U-Net on a grayscale crop; return probability HxW at `size`."""
    img = Image.fromarray(crop_u8, mode="L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)[0, 0].float().cpu().numpy()


@torch.no_grad()
def predict_local_mask(
    model: torch.nn.Module,
    crop_u8: np.ndarray,
    *,
    size: int = 512,
    threshold: float = 0.5,
    device: torch.device,
) -> np.ndarray:
    """Run local U-Net on a grayscale crop; return binary HxW at `size`."""
    prob = predict_local_prob(model, crop_u8, size=size, device=device)
    return (prob >= threshold).astype(np.uint8)


def crop_fg_stats(prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Foreground stats on a local probability crop (no GT)."""
    binary = (prob >= threshold).astype(np.uint8)
    crop_pixels = int(prob.size)
    fg_pixels = int(binary.sum())
    fg_frac = float(fg_pixels / max(crop_pixels, 1))
    cc_area = _largest_cc_area(binary)
    if fg_pixels > 0:
        mean_fg_prob = float(prob[binary > 0].mean())
    else:
        mean_fg_prob = 0.0
    max_prob = float(prob.max()) if prob.size else 0.0
    return {
        "fg_pixels": fg_pixels,
        "crop_pixels": crop_pixels,
        "fg_frac": fg_frac,
        "largest_cc_area": cc_area,
        "mean_fg_prob": mean_fg_prob,
        "max_prob": max_prob,
    }


def refine_boxes_with_local_unet(
    image_u8: np.ndarray,
    boxes_xyxy: np.ndarray,
    model: torch.nn.Module,
    *,
    pad_frac: float = 0.20,
    size: int = 512,
    threshold: float = 0.5,
    min_area: int = 64,
    device: torch.device,
    min_fg_frac: float | None = None,
    min_cc_area: int | None = None,
    min_mean_fg_prob: float | None = None,
    return_stats: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[dict]]:
    """One instance per box via local crop U-Net → paste → largest CC."""
    h, w = image_u8.shape[:2]
    instances: list[tuple[int, np.ndarray]] = []
    stats_out: list[dict] = []
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        empty: list[np.ndarray] = []
        return (empty, []) if return_stats else empty

    for bi, box in enumerate(np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)):
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
        exp = expand_xyxy([x1, y1, x2, y2], pad_frac, w, h)
        crop = image_u8[exp[1] : exp[3], exp[0] : exp[2]]
        if crop.size == 0:
            continue
        prob = predict_local_prob(model, crop, size=size, device=device)
        stats = crop_fg_stats(prob, threshold=threshold)
        stats["box_index"] = bi
        stats["box"] = [x1, y1, x2, y2]
        stats["exp_box"] = exp.tolist() if hasattr(exp, "tolist") else list(exp)

        if min_fg_frac is not None and stats["fg_frac"] < float(min_fg_frac):
            stats["kept"] = False
            stats["drop_reason"] = "min_fg_frac"
            if return_stats:
                stats_out.append(stats)
            continue
        if min_cc_area is not None and stats["largest_cc_area"] < int(min_cc_area):
            stats["kept"] = False
            stats["drop_reason"] = "min_cc_area"
            if return_stats:
                stats_out.append(stats)
            continue
        if min_mean_fg_prob is not None and stats["mean_fg_prob"] < float(min_mean_fg_prob):
            stats["kept"] = False
            stats["drop_reason"] = "min_mean_fg_prob"
            if return_stats:
                stats_out.append(stats)
            continue

        local_bin = (prob >= threshold).astype(np.uint8)
        canvas = paste_mask_to_canvas(local_bin, exp, h, w)
        roi = np.zeros_like(canvas)
        roi[exp[1] : exp[3], exp[0] : exp[2]] = canvas[exp[1] : exp[3], exp[0] : exp[2]]
        cc = _largest_cc(roi)
        if cc is None:
            stats["kept"] = False
            stats["drop_reason"] = "no_cc"
            if return_stats:
                stats_out.append(stats)
            continue
        area = int(cc.sum())
        if area < min_area:
            stats["kept"] = False
            stats["drop_reason"] = "min_area"
            if return_stats:
                stats_out.append(stats)
            continue
        stats["kept"] = True
        stats["instance_area"] = area
        if return_stats:
            stats_out.append(stats)
        instances.append((area, cc, stats if return_stats else None))

    instances.sort(key=lambda t: t[0], reverse=True)
    masks = [m for _, m, _ in instances]
    if return_stats:
        # Dropped stats first (original order), then kept stats in sorted mask order
        dropped = [s for s in stats_out if not s.get("kept")]
        kept_sorted = []
        for rank, (_, _, st) in enumerate(instances):
            st = dict(st)
            st["instance_index"] = rank
            kept_sorted.append(st)
        return masks, dropped + kept_sorted
    return masks


def local_hybrid_instances(
    image_u8: np.ndarray,
    boxes_xyxy: np.ndarray,
    local_model: torch.nn.Module,
    v2_prob: np.ndarray,
    *,
    device: torch.device,
    out_size: int = 2048,
    pad_frac: float = 0.20,
    local_size: int = 512,
    threshold: float = 0.5,
    orphan_threshold: float | None = None,
    min_area: int = 64,
    orphan_min_area: int = 256,
    max_orphan_overlap: float = 0.10,
    use_orphans: bool = True,
    min_fg_frac: float | None = None,
    min_cc_area: int | None = None,
    min_mean_fg_prob: float | None = None,
) -> list[np.ndarray]:
    """V6 hybrid: local refine + V2 orphans + exclusive masks."""
    h, w = image_u8.shape[:2]
    if h != out_size or w != out_size:
        img_pil = Image.fromarray(image_u8, mode="L").resize((out_size, out_size), Image.BILINEAR)
        image_u8 = np.asarray(img_pil, dtype=np.uint8)
        sx = out_size / float(w)
        sy = out_size / float(h)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        h = w = out_size
    else:
        boxes = (
            np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
            if len(boxes_xyxy)
            else np.zeros((0, 4), np.float32)
        )

    refined = refine_boxes_with_local_unet(
        image_u8,
        boxes,
        local_model,
        pad_frac=pad_frac,
        size=local_size,
        threshold=threshold,
        min_area=min_area,
        device=device,
        min_fg_frac=min_fg_frac,
        min_cc_area=min_cc_area,
        min_mean_fg_prob=min_mean_fg_prob,
    )

    othr = float(threshold if orphan_threshold is None else orphan_threshold)
    if use_orphans:
        prob_2048 = upsample_prob(v2_prob, out_size=out_size)
        orphans = orphan_components(
            prob_2048,
            refined,
            threshold=othr,
            orphan_min_area=orphan_min_area,
            max_overlap=max_orphan_overlap,
        )
        all_inst = refined + orphans
    else:
        all_inst = refined
    return make_masks_exclusive(all_inst, min_area=min_area)
