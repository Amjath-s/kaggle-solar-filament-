"""Instance Panoptic Quality and matched Dice helpers."""

from __future__ import annotations

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float((inter + eps) / (union + eps))


def mask_dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int(np.logical_and(a, b).sum())
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def compute_pq(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """Standard PQ matching: greedy IoU > threshold, one-to-one.

    Returns PQ, SQ, RQ, TP/FP/FN, and mean IoU/Dice over matched pairs.
    """
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        return {
            "pq": 1.0,
            "sq": 1.0,
            "rq": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "mean_matched_iou": 1.0,
            "mean_matched_dice": 1.0,
        }
    if n_gt == 0:
        return {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "mean_matched_iou": 0.0,
            "mean_matched_dice": 0.0,
        }
    if n_pred == 0:
        return {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "mean_matched_iou": 0.0,
            "mean_matched_dice": 0.0,
        }

    # IoU matrix
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            iou_mat[i, j] = mask_iou(g, p)

    # Greedy match by descending IoU
    pairs: list[tuple[int, int, float]] = []
    used_gt = set()
    used_pred = set()
    flat = [
        (iou_mat[i, j], i, j)
        for i in range(n_gt)
        for j in range(n_pred)
        if iou_mat[i, j] > iou_threshold
    ]
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        pairs.append((i, j, float(iou)))

    tp = len(pairs)
    fp = n_pred - tp
    fn = n_gt - tp

    if tp == 0:
        sq = 0.0
        mean_dice = 0.0
    else:
        sq = float(np.mean([p[2] for p in pairs]))
        mean_dice = float(
            np.mean([mask_dice(gt_masks[i], pred_masks[j]) for i, j, _ in pairs])
        )

    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    pq = sq * rq

    return {
        "pq": float(pq),
        "sq": float(sq),
        "rq": float(rq),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_matched_iou": float(sq),
        "mean_matched_dice": float(mean_dice),
    }
