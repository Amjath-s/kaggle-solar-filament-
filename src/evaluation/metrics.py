"""Validation metrics for binary masks."""

from __future__ import annotations

import torch


@torch.no_grad()
def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    preds_f = preds.reshape(preds.size(0), -1)
    targets_f = targets.reshape(targets.size(0), -1)
    inter = (preds_f * targets_f).sum(dim=1)
    pred_sum = preds_f.sum(dim=1)
    tgt_sum = targets_f.sum(dim=1)

    dice = ((2 * inter + eps) / (pred_sum + tgt_sum + eps)).mean().item()
    union = pred_sum + tgt_sum - inter
    iou = ((inter + eps) / (union + eps)).mean().item()
    return dice, iou
