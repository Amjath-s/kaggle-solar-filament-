"""Segmentation losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


class FocalDiceLoss(nn.Module):
    """Focal BCE (down-weights easy background pixels, focuses on thin edges) + Dice."""

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5, gamma: float = 2.0,
                 alpha: float = 0.25) -> None:
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        a_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets) if self.alpha >= 0 else 1.0
        focal = (a_t * (1.0 - p_t) ** self.gamma * bce).mean()
        return self.focal_weight * focal + self.dice_weight * self.dice(logits, targets)


class TverskyLoss(nn.Module):
    """Tversky loss; β>α penalizes FN more (recall-friendly for thin filaments)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()


class V8ASegLoss(nn.Module):
    """0.50 Tversky + 0.30 Dice + 0.20 boundary BCE (aux head)."""

    def __init__(
        self,
        tversky_weight: float = 0.50,
        dice_weight: float = 0.30,
        boundary_weight: float = 0.20,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.tversky_weight = float(tversky_weight)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        boundary_logits: torch.Tensor | None = None,
        boundary_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.tversky_weight * self.tversky(main_logits, targets) + self.dice_weight * self.dice(
            main_logits, targets
        )
        if boundary_logits is not None and boundary_targets is not None:
            loss = loss + self.boundary_weight * self.bce(boundary_logits, boundary_targets)
        return loss
