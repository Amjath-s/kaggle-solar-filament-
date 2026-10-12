from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util


def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def get_overlap_matrices(
    gt_layers: torch.Tensor,
    pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0.0), intersection / union)
    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0.0),
        2 * intersection / (gt_areas + pred_areas),
    )
    return iou_matrix, dice_matrix


def _image_key(series: pd.Series, ground_truth: bool) -> pd.Series:
    image_ids = series.str.split("_", n=1).str[0]
    if ground_truth:
        image_ids = image_ids.str.split("-", n=1).str[-1]
    return image_ids


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    _image_key(gt_df["filament_id"], ground_truth=True)  # validates ids
    pred_image_ids = _image_key(pred_df["filament_id"], ground_truth=False)
    rows = []

    for annotator_image in gt_df["filament_id"].str.split("_", n=1).str[0].unique():
        image_id = annotator_image.split("-", maxsplit=1)[-1]
        gt_rles = gt_df.loc[
            gt_df["filament_id"].str.split("_", n=1).str[0] == annotator_image,
            "segmentation_rle",
        ].tolist()
        pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
        gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
        rows.append(
            {
                "annotator_image": annotator_image,
                "iou_matrix": iou_matrix,
                "dice_matrix": dice_matrix,
                "n_gt": len(gt_rles),
                "n_pred": len(pred_rles),
            }
        )

    return pd.DataFrame(rows)


def _fp_count(hit_matrix: torch.Tensor) -> int:
    return int((hit_matrix.sum(dim=0) == 0).sum().item())


def _fn_count(hit_matrix: torch.Tensor) -> int:
    return int((hit_matrix.sum(dim=1) == 0).sum().item())


def pq_score(overlap_df: pd.DataFrame, iou_threshold: float = 0.5) -> float:
    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            fp_count += row.n_pred
            continue
        if row.n_pred == 0:
            fn_count += row.n_gt
            continue

        hit_matrix = row.iou_matrix > iou_threshold
        tp_iou_scores.extend(row.iou_matrix[hit_matrix].tolist())
        fp_count += _fp_count(hit_matrix)
        fn_count += _fn_count(hit_matrix)

    denominator = len(tp_iou_scores) + 0.5 * fp_count + 0.5 * fn_count
    return float(sum(tp_iou_scores) / denominator) if denominator > 0 else 0.0


def relation_counts(overlap_df: pd.DataFrame) -> tuple[Counter, Counter]:
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []
    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            pred_match_counts.extend([0] * row.n_pred)
            continue
        if row.n_pred == 0:
            gt_match_counts.extend([0] * row.n_gt)
            continue
        hit_matrix = row.iou_matrix > 0
        gt_match_counts.extend(hit_matrix.sum(dim=1).tolist())
        pred_match_counts.extend(hit_matrix.sum(dim=0).tolist())
    return Counter(gt_match_counts), Counter(pred_match_counts)

