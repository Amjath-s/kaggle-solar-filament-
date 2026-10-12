"""V61 OOF toolkit: 4-view flip generation, consensus, direct (no-refiner) PQ, bootstrap.

The V17 detector path (kaggle_inference/v17_compose.generate_views + detector_tta.cluster_views)
is REPLICATED here instead of imported because that module needs the Kaggle runtime. The replica
is pinned by `scripts/validate_v61_views.py`, which regenerates views with the pinned V16 seed-1
detector and compares masks/scores to the signed V17 caches BEFORE any new-model number is used.

Direct-detector scoring (no V11 refiner): the screen measures the detector, and the refiner adds
a stable gain measured earlier (+0.0157/+0.0160 tune for two independent detectors). Scoring
order matches V17: consensus -> pixel ownership by descending score -> confidence cut -> PQ via
the per-annotator-record iou>0.5 counts of optimize_inference.pq_counts.
"""
from __future__ import annotations

import gzip
import json

import numpy as np

FULL = 2048
MIN_AREA = 20
CONF_FLOOR = 0.1
FLIPS = ((), (1,), (0,), (0, 1))
MATCH_IOU = 0.5
POLICY = "flip_consensus"
CONFIDENCES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def flip_array(array, axes):
    return np.ascontiguousarray(np.flip(array, axis=axes)) if axes else array.copy()


def encode(mask):
    from pycocotools import mask as mask_utils
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def generate_views(model, image_path, iou=0.3, imgsz=1024, retina=True):
    """Four flip views -> [(rles, scores)] in original coordinates (V17's generate_views)."""
    import cv2
    from pycocotools import mask as mask_utils

    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (FULL, FULL):
        raise ValueError(f"bad image {image_path}")
    views = []
    for axes in FLIPS:
        result = model.predict(str(image_path) if not axes else flip_array(image, axes),
                               imgsz=imgsz, conf=CONF_FLOOR, agnostic_nms=True,
                               retina_masks=retina, iou=iou, verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != image.shape[:2]:
                    raise ValueError("masks must be in original image coordinates")
                rles.append(encode(restored))
        keep = [i for i, r in enumerate(rles) if mask_utils.area(r) > 0]
        views.append(([rles[i] for i in keep], [scores[i] for i in keep]))
    return views


def read_cache_views(path):
    """Read a V17-format signed cache file into the same structure (bytes counts)."""
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    return [([dict(size=list(r["size"]), counts=r["counts"].encode()) for r in v["rles"]],
             v["scores"]) for v in raw]


def cluster_views(views, match_iou=MATCH_IOU):
    """flip_consensus: strongest mask per cross-view group, score = sum/len(views)."""
    from pycocotools import mask as mask_utils

    proposals = []
    for view_id, (rles, scores) in enumerate(views):
        proposals.extend((rle, float(s), view_id) for rle, s in zip(rles, scores, strict=True))
    proposals.sort(key=lambda item: -item[1])
    groups = []
    for rle, score, view_id in proposals:
        eligible = [i for i, g in enumerate(groups) if view_id not in g["views"]]
        match = None
        if eligible:
            overlaps = mask_utils.iou([rle], [groups[i]["rle"] for i in eligible],
                                      [0] * len(eligible))[0]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= match_iou:
                match = eligible[best]
        if match is None:
            groups.append(dict(rle=rle, scores=[score], views={view_id}))
        else:
            groups[match]["scores"].append(score)
            groups[match]["views"].add(view_id)
    scores = [sum(g["scores"]) / len(views) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -scores[i])
    return [groups[i]["rle"] for i in order], [scores[i] for i in order]


def own_pixels(rles, scores, min_area=MIN_AREA):
    """Descending-score pixel ownership (optimize_inference.clean_predictions)."""
    from pycocotools import mask as mask_utils

    if not rles:
        return [], np.asarray([])
    occupied = None
    kept, kept_scores = [], []
    for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
        m = mask_utils.decode(rles[i]).astype(bool)
        if occupied is None:
            occupied = np.zeros_like(m)
        m &= ~occupied
        if m.sum() < min_area:
            continue
        occupied |= m
        kept.append(encode(m))
        kept_scores.append(scores[i])
    return kept, np.asarray(kept_scores)


def pq_counts(iou, n_gt, n_pred):
    """Identical to optimize_inference.pq_counts."""
    if not n_gt or not n_pred:
        return 0.0, 0, n_pred, n_gt
    hits = iou > 0.5
    return (float(iou[hits].sum()), int(hits.sum()),
            int((~hits.any(axis=0)).sum()), int((~hits.any(axis=1)).sum()))


def pq_value(total):
    iou, tp, fp, fn = total
    d = tp + 0.5 * (fp + fn)
    return float(iou / d) if d else 0.0


def score_file(rles, scores, gt_by_record, confidences=CONFIDENCES):
    """Per-confidence counts for one file. gt_by_record: list of GT RLE lists (one per record)."""
    from pycocotools import mask as mask_utils

    out = {}
    scores = np.asarray(scores)
    for c in confidences:
        keep = [r for r, s in zip(rles, scores, strict=True) if s >= c]
        total = np.zeros(4)
        for gt in gt_by_record:
            iou = (mask_utils.iou(keep, gt, [0] * len(gt)).T if keep and gt
                   else np.zeros((len(gt), len(keep))))
            total += pq_counts(iou, len(gt), len(keep))
        out[c] = total
    return out


def bootstrap_pq(per_file_counts, n=2000, seed=61):
    """95% CI of pooled PQ by resampling FILES. per_file_counts: (F,4) array of iou,tp,fp,fn."""
    arr = np.asarray(per_file_counts, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(arr), len(arr))
        vals.append(pq_value(arr[idx].sum(0)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
