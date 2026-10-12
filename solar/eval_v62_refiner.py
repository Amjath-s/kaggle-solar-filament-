"""V62 refiner test: baseline+V11 refiner vs p2first+V11 refiner (141 canonical held-out files).

The V11 refiner (and the V6/V12/V13/V17 line) was trained on the canonical 566 FIT files. The other
141 canonical files (94 tune + 47 audit) were never seen by it. Those 141 files are spread over the
five OOF folds, so for each of them the detector proposals come from a model that also never saw
the file (OOF). On that subset BOTH detector and refiner are unseen-data, so the comparison

    A: OOF baseline proposals  -> V11 refiner -> V17 scoring
    B: OOF p2first proposals   -> V11 refiner -> V17 scoring

is clean. (On the other 566 files the refiner has seen the labels: do not use them.)

Mirrors `reproduce_v17_from_cache.evaluate` exactly, except proposals are read from the V61 OOF
cache instead of the signed V17 caches:  cluster_views(flip_consensus) -> crop_probabilities ->
clean_refined_predictions(mode=refined_constrained) at thresholds t in THRESHOLDS and confidence
cuts c in CONFIDENCES; counts summed per annotator record with core.pq_counts.

Reports PQ grid per recipe, the V17-recipe cell (t0.45/c0.30) and each recipe's best cell, plus a
file-bootstrap of B-A on the same files. Selection of the cell is on these same files, so the
headline comparison is at the FIXED V17 cell (t0.45, c0.30) and at the matched best cell.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v62_refiner.py --recipes baseline p2first \
      --out outputs/v62_refiner_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_oof as O  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import v61_toolkit as K  # noqa: E402

THRESHOLDS = v50.THRESHOLDS
CONFIDENCES = v50.CONFIDENCES
V17_CELL = (0.45, 0.30)


def fold_of_file():
    return {n: int(k) for k, names in O.load_folds().items() for n in names}


def file_counts(rt, recipe, fold, name, gray, gt_by_image):
    """-> {(t, c): np.array(4)} for one file using cached OOF views of `recipe`."""
    from pycocotools import mask as mask_utils

    core, tta = rt["core"], rt["tta"]
    views = K.read_cache_views(O.cache_path(recipe, fold, name))
    proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
    out = {}
    if proposals:
        masks = [mask_utils.decode(r) for r in proposals]
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
        refined = {t: core.clean_refined_predictions(masks, scores, bounds, probs, t, v50.MODE)
                   for t in THRESHOLDS}
    else:
        refined = {t: ([], np.asarray([])) for t in THRESHOLDS}
    for t in THRESHOLDS:
        rles, final_scores = refined[t]
        for c in CONFIDENCES:
            keep = (np.asarray(final_scores) >= c if len(final_scores) else np.zeros(0, bool))
            kept = [r for r, flag in zip(rles, keep, strict=True) if flag]
            total = np.zeros(4)
            for gt in gt_by_image.values():
                iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                       else np.zeros((len(gt), len(kept))))
                total += core.pq_counts(iou, len(gt), len(kept))
            out[(t, c)] = total
    return out


def main():
    import cv2
    from pycocotools import mask as mask_utils

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipes", nargs="+", default=["baseline", "p2first"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N files only")
    args = ap.parse_args()

    rt = v50.runtime_state()
    core, base = rt["core"], rt["base"]
    held, tune, audit, _, by_file = rt["split"]
    canon = sorted(held)
    if len(canon) != 141:
        raise RuntimeError("canonical held-out population changed")
    fmap = fold_of_file()
    if args.limit:
        canon = canon[: args.limit]
    per = {r: {} for r in args.recipes}
    for i, name in enumerate(canon, 1):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [core.annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_image[image_id] = [r for r in gt if mask_utils.area(r) > 0]
        for r in args.recipes:
            per[r][name] = file_counts(rt, r, fmap[name], name, gray, gt_by_image)
        if i % 10 == 0:
            print(f"{i}/{len(canon)} {name}", flush=True)

    names = sorted(per[args.recipes[0]])
    cells = [(t, c) for t in THRESHOLDS for c in CONFIDENCES]
    rep = {"files": len(names), "recipes": {}, "reads_test": False,
           "note": "canonical 141 held-out files: unseen by refiner AND (OOF) by detector"}
    for r in args.recipes:
        grid = {}
        for cell in cells:
            tot = sum(per[r][n][cell] for n in names)
            grid[f"t{cell[0]:.2f}_c{cell[1]:.2f}"] = {
                "pq": K.pq_value(tot), "tp": int(tot[1]), "fp": int(tot[2]), "fn": int(tot[3])}
        best = max(grid, key=lambda k: grid[k]["pq"])
        rep["recipes"][r] = {"grid": grid, "best_cell": best, "best": grid[best],
                             "v17_cell": grid[f"t{V17_CELL[0]:.2f}_c{V17_CELL[1]:.2f}"]}
    if len(args.recipes) == 2:
        a, b = args.recipes
        rng = np.random.default_rng(62)
        key17 = V17_CELL

        def arr(r, cell):
            return np.asarray([per[r][n][cell] for n in names])

        def boot(ca, cb):
            d = []
            for _ in range(2000):
                idx = rng.integers(0, len(ca), len(ca))
                d.append(K.pq_value(cb[idx].sum(0)) - K.pq_value(ca[idx].sum(0)))
            return {"delta": K.pq_value(cb.sum(0)) - K.pq_value(ca.sum(0)),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "p_gt_0": float(np.mean(np.asarray(d) > 0))}
        ba = tuple(float(x) for x in rep["recipes"][a]["best_cell"].replace("t", "").replace(
            "c", "").split("_"))
        bb = tuple(float(x) for x in rep["recipes"][b]["best_cell"].replace("t", "").replace(
            "c", "").split("_"))
        rep["paired"] = {"at_v17_cell": boot(arr(a, key17), arr(b, key17)),
                         "each_own_best_cell": boot(arr(a, ba), arr(b, bb))}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({r: {k: v for k, v in rep["recipes"][r].items() if k != "grid"}
                      for r in args.recipes}, indent=2))
    print(json.dumps(rep.get("paired"), indent=2))


if __name__ == "__main__":
    main()
