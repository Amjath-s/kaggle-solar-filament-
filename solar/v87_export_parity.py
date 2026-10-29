"""V87 end-to-end export parity on fold-held-out TRAIN files (no cached views).

Runs the EXACT test-export path (v61_toolkit.generate_views on the image file -> export_v74.refine_image ->
export_v74 CSV writer) with the fold-k detector, on fold k's held-out train images, for two arms:
  v84: fold-k V84 soft refiner, t0.50, MIN200, fold-k cross-fitted demote (outputs/v84/export/parity)
  v65: fold-k V73 base (r3) refiner, t0.45, min-area 20, no demote (= V65 recipe OOF, V73 base k7)
Stage diffs per file: fresh views vs OOF cache views; per-file PQ counts vs the OOF references
(V84 bundle_ref.pkl; V73 base per_file k7/t0.45_c0.30). Writes outputs/v87/fold{k}_*.csv + .json.
  CUDA_VISIBLE_DEVICES=N python solar/v87_export_parity.py <fold> [limit]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import export_v74 as X  # noqa: E402
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v87"


def bare(src, dst):
    import torch
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save(sd["model"] if "model" in sd else sd, dst)
    return dst


def views_diff(a, b):
    """-> dict: identical (bool), per-view counts, max |score diff| on equal-count views, rle mismatches."""
    same, nmis, ds = True, 0, 0.0
    for (ra, sa), (rb, sb) in zip(a, b):
        if len(ra) != len(rb):
            same = False
            continue
        ds = max([ds] + [abs(x - y) for x, y in zip(sa, sb)])
        nmis += sum(x["counts"] != y["counts"] for x, y in zip(ra, rb))
    same = same and nmis == 0 and ds == 0.0
    return dict(identical=same, n_fresh=[len(v[0]) for v in a], n_cache=[len(v[0]) for v in b],
                max_dscore=ds, rle_mismatch=nmis)


def counts(core, preds, gts):
    from pycocotools import mask as mu
    tot = np.zeros(4)
    for g in gts:
        iou = np.asarray(mu.iou(g, preds, [0] * len(preds))) if g and preds else np.zeros((len(g), len(preds)))
        tot += core.pq_counts(iou, len(g), len(preds))
    return tot


def main():
    import torch
    from ultralytics import YOLO

    fold = int(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "tmp"
    tmp.mkdir(exist_ok=True)
    det = ROOT / f"work/v61/p2first_fold{fold}/mask_ratio1/final.pt"  # README training step 2
    arms = {
        "v84": dict(ref=bare(ROOT / f"outputs/v84/soft_fold{fold}_refiner.pt", tmp / f"soft{fold}.pt"),
                    args=Namespace(mask_threshold=0.50, confidence=0.30, min_area=200),
                    demote=ROOT / f"outputs/v84/export/parity/demote_fold{fold}.json"),
        "v65": dict(ref=bare(ROOT / f"outputs/v73/base_fold{fold}_refiner.pt", tmp / f"base{fold}.pt"),
                    args=Namespace(mask_threshold=0.45, confidence=0.30, min_area=20), demote=None),
    }
    rt = {}
    for a, c in arms.items():
        core, tta, base, ref = X.refiner_runtime(c["ref"], "cuda:0")
        rt[a] = (core, tta, base, ref, X.Demote(c["demote"], core) if c["demote"] else None)
    core = rt["v84"][0]
    ref84 = pickle.load(open(ROOT / "outputs/v84/export/parity/bundle_ref.pkl", "rb"))
    ref65 = json.loads((ROOT / f"outputs/v73/base_fold{fold}.json").read_text())["per_file"]
    gts = O.gt_by_file(json.loads(O.ANNOTATIONS.read_text()))
    names = O.load_folds()[fold]
    if limit:
        names = names[:limit]
    detector = YOLO(str(det))
    rows = {a: [] for a in arms}
    report, t0 = [], time.monotonic()
    for i, name in enumerate(names, 1):
        path = O.IMAGE_DIR / name
        fresh = K.generate_views(detector, path, iou=0.3, imgsz=1024)
        cache = K.read_cache_views(O.cache_path("p2first", fold, name))
        rec = dict(name=name, views=views_diff(fresh, cache))
        for a, c in arms.items():
            cc, tta, base, ref, dem = rt[a]
            with torch.no_grad():
                preds, kept, small, nd, _ = X.refine_image(cc, tta, base, ref, fresh, path, c["args"], dem)
                cnt = counts(core, preds, gts[name])
                want = np.asarray(ref84[name]["counts"] if a == "v84" else ref65[name]["k7"]["t0.45_c0.30"])
                rec[a] = dict(kept=len(preds), small=small, demoted=nd, counts=cnt.tolist(),
                              ref=want.tolist(), dcount=float(np.max(np.abs(cnt - want))))
                if not rec["views"]["identical"]:  # localize: same refine on the cached views
                    p2, *_ = X.refine_image(cc, tta, base, ref, cache, path, c["args"], dem)
                    rec[a]["counts_cacheviews"] = counts(core, p2, gts[name]).tolist()
            for n, rle in enumerate(preds, 1):  # export_v74.main writer
                rows[a].append({"filament_id": f"{path.stem}_{n}", "segmentation_rle": rle["counts"].decode()})
        report.append(rec)
        print(f"fold{fold} {i}/{len(names)} {time.monotonic() - t0:.0f}s views_identical={rec['views']['identical']} "
              f"d84={rec['v84']['dcount']:.4g} d65={rec['v65']['dcount']:.4g}", flush=True)
    summ = dict(fold=fold, files=len(report), detector=str(det),
                views_identical=sum(r["views"]["identical"] for r in report))
    for a in arms:
        pd.DataFrame(rows[a], columns=["filament_id", "segmentation_rle"]).to_csv(OUT / f"fold{fold}_{a}.csv", index=False)
        got = np.sum([r[a]["counts"] for r in report], 0)
        want = np.sum([r[a]["ref"] for r in report], 0)
        summ[a] = dict(pq_export=K.pq_value(got), pq_oof=K.pq_value(want), counts_export=got.tolist(),
                       counts_oof=want.tolist(), files_differing=sum(r[a]["dcount"] > 1e-6 for r in report))
    (OUT / f"fold{fold}.json").write_text(json.dumps(dict(summary=summ, files=report), indent=1, default=float) + "\n")
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
