"""V61 OOF evaluator: held-out inference for one fold + pooled PQ across folds.

  infer : run a fold's final detector on ITS held-out files only, cache the four signed flip
          views per file (V17 cell: iou 0.3, retina, agnostic NMS, imgsz 1024, conf floor 0.1).
  pool  : read cached views for every fold of a recipe -> consensus -> pixel ownership ->
          per-confidence counts per file (per-annotator-record V17 scoring) -> pooled PQ over
          all 707 files, per-fold PQ, and a file-bootstrap 95% CI. Direct detector only
          (no V11 refiner) - this is the detector screen.
  diff  : paired comparison of two recipes on the SAME files (bootstrap of per-file deltas).

Selection of the confidence cut is made on pooled OOF for the recipe being scored, from the
declared grid K.CONFIDENCES (8 values) - reported, not tuned per fold.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v61_oof.py infer --fold 0 --recipe baseline \
      --weights /mnt/v61/baseline_fold0/mask_ratio1/final.pt
  python scripts/eval_v61_oof.py pool --recipe baseline --out outputs/v61_oof_baseline.json
  python scripts/eval_v61_oof.py diff --a baseline --b <recipe> --out outputs/v61_diff.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v61_toolkit as K  # noqa: E402

ROOT = SCRIPTS.parent
DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = DATA_ROOT / "train/train_images"
CACHE = ROOT / "outputs/v61_oof_cache"
ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"


def cache_path(recipe, fold, name):
    return CACHE / recipe / f"fold{fold}" / (Path(name).stem + ".json.gz")


def write_views(path, views):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump([dict(rles=[dict(size=list(r["size"]), counts=r["counts"].decode())
                              for r in rles], scores=scores) for rles, scores in views], f)
    tmp.replace(path)


def load_folds(folds_path=ROOT / "configs/v61_folds.json"):
    m = json.loads(Path(folds_path).read_text())
    if m["annotation_sha256"] != ANNOTATION_SHA256:
        raise RuntimeError("fold manifest annotation mismatch")
    return {int(k): v for k, v in m["folds"].items()}


def gt_by_file(document):
    from pycocotools import mask as mask_utils

    id_file = {i["id"]: i for i in document["images"]}
    per_image = {}
    for a in document["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    out = {}
    for image_id, info in id_file.items():
        rles = []
        for a in per_image.get(image_id, []):
            r = mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], info["height"],
                                                        info["width"]))
            if mask_utils.area(r) > 0:
                rles.append(r)
        out.setdefault(info["file_name"], []).append(rles)
    return out


def cmd_infer(args):
    import hashlib

    from ultralytics import YOLO

    if hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest() != ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    held = load_folds()[args.fold]
    model = YOLO(str(args.weights))
    for i, name in enumerate(held, 1):
        p = cache_path(args.recipe, args.fold, name)
        if p.exists():
            continue
        if args.image_dir and not (args.image_dir / name).exists():
            continue  # V69: no neighbour frame for this file in this bucket
        write_views(p, K.generate_views(model, (args.image_dir or IMAGE_DIR) / name, iou=0.3, imgsz=args.imgsz))
        if i % 20 == 0:
            print(f"fold{args.fold} {i}/{len(held)}", flush=True)
    print(f"fold{args.fold} done {len(held)} files")


def file_counts(recipe, fold, name, gt):
    views = K.read_cache_views(cache_path(recipe, fold, name))
    rles, scores = K.cluster_views(views)
    rles, scores = K.own_pixels(rles, scores)
    return K.score_file(rles, scores, gt)


def pooled(recipe):
    """-> {conf: (F,4) per-file counts aligned to `files`}, files, fold_of_file."""
    doc = json.loads(ANNOTATIONS.read_text())
    gts = gt_by_file(doc)
    folds = load_folds()
    files, fold_of, rows = [], [], {c: [] for c in K.CONFIDENCES}
    for fold, names in sorted(folds.items()):
        for name in names:
            res = file_counts(recipe, fold, name, gts[name])
            files.append(name)
            fold_of.append(fold)
            for c in K.CONFIDENCES:
                rows[c].append(res[c])
    return {c: np.asarray(v) for c, v in rows.items()}, files, np.asarray(fold_of)


def cmd_pool(args):
    rows, files, fold_of = pooled(args.recipe)
    table = {}
    for c, arr in rows.items():
        tot = arr.sum(0)
        table[f"{c:.2f}"] = {"pq": K.pq_value(tot), "iou_sum": tot[0], "tp": int(tot[1]),
                             "fp": int(tot[2]), "fn": int(tot[3]),
                             "per_fold_pq": [K.pq_value(arr[fold_of == f].sum(0))
                                             for f in sorted(set(fold_of.tolist()))]}
    best = max(table, key=lambda k: table[k]["pq"])
    lo, hi = K.bootstrap_pq(rows[float(best)])
    rep = {"recipe": args.recipe, "files": len(files), "grid": table, "best_conf": best,
           "best_pq": table[best]["pq"], "best_ci95": [lo, hi],
           "best_per_fold_pq": table[best]["per_fold_pq"],
           "per_fold_sd": float(np.std(table[best]["per_fold_pq"])), "reads_test": False}
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "grid"}, indent=2))


def cmd_diff(args):
    ra, files, _ = pooled(args.a)
    rb, files_b, _ = pooled(args.b)
    if files != files_b:
        raise RuntimeError("recipes cover different files")
    ca, cb = float(args.conf_a), float(args.conf_b)
    rng = np.random.default_rng(61)
    a, b = ra[ca], rb[cb]
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(a), len(a))
        deltas.append(K.pq_value(b[idx].sum(0)) - K.pq_value(a[idx].sum(0)))
    rep = {"a": args.a, "b": args.b, "conf_a": ca, "conf_b": cb,
           "pq_a": K.pq_value(a.sum(0)), "pq_b": K.pq_value(b.sum(0)),
           "delta": K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
           "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
           "p_delta_gt_0": float(np.mean(np.asarray(deltas) > 0))}
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--fold", type=int, required=True, choices=range(5))
    i.add_argument("--recipe", required=True)
    i.add_argument("--weights", type=Path, required=True)
    i.add_argument("--imgsz", type=int, default=1024)
    i.add_argument("--image-dir", type=Path, default=None, dest="image_dir",
                   help="V69: read same-named images from here (neighbour frames)")
    p = sub.add_parser("pool")
    p.add_argument("--recipe", required=True)
    p.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("diff")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--conf-a", default="0.30", dest="conf_a")
    d.add_argument("--conf-b", default="0.30", dest="conf_b")
    d.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"infer": cmd_infer, "pool": cmd_pool, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()
