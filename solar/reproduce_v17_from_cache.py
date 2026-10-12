"""Reproduce V17's tune PQ from its signed Kaggle proposal caches (no detector run).

V17 (public 0.39, submission 56103401) = V16 arm_r1 seed1 detector proposals @ box
iou 0.3, 4-view flip_consensus, refined by V11 proposal_centered_e4, candidate
t0.45_c0.30 -> tune counts [481.00931292355114, 704, 306, 430] (PQ 0.448703).

This reads the pulled caches in outputs/kaggle_v17_cache_pull/proposal_cache, verifies
each one's signature + SHA-256 + cache_id, and re-runs only the refiner + PQ scoring.
Detector/library drift is therefore excluded; only refiner numerics on this runtime
can differ. Tune only by default (audit stays sealed unless --group audit).

Usage (one GPU per shard):
  CUDA_VISIBLE_DEVICES=N python scripts/reproduce_v17_from_cache.py eval --shard N
  python scripts/reproduce_v17_from_cache.py merge
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
CACHE_DIR = ROOT / "outputs/kaggle_v17_cache_pull/proposal_cache"
V6_SOURCE = Path(__file__).resolve().parent  # solar_base.py + v6_provenance.json
CANONICAL_REFINER = ROOT / "weights/training/best_refiner_coco.pt"
V11_REFINER = ROOT / "outputs/kaggle_v11_full/checkpoints/proposal_centered_e4.pt"
OUTPUT = ROOT / "outputs/v50_v17_reproduction"

PINS = {
    "annotations": "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1",
    "provenance": "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122",
    "canonical_refiner": "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1",
    "v11_refiner": "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2",
}
DETECTORS = {
    "seed1": "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef",
    "seed2": "097a57ed7b5fe56909393ab05897ba5c41b96f7c343cb88d2ba782dd9dfabbe7",
}
CELLS = {
    "mr1_iou030": dict(name="mr1_iou030", retina_masks=True, agnostic_nms=True, iou=0.3),
    "mr1_iou070": dict(name="mr1_iou070", retina_masks=True, agnostic_nms=True, iou=0.7),
}
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
POLICY, MATCH_IOU, MODE = "flip_consensus", 0.5, "refined_constrained"
SHARDS = 4
SELECTED = "seed1/mr1_iou030/t0.45_c0.30"
EXPECTED_TUNE = [481.00931292355114, 704, 306, 430]
V17_RESULTS = ROOT / "outputs/kaggle_v17_cache_pull/v17_results.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()  # chunked: VM runs Python 3.10 (no hashlib.file_digest)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pq(counts) -> float:
    denominator = counts[1] + 0.5 * (counts[2] + counts[3])
    return float(counts[0] / denominator) if denominator else 0.0


def cache_id(name: str, split: str, cell: str, detector_sha: str) -> str:
    token = f"{detector_sha}|mr1|{cell}|{split}|{name}|imgsz1024|conf0.1"
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def read_signed_views(name: str, split: str, cell: str, detector_sha: str):
    """Exactly V17's generate_views() validation path, minus generation."""
    from pycocotools import mask as mask_utils

    path = CACHE_DIR / cache_id(name, split, cell, detector_sha)
    marker = path.with_suffix(".signature.json")
    expected = dict(cell=CELLS[cell], split=split, file_name=name, mask_ratio=1,
                    detector_sha256=detector_sha, resolution=1024,
                    confidence_floor=0.1, views=4)
    recorded = json.loads(marker.read_text())
    if recorded["signature"] != expected or recorded["sha256"] != sha256(path):
        raise RuntimeError(f"Signature/hash mismatch for {path.name} ({name})")
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if len(raw) != 4:
        raise RuntimeError(f"Incomplete four-view cache {path.name}")
    views = []
    for view in raw:
        rles = [dict(size=list(r["size"]), counts=r["counts"].encode()) for r in view["rles"]]
        scores = view["scores"]
        if len(rles) != len(scores) or any(not 0 <= s <= 1 for s in scores):
            raise RuntimeError(f"Invalid scores in {path.name}")
        if any(r["size"] != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
            raise RuntimeError(f"Invalid geometry in {path.name}")
        views.append((rles, scores))
    return views


def runtime_state():
    import torch

    sys.path.insert(0, str(ROOT / "kaggle_inference"))
    import detector_tta as tta
    import optimize_inference as core
    import v11_refiner as v11

    annotations = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    pinned = ((annotations, "annotations"), (V6_SOURCE / "v6_provenance.json", "provenance"),
              (CANONICAL_REFINER, "canonical_refiner")) + (
        ((V11_REFINER, "v11_refiner"),) if V11_REFINER.exists() else ())
    for path, key in pinned:
        if sha256(path) != PINS[key]:
            raise RuntimeError(f"Pinned input changed: {path}")
    base = core.load_base(V6_SOURCE / "solar_base.py")
    base.CFG.root = DATA_ROOT
    base.CFG.train_dir = DATA_ROOT / "train/train_images"
    base.CFG.test_dir = DATA_ROOT / "test/test_images"
    base.CFG.train_json = annotations
    base.CFG.imgsz = 1024
    if (base.CFG.crop_size, base.CFG.crop_context, base.CFG.min_crop) != (256, 1.8, 96):
        raise RuntimeError("Crop geometry changed")
    base.seed_everything(42)
    provenance = json.loads((V6_SOURCE / "v6_provenance.json").read_text())
    _, images, annots = base.load_coco()
    split = core.validation_split(base, images, annots)
    held, tune, audit, _, by_file = split
    core.verify_split(provenance, held, images)
    if (len(held), len(tune), len(audit), sum(map(len, by_file.values()))) != (141, 94, 47, 217):
        raise RuntimeError("Canonical split mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the refiner")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    model = v11.build_model(base, CANONICAL_REFINER, False,
                          V11_REFINER if V11_REFINER.exists() else None)
    return dict(core=core, tta=tta, base=base, images=images, annotations=annots,
                split=split, model=model)


def evaluate(shard: int, group: str, seeds: list[str], limit: int) -> Path:
    from pycocotools import mask as mask_utils

    rt = runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, audit, _, by_file = rt["split"]
    names = sorted(tune if group == "tune" else audit)[shard::SHARDS]
    if limit:
        names = names[:limit]
    totals: dict[str, np.ndarray] = {}
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        image_path = base.CFG.train_dir / name
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"Cannot read {image_path}")
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [core.annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_image[image_id] = [r for r in gt if mask_utils.area(r) > 0]
        for seed in seeds:
            for cell in CELLS:
                views = read_signed_views(name, "train", cell, DETECTORS[seed])
                proposals, scores = tta.cluster_views(views, POLICY, MATCH_IOU)
                if proposals:
                    masks = [mask_utils.decode(r) for r in proposals]
                    bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
                    refined = {t: core.clean_refined_predictions(masks, scores, bounds, probs,
                                                                 t, MODE)
                               for t in THRESHOLDS}
                else:
                    refined = {t: ([], np.asarray([])) for t in THRESHOLDS}
                for t in THRESHOLDS:
                    rles, final_scores = refined[t]
                    for c in CONFIDENCES:
                        keep = (np.asarray(final_scores) >= c if len(final_scores)
                                else np.zeros(0, bool))
                        kept = [r for r, flag in zip(rles, keep, strict=True) if flag]
                        key = f"{seed}/{cell}/t{t:.2f}_c{c:.2f}"
                        total = totals.setdefault(key, np.zeros(4))
                        for gt in gt_by_image.values():
                            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                                   else np.zeros((len(gt), len(kept))))
                            total += core.pq_counts(iou, len(gt), len(kept))
        print(f"shard={shard} {index}/{len(names)} {name} "
              f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / f"{group}_shard{shard}.json"
    out.write_text(json.dumps(dict(group=group, shard=shard, files=names, seeds=seeds,
                                   limit=limit, totals={k: v.tolist() for k, v in totals.items()}),
                              indent=1) + "\n")
    return out


def merge(group: str) -> dict:
    parts = [json.loads((OUTPUT / f"{group}_shard{s}.json").read_text()) for s in range(SHARDS)]
    if any(p["limit"] for p in parts):
        raise RuntimeError("Refusing to merge smoke (limited) shards")
    files = sorted(f for p in parts for f in p["files"])
    if len(files) != (94 if group == "tune" else 47) or len(set(files)) != len(files):
        raise RuntimeError("Shards do not cover the population exactly once")
    totals: dict[str, np.ndarray] = {}
    for part in parts:
        for key, value in part["totals"].items():
            totals.setdefault(key, np.zeros(4))
            totals[key] += np.asarray(value)
    original = json.loads(V17_RESULTS.read_text())["composition"]
    diffs = []
    for key, value in totals.items():
        seed, cell_candidate = key.split("/", 1)
        ref = original[seed][cell_candidate][group]
        exact = bool(abs(value[0] - ref[0]) < 1e-6
                     and list(map(int, value[1:])) == list(map(int, ref[1:])))
        diffs.append(dict(key=key, pq=pq(value), ref_pq=pq(ref), exact=exact))
    selected = totals[SELECTED].tolist()
    report = dict(
        group=group, files=len(files),
        selected=SELECTED, selected_counts=selected, selected_pq=pq(selected),
        expected_counts=original["seed1"]["mr1_iou030/t0.45_c0.30"][group],
        selected_exact=next(d["exact"] for d in diffs if d["key"] == SELECTED),
        candidates=len(diffs), exact_candidates=sum(d["exact"] for d in diffs),
        max_abs_pq_diff=max(abs(d["pq"] - d["ref_pq"]) for d in diffs),
        best_by_seed={s: max((d for d in diffs if d["key"].startswith(s)), key=lambda d: d["pq"])
                      for s in {d["key"].split("/")[0] for d in diffs}},
    )
    (OUTPUT / f"{group}_merged.json").write_text(json.dumps(
        dict(report=report, totals={k: v.tolist() for k, v in totals.items()}), indent=1) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--shard", type=int, required=True, choices=range(SHARDS))
    ev.add_argument("--group", default="tune", choices=("tune", "audit"))
    ev.add_argument("--seeds", default="seed1,seed2")
    ev.add_argument("--limit", type=int, default=0)
    mg = sub.add_parser("merge")
    mg.add_argument("--group", default="tune", choices=("tune", "audit"))
    args = parser.parse_args()
    if args.command == "eval":
        seeds = [s for s in args.seeds.split(",") if s]
        if not set(seeds) <= set(DETECTORS):
            parser.error("unknown seed")
        print(evaluate(args.shard, args.group, seeds, args.limit))
    else:
        merge(args.group)


if __name__ == "__main__":
    main()
