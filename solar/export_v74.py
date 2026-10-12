"""V74 test export: V65 pipeline with a configurable detector, input size and minimum mask area.

Same graph as scripts/export_v62_shard.py (the V65 export): 4-flip views from the shared
`v61_toolkit.generate_views` (box iou 0.3, retina masks, agnostic NMS, floor 0.1) ->
`detector_tta.cluster_views` flip_consensus (match iou 0.5) -> refiner crops with flip TTA ->
`clean_refined_predictions` refined_constrained (score-ordered pixel ownership) -> confidence.
Changes vs V65, and ONLY these:
  * --imgsz (1024 = V65, 1280 = p2first_1280) is passed to generate_views;
  * --min-area (default 200, V69 audit / V72 "MIN200") drops FINAL kept masks below that area.
    It is drop-only, after ownership and confidence, as measured in V72. With --min-area 20 it is a
    no-op (cleaning already drops < 20) and the output equals V65;
  * --device cpu for smoke tests; --limit N keeps the first N files of the shard;
  * the refiner defaults to the V65 r3all refiner; SHAs are optional (recorded, verified if given).
  * V84: --demote-model <demote_all.json> (default off) applies the V83/V84 demote rule. It computes the
    V83 candidate features (v83_extract.refine_feats + candidate_rows, shared code) on the refined
    candidates and drops kept masks with yhat <= lambda/2. Use with --mask-threshold 0.50 and the V84 soft refiner.
It never submits. It reads only the 180 test IMAGES (no labels).

  CUDA_VISIBLE_DEVICES=N python scripts/export_v74.py --shard-index 0 --num-shards 4 \
      --detector <final.pt> --imgsz 1280 --output-dir outputs/v74/export
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "kaggle_inference"))
import reproduce_v17_from_cache as v50  # noqa: E402
import v61_toolkit as K  # noqa: E402
from export_v62_shard import TEST_DIR, sha256  # noqa: E402

ROOT = SCRIPTS.parent
V65_REFINER = ROOT / "weights/r3all_refiner_state.pt"


def refiner_runtime(path, device):
    """The parts of v50.runtime_state() the export uses, without its CUDA requirement."""
    import cv2
    import torch
    import detector_tta as tta
    import optimize_inference as core

    base = core.load_base(v50.V6_SOURCE / "solar_base.py")
    base.CFG.imgsz = 1024  # refiner crop geometry does not depend on the detector input size
    if (base.CFG.crop_size, base.CFG.crop_context, base.CFG.min_crop) != (256, 1.8, 96):
        raise RuntimeError("Crop geometry changed")
    base.seed_everything(42)
    cv2.setNumThreads(2)
    torch.set_num_threads(2 if device != "cpu" else 8)
    model = base.CropUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return core, tta, base, model.eval()


class Demote:
    """V84 demote rule: V83 LightGBM on the V83 per-candidate features, drop yhat <= lambda/2."""

    def __init__(self, path, core):
        env = os.environ.get("CUDA_VISIBLE_DEVICES")  # audit_v80_extract/v83_extract blank it on import
        import v83_extract as X
        if env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env
        import lightgbm as lgb
        self.X, self.E = X, X.E
        if self.E.RT is None:
            self.E.RT = {"core": core}
        self.meta = json.loads(Path(path).read_text())
        mpath = Path(path).parent / self.meta["model"]
        if sha256(mpath) != self.meta["model_sha256"]:
            raise RuntimeError("demote model identity changed")
        self.model, self.lam = lgb.Booster(model_file=str(mpath)), float(self.meta["lambda"])
        self.sha = sha256(Path(path))

    def groups(self, views, proposals, pscores):
        """V83 groups (with member scores/views); asserted identical to tta.cluster_views."""
        groups = self.E.groups_with_ids(views)
        a = sorted((float(s), r["counts"]) for r, s in zip(proposals, pscores))
        b = sorted((float(g["score"]), g["rle"]["counts"]) for g in groups)
        if a != b:
            raise RuntimeError("groups_with_ids differs from cluster_views")
        return groups

    def yhat(self, groups, masks, bounds, probs, gray, threshold, rles):
        """yhat per kept-candidate RLE counts; asserts refine_feats ownership == clean_refined_predictions."""
        self.X.T = threshold
        scores = [g["score"] for g in groups]
        post, feats = self.X.refine_feats(masks, scores, bounds, probs, gray, self.E.disk_fit(gray))
        alive = [i for i in range(len(groups)) if post[i] is not None]
        if sorted(post[i]["counts"] for i in alive) != sorted(r["counts"] for r in rles):
            raise RuntimeError("refine_feats ownership differs from clean_refined_predictions")
        cand, rows = self.X.candidate_rows(groups, scores, alive, feats)
        if not rows:
            return {}
        y = self.model.predict(np.asarray([[r[k] for k in self.meta["features"]] for r in rows], float))
        return {post[alive[c]]["counts"]: float(v) for c, v in zip(cand, y)}


def refine_image(core, tta, base, refiner, views, path, args, demote=None):
    """Post-detector graph for one image -> (kept RLEs, kept scores, dropped_small, demoted, yhat)."""
    import cv2
    proposals, pscores = tta.cluster_views(views, "flip_consensus", match_iou=0.5)
    preds, kept, yh = [], [], {}
    if proposals:
        if demote is not None:  # group order (same set, asserted) so features see V83's group fields
            groups = demote.groups(views, proposals, pscores)
            proposals, pscores = [g["rle"] for g in groups], [g["score"] for g in groups]
        masks = [mask_utils.decode(r) for r in proposals]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"invalid grayscale image {path}")
        bounds, _, probs = core.crop_probabilities(base, refiner, gray, masks, True)
        rles, kept = core.clean_refined_predictions(
            masks, pscores, bounds, probs, args.mask_threshold, "refined_constrained")
        if demote is not None:
            yh = demote.yhat(groups, masks, bounds, probs, gray, args.mask_threshold, rles)
        keep = [(r, s) for r, s in zip(rles, kept, strict=True) if s >= args.confidence]
        preds, kept = [r for r, _ in keep], [s for _, s in keep]
    n_before = len(preds)
    small = [int(mask_utils.area(r)) >= args.min_area for r in preds]
    preds, kept = [r for r, b in zip(preds, small) if b], [s for s, b in zip(kept, small) if b]
    dropped_small = n_before - len(preds)
    demoted = 0
    if demote is not None:
        ok = [yh[r["counts"]] > demote.lam / 2 for r in preds]
        demoted = len(ok) - sum(ok)
        preds, kept = [r for r, b in zip(preds, ok) if b], [s for s, b in zip(kept, ok) if b]
    return preds, kept, dropped_small, demoted, yh


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--detector-sha", default=None)
    ap.add_argument("--imgsz", type=int, choices=(1024, 1280), default=1024)
    ap.add_argument("--refiner", type=Path, default=V65_REFINER)
    ap.add_argument("--refiner-sha", default=None)
    ap.add_argument("--mask-threshold", type=float, default=0.45)
    ap.add_argument("--confidence", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--demote-model", type=Path, default=None, help="V84 demote rule json (default off)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--test-dir", type=Path, default=TEST_DIR, dest="test_dir")
    ap.add_argument("--limit", type=int, default=0, help="first N files of the shard (smoke tests)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")
    if args.min_area < 20:
        raise SystemExit("--min-area below the cleaning floor (20) has no effect")
    dsha, rsha = sha256(args.detector), sha256(args.refiner)
    for name, want, got in (("detector", args.detector_sha, dsha), ("refiner", args.refiner_sha, rsha)):
        if want and want != got:
            raise RuntimeError(f"{name} checkpoint identity changed")
    core, tta, base, refiner = refiner_runtime(args.refiner, args.device)
    demote = Demote(args.demote_model, core) if args.demote_model else None

    detector = YOLO(str(args.detector))
    all_images = sorted(args.test_dir.glob("*.jpeg"))
    if len(all_images) != 180:
        raise RuntimeError(f"expected 180 test images, found {len(all_images)}")
    selected = [p for i, p in enumerate(all_images) if i % args.num_shards == args.shard_index]
    if args.limit:
        selected = selected[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, coverage, dropped_small, demoted, started = [], {}, 0, 0, time.monotonic()
    for index, path in enumerate(selected, 1):
        views = K.generate_views(detector, path, iou=0.3, imgsz=args.imgsz)
        preds, _, n_small, n_dem, _ = refine_image(core, tta, base, refiner, views, path, args, demote)
        dropped_small += n_small
        demoted += n_dem
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(preds, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if int(mask.sum()) < args.min_area or np.any(occupied & mask):
                raise RuntimeError(f"invalid area/overlap for {path.name}")
            occupied |= mask
            rows.append({"filament_id": f"{path.stem}_{number}",
                         "segmentation_rle": rle["counts"].decode()})
        coverage[path.name] = len(preds)
        print(f"shard={args.shard_index}/{args.num_shards} images={index}/{len(selected)} "
              f"rows={len(rows)} elapsed={time.monotonic() - started:.0f}s", flush=True)

    csv_path = args.output_dir / f"shard_{args.shard_index}.csv"
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(csv_path, index=False)
    manifest = {
        "status": "complete" if not args.limit else "partial", "shard_index": args.shard_index,
        "num_shards": args.num_shards, "files": [p.name for p in selected], "rows": len(rows),
        "empty_images": [n for n, c in coverage.items() if c == 0], "coverage": coverage,
        "detector": str(args.detector), "detector_sha256": dsha,
        "refiner": str(args.refiner), "refiner_sha256": rsha, "device": args.device,
        "recipe": {"imgsz": args.imgsz, "box_iou": 0.3, "detector_floor": K.CONF_FLOOR, "views": 4,
                   "policy": "flip_consensus", "match_iou": 0.5, "refiner_tta": True,
                   "mode": "refined_constrained", "mask_threshold": args.mask_threshold,
                   "confidence": args.confidence, "min_area": args.min_area,
                   "dropped_by_min_area": dropped_small},
        "csv_sha256": sha256(csv_path), "seconds": time.monotonic() - started,
        "reads_labels": False}
    if demote is not None:
        manifest["recipe"].update(demote_model=str(args.demote_model), demote_sha256=demote.sha,
                                  demote_model_sha256=demote.meta["model_sha256"],
                                  demote_lambda=demote.lam, dropped_by_demote=demoted)
    (args.output_dir / f"shard_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("coverage", "files")}, indent=2))


if __name__ == "__main__":
    main()
