"""V17: v16b stack + Kaggle-faithful local scoring + inference fixes from the code review.

Presets (override any knob with --set key=value, e.g. --set yolo_weights=experiments/v17/yolo11s/weights/best.pt):
  v16b : exact v16b inference (LB 0.38) — use it as the A/B reference on the same data.
  v17  : v16b plus
           nms=mask           cross-view NMS on seed-mask IoU, so thin neighbours with overlapping
                              boxes are not suppressed (true duplicates still go in finish())
           upsample=bilinear  resize refiner probabilities before thresholding (no blocky edges)
           post=fill_largest  one piece, no holes (MAGFiLO mask rules)
           greedy=True        standard NMS in finish(); legacy code also let suppressed instances suppress
           min_keep_frac=0.5  drop instances that keep <50 % of their area after exclusive painting
           one_piece=True     keep the largest component of what survives painting
  Opt-in knobs (off in both presets until validated): agree=mask, partner_conf, fuse_seeds, limb_max_deg.

Local validation scores every plausible convention for how Kaggle builds the test GT, with pycocotools
rasterization by default (--raster pil reproduces the old local numbers):
  primary : the annotator with most filaments (all local scores so far)
  first   : the first annotator row in the JSON
  each    : every annotator set scored as its own image
  pooled  : all annotators' filaments in one GT set
and reports both pooled PQ (ΣIoU/TP × RQ) and the mean of per-image PQ.

    python -u scripts/make_submission_v17.py --preset v16b --skip-test        # reference, val only
    python -u scripts/make_submission_v17.py --preset v17 --skip-test         # candidate, val only
    python -u scripts/make_submission_v17.py --preset v17 --skip-val          # test CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_submission_v12_indep_iou050_excl as pipe
import v13_inference_sweep as sw
from src.data.annotations import RASTERS, anns_to_instance_masks, build_file_to_ann_sets, load_coco
from src.data.split import read_split_list
from src.inference.limb import central_meridian_deg, fit_disk
from src.inference.rle import mask_to_rle_counts, rle_counts_to_mask, roundtrip_ok
from src.inference.v12_consensus import box_iou_matrix, dilate_mask
from src.utils.config import load_config, resolve_path

V16B = {
    "yolo_weights": "experiments/v12/yolo11s/weights/best.pt",
    "refiner": "experiments/v16/r18aug_refiner/best_slim.pt",
    "refiner_config": "experiments/v16/configs/r18aug_refiner.yaml",
    "yolo_imgsz": 1024,
    "yolo_conf": 0.35,
    "yolo_iou": 0.70,
    "match_iou": 0.50,
    "nms_iou": 0.50,
    "agree": "box",
    "nms": "box",
    "partner_conf": None,
    "fuse_seeds": False,
    "pad_frac": 0.20,
    "image_size": 256,
    "threshold": 0.50,
    "seed_dilate": 7,
    "upsample": "nearest",
    "post": "none",
    "min_area": 80,
    "supp_iou": 0.25,
    "greedy": False,
    "min_keep_frac": 0.0,
    "one_piece": False,
    "limb_max_deg": None,
}
PRESETS = {
    "v16b": V16B,
    "v17": V16B | {"nms": "mask", "upsample": "bilinear", "post": "fill_largest", "greedy": True,
                   "min_keep_frac": 0.5, "one_piece": True},
}
GT_MODES = ("primary", "first", "each", "pooled")


def parse_sets(pairs: list[str], knobs: dict) -> dict:
    out = dict(knobs)
    for kv in pairs:
        key, _, val = kv.partition("=")
        if key not in out:
            raise SystemExit(f"--set: unknown knob {key!r} (known: {', '.join(out)})")
        out[key] = yaml.safe_load(val)
    return out


def predict(k: dict, yolo, model, img: np.ndarray, device) -> list[sw.Inst]:
    h, w = img.shape[:2]
    conf_min = k["yolo_conf"] if k["partner_conf"] is None else min(k["yolo_conf"], k["partner_conf"])
    boxes, confs, views, seeds = sw.detect_views(yolo, img, k["yolo_imgsz"], conf_min, device, iou=k["yolo_iou"])
    if not len(boxes):
        return []
    biou = box_iou_matrix(boxes, boxes)
    miou = sw.seed_iou_matrix(seeds, biou) if "mask" in (k["agree"], k["nms"]) else None
    idx, members = sw.select(
        boxes, confs, views, biou, k["yolo_conf"], float("inf"), k["match_iou"], k["nms_iou"],
        agree_mat=miou if k["agree"] == "mask" else None,
        nms_mat=miou if k["nms"] == "mask" else None,
        partner_conf=k["partner_conf"], clusters=True,
    )
    disk = fit_disk(img) if k["limb_max_deg"] is not None else None
    insts, scores = [], []
    for i in idx:
        box, seed = boxes[i], seeds[i]
        if k["fuse_seeds"] and len(members[i]) > 1:
            grp = members[i]
            box = np.array([boxes[grp, 0].min(), boxes[grp, 1].min(), boxes[grp, 2].max(), boxes[grp, 3].max()])
            seed = np.logical_or.reduce([seeds[j] > 0 for j in grp]).astype(np.uint8)
        exp, probs = sw.refine_probs({"r": model}, img, box, k["image_size"], k["pad_frac"], device)
        sc = seed[exp[1] : exp[3], exp[0] : exp[2]]
        allow = dilate_mask((sc > 0).astype(np.uint8), k["seed_dilate"]).astype(bool)
        inst, mp = sw.build_mask(probs["r_tta"], k["threshold"], exp, allow, k["post"], k["upsample"])
        if disk is not None and inst.area:
            ys, xs = np.nonzero(inst.m)
            if central_meridian_deg(inst.x0 + xs.mean(), inst.y0 + ys.mean(), disk) > k["limb_max_deg"]:
                continue
        insts.append(inst)
        scores.append(float(confs[i]) * mp)
    return sw.finish(insts, np.asarray(scores, np.float32), k["min_area"], k["supp_iou"], h, w,
                     greedy=k["greedy"], min_keep_frac=k["min_keep_frac"], one_piece=k["one_piece"])


def gt_sets_for(mode: str, sets: list[list[dict]]) -> list[list[dict]]:
    if not sets:
        return [[]]
    if mode == "primary":
        return [max(sets, key=len)]  # first max, same as build_file_to_anns_primary
    if mode == "first":
        return [sets[0]]
    if mode == "each":
        return sets
    return [[a for s in sets for a in s]]  # pooled


def evaluate(k, yolo, model, device, names, images_dir, file_to_sets, raster) -> dict:
    acc = {m: {"tp": 0, "fp": 0, "fn": 0, "s": 0.0, "per_image": []} for m in GT_MODES}
    for name in tqdm(names, desc="val"):
        img = np.asarray(Image.open(images_dir / name).convert("L"), dtype=np.uint8)
        h, w = img.shape[:2]
        preds = predict(k, yolo, model, img, device)
        for mode in GT_MODES:
            for anns in gt_sets_for(mode, file_to_sets.get(name, [])):
                gts = [sw.Inst(0, 0, g > 0) for g in anns_to_instance_masks(anns, h, w, raster=raster)]
                tp, fp, fn, s = sw.pq_counts(gts, preds)
                a = acc[mode]
                a["tp"] += tp; a["fp"] += fp; a["fn"] += fn; a["s"] += s
                denom = tp + 0.5 * fp + 0.5 * fn
                a["per_image"].append(1.0 if denom == 0 else s / denom)  # PQ = ΣIoU / (TP + FP/2 + FN/2)
    out = {}
    for mode, a in acc.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        sq = a["s"] / tp if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        out[mode] = {"pq": round(sq * rq, 6), "sq": round(sq, 6), "rq": round(rq, 6), "tp": tp, "fp": fp, "fn": fn,
                     "pq_mean_per_image": round(float(np.mean(a["per_image"])), 6) if a["per_image"] else 0.0,
                     "n_scored": len(a["per_image"])}
    return out


def to_full(inst: sw.Inst, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[inst.y0 : inst.y0 + inst.m.shape[0], inst.x0 : inst.x0 + inst.m.shape[1]] = inst.m
    return m


def write_submission(k, yolo, model, device, images: list[Path], out_csv: Path) -> dict:
    rows, empty = [], 0
    for path in tqdm(images, desc="test"):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        h, w = img.shape[:2]
        assert (h, w) == (2048, 2048), f"{path.name} is {h}x{w}"
        masks = [to_full(i, h, w) for i in predict(k, yolo, model, img, device)]
        if not pipe.assert_no_overlap(masks):
            raise AssertionError(f"Overlap detected for {path.name}")
        empty += not masks
        for n, m in enumerate(masks, start=1):
            rle = mask_to_rle_counts(m)
            if rle:
                assert np.array_equal(rle_counts_to_mask(rle, height=h, width=w) > 0, m > 0), f"RLE {path.stem}_{n}"
                rows.append({"filament_id": f"{path.stem}_{n}", "segmentation_rle": rle})
    ids = [r["filament_id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate filament_id"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        wr.writeheader()
        wr.writerows(rows)
    return {"output": os.path.relpath(out_csv, ROOT), "n_rows": len(rows), "n_images": len(images),
            "empty_pred_images": empty, "strict_nonoverlap_verified": True}


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v17")
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE")
    ap.add_argument("--raster", choices=RASTERS, default="coco", help="GT rasterization for local scoring")
    ap.add_argument("--images-dir", default=None, help="default: refiner config data.images_dir")
    ap.add_argument("--annotations", default=None, help="default: refiner config data.annotations")
    ap.add_argument("--val-list", default=None, help="default: <splits_dir>/val.txt")
    ap.add_argument("--test-dir", default="data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--output", default=None, help="default: submissions/submission_<preset>.csv")
    ap.add_argument("--report-dir", default=None, help="default: experiments/v17/<preset>")
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    k = parse_sets(args.set, PRESETS[args.preset])
    toy = np.zeros((32, 32), dtype=np.uint8)
    toy[8:24, 8:24] = 1
    if not roundtrip_ok(toy):
        raise RuntimeError("RLE roundtrip failed")

    cfg = load_config(resolve_path(ROOT, k["refiner_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(resolve_path(ROOT, k["yolo_weights"])))
    refiner_path = resolve_path(ROOT, k["refiner"])
    if not refiner_path.exists():
        raise FileNotFoundError(f"{refiner_path} missing; run scripts/export_refiner_weights.py on the trained best.pt")
    model = pipe.load_refiner(refiner_path, cfg, device)
    report_dir = resolve_path(ROOT, args.report_dir or f"experiments/v17/{args.preset}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"preset": args.preset, "knobs": k}

    if not args.skip_val:
        images_dir = resolve_path(ROOT, args.images_dir or cfg["data"]["images_dir"])
        coco = load_coco(resolve_path(ROOT, args.annotations or cfg["data"]["annotations"]))
        names = read_split_list(resolve_path(ROOT, args.val_list or Path(cfg["data"]["splits_dir"]) / "val.txt"))
        if args.limit:
            names = names[: args.limit]
        val = evaluate(k, yolo, model, device, names, images_dir, build_file_to_ann_sets(coco), args.raster)
        report["val"] = {"raster": args.raster, "n_images": len(names), "modes": val}
        print(f"\nVAL preset={args.preset} raster={args.raster} n={len(names)}")
        print(f"{'gt mode':8} {'PQ':>8} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'PQ/img':>8}")
        for mode, r in val.items():
            print(f"{mode:8} {r['pq']:8.4f} {r['sq']:7.4f} {r['rq']:7.4f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} "
                  f"{r['pq_mean_per_image']:8.4f}")

    if not args.skip_test:
        test_dir = resolve_path(ROOT, args.test_dir)
        images = sorted({p.name: p for p in list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg"))}.values())
        if args.limit:
            images = images[: args.limit]
        out_csv = resolve_path(ROOT, args.output or f"submissions/submission_{args.preset}.csv")
        report["test"] = write_submission(k, yolo, model, device, images, out_csv)
        print(f"Wrote {out_csv}  images={len(images)} rows={report['test']['n_rows']} "
              f"empty={report['test']['empty_pred_images']} nonoverlap=OK")

    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
