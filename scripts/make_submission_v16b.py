"""V16b ablation: v12 YOLO11s only + V16 aug ResNet18 refiner (4-flip TTA), local val PQ 0.4581.

Same as v13 tuned (Kaggle 0.38) except the refiner, to test whether the v14 YOLO or the aug refiner
caused the Kaggle drop of v16 (0.37). Runs on CPU when CUDA_VISIBLE_DEVICES is empty.

Copy of make_submission_v14.py; knobs from experiments/v16/sweep_r18aug (r18aug_tta, det=both).
Original V13 notes:

Changes vs make_submission_v12_indep_iou050_excl.py (sweep: experiments/v13/inference_sweep/):
  - refiner 4-flip TTA (hard_tta)
  - yolo conf 0.40 → 0.35, refiner thr 0.45 → 0.50
  - whole-instance duplicate suppression IoU 0.50 → 0.25
  - suppress/exclusive ordering score: conf → conf × mean refiner prob

Uses the same functions as scripts/v13_inference_sweep.py, so val PQ here must match the sweep.

    .venv\\Scripts\\python.exe -u scripts\\make_submission_v13_tuned.py            # val check + test CSV
    .venv\\Scripts\\python.exe -u scripts\\make_submission_v13_tuned.py --skip-val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_v12_pq as v12
import make_submission_v12_indep_iou050_excl as pipe
import v13_inference_sweep as sw
from src.data.annotations import build_file_to_anns_primary, load_coco
from src.data.split import read_split_list
from src.inference.rle import mask_to_rle_counts, rle_counts_to_mask, roundtrip_ok
from src.inference.v12_consensus import box_iou_matrix, dilate_mask
from src.utils.config import load_config, resolve_path

KNOBS = {
    "yolo_weights": "experiments/v12/yolo11s/weights/best.pt",
    "yolo_weights2": "experiments/v14/yolo11s/weights/best.pt",
    "det": "single",
    "refiner": "experiments/v16/r18aug_refiner/best.pt",
    "refiner_tta": "4flip",
    "yolo_imgsz": 1024,
    "yolo_conf": 0.35,
    "yolo_iou": 0.70,
    "match_iou": 0.50,
    "nms_iou": 0.50,
    "pad_frac": 0.20,
    "image_size": 256,
    "threshold": 0.50,
    "seed_dilate": 7,
    "min_area": 80,
    "supp_iou": 0.25,
    "score": "conf_x_mean_prob",
}
EXPECTED_VAL = {"pq": 0.45814, "tp": 692, "fp": 272, "fn": 453}
DEFAULT_OUT = "submissions/submission_v16b_v12yolo_r18aug_c035_t050_s025.csv"
REPORT_DIR = ROOT / "experiments/v16/submission_b"


def predict(yolos, models, img: np.ndarray, device) -> list[sw.Inst]:
    k = KNOBS
    h, w = img.shape[:2]
    boxes, confs, views, seeds = sw.detect_views(yolos[0], img, k["yolo_imgsz"], k["yolo_conf"], device)
    b2 = []
    if k["det"] != "single":  # single uses only the first detector; skip the second (halves CPU time)
        b2, c2, v2, s2 = sw.detect_views(yolos[1], img, k["yolo_imgsz"], k["yolo_conf"], device)
    if len(b2):  # second detector's views are offset by 4 so select(det="both") can tell them apart
        boxes = np.concatenate([boxes, b2]) if len(boxes) else b2
        confs = np.concatenate([confs, c2])
        views = np.concatenate([views, v2 + len(sw.FLIP_VIEWS)])
        seeds = seeds + s2
    if not len(boxes):
        return []
    biou = box_iou_matrix(boxes, boxes)
    idx = sw.select(boxes, confs, views, biou, k["yolo_conf"], float("inf"), k["match_iou"], k["nms_iou"], det=k["det"])
    insts, scores = [], []
    for i in idx:
        exp, probs = sw.refine_probs(models, img, boxes[i], k["image_size"], k["pad_frac"], device)
        sc = seeds[i][exp[1] : exp[3], exp[0] : exp[2]]
        allow = dilate_mask((sc > 0).astype(np.uint8), k["seed_dilate"]).astype(bool)
        inst, mp = sw.build_mask(probs["r18aug_tta"], k["threshold"], exp, allow, "none")
        insts.append(inst)
        scores.append(float(confs[i]) * mp)
    return sw.finish(insts, np.asarray(scores, np.float32), k["min_area"], k["supp_iou"], h, w)


def to_full(inst: sw.Inst, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[inst.y0 : inst.y0 + inst.m.shape[0], inst.x0 : inst.x0 + inst.m.shape[1]] = inst.m
    return m


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    toy = np.zeros((32, 32), dtype=np.uint8)
    toy[8:24, 8:24] = 1
    if not roundtrip_ok(toy):
        raise RuntimeError("RLE roundtrip failed")

    cfg = load_config(ROOT / sw.REFINERS["r18aug"][1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = [YOLO(str(resolve_path(ROOT, KNOBS[w]))) for w in ("yolo_weights", "yolo_weights2")]
    models = {"r18aug": pipe.load_refiner(resolve_path(ROOT, KNOBS["refiner"]), cfg, device)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    val = None
    if not args.skip_val:
        names = read_split_list(resolve_path(ROOT, cfg["data"]["splits_dir"]) / "val.txt")
        if args.limit:
            names = names[: args.limit]
        f2a = build_file_to_anns_primary(load_coco(resolve_path(ROOT, cfg["data"]["annotations"])))
        images_dir = resolve_path(ROOT, cfg["data"]["images_dir"])
        tp = fp = fn = 0
        s = 0.0
        for name in tqdm(names, desc="val"):
            img, _, gt_full, _, _ = v12.load_image_and_gt(images_dir / name, f2a.get(name, []), out_size=2048)
            preds = predict(yolo, models, img, device)
            a, b, c, v = sw.pq_counts([sw.Inst(0, 0, g > 0) for g in gt_full], preds)
            tp += a; fp += b; fn += c; s += v
        sq = s / tp if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        val = {"pq": round(sq * rq, 6), "sq": round(sq, 6), "rq": round(rq, 6), "tp": tp, "fp": fp, "fn": fn,
               "n_val": len(names)}
        print("VAL", val, flush=True)
        if not args.limit and (abs(val["pq"] - EXPECTED_VAL["pq"]) > 2e-3 or abs(val["tp"] - EXPECTED_VAL["tp"]) > 2):
            raise AssertionError(f"val mismatch vs sweep {EXPECTED_VAL}")
        (REPORT_DIR / "val_verification.json").write_text(json.dumps(val, indent=2), encoding="utf-8")

    test_dir = resolve_path(ROOT, args.test_dir)
    images = sorted({p.name: p for p in list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg"))}.values())
    if args.limit:
        images = images[: args.limit]
    rows, empty = [], 0
    for path in tqdm(images, desc="test"):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        h, w = img.shape[:2]
        assert (h, w) == (2048, 2048), f"{path.name} is {h}x{w}"
        masks = [to_full(i, h, w) for i in predict(yolo, models, img, device)]
        if not pipe.assert_no_overlap(masks):
            raise AssertionError(f"Overlap detected for {path.name}")
        if not masks:
            empty += 1
        for k, m in enumerate(masks, start=1):
            rle = mask_to_rle_counts(m)
            if rle:
                rows.append({"filament_id": f"{path.stem}_{k}", "segmentation_rle": rle})

    out_csv = resolve_path(ROOT, args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        wr.writeheader()
        wr.writerows(rows)

    ids = [r["filament_id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate filament_id"
    stems = {p.stem for p in images}
    assert all(i.rsplit("_", 1)[0] in stems for i in ids)
    by_stem: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_stem[r["filament_id"].rsplit("_", 1)[0]].append(r["segmentation_rle"])
    for stem, rles in by_stem.items():
        if not pipe.assert_no_overlap([rle_counts_to_mask(c, height=2048, width=2048) for c in rles]):
            raise AssertionError(f"RLE overlap for stem={stem}")

    checks = {"output": out_csv.relative_to(ROOT).as_posix(), "n_rows": len(rows), "n_images": len(images),
              "empty_pred_images": empty, "strict_nonoverlap_verified": True, "val": val, "knobs": KNOBS}
    (REPORT_DIR / "submission_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if not args.limit:
        log_path = resolve_path(ROOT, "submissions/submission_log.csv")
        with log_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                out_csv.name, "v16b", "yolo11s_v12@1024+consensus+resnet18_aug_tta_nonoverlap",
                "scripts/make_submission_v16b.py",
                f"local_val_pq={val['pq'] if val else EXPECTED_VAL['pq']}", "",
                f"rows={len(rows)}; empty_images={empty}; yolo_conf=0.35; det=single; thr=0.50; supp_iou=0.25; "
                f"score=conf*mean_prob; refiner_tta=4flip; min_area=80; nonoverlap=1",
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ])
    print(f"Wrote {out_csv}  images={len(images)} rows={len(rows)} empty={empty} nonoverlap=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
