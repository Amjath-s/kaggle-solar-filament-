"""V12 submission: independent → iou_ge_0.50 → score-priority exclusive → min_area=80.

Promoted local val PQ ≈ 0.437318 (non-overlapping / Kaggle-legal).
Uses locked YOLO best.pt (not kaggle_ep80). Does not modify make_submission_v12.py.

Usage:

    .venv\\Scripts\\python.exe -u scripts\\make_submission_v12_indep_iou050_excl.py
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.local_crops import expand_xyxy, resize_image_mask
from src.inference.rle import mask_to_rle_counts, rle_counts_to_mask, roundtrip_ok
from src.inference.v12_consensus import detect_consensus_4flip, dilate_mask
from src.models.unet import build_model
from src.utils.config import load_config, resolve_path

OUT_SIZE = 2048
MIN_AREA = 80
YOLO_CONF = 0.40
IOU_THR = 0.50
DEFAULT_OUT = "submissions/submission_v12_indep_iou050_excl_c040_a080.csv"
REPORT_DIR = ROOT / "experiments/v13/submission_integration_nonoverlap"
LOCAL_VAL_PQ = 0.437318


def load_refiner(checkpoint_path: Path, cfg: dict, device: torch.device):
    model_cfg = cfg.get("model", {})
    model = build_model(
        model_cfg.get("name", "resnet18_unet"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=1,
        pretrained=False,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def refine_one(
    model,
    img_u8: np.ndarray,
    box: np.ndarray,
    seed_mask: np.ndarray,
    *,
    image_size: int,
    pad_frac: float,
    threshold: float,
    seed_dilate: int,
    device: torch.device,
) -> np.ndarray:
    h, w = img_u8.shape[:2]
    exp = expand_xyxy(box, pad_frac, w, h)
    crop = img_u8[exp[1] : exp[3], exp[0] : exp[2]]
    seed_crop = seed_mask[exp[1] : exp[3], exp[0] : exp[2]]
    img_r, _ = resize_image_mask(crop, None, size=image_size)
    x = torch.from_numpy(img_r[None, None, ...]).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8)
    pred_crop = np.asarray(
        Image.fromarray(pred * 255, mode="L").resize(
            (int(exp[2] - exp[0]), int(exp[3] - exp[1])), Image.NEAREST
        )
    )
    pred_crop = (pred_crop > 127).astype(np.uint8)
    allow = dilate_mask((seed_crop > 0).astype(np.uint8), seed_dilate)
    clamped = np.logical_and(pred_crop, allow).astype(np.uint8)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[exp[1] : exp[3], exp[0] : exp[2]] = clamped
    return canvas


def resize_mask_to(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return (mask > 0).astype(np.uint8)
    return (
        np.asarray(
            Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").resize(
                (size, size), Image.NEAREST
            )
        )
        > 127
    ).astype(np.uint8)


def independent_keep(
    masks: list[np.ndarray], confs: np.ndarray, min_area: int
) -> tuple[list[np.ndarray], np.ndarray]:
    idx = [i for i, m in enumerate(masks) if int((m > 0).sum()) >= min_area]
    kept = [(masks[i] > 0).astype(np.uint8) for i in idx]
    confs_k = confs[idx] if len(idx) else np.zeros(0, dtype=np.float32)
    return kept, confs_k


def pairwise_max_iou_higher_conf(masks: list[np.ndarray], confs: np.ndarray) -> np.ndarray:
    n = len(masks)
    max_iou_higher = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return max_iou_higher
    confs = np.asarray(confs, dtype=np.float32).reshape(-1)
    for j in range(n):
        mj = masks[j] > 0
        if int(mj.sum()) == 0:
            continue
        best = 0.0
        for i in range(n):
            if i == j:
                continue
            higher = confs[i] > confs[j] + 1e-12 or (
                abs(float(confs[i] - confs[j])) <= 1e-12 and i < j
            )
            if not higher:
                continue
            mi = masks[i] > 0
            inter = int(np.logical_and(mj, mi).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(mj, mi).sum())
            iou = inter / max(union, 1)
            if iou > best:
                best = iou
        max_iou_higher[j] = best
    return max_iou_higher


def suppress_iou_ge_050(
    masks: list[np.ndarray], confs: np.ndarray, thr: float = IOU_THR
) -> tuple[list[np.ndarray], np.ndarray]:
    if not masks:
        return [], np.zeros(0, dtype=np.float32)
    iou_h = pairwise_max_iou_higher_conf(masks, confs)
    keep_m, keep_c = [], []
    for j, m in enumerate(masks):
        if float(iou_h[j]) >= thr:
            continue
        keep_m.append(m)
        keep_c.append(float(confs[j]))
    return keep_m, np.asarray(keep_c, dtype=np.float32)


def exclusive_paint_then_min_area(
    masks: list[np.ndarray], scores: np.ndarray, min_area: int
) -> list[np.ndarray]:
    if not masks:
        return []
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    h, w = masks[0].shape
    claimed = np.zeros((h, w), dtype=np.uint8)
    clipped: list[np.ndarray] = []
    for idx in order:
        binary = (masks[int(idx)] > 0).astype(np.uint8)
        kept = np.logical_and(binary, claimed == 0).astype(np.uint8)
        if int(kept.sum()) > 0:
            clipped.append(kept)
            claimed = np.maximum(claimed, kept)
    return [m for m in clipped if int(m.sum()) >= min_area]


def assert_no_overlap(masks: list[np.ndarray]) -> bool:
    if len(masks) <= 1:
        return True
    acc = np.zeros_like(masks[0], dtype=np.uint16)
    for m in masks:
        acc = acc + (m > 0).astype(np.uint16)
        if int((acc > 1).sum()) > 0:
            return False
    return True


def predict_nonoverlap(
    yolo,
    refiner,
    img_u8: np.ndarray,
    *,
    device: torch.device,
    yolo_imgsz: int,
    yolo_conf: float,
    yolo_iou: float,
    match_iou: float,
    nms_iou: float,
    image_size: int,
    pad_frac: float,
    threshold: float,
    seed_dilate: int,
    min_area: int,
    out_size: int,
) -> list[np.ndarray]:
    boxes, confs, seeds = detect_consensus_4flip(
        yolo,
        img_u8,
        imgsz=yolo_imgsz,
        conf=yolo_conf,
        iou=yolo_iou,
        match_iou=match_iou,
        nms_iou=nms_iou,
        device=device,
    )
    if not len(seeds):
        return []
    refined = [
        refine_one(
            refiner,
            img_u8,
            box,
            seed,
            image_size=image_size,
            pad_frac=pad_frac,
            threshold=threshold,
            seed_dilate=seed_dilate,
            device=device,
        )
        for box, seed in zip(boxes, seeds)
    ]
    masks_out = [resize_mask_to(m, out_size) for m in refined]
    indep, confs_i = independent_keep(masks_out, confs, min_area)
    after_iou, confs_f = suppress_iou_ge_050(indep, confs_i, IOU_THR)
    return exclusive_paint_then_min_area(after_iou, confs_f, min_area)


def main() -> int:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/v12/configs/resnet18_refiner.yaml")
    parser.add_argument("--test-dir", type=str, default="data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--out-size", type=int, default=OUT_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    toy = np.zeros((32, 32), dtype=np.uint8)
    toy[8:24, 8:24] = 1
    if not roundtrip_ok(toy):
        raise RuntimeError("RLE roundtrip failed")

    cfg = load_config(ROOT / args.config)
    infer = cfg.get("infer", {})
    # Locked detector — never kaggle_ep80 unless explicitly requested later
    yolo_weights = resolve_path(ROOT, "experiments/v12/yolo11s/weights/best.pt")
    local_ckpt = resolve_path(
        ROOT, infer.get("local_checkpoint", "experiments/v12/resnet18_refiner/best.pt")
    )
    if not yolo_weights.is_file():
        raise FileNotFoundError(yolo_weights)
    if not local_ckpt.is_file():
        raise FileNotFoundError(local_ckpt)

    yolo_conf = YOLO_CONF
    yolo_iou = 0.70
    yolo_imgsz = 1024
    pad_frac = float(cfg.get("pad_frac", infer.get("pad_frac", 0.20)))
    threshold = float(infer.get("threshold", cfg.get("threshold", 0.45)))
    seed_dilate = int(infer.get("seed_dilate", 7))
    match_iou = float(infer.get("match_iou", 0.50))
    nms_iou = float(infer.get("nms_iou", 0.50))
    image_size = int(cfg.get("image_size", infer.get("image_size", 256)))
    min_area = int(infer.get("min_area", MIN_AREA))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(yolo_weights))
    refiner = load_refiner(local_ckpt, cfg, device)

    test_dir = resolve_path(ROOT, args.test_dir)
    out_csv = resolve_path(ROOT, args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(test_dir.glob("*.jpeg")) + sorted(test_dir.glob("*.jpg"))
    seen: set[str] = set()
    unique_images: list[Path] = []
    for p in images:
        if p.name not in seen:
            seen.add(p.name)
            unique_images.append(p)
    if args.limit > 0:
        unique_images = unique_images[: args.limit]
    if not unique_images:
        raise FileNotFoundError(f"No test images in {test_dir}")

    print(
        f"device={device} yolo={yolo_weights.name} conf={yolo_conf} "
        f"image_size={image_size} pipeline=indep→iou{IOU_THR}→excl→a{min_area} "
        f"n_test={len(unique_images)}",
        flush=True,
    )

    rows: list[dict[str, str]] = []
    empty_images = 0
    overlap_fail = 0
    for path in tqdm(unique_images, desc="v12-nonoverlap-submit"):
        stem = path.stem
        img_u8 = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        instances = predict_nonoverlap(
            yolo,
            refiner,
            img_u8,
            device=device,
            yolo_imgsz=yolo_imgsz,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            match_iou=match_iou,
            nms_iou=nms_iou,
            image_size=image_size,
            pad_frac=pad_frac,
            threshold=threshold,
            seed_dilate=seed_dilate,
            min_area=min_area,
            out_size=args.out_size,
        )
        if not assert_no_overlap(instances):
            overlap_fail += 1
            raise AssertionError(f"Overlap detected for {path.name} — aborting submit")
        if not instances:
            empty_images += 1
        for k, inst in enumerate(instances, start=1):
            rle = mask_to_rle_counts(inst)
            if not rle:
                continue
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": rle})

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        w.writeheader()
        w.writerows(rows)

    filament_ids = [r["filament_id"] for r in rows]
    assert len(filament_ids) == len(set(filament_ids)), "Duplicate filament_id"
    assert all(r["segmentation_rle"] for r in rows), "Empty RLE"
    stems = {p.stem for p in unique_images}
    for fid in filament_ids:
        assert fid.rsplit("_", 1)[0] in stems

    # Decode-based overlap check on RLE (same as exclusive submitter)
    by_stem: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_stem[r["filament_id"].rsplit("_", 1)[0]].append(r["segmentation_rle"])
    for stem, rles in by_stem.items():
        masks = [rle_counts_to_mask(c, height=args.out_size, width=args.out_size) for c in rles]
        if not assert_no_overlap(masks):
            raise AssertionError(f"RLE overlap for stem={stem}")
    for r in rows[:20]:
        m = rle_counts_to_mask(r["segmentation_rle"], height=args.out_size, width=args.out_size)
        assert int(m.sum()) > 0 and roundtrip_ok(m)

    checks = {
        "n_rows": len(rows),
        "n_images": len(unique_images),
        "n_images_with_preds": len(by_stem),
        "empty_pred_images": empty_images,
        "overlap_fail_images": overlap_fail,
        "columns": ["filament_id", "segmentation_rle"],
        "strict_nonoverlap_verified": True,
        "local_val_pq": LOCAL_VAL_PQ,
        "yolo_weights": str(yolo_weights.relative_to(ROOT)).replace("\\", "/"),
        "pipeline": "indep_min_area → iou_ge_0.50 → exclusive_paint → min_area",
    }
    with (REPORT_DIR / "submission_checks.json").open("w", encoding="utf-8") as f:
        json.dump({"output": str(out_csv.relative_to(ROOT)).replace("\\", "/"), **checks}, f, indent=2)
    (REPORT_DIR / "REPORT.md").write_text(
        f"""# V12 Non-Overlap Submission

- Path: `{out_csv.relative_to(ROOT).as_posix()}`
- Local val PQ (promoted): {LOCAL_VAL_PQ}
- Rows: {len(rows)}
- Images: {len(unique_images)} (empty preds: {empty_images})
- Strict non-overlap: verified on instances + decoded RLE
- YOLO: `{yolo_weights.name}` conf={yolo_conf} min_area={min_area}
- Pipeline: independent → iou≥0.50 → exclusive → min_area
""",
        encoding="utf-8",
    )

    log_path = resolve_path(ROOT, "submissions/submission_log.csv")
    log_exists = log_path.is_file() and log_path.stat().st_size > 0
    with log_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not log_exists:
            w.writerow(
                [
                    "submission_id",
                    "version",
                    "model",
                    "config",
                    "local_metric",
                    "kaggle_score",
                    "notes",
                    "date",
                ]
            )
        w.writerow(
            [
                out_csv.name,
                "v12_indep_iou050_excl",
                "yolo11s@1024+consensus+resnet18_indep_iou050_excl",
                "experiments/v12/configs/resnet18_refiner.yaml",
                f"local_val_pq={LOCAL_VAL_PQ}",
                "",
                (
                    f"rows={len(rows)}; empty_images={empty_images}; "
                    f"yolo_conf={yolo_conf}; min_area={min_area}; "
                    f"iou_thr={IOU_THR}; exclusive_after_iou=1; nonoverlap=1"
                ),
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ]
        )

    print(f"\nWrote {out_csv}", flush=True)
    print(
        f"  images={len(unique_images)} rows={len(rows)} empty={empty_images} "
        f"nonoverlap=OK",
        flush=True,
    )
    print(f"  report -> {REPORT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
