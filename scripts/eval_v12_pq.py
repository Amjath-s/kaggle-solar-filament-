"""V12 val PQ: YOLO11s-seg@1024 (+ optional consensus TTA) (+ optional ResNet18 clamp).

Modes:
  detector   — single-view YOLO masks, score-order exclusive
  consensus  — 4-flip consensus TTA, score-order exclusive
  refined    — consensus + ResNet18@256 refine, clamp to dilated YOLO seed
  rescue     — main consensus@yolo_conf + single-view low-conf gated rescue + refine

Usage:
  .venv\\Scripts\\python.exe -u scripts\\eval_v12_pq.py --mode detector
  .venv\\Scripts\\python.exe -u scripts\\eval_v12_pq.py --mode consensus
  .venv\\Scripts\\python.exe -u scripts\\eval_v12_pq.py --mode refined
  .venv\\Scripts\\python.exe -u scripts\\eval_v12_pq.py --mode rescue --yolo-conf 0.40 --rescue-conf 0.20 --min-area 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.annotations import anns_to_instance_masks, build_file_to_anns_primary, load_coco
from src.data.split import read_split_list
from src.evaluation.pq import compute_pq
from src.inference.local_crops import expand_xyxy, resize_image_mask
from src.inference.v12_consensus import (
    box_iou_matrix,
    detect_consensus_4flip,
    detect_single,
    dilate_mask,
)
from src.models.unet import build_model
from src.utils.config import load_config, resolve_path

OUT_SIZE = 2048
PQ_IOU = 0.5
V7_REF = 0.379202
BASELINE_C040_A064_PQ = 0.425411


def make_masks_exclusive_by_score(
    masks: list[np.ndarray],
    scores: np.ndarray,
    *,
    min_area: int = 20,
) -> list[np.ndarray]:
    if not masks:
        return []
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    h, w = masks[0].shape
    claimed = np.zeros((h, w), dtype=np.uint8)
    out: list[np.ndarray] = []
    for idx in order:
        binary = (masks[int(idx)] > 0).astype(np.uint8)
        kept = np.logical_and(binary, claimed == 0).astype(np.uint8)
        if int(kept.sum()) < min_area:
            continue
        out.append(kept)
        claimed = np.maximum(claimed, kept)
    return out


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


def load_image_and_gt(
    path: Path,
    anns: list,
    out_size: int = OUT_SIZE,
    raster: str = "pil",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float, float]:
    img_u8 = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h0, w0 = img_u8.shape[:2]
    gt_native = anns_to_instance_masks(anns, height=h0, width=w0, raster=raster)
    if h0 == out_size and w0 == out_size:
        return img_u8, img_u8, gt_native, 1.0, 1.0
    img_out = np.asarray(
        Image.fromarray(img_u8, mode="L").resize((out_size, out_size), Image.BILINEAR),
        dtype=np.uint8,
    )
    sx, sy = out_size / float(w0), out_size / float(h0)
    gt_out = [
        np.asarray(
            Image.fromarray((g > 0).astype(np.uint8) * 255, mode="L").resize(
                (out_size, out_size), Image.NEAREST
            )
        )
        > 127
        for g in gt_native
    ]
    gt_out = [g.astype(np.uint8) for g in gt_out]
    return img_u8, img_out, gt_out, sx, sy


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
    # Resize pred back to crop, clamp to dilated seed
    pred_crop = np.asarray(
        Image.fromarray(pred * 255, mode="L").resize(
            (int(exp[2] - exp[0]), int(exp[3] - exp[1])), Image.NEAREST
        )
    )
    pred_crop = (pred_crop > 127).astype(np.uint8)
    seed_bin = (seed_crop > 0).astype(np.uint8)
    allow = dilate_mask(seed_bin, seed_dilate)
    clamped = np.logical_and(pred_crop, allow).astype(np.uint8)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[exp[1] : exp[3], exp[0] : exp[2]] = clamped
    return canvas


def scale_instances_to_out(
    masks: list[np.ndarray],
    boxes: np.ndarray,
    confs: np.ndarray,
    sx: float,
    sy: float,
    out_size: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    out_masks = [resize_mask_to(m, out_size) for m in masks]
    scaled = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(scaled):
        scaled[:, [0, 2]] *= sx
        scaled[:, [1, 3]] *= sy
    return out_masks, scaled, confs


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def filter_rescue_by_main_iou(
    rescue_boxes: np.ndarray,
    rescue_confs: np.ndarray,
    rescue_masks: list[np.ndarray],
    main_boxes: np.ndarray,
    *,
    max_iou: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Keep rescue dets whose max box-IoU vs any main box is < max_iou."""
    rescue_boxes = np.asarray(rescue_boxes, dtype=np.float32).reshape(-1, 4)
    rescue_confs = np.asarray(rescue_confs, dtype=np.float32).reshape(-1)
    main_boxes = np.asarray(main_boxes, dtype=np.float32).reshape(-1, 4)
    if len(rescue_boxes) == 0:
        return rescue_boxes, rescue_confs, []
    if len(main_boxes) == 0:
        return rescue_boxes, rescue_confs, list(rescue_masks)
    ious = box_iou_matrix(rescue_boxes, main_boxes)
    keep = [i for i in range(len(rescue_boxes)) if float(ious[i].max()) < max_iou]
    if not keep:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    return (
        rescue_boxes[keep],
        rescue_confs[keep],
        [rescue_masks[i] for i in keep],
    )


def refine_instances(
    refiner,
    img_u8: np.ndarray,
    boxes: np.ndarray,
    seeds: list[np.ndarray],
    *,
    image_size: int,
    pad_frac: float,
    threshold: float,
    seed_dilate: int,
    device: torch.device,
) -> list[np.ndarray]:
    if not len(seeds):
        return []
    return [
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


def gate_rescue_refined(
    boxes: np.ndarray,
    confs: np.ndarray,
    masks: list[np.ndarray],
    *,
    min_area: int,
    min_fg_frac: float,
    topk: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    keep_boxes: list[np.ndarray] = []
    keep_confs: list[float] = []
    keep_masks: list[np.ndarray] = []
    for box, conf, mask in zip(boxes, confs, masks):
        area = int((mask > 0).sum())
        if area < min_area:
            continue
        ba = box_area(box)
        fg = float(area) / max(ba, 1.0)
        if fg < min_fg_frac:
            continue
        keep_boxes.append(np.asarray(box, dtype=np.float32))
        keep_confs.append(float(conf))
        keep_masks.append(mask)
    if not keep_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    order = np.argsort(-np.asarray(keep_confs, dtype=np.float32), kind="stable")
    if topk and topk > 0:
        order = order[: int(topk)]
    keep_boxes = [keep_boxes[i] for i in order]
    keep_confs = [keep_confs[i] for i in order]
    keep_masks = [keep_masks[i] for i in order]
    return np.stack(keep_boxes, axis=0), np.asarray(keep_confs, dtype=np.float32), keep_masks


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/v12/configs/resnet18_refiner.yaml")
    parser.add_argument(
        "--mode",
        choices=("detector", "consensus", "refined", "rescue"),
        default="detector",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional val image cap for smoke tests")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Override config infer.yolo_weights (does not modify config file)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix appended to output stem (e.g. kaggle_ep80)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=None,
        help="Main-stream YOLO conf (writes results as v12_{mode}_cXXX_*)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="Override config infer.min_area (appends _aXXX to output stem when set)",
    )
    parser.add_argument(
        "--rescue-conf",
        type=float,
        default=0.20,
        help="Rescue-stream single-view YOLO conf (mode=rescue)",
    )
    parser.add_argument(
        "--rescue-max-iou",
        type=float,
        default=0.30,
        help="Drop rescue boxes with max IoU vs main >= this",
    )
    parser.add_argument(
        "--rescue-min-fg",
        type=float,
        default=0.05,
        help="Min refined mask area / box area for rescue keep",
    )
    parser.add_argument(
        "--rescue-min-area",
        type=int,
        default=None,
        help="Min refined mask area for rescue keep (defaults to --min-area)",
    )
    parser.add_argument(
        "--rescue-topk",
        type=int,
        default=0,
        help="Keep at most this many rescue dets per image by score (0=all)",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))
    infer = cfg.get("infer", {})
    images_dir = resolve_path(ROOT, cfg["data"]["images_dir"])
    annotations = resolve_path(ROOT, cfg["data"]["annotations"])
    splits_dir = resolve_path(ROOT, cfg["data"]["splits_dir"])
    names = read_split_list(splits_dir / "val.txt")
    if args.limit and args.limit > 0:
        names = names[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_w = resolve_path(
        ROOT,
        args.yolo_weights
        if args.yolo_weights
        else infer.get("yolo_weights", cfg["yolo"]["weights"]),
    )
    if not yolo_w.is_file():
        raise FileNotFoundError(f"YOLO weights missing: {yolo_w}")
    yolo = YOLO(str(yolo_w))
    yolo_conf = float(args.yolo_conf if args.yolo_conf is not None else infer.get("yolo_conf", 0.30))
    conf_tag = f"c{int(round(yolo_conf * 100)):03d}"
    rescue_conf = float(args.rescue_conf)
    rescue_tag = f"r{int(round(rescue_conf * 100)):03d}"
    yolo_iou = float(infer.get("yolo_iou", 0.7))
    yolo_imgsz = int(infer.get("yolo_imgsz", 1024))
    min_area = int(args.min_area if args.min_area is not None else infer.get("min_area", 20))
    match_iou = float(infer.get("match_iou", 0.5))
    nms_iou = float(infer.get("nms_iou", 0.5))
    pad_frac = float(infer.get("pad_frac", 0.20))
    threshold = float(infer.get("threshold", 0.45))
    seed_dilate = int(infer.get("seed_dilate", 7))
    image_size = int(cfg.get("image_size", 256))
    rescue_max_iou = float(args.rescue_max_iou)
    rescue_min_fg = float(args.rescue_min_fg)
    rescue_min_area = int(args.rescue_min_area if args.rescue_min_area is not None else min_area)
    rescue_topk = int(args.rescue_topk)

    refiner = None
    if args.mode in ("refined", "rescue"):
        ckpt_path = resolve_path(
            ROOT, args.checkpoint or infer.get("local_checkpoint", "experiments/v12/resnet18_refiner/best.pt")
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Refiner checkpoint missing: {ckpt_path}. Train first.")
        model_cfg = cfg.get("model", {})
        refiner = build_model(
            model_cfg.get("name", "resnet18_unet"),
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=1,
            pretrained=False,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt["model_state"])
        refiner.eval()
        print(f"loaded refiner {ckpt_path}", flush=True)

    coco = load_coco(annotations)
    file_to_anns = build_file_to_anns_primary(coco)

    out_dir = resolve_path(ROOT, cfg.get("output", {}).get("dir", "experiments/v12")) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"v12_{args.mode}_{conf_tag}"
    if args.mode == "rescue":
        stem = f"{stem}_{rescue_tag}"
        if abs(rescue_min_fg - 0.05) > 1e-9:
            stem = f"{stem}_fg{int(round(rescue_min_fg * 100)):03d}"
        if abs(rescue_max_iou - 0.30) > 1e-9:
            stem = f"{stem}_miou{int(round(rescue_max_iou * 100)):03d}"
        if args.rescue_min_area is not None:
            stem = f"{stem}_ra{rescue_min_area:03d}"
        if rescue_topk > 0:
            stem = f"{stem}_topk{rescue_topk}"
    if args.min_area is not None:
        stem = f"{stem}_a{min_area:03d}"
    if args.tag:
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.tag.strip())
        if safe_tag:
            stem = f"{stem}_{safe_tag}"
    log_path = out_dir / f"{stem}_pq.log"

    tp = fp = fn = 0
    sq_sum = 0.0
    n_pred = 0
    n_main_total = 0
    n_rescue_kept_total = 0
    t0 = time.time()
    rows = []

    print(
        f"mode={args.mode} device={device} n_val={len(names)} "
        f"yolo={yolo_w.name} conf={yolo_conf} imgsz={yolo_imgsz} min_area={min_area}"
        + (
            f" rescue_conf={rescue_conf} rescue_max_iou={rescue_max_iou} "
            f"rescue_min_fg={rescue_min_fg} rescue_min_area={rescue_min_area} rescue_topk={rescue_topk}"
            if args.mode == "rescue"
            else ""
        ),
        flush=True,
    )

    tqdm_desc = f"v12-{args.mode}-{conf_tag}"
    if args.mode == "rescue":
        tqdm_desc = f"{tqdm_desc}-{rescue_tag}"
    if args.min_area is not None:
        tqdm_desc = f"{tqdm_desc}-a{min_area:03d}"
    for name in tqdm(names, desc=tqdm_desc):
        path = images_dir / name
        anns = file_to_anns.get(name, [])
        img_native, _img_out, gt_masks, sx, sy = load_image_and_gt(path, anns)

        n_main = 0
        n_rescue_kept = 0

        if args.mode == "detector":
            boxes, confs, masks = detect_single(
                yolo, img_native, imgsz=yolo_imgsz, conf=yolo_conf, iou=yolo_iou, device=device
            )
        elif args.mode == "consensus":
            boxes, confs, masks = detect_consensus_4flip(
                yolo,
                img_native,
                imgsz=yolo_imgsz,
                conf=yolo_conf,
                iou=yolo_iou,
                match_iou=match_iou,
                nms_iou=nms_iou,
                device=device,
            )
        elif args.mode == "refined":
            boxes, confs, masks = detect_consensus_4flip(
                yolo,
                img_native,
                imgsz=yolo_imgsz,
                conf=yolo_conf,
                iou=yolo_iou,
                match_iou=match_iou,
                nms_iou=nms_iou,
                device=device,
            )
            masks = refine_instances(
                refiner,
                img_native,
                boxes,
                masks,
                image_size=image_size,
                pad_frac=pad_frac,
                threshold=threshold,
                seed_dilate=seed_dilate,
                device=device,
            )
        else:
            # rescue: main consensus@yolo_conf + single-view rescue@rescue_conf
            main_boxes, main_confs, main_seeds = detect_consensus_4flip(
                yolo,
                img_native,
                imgsz=yolo_imgsz,
                conf=yolo_conf,
                iou=yolo_iou,
                match_iou=match_iou,
                nms_iou=nms_iou,
                device=device,
            )
            main_masks = refine_instances(
                refiner,
                img_native,
                main_boxes,
                main_seeds,
                image_size=image_size,
                pad_frac=pad_frac,
                threshold=threshold,
                seed_dilate=seed_dilate,
                device=device,
            )
            n_main = len(main_masks)

            res_boxes, res_confs, res_seeds = detect_single(
                yolo,
                img_native,
                imgsz=yolo_imgsz,
                conf=rescue_conf,
                iou=yolo_iou,
                device=device,
            )
            res_boxes, res_confs, res_seeds = filter_rescue_by_main_iou(
                res_boxes, res_confs, res_seeds, main_boxes, max_iou=rescue_max_iou
            )
            res_masks = refine_instances(
                refiner,
                img_native,
                res_boxes,
                res_seeds,
                image_size=image_size,
                pad_frac=pad_frac,
                threshold=threshold,
                seed_dilate=seed_dilate,
                device=device,
            )
            res_boxes, res_confs, res_masks = gate_rescue_refined(
                res_boxes,
                res_confs,
                res_masks,
                min_area=rescue_min_area,
                min_fg_frac=rescue_min_fg,
                topk=rescue_topk,
            )
            n_rescue_kept = len(res_masks)

            if n_main and n_rescue_kept:
                boxes = np.concatenate([main_boxes, res_boxes], axis=0)
                confs = np.concatenate([main_confs, res_confs], axis=0)
                masks = list(main_masks) + list(res_masks)
            elif n_main:
                boxes, confs, masks = main_boxes, main_confs, main_masks
            else:
                boxes, confs, masks = res_boxes, res_confs, res_masks

            n_main_total += n_main
            n_rescue_kept_total += n_rescue_kept

        masks_out, _boxes_out, confs_out = scale_instances_to_out(
            masks, boxes, confs, sx, sy, OUT_SIZE
        )
        if masks_out and masks_out[0].shape[0] != OUT_SIZE:
            masks_out = [resize_mask_to(m, OUT_SIZE) for m in masks_out]

        pred = make_masks_exclusive_by_score(masks_out, confs_out, min_area=min_area)
        metrics = compute_pq(gt_masks, pred, iou_threshold=PQ_IOU)
        tp += int(metrics["tp"])
        fp += int(metrics["fp"])
        fn += int(metrics["fn"])
        sq_sum += float(metrics.get("sq", 0.0)) * int(metrics["tp"])
        n_pred += len(pred)
        row = {
            "file_name": name,
            "n_gt": len(gt_masks),
            "n_pred": len(pred),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "pq": float(metrics["pq"]),
        }
        if args.mode == "rescue":
            row["n_main"] = n_main
            row["n_rescue_kept"] = n_rescue_kept
        rows.append(row)

    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
    sq = sq_sum / max(tp, 1)
    pq = sq * rq
    elapsed = time.time() - t0
    summary = {
        "mode": args.mode,
        "n_images": len(names),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sq": sq,
        "rq": rq,
        "pq": pq,
        "n_pred_total": n_pred,
        "v7_ref_pq": V7_REF,
        "delta_vs_v7": pq - V7_REF,
        "baseline_c040_a064_pq": BASELINE_C040_A064_PQ,
        "delta_vs_baseline": pq - BASELINE_C040_A064_PQ,
        "elapsed_sec": elapsed,
        "yolo_weights": str(yolo_w),
        "yolo_conf": yolo_conf,
        "yolo_imgsz": yolo_imgsz,
        "min_area": min_area,
        "gate": (
            "SUBMIT_READY"
            if pq >= float(cfg.get("gates", {}).get("submit_ready_pq", 0.45))
            else "MILESTONE"
            if pq >= float(cfg.get("gates", {}).get("milestone_pq", 0.42))
            else "PROMOTE"
            if pq > V7_REF
            else "HOLD"
        ),
    }
    if args.mode == "rescue":
        summary.update(
            {
                "rescue_conf": rescue_conf,
                "rescue_max_iou": rescue_max_iou,
                "rescue_min_fg": rescue_min_fg,
                "rescue_min_area": rescue_min_area,
                "rescue_topk": rescue_topk,
                "n_main_total": n_main_total,
                "n_rescue_kept_total": n_rescue_kept_total,
                "promote_rescue": bool(pq - BASELINE_C040_A064_PQ >= 0.002),
            }
        )

    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{stem}_per_image.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Keep untagged names only for the default conf=0.30 baseline (do not overwrite sweeps).
    if abs(yolo_conf - 0.30) < 1e-9 and args.mode != "rescue":
        (out_dir / f"v12_{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / f"v12_{args.mode}_per_image.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    extra = ""
    if args.mode == "rescue":
        extra = (
            f"  rescue_conf={rescue_conf:.2f} n_main={n_main_total} n_rescue_kept={n_rescue_kept_total}  "
            f"Δbase={pq - BASELINE_C040_A064_PQ:+.6f} promote={summary['promote_rescue']}"
        )
    msg = (
        f"V12 {args.mode} conf={yolo_conf:.2f} min_area={min_area}: PQ={pq:.6f}  SQ={sq:.6f}  RQ={rq:.6f}  "
        f"TP={tp} FP={fp} FN={fn}  n_pred={n_pred}  ΔV7={pq - V7_REF:+.6f}  "
        f"gate={summary['gate']}  ({elapsed:.0f}s){extra}\n"
    )
    print(msg, flush=True)
    log_path.write_text(msg + json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

