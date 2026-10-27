"""Build V6-B 512 crops from frozen YOLO proposals (Option A).

Usage:
    python scripts/build_yolo_crops.py --config experiments/v6b/configs/local_unet_yolo_crops.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.annotations import (
    anns_to_instance_masks,
    build_file_to_anns_primary,
    load_coco,
    polygons_to_mask,
)
from src.data.split import read_split_list
from src.inference.local_crops import expand_xyxy, resize_image_mask
from src.utils.config import load_config, resolve_path


def box_iou_xyxy(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float((inter + eps) / (area_a + area_b - inter + eps))


def mask_to_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], dtype=np.float32)


def classify_proposal(
    box: np.ndarray,
    gt_masks: list[np.ndarray],
    *,
    hard_neg_max_iou: float,
    hard_neg_max_inter_box: float,
    positive_min_iou: float,
    positive_min_inter_gt: float,
) -> tuple[str, dict]:
    """Return (kind, stats) for one YOLO box vs GT instances."""
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    box_area = float(bw * bh)

    best_iou = 0.0
    best_inter_box = 0.0
    best_inter_gt = 0.0
    any_inter = False

    for g in gt_masks:
        gbox = mask_to_xyxy(g)
        if gbox is None:
            continue
        iou = box_iou_xyxy(box.astype(np.float32), gbox)
        crop_g = g[y1:y2, x1:x2]
        inter = float(crop_g.sum())
        gt_area = float(max(int(g.sum()), 1))
        inter_box = inter / box_area
        inter_gt = inter / gt_area
        if inter > 0:
            any_inter = True
        if iou > best_iou:
            best_iou = iou
        if inter_box > best_inter_box:
            best_inter_box = inter_box
        if inter_gt > best_inter_gt:
            best_inter_gt = inter_gt

    stats = {
        "best_gt_iou": best_iou,
        "best_inter_box": best_inter_box,
        "best_inter_gt": best_inter_gt,
    }

    if best_iou < hard_neg_max_iou and best_inter_box < hard_neg_max_inter_box:
        return "hard_neg", stats
    if best_iou >= positive_min_iou or best_inter_gt >= positive_min_inter_gt:
        return "positive", stats
    if any_inter:
        return "partial", stats
    return "hard_neg", stats


def yolo_boxes_xyxy(model, image_path: Path, imgsz: int, conf: float, device, iou: float):
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return r0.boxes.xyxy.detach().float().cpu().numpy()


def build_split(
    split: str,
    *,
    yolo,
    yolo_device,
    images_dir: Path,
    file_to_anns: dict,
    names: list[str],
    out_dir: Path,
    image_size: int,
    pad_frac: float,
    yolo_imgsz: int,
    yolo_conf: float,
    yolo_iou: float,
    ov: dict,
    seed: int,
    raster: str = "pil",
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    split_dir = out_dir / split
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    raw: list[dict] = []
    per_image_counts: list[int] = []
    census = {"positive": 0, "partial": 0, "hard_neg": 0, "n_images": len(names), "total_proposals": 0}

    for name in tqdm(names, desc=f"yolo-crops-{split}"):
        path = images_dir / name
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        h, w = img.shape[:2]
        anns = file_to_anns.get(name, [])
        gt_masks = anns_to_instance_masks(anns, height=h, width=w, raster=raster)
        union = polygons_to_mask(anns, height=h, width=w, raster=raster)

        boxes = yolo_boxes_xyxy(yolo, path, yolo_imgsz, yolo_conf, yolo_device, yolo_iou)
        per_image_counts.append(int(len(boxes)))
        census["total_proposals"] += int(len(boxes))

        for box0 in boxes:
            x1, y1, x2, y2 = [float(v) for v in box0]
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            exp = expand_xyxy([x1, y1, x2, y2], pad_frac, w, h)
            kind, stats = classify_proposal(
                exp.astype(np.float32),
                gt_masks,
                hard_neg_max_iou=float(ov["hard_neg_max_iou"]),
                hard_neg_max_inter_box=float(ov["hard_neg_max_inter_box"]),
                positive_min_iou=float(ov["positive_min_iou"]),
                positive_min_inter_gt=float(ov["positive_min_inter_gt"]),
            )
            census[kind] = census.get(kind, 0) + 1

            crop_img = img[exp[1] : exp[3], exp[0] : exp[2]]
            if kind == "hard_neg":
                crop_mask = np.zeros_like(crop_img, dtype=np.uint8)
            else:
                crop_mask = union[exp[1] : exp[3], exp[0] : exp[2]].astype(np.uint8)

            img_r, mask_r = resize_image_mask(crop_img, crop_mask, size=image_size)
            assert mask_r is not None
            if kind == "hard_neg":
                mask_r = np.zeros((image_size, image_size), dtype=np.float32)

            raw.append(
                {
                    "file_name": name,
                    "kind": kind,
                    "box": exp.tolist(),
                    "stats": stats,
                    "img_arr": (img_r * 255).astype(np.uint8),
                    "mask_arr": (mask_r > 0.5).astype(np.uint8) * 255,
                }
            )

    # Balance hard negs
    pos_n = sum(1 for r in raw if r["kind"] in ("positive", "partial"))
    neg_items = [r for r in raw if r["kind"] == "hard_neg"]
    keep_items = [r for r in raw if r["kind"] != "hard_neg"]
    cap = float(ov.get("hard_neg_cap_vs_pos", 1.5))
    max_neg = max(1, int(round(pos_n * cap))) if pos_n > 0 else len(neg_items)
    if len(neg_items) > max_neg:
        rng.shuffle(neg_items)
        neg_items = neg_items[:max_neg]
    kept = keep_items + neg_items
    rng.shuffle(kept)

    records: list[dict] = []
    for idx, rec in enumerate(kept):
        stem = f"{idx:06d}"
        Image.fromarray(rec["img_arr"], mode="L").save(img_dir / f"{stem}.png")
        Image.fromarray(rec["mask_arr"], mode="L").save(mask_dir / f"{stem}.png")
        records.append(
            {
                "id": stem,
                "file_name": rec["file_name"],
                "kind": rec["kind"],
                "box": rec["box"],
                "stats": rec["stats"],
            }
        )

    census["kept"] = {
        "n": len(records),
        "positive": sum(1 for r in records if r["kind"] == "positive"),
        "partial": sum(1 for r in records if r["kind"] == "partial"),
        "hard_neg": sum(1 for r in records if r["kind"] == "hard_neg"),
    }
    census["proposals_per_image"] = {
        "mean": float(np.mean(per_image_counts)) if per_image_counts else 0.0,
        "min": int(min(per_image_counts)) if per_image_counts else 0,
        "max": int(max(per_image_counts)) if per_image_counts else 0,
    }

    with (split_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"split": split, "n": len(records), "items": records}, f)
    with (split_dir / "census.json").open("w", encoding="utf-8") as f:
        json.dump(census, f, indent=2)
    return records, census


def main() -> None:
    from ultralytics import YOLO
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/v6b/configs/local_unet_yolo_crops.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))

    images_dir = resolve_path(ROOT, cfg["data"]["images_dir"])
    annotations = resolve_path(ROOT, cfg["data"]["annotations"])
    splits_dir = resolve_path(ROOT, cfg["data"]["splits_dir"])
    out_dir = resolve_path(ROOT, cfg["data"].get("crop_root", "experiments/v6b/crops"))
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_cfg = cfg.get("yolo", {})
    yolo_weights = resolve_path(ROOT, yolo_cfg.get("weights", "experiments/v3/yolo8n_seg/weights/best.pt"))
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(yolo_weights))

    coco = load_coco(annotations)
    file_to_anns = build_file_to_anns_primary(coco)
    ov = cfg.get("overlap", {})

    meta = {
        "pad_frac": float(cfg.get("pad_frac", 0.20)),
        "image_size": int(cfg.get("image_size", 512)),
        "yolo_weights": str(yolo_weights),
        "yolo_conf": float(yolo_cfg.get("conf", 0.25)),
        "yolo_iou": float(yolo_cfg.get("iou", 0.7)),
        "yolo_imgsz": int(yolo_cfg.get("imgsz", 640)),
        "normalization": "uint8/255",
        "raster": cfg["data"].get("raster", "pil"),
        "overlap": ov,
        "splits": {},
    }

    for split, seed_off in (("train", 0), ("val", 1)):
        names = read_split_list(splits_dir / f"{split}.txt")
        _, census = build_split(
            split,
            yolo=yolo,
            yolo_device=device,
            images_dir=images_dir,
            file_to_anns=file_to_anns,
            names=names,
            out_dir=out_dir,
            image_size=int(cfg.get("image_size", 512)),
            pad_frac=float(cfg.get("pad_frac", 0.20)),
            yolo_imgsz=int(yolo_cfg.get("imgsz", 640)),
            yolo_conf=float(yolo_cfg.get("conf", 0.25)),
            yolo_iou=float(yolo_cfg.get("iou", 0.7)),
            ov=ov,
            seed=int(cfg.get("seed", 42)) + seed_off,
            raster=meta["raster"],
        )
        meta["splits"][split] = census
        print(f"{split} census:", json.dumps(census, indent=2))

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("V6-B YOLO crop cache done ->", out_dir)


if __name__ == "__main__":
    main()
