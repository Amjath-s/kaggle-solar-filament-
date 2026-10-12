"""V6 checkpoint ablations: resolution, crop TTA, and segmentation thresholds."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils


RESOLUTIONS = (1024, 1536, 2048)
CONFIDENCES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)
THRESHOLDS = (0.5, 0.65, 0.8)
MIN_AREA = 20
DEFAULT_REFINER_EPOCHS = 0


def load_base(source):
    spec = importlib.util.spec_from_file_location("solar_base", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def annotation_rle(info, annotation):
    polygons = annotation["segmentation"]
    return mask_utils.merge(mask_utils.frPyObjects(polygons, info["height"], info["width"]))


def rounded_legacy_rle(info, annotation):
    mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for polygon in annotation["segmentation"]:
        points = np.round(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return encode(mask)


def encode_crop(crop, bounds, image_shape):
    """Encode a placed crop without allocating/scanning a full-resolution mask."""
    x0, y0, x1, y1 = bounds
    height, width = image_shape
    if crop.shape != (y1 - y0, x1 - x0) or not (
        0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height
    ):
        raise ValueError("Crop bounds do not match image dimensions")
    transitions = np.diff(np.pad(crop.astype(np.int8), ((1, 1), (0, 0))), axis=0).T
    columns, ys = np.where(transitions == 1)
    end_columns, ends = np.where(transitions == -1)
    starts = (columns + x0) * height + ys + y0
    ends = (end_columns + x0) * height + ends + y0
    if not len(starts):
        counts = [height * width]
    else:
        # Merge runs touching at a full-height column boundary.
        new_run = np.r_[True, starts[1:] != ends[:-1]]
        starts, ends = starts[new_run], ends[np.r_[new_run[1:], True]]
        counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
        counts[0] = starts[0]
        counts[1::2] = ends - starts
        counts[2:-1:2] = starts[1:] - ends[:-1]
        counts[-1] = height * width - ends[-1]
        counts = counts.tolist()
    return mask_utils.frPyObjects({"size": [height, width], "counts": counts}, height, width)


def clean_predictions(masks, scores):
    # A confidence cut removes a suffix, so lower-score masks cannot alter survivors.
    if len(masks) != len(scores):
        raise ValueError("Every proposal must have a confidence score")
    occupied = np.zeros_like(masks[0], dtype=bool) if masks else None
    kept, kept_scores = [], []
    for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
        mask = masks[index].astype(bool) & ~occupied
        if mask.sum() < MIN_AREA:
            continue
        occupied |= mask
        kept.append(encode(mask))
        kept_scores.append(scores[index])
    return kept, np.asarray(kept_scores)


def pq_counts(iou, n_gt, n_pred):
    if not n_gt or not n_pred:
        return 0.0, 0, n_pred, n_gt
    hits = iou > 0.5
    return (
        float(iou[hits].sum()), int(hits.sum()),
        int((~hits.any(axis=0)).sum()), int((~hits.any(axis=1)).sum()),
    )


def pq_value(total):
    iou, tp, fp, fn = total
    denominator = tp + 0.5 * (fp + fn)
    return float(iou / denominator) if denominator else 0.0


def variant_key(resolution, mode, threshold, tta, confidence):
    return f"r{resolution}_{mode}_t{threshold:.2f}_tta{int(tta)}_c{confidence:.2f}"


@torch.no_grad()
def crop_probabilities(base, refiner, gray, masks, use_tta):
    bounds = [base.square_bounds(mask) for mask in masks]
    crops = [cv2.resize(gray[y0:y1, x0:x1], (base.CFG.crop_size, base.CFG.crop_size))
             for x0, y0, x1, y1 in bounds]
    normal, averaged = [], []
    device = next(refiner.parameters()).device
    for start in range(0, len(crops), 8):
        batch = torch.from_numpy(np.stack(crops[start:start + 8])).float()
        batch = batch[:, None].repeat(1, 3, 1, 1).to(device) / 255
        original = refiner(batch).sigmoid()
        normal.extend(original[:, 0].cpu().numpy())
        if use_tta:
            total = original.clone()
            for dims in ((2,), (3,), (2, 3)):
                total += refiner(batch.flip(dims)).sigmoid().flip(dims)
            averaged.extend((total[:, 0] / 4).cpu().numpy())
    return bounds, normal, averaged


def materialize_masks(masks, bounds, probabilities, threshold, mode):
    refined = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for mask, (x0, y0, x1, y1), probability in zip(masks, bounds, probabilities):
        crop = cv2.resize(probability, (x1 - x0, y1 - y0)) >= threshold
        if mode == "refined_constrained":
            crop &= cv2.dilate(mask[y0:y1, x0:x1], kernel) > 0
        if not crop.any():
            refined.append(mask)
            continue
        full = np.zeros_like(mask)
        full[y0:y1, x0:x1] = crop
        refined.append(full)
    return refined


def clean_refined_predictions(masks, scores, bounds, probabilities, threshold, mode):
    if not (len(masks) == len(scores) == len(bounds) == len(probabilities)):
        raise ValueError("Refinement inputs must have matching lengths")
    if not masks:
        return [], np.asarray([])
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rles, kept_scores = [], []
    for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
        x0, y0, x1, y1 = bounds[index]
        proposal = masks[index][y0:y1, x0:x1]
        crop = cv2.resize(probabilities[index], (x1 - x0, y1 - y0)) >= threshold
        if mode == "refined_constrained":
            crop &= cv2.dilate(proposal, kernel) > 0
        if not crop.any():
            crop = proposal.astype(bool)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        rles.append(encode_crop(crop, bounds[index], occupied.shape))
        kept_scores.append(scores[index])
    return rles, np.asarray(kept_scores)


def prediction_variants(base, detector, refiner, image_path, resolution, selected=None):
    base.CFG.imgsz = resolution
    confidence = selected["confidence"] if selected is not None else min(CONFIDENCES)
    masks, scores = base.predict_yolo(detector, image_path, confidence)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {image_path}")
    if selected is None or selected["mode"] == "yolo_only":
        yield "yolo_only", 0.0, False, *clean_predictions(masks, scores)
        if selected is not None:
            return
    use_tta = selected is None or selected["tta"]
    bounds, normal, averaged = crop_probabilities(base, refiner, gray, masks, use_tta)
    settings = (
        [(selected["mode"], selected["threshold"], selected["tta"])]
        if selected is not None else
        [(mode, threshold, tta) for tta in (False, True)
         for threshold in THRESHOLDS for mode in ("refined_free", "refined_constrained")]
    )
    for mode, threshold, tta in settings:
        rles, kept_scores = clean_refined_predictions(
            masks, scores, bounds, averaged if tta else normal, threshold, mode)
        yield mode, threshold, tta, rles, kept_scores


def validation_split(base, images, annotations):
    best_ids = base.best_annotator_ids(images, annotations)
    files = sorted(best_ids)
    random.Random(42).shuffle(files)
    held_out = files[:round(len(files) * 0.2)]
    tune = set(held_out[:94])
    audit = set(held_out[94:])
    by_file = {name: [] for name in held_out}
    for image_id, info in images.items():
        if info["file_name"] in by_file:
            by_file[info["file_name"]].append(image_id)
    assert tune.isdisjoint(audit)
    return held_out, tune, audit, best_ids, by_file


def verify_split(provenance, held_out, images):
    expected = set(provenance["validation_files"])
    training = set(provenance["training_files"])
    if expected != set(held_out) or training & expected:
        raise ValueError("Checkpoint split does not match inference validation split")
    if training | expected != {info["file_name"] for info in images.values()}:
        raise ValueError("Checkpoint provenance does not cover the current dataset")


def finetune_refiner(base, refiner, images, annotations, provenance, output, epochs):
    """Continue V6 on canonical targets; select epochs using tuning files only."""
    _, tune, audit, best_ids, _ = validation_split(base, images, annotations)
    training = set(provenance["training_files"])
    train_ids = [image_id for image_id, info in images.items() if info["file_name"] in training]
    valid_ids = [best_ids[name] for name in sorted(tune)]
    assert training.isdisjoint(tune | audit)
    base.annotation_mask = lambda shape, ann: mask_utils.decode(annotation_rle(
        {"height": shape[0], "width": shape[1]}, ann))
    train_images, train_masks = base.build_crop_arrays(train_ids, images, annotations)
    valid_images, valid_masks = base.build_crop_arrays(valid_ids, images, annotations)
    train_loader = torch.utils.data.DataLoader(
        base.CropDataset(train_images, train_masks, augment=True),
        batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        base.CropDataset(valid_images, valid_masks),
        batch_size=32, num_workers=2, pin_memory=True)
    device = next(refiner.parameters()).device
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = output / "best_refiner_coco.pt"

    def validation_dice():
        refiner.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for batch, target in valid_loader:
                batch, target = batch.to(device), target.to(device)
                loss = base.soft_dice_loss(refiner(batch), target)
                total += (1.0 - float(loss)) * len(batch)
                count += len(batch)
        return total / count

    best_dice = validation_dice()
    best_epoch = 0
    torch.save(refiner.state_dict(), checkpoint)
    history = [dict(epoch=0, tune_crop_soft_dice=best_dice)]
    print(f"refiner baseline tune_crop_soft_dice={best_dice:.6f}", flush=True)
    for epoch in range(1, epochs + 1):
        refiner.train()
        for batch, target in train_loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = refiner(batch)
                loss = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                        + 0.5 * base.soft_dice_loss(logits, target))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        score = validation_dice()
        history.append(dict(epoch=epoch, tune_crop_soft_dice=score))
        if score > best_dice:
            best_dice, best_epoch = score, epoch
            torch.save(refiner.state_dict(), checkpoint)
        scheduler.step()
        pd.DataFrame(history).to_csv(output / "refiner_finetune_history.csv", index=False)
        print(f"refiner epoch={epoch}/{epochs} tune_crop_soft_dice={score:.6f} "
              f"best_epoch={best_epoch}", flush=True)
    refiner.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    refiner.eval()
    return dict(epochs=epochs, selected_epoch=best_epoch, learning_rate=5e-5,
                selection="tune crop soft Dice; end-to-end PQ evaluated separately",
                tune_crop_soft_dice=best_dice, training_crops=len(train_images),
                tuning_crops=len(valid_images), checkpoint_sha256=hashlib.sha256(
                    checkpoint.read_bytes()).hexdigest())


def sweep(base, detector, refiner, images, annotations, output, resolutions, limit):
    held_out, tune, audit, best_ids, by_file = validation_split(base, images, annotations)
    if limit:
        held_out = held_out[:limit]
    totals, configs, diagnostics = {}, {}, []
    cache_dir = output / "prediction_cache"
    cache_dir.mkdir(exist_ok=True)
    for resolution in resolutions:
        started = time.monotonic()
        for index, name in enumerate(held_out):
            gt_entries = []
            for image_id in by_file[name]:
                info = images[image_id]
                gt = [annotation_rle(info, ann) for ann in annotations.get(image_id, [])]
                gt_entries.append(("coco", image_id,
                                   [rle for rle in gt if mask_utils.area(rle) > 0]))
                if image_id == best_ids[name]:
                    rounded = [rounded_legacy_rle(info, ann)
                               for ann in annotations.get(image_id, [])]
                    gt_entries.append(("rounded_legacy", image_id,
                                       [r for r in rounded if mask_utils.area(r) >= MIN_AREA]))
            cached = []
            for mode, threshold, tta, rles, scores in prediction_variants(
                base, detector, refiner, base.CFG.train_dir / name, resolution
            ):
                cached.append(dict(mode=mode, threshold=threshold, tta=tta,
                                   scores=scores.tolist(), rles=[
                                       dict(size=rle["size"], counts=rle["counts"].decode("utf-8"))
                                       for rle in rles]))
                for target, image_id, gt in gt_entries:
                    iou = (mask_utils.iou(rles, gt, [0] * len(gt)).T if rles and gt
                           else np.zeros((len(gt), len(rles))))
                    for confidence in CONFIDENCES:
                        key = variant_key(resolution, mode, threshold, tta, confidence)
                        configs[key] = dict(resolution=resolution, mode=mode, threshold=threshold,
                                            tta=tta, confidence=confidence)
                        keep = scores >= confidence
                        counts = pq_counts(iou[:, keep], len(gt), int(keep.sum()))
                        current = totals.setdefault(key, {group: np.zeros(4) for group in
                                                         ("all", "legacy", "tune", "audit",
                                                          "canonical_best")})
                        group = "tune" if name in tune else "audit"
                        if target == "rounded_legacy":
                            current["legacy"] += counts
                            group = "legacy"
                        else:
                            current["all"] += counts
                            current[group] += counts
                            if image_id == best_ids[name]:
                                current["canonical_best"] += counts
                        diagnostics.append(dict(candidate=key, image_id=image_id, file_name=name,
                                                group=group, target=target,
                                                is_legacy=image_id == best_ids[name],
                                                iou_sum=counts[0], tp=counts[1], fp=counts[2], fn=counts[3]))
            with gzip.open(cache_dir / f"r{resolution}_{Path(name).stem}.json.gz", "wt") as file:
                json.dump(dict(file_name=name, resolution=resolution, variants=cached), file)
            print(f"resolution={resolution} validation={index + 1}/{len(held_out)} "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        write_results(totals, configs, diagnostics, output)
    return totals, configs


def write_results(totals, configs, diagnostics, output):
    rows = []
    for key, groups in totals.items():
        row = dict(candidate=key, **configs[key])
        for group, total in groups.items():
            row[f"{group}_pq"] = pq_value(total)
            row.update({f"{group}_{field}": value for field, value in
                        zip(("iou_sum", "tp", "fp", "fn"), total)})
        rows.append(row)
    pd.DataFrame(rows).sort_values("tune_pq", ascending=False).to_csv(
        output / "inference_candidates.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "inference_image_diagnostics.csv", index=False)


def select_candidates(totals, configs):
    # Audit values are deliberately excluded from selection.
    ranked = sorted(totals, key=lambda key: (-pq_value(totals[key]["tune"]), key))
    selected = ranked[:1]
    control = variant_key(1024, "refined_free", 0.65, False, 0.4)
    if control in configs and control not in selected:
        selected.append(control)
    return selected


def export_submission(base, detector, refiner, config, output):
    rows, counts = [], []
    for image_index, image_path in enumerate(sorted(base.CFG.test_dir.glob("*.jpeg")), start=1):
        count = 0
        for _, _, _, rles, scores in prediction_variants(
            base, detector, refiner, image_path, config["resolution"], selected=config
        ):
            for rle, score in zip(rles, scores):
                if score < config["confidence"]:
                    continue
                count += 1
                rows.append({"filament_id": f"{image_path.stem}_{count}",
                             "segmentation_rle": rle["counts"].decode("utf-8")})
        counts.append(count)
        if image_index % 30 == 0:
            print(f"export={output.name} images={image_index}/180 rows={len(rows)}", flush=True)
    assert len(counts) == 180, f"Expected 180 test images, found {len(counts)}"
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(output, index=False)
    return dict(file=output.name, rows=len(rows), populated_images=sum(n > 0 for n in counts),
                missing_images=sum(n == 0 for n in counts), **config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test validation image limit")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(RESOLUTIONS))
    parser.add_argument("--refiner-epochs", type=int, default=DEFAULT_REFINER_EPOCHS)
    args = parser.parse_args()
    if (args.limit < 0 or args.refiner_epochs < 0 or not args.resolutions
            or any(r <= 0 or r % 32 for r in args.resolutions)):
        parser.error("Use a nonnegative limit and resolutions divisible by 32")
    source = args.source
    if source is None:
        matches = list(Path("/kaggle/input").rglob("v6_provenance.json"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one V6 checkpoint source, found {matches}")
        source = matches[0].parent
    provenance = json.loads((source / "v6_provenance.json").read_text())
    for name, expected in provenance["sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    base = load_base(source / "solar_base.py")
    base.seed_everything(42)
    args.output.mkdir(parents=True, exist_ok=True)
    _, images, annotations = base.load_coco()
    held_out, tune, audit, best_ids, _ = validation_split(base, images, annotations)
    verify_split(provenance, held_out, images)
    manifest = dict(experiment="v9_v6_resolution_refiner_calibration", source=provenance,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    target_rasterization="COCO polygons; rounded OpenCV only for legacy comparison",
                    resolutions=args.resolutions, thresholds=THRESHOLDS, confidences=CONFIDENCES,
                    tune_files=sorted(tune), audit_files=sorted(audit),
                    legacy_ids=[best_ids[name] for name in held_out], smoke_limit=args.limit,
                    refiner_epochs=args.refiner_epochs,
                    audit_caveat="Excluded from inference tuning; used in V6 checkpoint selection",
                    versions={name: base.version(name) for name in
                              ("ultralytics", "torch", "torchvision", "pycocotools", "numpy")})
    (args.output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    # Large OpenCV/BLAS thread pools slow the many small CPU mask operations on Kaggle.
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    detector = base.YOLO(str(source / "yolo_best.pt"))
    refiner = base.CropUNet(pretrained=False).to("cuda:0")
    refiner.load_state_dict(torch.load(source / "best_refiner.pt", map_location="cuda:0", weights_only=True))
    refiner.eval()
    if args.refiner_epochs:
        manifest["finetuning"] = finetune_refiner(
            base, refiner, images, annotations, provenance, args.output, args.refiner_epochs)
        (args.output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2))
    totals, configs = sweep(base, detector, refiner, images, annotations,
                            args.output, args.resolutions, args.limit)
    if args.limit:
        print("Smoke test complete; no submission exported", flush=True)
        return
    selected = select_candidates(totals, configs)
    if args.refiner_epochs:
        selected = selected[:1]
    exports = []
    for index, key in enumerate(selected):
        filename = "submission.csv" if index == 0 else f"submission_candidate_{index + 1}.csv"
        diagnostic = export_submission(base, detector, refiner, configs[key], args.output / filename)
        diagnostic.update(candidate=key, **{f"{group}_pq": pq_value(total)
                                           for group, total in totals[key].items()})
        exports.append(diagnostic)
        (args.output / "submission_candidates.json").write_text(json.dumps(exports, indent=2))
    print(json.dumps(exports, indent=2), flush=True)


if __name__ == "__main__":
    main()
