"""Bounded V11 proposal-aware refiner experiment; never submits predictions.

CPU-importable geometry/protocol helpers; GPU imports live inside runtime paths.
The stager embeds frozen source/cache identities. See docs/v11_execution_2026-09-06.md.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


CONTROL = "v11_frozen_control"
ARMS = ("gt_centered", "proposal_centered", "proposal_conditioned")
EPOCHS = 4
BATCH_SIZE = 16
CROP_SIZE = 256
CONFIDENCE = 0.4
THRESHOLD = 0.4
RAW_CACHE_SHA256 = None
MODULE_SHA256 = None
V10_REFERENCE = {
    "candidate": "v10_quality_rank_rescue40",
    "diagnostics_sha256": "8a334947ee5eb5876b618a1f00e23b023af2e2f8ed4be22b62067f3af4b8375a",
    "tune": [430.3746937713222, 632, 217, 502],
    "audit": [184.34953270420428, 267, 90, 192],
}
COUNTS = ("iou_sum", "tp", "fp", "fn")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic progress publication, leaving no partially written resume manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pq(counts):
    iou, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return float(iou / denominator) if denominator else 0.0


def declared_candidates(epochs=EPOCHS):
    if epochs not in (1, EPOCHS):
        raise ValueError("Only declared full or smoke epoch budgets are allowed")
    configs = {CONTROL: dict(arm="frozen", epoch=0, conditioned=False,
                            mode="refined_constrained", priority=0)}
    for arm_index, arm in enumerate(ARMS, 1):
        for epoch in range(1, epochs + 1):
            for mode in (("refined_constrained", "refined_free") if arm == ARMS[2]
                         else ("refined_constrained",)):
                key = f"v11_{arm}_e{epoch}_{mode}"
                configs[key] = dict(arm=arm, epoch=epoch, conditioned=arm == ARMS[2],
                                    mode=mode, priority=4 if mode == "refined_free" else arm_index)
    return configs


def protocol(smoke=False, train_limit=3, tune_limit=3, sample_limit=32):
    return dict(
        experiment="v11_proposal_refiner", smoke=bool(smoke), epochs=1 if smoke else EPOCHS,
        batch_size=BATCH_SIZE, crop_size=CROP_SIZE, learning_rate=5e-5, weight_decay=1e-4,
        training_precision="float32; autocast and gradient scaling disabled for all arms",
        scheduler="cosine over declared epochs", loss="0.5 BCEWithLogits + 0.5 soft Dice",
        confidence=CONFIDENCE, threshold=THRESHOLD, min_prediction_area=20,
        detector_resolution=1024, detector_views=4, crop_views=4,
        matching=dict(min_iou=0.30, max_second_iou_exclusive=0.20, min_margin=0.15,
                      one_proposal_per_annotation=True, per_annotator=True,
                      unmatched="ignore", all_categories=True),
        jitter=dict(scale=[0.9, 1.1], center_fraction=0.1),
        training_proposals="in-sample V6; training filenames only; optimistic screening",
        selection="final-image global tune PQ; ties prefer simpler arm then earlier epoch",
        promotion=dict(min_tune_delta_v6=0.01, min_tune_delta_v10=0.0,
                       min_audit_delta_v6=0.0, min_audit_delta_v10=0.0),
        candidates=declared_candidates(1 if smoke else EPOCHS),
        smoke_limits=dict(training_files=train_limit, tuning_files=tune_limit,
                          training_samples=sample_limit) if smoke else None,
        audit="winner and frozen control once, only after selection is saved",
        audit_caveat="Historical audit influenced earlier checkpoint selection; not untouched",
        public_score="unmeasured", automatic_submission=False)


def verify_training_names(names, provenance, tune, audit, require_full=False):
    names = list(names)
    train = set(provenance["training_files"])
    held = set(provenance["validation_files"])
    if len(names) != len(set(names)) or not names or not set(names) <= train:
        raise ValueError("Training requires unique filenames from the pinned training split")
    if train & held or train & (set(tune) | set(audit)) or set(tune) & set(audit):
        raise ValueError("Training/tuning/audit overlap")
    if held != set(tune) | set(audit):
        raise ValueError("Tuning and audit must cover the pinned validation split")
    if require_full and set(names) != train:
        raise ValueError("Full experiment must cover every training filename")
    return names


def match_positive_proposals(iou, scores, min_iou=0.30, max_second=0.20, min_margin=0.15):
    """Rows=GT of ONE annotator, columns=proposals; return one positive per GT.

    Duplicate proposals never increase a target's sample weight. No background
    targets are manufactured, including proposals unmatched by this annotator.
    """
    iou = np.asarray(iou, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if iou.ndim != 2 or iou.shape[1] != len(scores):
        raise ValueError("Expected GT-by-proposal IoU matrix and matching scores")
    if not np.isfinite(iou).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite matching input")
    if ((iou < 0) | (iou > 1)).any():
        raise ValueError("IoU outside [0, 1]")
    best_for_gt = {}
    if not iou.size:
        return []
    for proposal in range(iou.shape[1]):
        order = np.argsort(-iou[:, proposal], kind="stable")
        target = int(order[0])
        best = float(iou[target, proposal])
        second = float(iou[order[1], proposal]) if len(order) > 1 else 0.0
        if best < min_iou or second >= max_second or best - second < min_margin:
            continue
        rank = (best, float(scores[proposal]), -proposal)
        if target not in best_for_gt or rank > best_for_gt[target][0]:
            best_for_gt[target] = (rank, proposal)
    return [(target, best_for_gt[target][1]) for target in sorted(best_for_gt)]


def jitter_bounds(bounds, shape, rng):
    """Jitter native-image square geometry before raster resize."""
    x0, y0, x1, y1 = bounds
    height, width = shape
    side = max(x1 - x0, y1 - y0)
    new_side = min(max(1, round(side * float(rng.uniform(0.9, 1.1)))), height, width)
    cx = (x0 + x1) / 2 + float(rng.uniform(-0.1, 0.1)) * side
    cy = (y0 + y1) / 2 + float(rng.uniform(-0.1, 0.1)) * side
    left = min(max(round(cx - new_side / 2), 0), width - new_side)
    top = min(max(round(cy - new_side / 2), 0), height - new_side)
    return int(left), int(top), int(left + new_side), int(top + new_side)


def crop_triplet(gray, seed, target, bounds, size=CROP_SIZE):
    x0, y0, x1, y1 = bounds
    if gray.ndim != 2 or gray.shape != seed.shape or gray.shape != target.shape:
        raise ValueError("Image, real proposal seed, and target must share full-image geometry")
    if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
        raise ValueError("Invalid native-image crop bounds")
    image = cv2.resize(gray[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)
    masks = [cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (size, size),
                        interpolation=cv2.INTER_NEAREST) for m in (seed, target)]
    return image.astype(np.float32) / 255.0, masks[0], masks[1]


def augment_triplet(image, seed, target, rng):
    rotations, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
    transformed = [np.rot90(array, rotations) for array in (image, seed, target)]
    if flip:
        transformed = [np.flip(array, axis=1) for array in transformed]
    image, seed, target = transformed
    if rng.random() < 0.5:
        image = image ** float(rng.uniform(0.85, 1.2))
    if rng.random() < 0.5:
        image = np.clip(image * float(rng.uniform(0.9, 1.1)), 0, 1)
    return tuple(np.ascontiguousarray(array) for array in (image, seed, target))


def epoch_sample_order(records, epoch):
    """Shuffle filenames then their samples; identical arms and cheap image reuse."""
    by_file = {}
    for index, record in enumerate(records):
        by_file.setdefault(record["file_name"], []).append(index)
    rng = np.random.default_rng(42 + epoch)
    names = list(sorted(by_file))
    rng.shuffle(names)
    order = []
    for name in names:
        indices = by_file[name].copy()
        rng.shuffle(indices)
        order.extend(indices)
    return order


def select_candidate(configs, tuning_counts):
    if set(configs) != set(tuning_counts):
        raise ValueError("Selection requires every declared tuning candidate")
    scores = {key: pq(value) for key, value in tuning_counts.items()}
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite tuning score")
    maximum = max(scores.values())
    tied = [key for key in configs if maximum - scores[key] <= 1e-12]
    return min(tied, key=lambda key: (configs[key]["priority"], configs[key]["epoch"], key))


def audit_candidates(selection, configs, smoke=False):
    if smoke:
        raise ValueError("Smoke runs must never evaluate audit images")
    if selection not in configs:
        raise ValueError("Audit requires a frozen, declared tuning selection")
    return list(dict.fromkeys((CONTROL, selection)))


def promotion_gate(selected, tuning_counts, audit_counts, smoke=False):
    if smoke:
        return dict(promotable=False, reason="smoke runs cannot promote")
    if selected not in tuning_counts or selected not in audit_counts or CONTROL not in audit_counts:
        raise ValueError("Promotion requires complete selected/control tuning and audit counts")
    deltas = dict(tune_v6=pq(tuning_counts[selected]) - pq(tuning_counts[CONTROL]),
                  tune_v10=pq(tuning_counts[selected]) - pq(V10_REFERENCE["tune"]),
                  audit_v6=pq(audit_counts[selected]) - pq(audit_counts[CONTROL]),
                  audit_v10=pq(audit_counts[selected]) - pq(V10_REFERENCE["audit"]))
    promote = (selected != CONTROL and deltas["tune_v6"] >= 0.01
               and deltas["tune_v10"] >= 0 and deltas["audit_v6"] >= 0
               and deltas["audit_v10"] >= 0)
    return dict(promotable=promote, deltas=deltas,
                reason="all predeclared gates passed" if promote else "predeclared improvement gates failed")


def add_seed_channel(model):
    """Load the old 3-channel state FIRST; expand just conv1 and normalization."""
    import torch
    old = model.stem[0]
    if old.in_channels != 3 or model.mean.shape[1] != 3 or model.std.shape[1] != 3:
        raise ValueError("Seed expansion requires the unchanged RGB refiner")
    new = torch.nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                          old.dilation, old.groups, old.bias is not None, old.padding_mode,
                          device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.stem[0] = new
    model.mean = torch.cat((model.mean, torch.zeros_like(model.mean[:, :1])), dim=1)
    model.std = torch.cat((model.std, torch.ones_like(model.std[:, :1])), dim=1)
    return model.eval()


def build_model(base, checkpoint, conditioned=False, candidate_checkpoint=None):
    import torch
    model = base.CropUNet(pretrained=False).to("cuda:0")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=True))
    if conditioned:
        add_seed_channel(model)
    if candidate_checkpoint is not None:
        model.load_state_dict(torch.load(candidate_checkpoint, map_location="cuda:0", weights_only=True)["model"])
    return model.eval()


def numerical_seed_test(base, checkpoint, image, seed, target):
    """Actual GPU logits/loss, proving zero initialization and learnable seed input."""
    import torch
    original = build_model(base, checkpoint).eval()
    conditioned = build_model(base, checkpoint, True).eval()
    rgb = torch.from_numpy(image).to("cuda:0")[None, None].repeat(1, 3, 1, 1)
    spatial = torch.from_numpy(seed.astype(np.float32)).to("cuda:0")[None, None]
    gt = torch.from_numpy(target.astype(np.float32)).to("cuda:0")[None, None]
    four = torch.cat((rgb, spatial), 1)
    with torch.no_grad():
        old_logits, new_logits = original(rgb), conditioned(four)
        torch.testing.assert_close(old_logits, new_logits, rtol=1e-4, atol=1e-4)
        error = float((old_logits - new_logits).abs().max())
    logits = conditioned(four)
    loss = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, gt)
            + 0.5 * base.soft_dice_loss(logits, gt))
    loss.backward()
    gradient = conditioned.stem[0].weight.grad[:, 3]
    gradient_l1 = float(gradient.abs().sum())
    if not torch.isfinite(gradient).all() or gradient_l1 <= 0:
        raise ValueError("Fourth-channel weights received no finite learning signal")
    del original, conditioned, logits, loss
    torch.cuda.empty_cache()
    return dict(initial_logits_max_abs_error=error, rtol=1e-4, atol=1e-4,
                seed_weight_gradient_l1=gradient_l1, passed=True)


def load_runtime(args):
    """Resolve every pinned input before any training; no detector validation rerun."""
    global core, tta, v10
    import optimize_inference as core
    import detector_tta as tta
    import v10_quality as v10
    import torch
    source_paths = list(args.input_root.rglob("v6_provenance.json"))
    refiner_paths = list(args.input_root.rglob("best_refiner_coco.pt"))
    if len(source_paths) != 1 or len(refiner_paths) != 1:
        raise ValueError("Expected exactly one pinned V6 source and canonical refiner")
    provenance_path, refiner_path = source_paths[0], refiner_paths[0]
    if sha256(provenance_path) != v10.PROVENANCE_SHA256 or sha256(refiner_path) != v10.REFINER_SHA256:
        raise ValueError("Changed V6 provenance or canonical refiner")
    provenance = json.loads(provenance_path.read_text())
    for name, digest in provenance["sha256"].items():
        if sha256(provenance_path.parent / name) != digest:
            raise ValueError(f"Changed V6 source artifact {name}")
    base = core.load_base(provenance_path.parent / "solar_base.py")
    base.seed_everything(42)
    if sha256(base.CFG.train_json) != provenance["annotation_sha256"]:
        raise ValueError("Changed canonical annotations")
    _, images, annotations = base.load_coco()
    split = core.validation_split(base, images, annotations)
    held, tune, audit, _, by_file = split
    core.verify_split(provenance, held, images)
    if (len(provenance["training_files"]), len(tune), len(audit), sum(map(len, by_file.values()))) != (566, 94, 47, 217):
        raise ValueError("Unexpected training/tuning/audit/annotator split")
    if (base.CFG.crop_size, base.CFG.crop_context, base.CFG.min_crop) != (256, 1.8, 96):
        raise ValueError("Changed crop geometry")
    manifests = [p for p in args.input_root.rglob("inference_manifest.json")
                 if sha256(p) == v10.SOURCE_PINS["v6"]["manifest"]]
    if len(manifests) != 1:
        raise ValueError("Expected one exact frozen V6 cache manifest")
    previous = json.loads(manifests[0].read_text())
    v10.verify_manifest(previous, "v6", provenance, tune, audit)
    versions = {name: base.version(name) for name in previous["versions"]}
    if versions != previous["versions"]:
        raise ValueError(f"Runtime versions must reproduce frozen V6: {versions}")
    test = sorted(p.name for p in base.CFG.test_dir.glob("*.jpeg"))
    if len(test) != 180:
        raise ValueError("Expected 180 test images")
    specs = [("train", name) for name in held] + [("test", name) for name in test]
    expected = {v10.raw_cache_id(name, split_name) for split_name, name in specs}
    if not RAW_CACHE_SHA256 or set(RAW_CACHE_SHA256) != expected or len(expected) != 321:
        raise ValueError("Staging must pin all 321 validation/test raw caches")
    inventory = {}
    for split_name, name in specs:
        cache_name = v10.raw_cache_id(name, split_name)
        path = manifests[0].parent / "detector_proposal_cache" / cache_name
        if not path.is_file() or sha256(path) != RAW_CACHE_SHA256[cache_name]:
            raise ValueError(f"Missing or changed raw cache: {split_name}/{name}")
        v10.read_raw_views(path)
        inventory[(split_name, name)] = path
    baseline = manifests[0].parent / "inference_image_diagnostics.csv"
    if sha256(baseline) != v10.BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed frozen-control diagnostics")
    if not torch.cuda.is_available():
        raise RuntimeError("V11 smoke and training require a CUDA GPU")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    base.CFG.imgsz = 1024
    return dict(base=base, images=images, annotations=annotations, split=split,
                provenance=provenance, source_root=provenance_path.parent,
                refiner=refiner_path, inventory=inventory, baseline=baseline, versions=versions)


def build_training_cohort(runtime, names, output, identity, sample_limit=0):
    """Only train names reach the detector; cache each image independently."""
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    _, tune, audit, _, _ = runtime["split"]
    verify_training_names(names, runtime["provenance"], tune, audit)
    destination = output / "training_cohort.json"
    signature = dict(run_identity=identity, training_files=names, sample_limit=sample_limit)
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["signature"] != signature:
            raise ValueError("Cannot resume a different training cohort")
        # Revalidate source image and generated proposal identities, not only JSON.
        for name, entry in saved["training_sources"].items():
            if sha256(base.CFG.train_dir / name) != entry["image_sha256"]:
                raise ValueError("Training image changed during resume")
            if sha256(output / entry["cache"]) != entry["cache_sha256"]:
                raise ValueError("Training proposal cache changed during resume")
        return saved
    cache_root = output / "training_proposal_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    base.CFG.proposal_cache = cache_root
    detector = base.YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    records, source_inventory, annotation_count = [], {}, 0
    image_ids = {}
    for image_id, info in images.items():
        if info["file_name"] in set(names):
            image_ids.setdefault(info["file_name"], []).append(image_id)
    for file_index, name in enumerate(names, 1):
        path = base.CFG.train_dir / name
        cache = cache_root / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + ".json.gz")
        marker = cache.with_suffix(".signature.json")
        cache_signature = dict(run_identity=identity, file_name=name, image_sha256=sha256(path),
                               detector_sha256=v10.SOURCE_PINS["v6"]["detector"],
                               resolution=1024, confidence_floor=0.1, views=4)
        if cache.exists():
            if not marker.exists() or json.loads(marker.read_text())["signature"] != cache_signature:
                raise ValueError("Unsigned/stale training proposal cache; use a fresh output directory")
            if sha256(cache) != json.loads(marker.read_text())["sha256"]:
                raise ValueError("Changed generated training cache")
            views = v10.read_raw_views(cache)
        else:
            views = tta.proposal_views(base, detector, path)
            write_json(marker, dict(signature=cache_signature, sha256=sha256(cache)))
        proposals, scores = tta.cluster_views(views, "flip_consensus")
        source_inventory[name] = dict(image_sha256=cache_signature["image_sha256"],
                                      cache=str(cache.relative_to(output)), cache_sha256=sha256(cache))
        for image_id in sorted(image_ids.get(name, []), key=str):
            info = images[image_id]
            targets = [(ann, core.annotation_rle(info, ann)) for ann in annotations.get(image_id, [])]
            targets = [(ann, rle) for ann, rle in targets if mask_utils.area(rle) > 0]
            annotation_count += len(targets)
            iou = (mask_utils.iou([rle for _, rle in targets], proposals, [0] * len(proposals))
                   if targets and proposals else np.zeros((len(targets), len(proposals))))
            for target_index, proposal_index in match_positive_proposals(iou, scores):
                ann, target = targets[target_index]
                proposal = proposals[proposal_index]
                target_mask, seed_mask = mask_utils.decode(target), mask_utils.decode(proposal)
                records.append(dict(file_name=name, image_id=image_id, annotation_id=ann["id"],
                                    category_id=ann["category_id"], proposal_id=proposal_index,
                                    matched_iou=float(iou[target_index, proposal_index]),
                                    confidence=float(scores[proposal_index]),
                                    gt_bounds=list(base.square_bounds(target_mask)),
                                    proposal_bounds=list(base.square_bounds(seed_mask)),
                                    target=v10.serialized_rle(target), seed=v10.serialized_rle(proposal)))
        if file_index % 10 == 0 or file_index == len(names):
            print(f"training proposals files={file_index}/{len(names)} matched_targets={len(records)}", flush=True)
    del detector
    import torch
    torch.cuda.empty_cache()
    if sample_limit:
        records = records[:sample_limit]
    if not records or len({(r["image_id"], r["annotation_id"]) for r in records}) != len(records):
        raise ValueError("Training cohort empty or duplicates annotation sampling")
    saved = dict(signature=signature, training_sources=source_inventory, records=records,
                 matched_samples=len(records), all_training_annotations=annotation_count,
                 matching_policy=protocol()["matching"], cohort_sha256=digest_json(records))
    write_json(destination, saved)
    return saved


class ProposalDataset:
    """Deterministic per-epoch triples; runtime torch import keeps CPU tests small."""
    def __init__(self, records, train_dir, arm, epoch):
        if arm not in ARMS:
            raise ValueError("Unknown training arm")
        self.records, self.train_dir, self.arm, self.epoch = records, train_dir, arm, epoch
        self.image_cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def arrays(self, index, augment=True):
        record = self.records[index]
        name = record["file_name"]
        if name not in self.image_cache:
            gray = cv2.imread(str(self.train_dir / name), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise ValueError(f"Cannot read training image {name}")
            self.image_cache[name] = gray
            if len(self.image_cache) > 4:
                self.image_cache.popitem(last=False)
        gray = self.image_cache[name]
        self.image_cache.move_to_end(name)
        seed = mask_utils.decode(dict(size=record["seed"]["size"], counts=record["seed"]["counts"].encode()))
        target = mask_utils.decode(dict(size=record["target"]["size"], counts=record["target"]["counts"].encode()))
        bounds = record["gt_bounds" if self.arm == ARMS[0] else "proposal_bounds"]
        rng = np.random.default_rng(42 + self.epoch * 1_000_003 + index)
        if augment:
            jittered = jitter_bounds(bounds, gray.shape, rng)
            x0, y0, x1, y1 = jittered
            if seed[y0:y1, x0:x1].any() and target[y0:y1, x0:x1].any():
                bounds = jittered
        triplet = crop_triplet(gray, seed, target, bounds)
        if not triplet[1].any() or not triplet[2].any():
            # Very thin masks can vanish on resize; preserve supervision by
            # rejecting the cohort up front rather than silently change budgets.
            raise ValueError(f"Training seed/target vanished on resize: {name}/{record['annotation_id']}")
        return augment_triplet(*triplet, rng) if augment else triplet

    def __getitem__(self, index):
        import torch
        image, seed, target = self.arrays(index)
        channels = np.repeat(image[None], 3, axis=0)
        if self.arm == ARMS[2]:
            channels = np.concatenate((channels, seed.astype(np.float32)[None]), axis=0)
        return torch.from_numpy(channels), torch.from_numpy(target.astype(np.float32)[None])


def conditioned_crop_probabilities(base, model, gray, masks):
    import torch
    bounds = [base.square_bounds(mask) for mask in masks]
    crops = [cv2.resize(gray[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE))
             for x0, y0, x1, y1 in bounds]
    seeds = [cv2.resize(mask[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
             for mask, (x0, y0, x1, y1) in zip(masks, bounds)]
    averaged = []
    with torch.no_grad():
        for start in range(0, len(crops), 8):
            image = torch.from_numpy(np.stack(crops[start:start + 8])).float()
            image = image[:, None].repeat(1, 3, 1, 1).to("cuda:0") / 255
            seed = torch.from_numpy(np.stack(seeds[start:start + 8])).float()[:, None].to("cuda:0")
            batch = torch.cat((image, seed), dim=1)
            total = model(batch).sigmoid()
            for dims in ((2,), (3,), (2, 3)):
                total += model(batch.flip(dims)).sigmoid().flip(dims)
            averaged.extend((total[:, 0] / 4).cpu().numpy())
    return bounds, averaged


def predict_modes(runtime, model, image_path, split_name, conditioned, modes):
    base = runtime["base"]
    views = v10.read_raw_views(runtime["inventory"][(split_name, image_path.name)])
    proposals, scores = tta.cluster_views(views, "flip_consensus")
    masks = [mask_utils.decode(rle) for rle in proposals]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise ValueError(f"Cannot read inference image {image_path}")
    # ALL groups retain original batching before confidence filtering: this is
    # essential to reproduce the accepted control's floating-point behavior.
    if conditioned:
        bounds, probabilities = conditioned_crop_probabilities(base, model, gray, masks)
    else:
        bounds, _, probabilities = core.crop_probabilities(base, model, gray, masks, True)
    predictions = {}
    for mode in modes:
        rles, kept_scores = core.clean_refined_predictions(masks, scores, bounds, probabilities, THRESHOLD, mode)
        predictions[mode] = [rle for rle, score in zip(rles, kept_scores) if score >= CONFIDENCE]
    return predictions


def evaluate(runtime, model, names, configs, group, output, identity, model_sha):
    """Canonical per-annotator global counts, shared tune/audit/test geometry."""
    if group not in ("tune", "audit") or not names:
        raise ValueError("Evaluation requires nonempty tuning or audit files")
    allowed = runtime["split"][1 if group == "tune" else 2]
    if not set(names) <= allowed or len(names) != len(set(names)):
        raise ValueError("Evaluation files violate declared split")
    signature = dict(run_identity=identity, checkpoint_sha256=model_sha, group=group,
                     files=names, configs=configs)
    path = output / "evaluations" / f"{next(iter(configs))}_{group}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["signature"] != signature:
            raise ValueError("Stale evaluation artifact")
        return saved
    models = {c["conditioned"] for c in configs.values()}
    if len(models) != 1:
        raise ValueError("One evaluation pass must use one model")
    totals = {key: np.zeros(4, dtype=float) for key in configs}
    diagnostics = []
    model.eval()
    for index, name in enumerate(names, 1):
        predicted = predict_modes(runtime, model, runtime["base"].CFG.train_dir / name, "train",
                                  next(iter(models)), {c["mode"] for c in configs.values()})
        for image_id in runtime["split"][4][name]:
            gt = [core.annotation_rle(runtime["images"][image_id], ann)
                  for ann in runtime["annotations"].get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            for key, config in configs.items():
                rles = predicted[config["mode"]]
                iou = (mask_utils.iou(gt, rles, [0] * len(rles)) if gt and rles
                       else np.zeros((len(gt), len(rles))))
                counts = core.pq_counts(iou, len(gt), len(rles))
                totals[key] += counts
                diagnostics.append(dict(candidate=key, file_name=name, image_id=image_id,
                                        group=group, target="coco", **dict(zip(COUNTS, counts))))
        if index % 10 == 0 or index == len(names):
            print(f"evaluate group={group} model={next(iter(configs))} images={index}/{len(names)}", flush=True)
    saved = dict(signature=signature, totals={key: value.tolist() for key, value in totals.items()},
                 diagnostics=diagnostics)
    write_json(path, saved)
    return saved


def verify_control(runtime, result, names):
    v10.verify_control_counts(result["diagnostics"], runtime["baseline"], names,
                              candidate=CONTROL, baseline_candidate=v10.OLD_CONTROL)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, identity, cohort_hash):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                    run_identity=identity, cohort_sha256=cohort_hash,
                    torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all(),
                    python_rng_state=random.getstate()), temporary)
    temporary.replace(path)


def train_arm(runtime, arm, cohort, tune_names, configs, output, identity, epochs):
    import torch
    base = runtime["base"]
    base.seed_everything(42)
    model = build_model(base, runtime["refiner"], arm == ARMS[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # The historical continuation used AMP. This controlled experiment chooses
    # common FP32 before observing outcomes so each batch performs exactly one
    # optimizer update; disabled scaler retains a simple empty resume state.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    all_results, history = [], []
    for epoch in range(1, epochs + 1):
        checkpoint = output / "checkpoints" / f"{arm}_e{epoch}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
            if (saved["run_identity"] != identity or saved["cohort_sha256"] != cohort["cohort_sha256"]
                    or saved["epoch"] != epoch):
                raise ValueError("Checkpoint cannot resume another experiment or cohort")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
            random.setstate(saved["python_rng_state"])
            training = json.loads(checkpoint.with_suffix(".json").read_text())
        else:
            dataset = ProposalDataset(cohort["records"], base.CFG.train_dir, arm, epoch)
            order = epoch_sample_order(cohort["records"], epoch)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=order,
                                                num_workers=0, pin_memory=True, drop_last=False)
            model.train()
            loss_sum, samples, updates = 0.0, 0, 0
            for batch, target in loader:
                batch, target = batch.to("cuda:0"), target.to("cuda:0")
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=False):
                    logits = model(batch)
                    if logits.dtype != torch.float32:
                        raise ValueError("All continuation arms require float32 training")
                    loss = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                            + 0.5 * base.soft_dice_loss(logits, target))
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {arm} epoch {epoch} loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(batch)
                samples += len(batch)
                updates += 1
                if updates % 50 == 0:
                    print(f"train arm={arm} epoch={epoch}/{epochs} updates={updates}/{len(loader)} loss={loss_sum/samples:.6f}", flush=True)
            if samples != len(cohort["records"]) or updates != math.ceil(samples / BATCH_SIZE):
                raise ValueError("Training arm violated the matched sample/update budget")
            training = dict(arm=arm, epoch=epoch, samples=samples, updates=updates,
                            learning_rate=optimizer.param_groups[0]["lr"], mean_loss=loss_sum / samples,
                            sample_order_sha256=digest_json(order), run_identity=identity)
            scheduler.step()
            # Publish metadata first: a checkpoint is resumable only when both
            # files exist; the checkpoint itself is the atomic commit marker.
            write_json(checkpoint.with_suffix(".json"), training)
            save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, epoch, identity, cohort["cohort_sha256"])
        model_hash = sha256(checkpoint)
        candidates = {key: value for key, value in configs.items() if value["arm"] == arm and value["epoch"] == epoch}
        result = evaluate(runtime, model, tune_names, candidates, "tune", output, identity, model_hash)
        all_results.append(result)
        history.append(dict(**training, checkpoint_sha256=model_hash,
                            tune_pq={key: pq(value) for key, value in result["totals"].items()}))
        write_json(output / f"training_history_{arm}.json", history)
        print(json.dumps(dict(arm=arm, epoch=epoch, tune_pq=history[-1]["tune_pq"])), flush=True)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return all_results, history


def validate_matched_budgets(histories, samples, epochs):
    if set(histories) != set(ARMS):
        raise ValueError("All three continuation arms must finish")
    for epoch in range(epochs):
        rows = [histories[arm][epoch] for arm in ARMS]
        expected = (samples, math.ceil(samples / BATCH_SIZE), rows[0]["sample_order_sha256"])
        if any((row["samples"], row["updates"], row["sample_order_sha256"]) != expected for row in rows):
            raise ValueError("Continuation arms did not receive matched training budgets")


def write_results(output, results, configs):
    rows, diagnostic_rows = [], []
    totals = {}
    for result in results:
        group = result["signature"]["group"]
        diagnostic_rows.extend(result["diagnostics"])
        for key, counts in result["totals"].items():
            if (key, group) in totals:
                raise ValueError("Duplicate candidate/group evaluation")
            totals[(key, group)] = counts
    for key, config in configs.items():
        row = dict(candidate=key, **config, evaluation_scope="tune_only")
        for group in ("tune", "audit"):
            values = totals.get((key, group))
            row[f"{group}_pq"] = pq(values) if values is not None else None
            for count_index, count in enumerate(COUNTS):
                row[f"{group}_{count}"] = values[count_index] if values is not None else None
        if (key, "audit") in totals:
            row["evaluation_scope"] = "full_validation"
            row["all_pq"] = pq(np.asarray(totals[(key, "tune")]) + totals[(key, "audit")])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "inference_candidates.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output / "inference_image_diagnostics.csv", index=False)


def export_candidate(runtime, model, config, output, filename):
    model.eval()
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(runtime["base"].CFG.test_dir.glob("*.jpeg")), 1):
        predictions = predict_modes(runtime, model, path, "test", config["conditioned"], [config["mode"]])[config["mode"]]
        image_counts[path.name] = len(predictions)
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(predictions, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if mask.sum() < 20 or (mask & occupied).any():
                raise ValueError("Invalid test prediction area or pixel ownership")
            occupied |= mask
            rows.append(dict(filament_id=f"{path.stem}_{number}", segmentation_rle=rle["counts"].decode()))
        if index % 20 == 0:
            print(f"export={filename} images={index}/180 rows={len(rows)}", flush=True)
    if len(image_counts) != 180:
        raise ValueError("Incomplete test coverage during export")
    destination = output / filename
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(destination, index=False)
    return dict(file=filename, rows=len(rows), sha256=sha256(destination),
                evaluated_images=180, empty_images=[name for name, count in image_counts.items() if not count],
                structural_validation="all masks decoded, area>=20, per-image ownership disjoint")


def run_v11():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-limit", type=int, default=3)
    parser.add_argument("--tune-limit", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=32)
    args = parser.parse_args()
    if min(args.train_limit, args.tune_limit, args.sample_limit) < 1:
        parser.error("Smoke limits must be positive")
    if not args.smoke and (args.train_limit, args.tune_limit, args.sample_limit) != (3, 3, 32):
        parser.error("Smoke limits cannot change a full run")
    if not MODULE_SHA256:
        raise ValueError("Run scripts/prepare_v11_refiner.py to stage a pinned experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = protocol(args.smoke, args.train_limit, args.tune_limit, args.sample_limit)
    identity_payload = dict(script_sha256=sha256(Path(__file__)), modules=MODULE_SHA256,
                            raw_cache_sha256=RAW_CACHE_SHA256, protocol=recipe, v10_reference=V10_REFERENCE)
    identity = digest_json(identity_payload)
    manifest_path = args.output / "inference_manifest.json"
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())
        if old_manifest["run_identity"] != identity:
            raise ValueError("Output directory belongs to a different experiment")
        if old_manifest.get("complete"):
            print(json.dumps(dict(already_complete=True, candidate=old_manifest.get("selected_candidate"),
                                  promotable=old_manifest.get("promotable"))), flush=True)
            return
    elif any((args.output / name).exists() for name in ("submission.csv", "submission_control.csv")):
        raise ValueError("Unsigned existing submission artifacts require a fresh output directory")
    manifest = dict(**recipe, run_identity=identity, identity=identity_payload,
                    promotable=False, complete=False, full_control_validated=False)
    write_json(manifest_path, manifest)
    runtime = load_runtime(args)
    held, tune, audit, _, _ = runtime["split"]
    train_names = sorted(runtime["provenance"]["training_files"])
    tune_names = [name for name in held if name in tune]
    if args.smoke:
        train_names, tune_names = train_names[:args.train_limit], tune_names[:args.tune_limit]
    verify_training_names(train_names, runtime["provenance"], tune, audit, require_full=not args.smoke)
    manifest.update(source=runtime["provenance"], versions=runtime["versions"],
                    training_files=train_names, tune_files=tune_names,
                    audit_files=sorted(audit) if not args.smoke else [],
                    detector_validation_inference=False, detector_test_inference=False)
    write_json(manifest_path, manifest)
    import torch
    configs, results = recipe["candidates"], []
    control = build_model(runtime["base"], runtime["refiner"])
    control_result = evaluate(runtime, control, tune_names, {CONTROL: configs[CONTROL]}, "tune",
                              args.output, identity, v10.REFINER_SHA256)
    verify_control(runtime, control_result, tune_names)
    results.append(control_result)
    manifest["control_tuning_verified"] = True
    write_json(manifest_path, manifest)
    del control
    torch.cuda.empty_cache()
    cohort = build_training_cohort(runtime, train_names, args.output, identity,
                                   args.sample_limit if args.smoke else 0)
    # Validate every native crop under the complete declared augmentation schedule
    # before any arm learns, so rare resize failures cannot change sample budgets.
    geometry_signature = dict(run_identity=identity, cohort_sha256=cohort["cohort_sha256"],
                              epochs=recipe["epochs"], samples=len(cohort["records"]))
    geometry_path = args.output / "training_geometry_validation.json"
    if geometry_path.exists():
        if json.loads(geometry_path.read_text()) != dict(signature=geometry_signature, passed=True):
            raise ValueError("Stale training geometry validation")
    else:
        for arm in (ARMS[0], ARMS[1]):
            for epoch in range(1, recipe["epochs"] + 1):
                dataset = ProposalDataset(cohort["records"], runtime["base"].CFG.train_dir, arm, epoch)
                for position, index in enumerate(epoch_sample_order(cohort["records"], epoch), 1):
                    dataset.arrays(index)
                    if position % 500 == 0 or position == len(cohort["records"]):
                        print(f"validate training geometry arm={arm} epoch={epoch} samples={position}/{len(cohort['records'])}", flush=True)
        write_json(geometry_path, dict(signature=geometry_signature, passed=True))
    first = ProposalDataset(cohort["records"], runtime["base"].CFG.train_dir, ARMS[2], 0).arrays(0, False)
    numerical = numerical_seed_test(runtime["base"], runtime["refiner"], *first)
    write_json(args.output / "numerical_seed_test.json", numerical)
    histories = {}
    for arm in ARMS:
        arm_results, histories[arm] = train_arm(runtime, arm, cohort, tune_names, configs,
                                                args.output, identity, recipe["epochs"])
        results.extend(arm_results)
        write_results(args.output, results, configs)
    validate_matched_budgets(histories, len(cohort["records"]), recipe["epochs"])
    tuning_counts = {key: counts for result in results for key, counts in result["totals"].items()}
    selected = select_candidate(configs, tuning_counts)
    selection = dict(candidate=selected, config=configs[selected], selected_on="tune global PQ only",
                     run_identity=identity, tune_counts=tuning_counts,
                     frozen_before_audit=True, smoke=args.smoke)
    selection_path = args.output / "selected_config.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Resume attempted to alter the frozen pre-audit selection")
    write_json(selection_path, selection)
    if args.smoke:
        manifest.update(complete=True, smoke_passed=True, numerical_seed_test=numerical,
                        selected_candidate=selected, training_samples=len(cohort["records"]),
                        reason="Smoke verifies execution only; no audit, test export, or promotion")
        write_json(manifest_path, manifest)
        print("V11 smoke passed; no audit, test inference, export, or submission", flush=True)
        return
    audit_names = [name for name in held if name in audit]
    audit_counts = {}
    for key in audit_candidates(selected, configs):
        config = configs[key]
        checkpoint = (None if key == CONTROL else args.output / "checkpoints" / f"{config['arm']}_e{config['epoch']}.pt")
        model = build_model(runtime["base"], runtime["refiner"], config["conditioned"], checkpoint)
        result = evaluate(runtime, model, audit_names, {key: config}, "audit", args.output, identity,
                          v10.REFINER_SHA256 if key == CONTROL else sha256(checkpoint))
        if key == CONTROL:
            verify_control(runtime, result, audit_names)
        audit_counts.update(result["totals"])
        results.append(result)
        del model
        torch.cuda.empty_cache()
    write_results(args.output, results, configs)
    gate = promotion_gate(selected, tuning_counts, audit_counts)
    manifest.update(selected_candidate=selected, gate=gate, control_validation_verified=True,
                    training_samples=len(cohort["records"]), numerical_seed_test=numerical)
    write_json(args.output / "inference_comparison.json",
               dict(selection=selection, gate=gate, v10_reference=V10_REFERENCE,
                    audit_counts=audit_counts, audit_caveat=recipe["audit_caveat"]))
    write_json(manifest_path, manifest)
    if gate["promotable"]:
        control = build_model(runtime["base"], runtime["refiner"])
        control_export = export_candidate(runtime, control, configs[CONTROL], args.output, "submission_control.csv")
        v10.verify_control_export(args.output / "submission_control.csv")
        del control
        torch.cuda.empty_cache()
        config = configs[selected]
        checkpoint = args.output / "checkpoints" / f"{config['arm']}_e{config['epoch']}.pt"
        model = build_model(runtime["base"], runtime["refiner"], config["conditioned"], checkpoint)
        candidate_export = export_candidate(runtime, model, config, args.output, "submission.csv")
        write_json(args.output / "submission_candidates.json",
                   [dict(control_export, candidate=CONTROL), dict(candidate_export, candidate=selected,
                    promotable=True, full_control_validated=True, requires_independent_csv_validation=True)])
        manifest.update(promotable=True, full_control_validated=True,
                        selected_checkpoint_sha256=sha256(checkpoint))
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    print(json.dumps(dict(candidate=selected, **gate, complete=True)), flush=True)


if __name__ == "__main__":
    run_v11()
