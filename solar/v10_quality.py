"""Frozen V10: exact V6 control, bounded mask-quality ablations, V8-only rescue.

Quality motivation: https://arxiv.org/abs/1903.00241 (Mask Scoring R-CNN).
Foreground-only rank proxy follows the idea in Mask2Former's instance_inference:
https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
Our foreground probability is an untrained heuristic, NOT its learned IoU head;
the paper's COCO AP gains are not evidence of gains on this competition's PQ.
No detector inference, training, submission, or automatic component splitting.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


CONTROL = "v10_control"
OLD_CONTROL = "flip_consensus_r1024_refined_constrained_t0.40_tta1_c0.40"
CONTROL_SHA256 = "bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9"
REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
BASELINE_DIAGNOSTICS_SHA256 = "4ee7b1b2c3180f52866b4eec4ea27fe0508ffb48d3d5ebb2299a7348d8fc876c"
V8_DIAGNOSTICS_SHA256 = "e6d0e098c8705498fcca2775a68bcd193d16b2027eff235bd103115a858844f0"
V8_CONTROL = "v8_source_identity_only"
V8_OLD_CONTROL = "flip_consensus_r1024_refined_constrained_t0.45_tta1_c0.25"
SOURCE_PINS = {
    "v6": dict(experiment="coco_refiner_detector_flip_tta",
               manifest="cf60725badfcba0b4b9a8c5d70e8eeda9c8c1204a030c99de6c27f4b932c0c96",
               detector="0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8"),
    "v8": dict(experiment="v8_coco_refiner_detector_flip_tta",
               manifest="10724b0aedb945b0e4297fd7bab615bc01d0f54fa80a2a9ca506698f091d1e90",
               detector="19c713e327e7bf39e839d428712d243d25ce746628237663a64abbf58f6e9359"),
}
TARGET = "COCO polygons; rounded OpenCV only for legacy comparison"
ORIGINAL_ROOT = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
RAW_CACHE_SHA256 = None  # Stager pins every one of the 642 existing raw caches.
COUNTS = ["iou_sum", "tp", "fp", "fn"]
MIN_AREA = 20
QUALITY = ("none", "drop_empty", "proposal50", "views3", "quality_rank")
RESCUE = {"none": None, "rescue35": 0.35, "rescue40": 0.40}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def serialized_rle(rle):
    return dict(size=list(rle["size"]), counts=rle["counts"].decode())


def restored_rle(rle):
    return dict(size=rle["size"], counts=rle["counts"].encode())


def declared_candidates():
    """15 fixed recipes; no threshold search or data-dependent gate creation."""
    configs = {}
    for quality in QUALITY:
        for rescue in RESCUE:
            name = CONTROL if (quality, rescue) == ("none", "none") else f"v10_{quality}_{rescue}"
            configs[name] = dict(
                resolution=1024, mode="refined_constrained", threshold=0.40,
                tta=True, confidence=0.40, detector_policy="flip_consensus",
                quality=quality, rescue=rescue, rescue_confidence=RESCUE[rescue],
                rescue_threshold=0.45, rescue_min_views=3, rescue_min_proposal_support=0.50,
                rescue_min_residual_fraction=0.90,
                duplicate_iou=0.50, duplicate_containment=0.80,
                complexity=int(quality != "none") + int(rescue != "none"))
    return configs


def isolated_candidates(configs):
    return {key: value for key, value in configs.items() if value["complexity"] <= 1}


def eligible_candidates(configs, tune_scores):
    eligible = isolated_candidates(configs)
    baseline = tune_scores[CONTROL]
    for key, config in configs.items():
        if config["complexity"] != 2:
            continue
        components = (f"v10_{config['quality']}_none", f"v10_none_{config['rescue']}")
        if all(tune_scores[c] > baseline + 1e-12 for c in components):
            eligible[key] = config
    return eligible


def select_candidate(configs, tune_scores):
    """No audit argument exists; ties prefer control, then fewer interventions."""
    best = max(tune_scores[key] for key in configs)
    ties = [key for key in configs if best - tune_scores[key] <= 1e-12]
    return min(ties, key=lambda key: (configs[key]["complexity"], key))


def group_views(views):
    """Exact fixed-anchor cluster_views geometry, with retained vote metadata."""
    if len(views) != 4:
        raise ValueError("Exactly four raw detector views are required")
    proposals = []
    for view_id, (rles, scores) in enumerate(views):
        if len(rles) != len(scores):
            raise ValueError("Every raw proposal needs a score")
        proposals.extend((rle, float(score), view_id) for rle, score in zip(rles, scores))
    proposals.sort(key=lambda item: -item[1])
    groups = []
    for rle, score, view_id in proposals:
        eligible = [i for i, group in enumerate(groups) if view_id not in group["views"]]
        match = None
        if eligible:
            overlaps = mask_utils.iou([rle], [groups[i]["rle"] for i in eligible], [0] * len(eligible))[0]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= 0.5:
                match = eligible[best]
        if match is None:
            groups.append(dict(rle=rle, scores=[score], views=[view_id], view_scores={view_id: score}, anchor_ious=[1.0]))
        else:
            groups[match]["scores"].append(score)
            groups[match]["views"].append(view_id)
            groups[match]["view_scores"][view_id] = score
            groups[match]["anchor_ious"].append(float(overlaps[best]))
    for group in groups:
        group["consensus_confidence"] = sum(group.pop("scores")) / 4
        group["support_count"] = len(group["views"])
    return sorted(groups, key=lambda group: -group["consensus_confidence"])


def make_record(group, mask, bounds, probability, threshold, detector, index):
    """Keep untrimmed crop geometry; never cache clipped masks for re-ranking."""
    x0, y0, x1, y1 = bounds
    proposal = mask[y0:y1, x0:x1].astype(bool)
    probability = cv2.resize(probability, (x1 - x0, y1 - y0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined = (probability >= threshold) & (cv2.dilate(proposal.astype(np.uint8), kernel) > 0)
    empty = not refined.any()
    untrimmed = proposal if empty else refined
    return dict(
        detector=detector, proposal_id=index, bounds=list(bounds), threshold=threshold,
        consensus_confidence=group["consensus_confidence"], support_count=group["support_count"],
        views=group["views"], view_scores=group["view_scores"], empty_fallback=empty,
        anchor_member_ious=group.get("anchor_ious", []),
        foreground_probability=float(probability[refined].mean()) if refined.any() else 0.0,
        proposal_foreground_probability=float(probability[proposal].mean()) if proposal.any() else 0.0,
        retention_fraction=float((refined & proposal).sum() / max(1, proposal.sum())),
        proposal_area=int(proposal.sum()), untrimmed_area=int(untrimmed.sum()),
        crop_rle=serialized_rle(mask_utils.encode(np.asfortranarray(untrimmed, dtype=np.uint8))))


def quality_accepts(record, quality):
    if quality not in QUALITY:
        raise ValueError(f"Unknown quality rule: {quality}")
    if quality == "drop_empty":
        return not record["empty_fallback"]
    if quality == "proposal50":
        return record["proposal_foreground_probability"] >= 0.50
    if quality == "views3":
        return record["support_count"] >= 3
    return True


def duplicate_rescue(rle, anchors, config):
    if not anchors:
        return False
    others = [anchor["rle"] for anchor in anchors]
    iou = mask_utils.iou([rle], others, [0] * len(others))[0]
    area = float(mask_utils.area(rle))
    other_area = mask_utils.area(others).astype(float)
    intersection = iou * (area + other_area) / (1 + iou)
    containment = intersection / np.maximum(1, np.minimum(area, other_area))
    return bool(np.any((iou >= config["duplicate_iou"]) |
                       (containment >= config["duplicate_containment"])))


def build_predictions(records, config, shape):
    """Shared validation/export path. Separate model scores; V6 owns pixels first."""
    occupied = np.zeros(shape, dtype=bool)
    kept = []
    for detector in ("v6", "v8"):
        if detector == "v8" and config["rescue"] == "none":
            continue
        cutoff = config["confidence"] if detector == "v6" else config["rescue_confidence"]
        candidates = [r for r in records[detector] if r["consensus_confidence"] >= cutoff]
        if detector == "v6":
            candidates = [r for r in candidates if quality_accepts(r, config["quality"])]
        else:
            candidates = [r for r in candidates if r["support_count"] >= config["rescue_min_views"]
                          and r["proposal_foreground_probability"] >= config["rescue_min_proposal_support"]
                          and not r["empty_fallback"]]
        def ordering(record):
            confidence = record["consensus_confidence"]
            if detector == "v6" and config["quality"] == "quality_rank":
                confidence *= record["foreground_probability"]
            return -confidence
        for record in sorted(candidates, key=ordering):
            x0, y0, x1, y1 = record["bounds"]
            crop = mask_utils.decode(restored_rle(record["crop_rle"])).astype(bool)
            if detector == "v8":
                full_rle = core.encode_crop(crop, record["bounds"], shape)
                if duplicate_rescue(full_rle, kept, config):
                    continue
            crop &= ~occupied[y0:y1, x0:x1]
            if crop.sum() < MIN_AREA:
                continue
            if detector == "v8" and crop.sum() / max(1, record["untrimmed_area"]) < config["rescue_min_residual_fraction"]:
                continue
            occupied[y0:y1, x0:x1] |= crop
            kept.append(dict(rle=core.encode_crop(crop, record["bounds"], shape),
                             detector=detector, proposal_id=record["proposal_id"],
                             empty_fallback=record["empty_fallback"]))
    return kept


def raw_cache_id(name, split):
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    path = f"{ORIGINAL_ROOT}/{split}/{split}_images/{name}"
    return hashlib.sha256(path.encode()).hexdigest()[:16] + ".json.gz"


def read_raw_views(path):
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Incomplete four-view cache: {path}")
    views = []
    for view in raw:
        rles, scores = [restored_rle(r) for r in view["rles"]], view["scores"]
        if len(rles) != len(scores) or any(not np.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError(f"Invalid cache scores: {path}")
        if any(r["size"] != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
            raise ValueError(f"Invalid proposal geometry: {path}")
        views.append((rles, scores))
    return views


def verify_manifest(metadata, detector, provenance, tune, audit):
    pin = SOURCE_PINS[detector]
    if metadata.get("experiment") != pin["experiment"] or metadata.get("source") != provenance:
        raise ValueError(f"Wrong {detector} experiment or checkpoint provenance")
    actual_detector = (metadata["source"]["sha256"]["yolo_best.pt"] if detector == "v6"
                       else metadata["detector"]["sha256"])
    if actual_detector != pin["detector"]:
        raise ValueError(f"Wrong {detector} cache identity")
    expected = dict(tune_files=sorted(tune), audit_files=sorted(audit), target_rasterization=TARGET,
                    smoke_limit=0, refiner_checkpoint_sha256=REFINER_SHA256,
                    detector_confidence_floor=0.1, resolutions=[1024], mask_match_iou=0.5)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible {detector} cache experiment settings or split")
    if (set(tune) & set(audit) or set(tune) | set(audit) != set(provenance["validation_files"])
            or set(provenance["training_files"]) & (set(tune) | set(audit))):
        raise ValueError("Leaking or incomplete split")


def resolve_cache_sources(input_root, provenance, tune, audit, image_specs, cache_hashes):
    """Hash and structurally validate ALL val/test caches before using the GPU."""
    if not cache_hashes:
        raise ValueError("Run scripts/prepare_v10_quality.py to pin complete cache inventories")
    manifests = list(input_root.rglob("inference_manifest.json"))
    roots, inventory = {}, {}
    expected_names = {raw_cache_id(name, split) for split, name in image_specs}
    if len(expected_names) != len(image_specs):
        raise ValueError("Duplicate or colliding cache image IDs")
    for detector, pin in SOURCE_PINS.items():
        matches = [(path, json.loads(path.read_text())) for path in manifests
                   if json.loads(path.read_text()).get("experiment") == pin["experiment"]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one pinned {detector} input manifest")
        path, metadata = matches[0]
        if sha256(path) != pin["manifest"]:
            raise ValueError(f"Changed {detector} input manifest")
        verify_manifest(metadata, detector, provenance, tune, audit)
        cache_root = path.parent / "detector_proposal_cache"
        if detector == "v8":
            cache_root /= pin["detector"][:16]
        if set(cache_hashes[detector]) != expected_names:
            raise ValueError(f"Pinned {detector} inventory does not cover every validation/test image")
        inventory[detector] = {}
        for split, name in image_specs:
            cache_name = raw_cache_id(name, split)
            cache = cache_root / cache_name
            if not cache.is_file() or sha256(cache) != cache_hashes[detector][cache_name]:
                raise ValueError(f"Missing/changed {detector} raw cache for {split}/{name}")
            read_raw_views(cache)
            inventory[detector][(split, name)] = cache
        roots[detector] = path.parent
    return roots, inventory


def verify_control_counts(diagnostics, baseline_path, files, candidate=CONTROL, baseline_candidate=OLD_CONTROL):
    expected = pd.read_csv(baseline_path)
    keys = ["file_name", "image_id", "group", "target"]
    actual = pd.DataFrame(diagnostics)
    actual = actual[(actual.candidate == candidate) & (actual.target == "coco")]
    expected = expected[(expected.candidate == baseline_candidate) & (expected.target == "coco")
                        & expected.file_name.isin(files)]
    if set(actual.file_name) != set(files) or expected.empty:
        raise ValueError("Missing control validation files")
    a = actual[keys + COUNTS].sort_values(keys).reset_index(drop=True)
    b = expected[keys + COUNTS].sort_values(keys).reset_index(drop=True)
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate annotation-image control rows")
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
    np.testing.assert_allclose(a[COUNTS], b[COUNTS], rtol=0, atol=1e-10)


def verify_control_export(path):
    if sha256(path) != CONTROL_SHA256:
        raise ValueError("Control submission hash mismatch; selected export is NOT authorized by this run")


def prepare_records(base, refiner, image_path, split, inventory, output, identity):
    """Persist lossless RLE crops and scalar diagnostics; no repeated dense variant caches."""
    records = {}
    gray = None
    for detector, pin in SOURCE_PINS.items():
        source_path = inventory[detector][(split, image_path.name)]
        cache_dir = output / "refined_untrimmed_cache" / pin["detector"][:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / raw_cache_id(image_path.name, split)
        signature = dict(identity=identity, raw_sha256=sha256(source_path), detector=detector,
                         file_name=image_path.name, split=split, threshold=0.4 if detector == "v6" else 0.45)
        if cache_path.exists():
            with gzip.open(cache_path, "rt") as stream:
                saved = json.load(stream)
            if saved["signature"] != signature:
                raise ValueError(f"Stale refined cache: {cache_path}")
            records[detector] = saved["proposals"]
            continue
        views = read_raw_views(source_path)
        groups = group_views(views)
        expected_rles, expected_scores = tta.cluster_views(views, "flip_consensus")
        if ([g["rle"] for g in groups] != expected_rles or
                [g["consensus_confidence"] for g in groups] != expected_scores):
            raise ValueError("V10 vote metadata changed established grouping geometry")
        if gray is None:
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise ValueError(f"Cannot read full-resolution image {image_path}")
        masks = [mask_utils.decode(group["rle"]) for group in groups]
        # Existing helper is @torch.no_grad(); preserve its original batching and four flips.
        bounds, _, averaged = core.crop_probabilities(base, refiner, gray, masks, True)
        records[detector] = [make_record(g, m, b, p, signature["threshold"], detector, i)
                             for i, (g, m, b, p) in enumerate(zip(groups, masks, bounds, averaged))]
        with gzip.open(cache_path, "wt") as stream:
            json.dump(dict(signature=signature, proposals=records[detector]), stream, allow_nan=False)
        diagnostics = [{k: v for k, v in record.items() if k != "crop_rle"}
                       for record in records[detector]]
        diagnostic_path = output / "proposal_diagnostics" / detector / f"{split}_{image_path.stem}.json.gz"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(diagnostic_path, "wt") as stream:
            json.dump(dict(file_name=image_path.name, split=split, proposals=diagnostics), stream)
    return records


def evaluate_files(base, refiner, names, configs, images, annotations, split_info,
                   inventory, output, identity, totals, diagnostics, source_diagnostics):
    _, tune, _, best_ids, by_file = split_info
    for index, name in enumerate(names, 1):
        records = prepare_records(base, refiner, base.CFG.train_dir / name, "train", inventory, output, identity)
        targets = []
        for image_id in by_file[name]:
            info = images[image_id]
            gt = [core.annotation_rle(info, ann) for ann in annotations.get(image_id, [])]
            targets.append((image_id, "coco", [r for r in gt if mask_utils.area(r) > 0]))
            if image_id == best_ids[name]:
                gt = [core.rounded_legacy_rle(info, ann) for ann in annotations.get(image_id, [])]
                targets.append((image_id, "rounded_legacy", [r for r in gt if mask_utils.area(r) >= MIN_AREA]))
        if CONTROL in configs:
            # V8 identity check is never a promotion candidate. Reuse its .45 masks,
            # .25 confidence and exact original score-ordered cleanup without rescue gates.
            v8_config = dict(configs[CONTROL], confidence=0.25)
            v8_predictions = build_predictions({"v6": records["v8"], "v8": []}, v8_config, (2048, 2048))
            v8_rles = [p["rle"] for p in v8_predictions]
            for image_id, target, gt in targets:
                if target != "coco":
                    continue
                iou = (mask_utils.iou(v8_rles, gt, [0] * len(gt)).T if v8_rles and gt
                       else np.zeros((len(gt), len(v8_rles))))
                counts = core.pq_counts(iou, len(gt), len(v8_rles))
                source_diagnostics.append(dict(candidate=V8_CONTROL, file_name=name, image_id=image_id,
                                          group="tune" if name in tune else "audit", target=target,
                                          **dict(zip(COUNTS, counts))))
        for key, config in configs.items():
            predicted = build_predictions(records, config, (2048, 2048))
            rles = [p["rle"] for p in predicted]
            groups = totals.setdefault(key, {g: np.zeros(4) for g in
                                            ("all", "legacy", "tune", "audit", "canonical_best")})
            for image_id, target, gt in targets:
                iou = (mask_utils.iou(rles, gt, [0] * len(gt)).T if rles and gt
                       else np.zeros((len(gt), len(rles))))
                counts = core.pq_counts(iou, len(gt), len(rles))
                source_metrics = {}
                for label, selected_columns in (
                        ("rescue", np.asarray([p["detector"] == "v8" for p in predicted], dtype=bool)),
                        ("fallback", np.asarray([p["empty_fallback"] for p in predicted], dtype=bool))):
                    hits = iou[:, selected_columns] > 0.5
                    source_metrics[f"{label}_tp"] = int(hits.sum())
                    source_metrics[f"{label}_fp"] = int((~hits.any(axis=0)).sum())
                group = "tune" if name in tune else "audit"
                if target == "rounded_legacy":
                    group = "legacy"
                else:
                    groups["all"] += counts
                    if image_id == best_ids[name]:
                        groups["canonical_best"] += counts
                groups[group] += counts
                diagnostics.append(dict(candidate=key, file_name=name, image_id=image_id,
                                        group=group, target=target, is_legacy=image_id == best_ids[name],
                                        **dict(zip(COUNTS, counts)), **source_metrics))
        print(f"evaluate configs={len(configs)} image={index}/{len(names)} {name}", flush=True)


def paired_comparison(diagnostics, candidate, group, samples=4000):
    """Same filename-clustered bootstrap as scripts/analyze_inference_sweep.py."""
    frame = pd.DataFrame(diagnostics)
    frame = frame[(frame.target == "coco") & frame.group.isin(
        ("tune", "audit") if group == "all" else (group,))]
    a = frame[frame.candidate == candidate].groupby("file_name")[COUNTS].sum().sort_index()
    b = frame[frame.candidate == CONTROL].groupby("file_name")[COUNTS].sum().sort_index()
    if a.empty or not a.index.equals(b.index):
        raise ValueError("Incomplete paired evaluation")
    a, b = a.to_numpy(float), b.to_numpy(float)
    def pq(values):
        denominator = values[..., 1] + 0.5 * (values[..., 2] + values[..., 3])
        return np.divide(values[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    indices = np.random.default_rng(42).integers(0, len(a), size=(samples, len(a)))
    differences = pq(a[indices].sum(axis=1)) - pq(b[indices].sum(axis=1))
    return dict(images=len(a), candidate_pq=float(pq(a.sum(axis=0))), control_pq=float(pq(b.sum(axis=0))),
                delta_pq=float(pq(a.sum(axis=0)) - pq(b.sum(axis=0))),
                delta_95pct_interval=np.quantile(differences, [0.025, 0.975]).tolist(),
                delta_counts=dict(zip(COUNTS, (a.sum(axis=0) - b.sum(axis=0)).tolist())))


def write_v10_results(totals, configs, diagnostics, output, full_candidates=()):
    """Uninspected audit/full metrics are missing, never misleading zero/partial scores."""
    core.write_results(totals, configs, diagnostics, output)
    frame = pd.read_csv(output / "inference_candidates.csv")
    complete = frame.candidate.isin(full_candidates)
    frame["evaluation_scope"] = np.where(complete, "full_validation", "tune_only")
    for column in frame:
        if column.startswith(("all_", "audit_", "legacy_", "canonical_best_")):
            frame.loc[~complete, column] = np.nan
    frame.to_csv(output / "inference_candidates.csv", index=False)


def export_config(base, refiner, config, inventory, output, identity, filename):
    rows = []
    image_paths = sorted(base.CFG.test_dir.glob("*.jpeg"))
    if len(image_paths) != 180:
        raise ValueError("Expected exactly 180 test images")
    for index, path in enumerate(image_paths, 1):
        records = prepare_records(base, refiner, path, "test", inventory, output, identity)
        predicted = build_predictions(records, config, (2048, 2048))
        rows.extend(dict(filament_id=f"{path.stem}_{i}", segmentation_rle=p["rle"]["counts"].decode())
                    for i, p in enumerate(predicted, 1))
        if index % 20 == 0:
            print(f"export={filename} images={index}/180 rows={len(rows)}", flush=True)
    destination = output / filename
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(destination, index=False)
    return dict(file=filename, rows=len(rows), sha256=sha256(destination))


def run_v10():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--limit", type=int, default=0, help="Tuning-image smoke limit; never exports/promotes")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if any((args.output / filename).exists() for filename in ("submission.csv", "submission_control.csv")):
        raise ValueError("Use a fresh output directory; existing exports must not survive a failed/smoke run")
    global core, tta
    import optimize_inference as core
    import detector_tta as tta
    import torch
    sources = list(args.input_root.rglob("v6_provenance.json"))
    refiners = list(args.input_root.rglob("best_refiner_coco.pt"))
    if len(sources) != 1 or len(refiners) != 1:
        raise ValueError("Expected exactly one V6 dataset source and one COCO refiner")
    if sha256(sources[0]) != PROVENANCE_SHA256 or sha256(refiners[0]) != REFINER_SHA256:
        raise ValueError("Changed source provenance or COCO refiner")
    provenance = json.loads(sources[0].read_text())
    for name, digest in provenance["sha256"].items():
        if sha256(sources[0].parent / name) != digest:
            raise ValueError(f"Changed V6 source: {name}")
    base = core.load_base(sources[0].parent / "solar_base.py")
    base.seed_everything(42)
    if sha256(base.CFG.train_json) != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation file changed")
    _, images, annotations = base.load_coco()
    split_info = core.validation_split(base, images, annotations)
    held_out, tune, audit, _, by_file = split_info
    core.verify_split(provenance, held_out, images)
    if (len(tune), len(audit), sum(map(len, by_file.values()))) != (94, 47, 217):
        raise ValueError("Expected tune94/audit47 and all 217 annotator entries")
    test_names = sorted(p.name for p in base.CFG.test_dir.glob("*.jpeg"))
    specs = [("train", name) for name in held_out] + [("test", name) for name in test_names]
    if len(test_names) != 180 or any(not (base.CFG.train_dir / name).is_file() for name in held_out):
        raise ValueError("Missing validation/test images")
    roots, inventory = resolve_cache_sources(args.input_root, provenance, tune, audit, specs, RAW_CACHE_SHA256)
    baseline_path = roots["v6"] / "inference_image_diagnostics.csv"
    if sha256(baseline_path) != BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed gold control diagnostics")
    v8_baseline_path = roots["v8"] / "inference_image_diagnostics.csv"
    if sha256(v8_baseline_path) != V8_DIAGNOSTICS_SHA256:
        raise ValueError("Changed V8 source-control diagnostics")
    previous = json.loads((roots["v6"] / "inference_manifest.json").read_text())
    versions = {name: base.version(name) for name in previous["versions"]}
    if versions != previous["versions"]:
        raise ValueError(f"Refiner runtime must reproduce V6 exactly: {versions}")
    args.output.mkdir(parents=True, exist_ok=True)
    configs = declared_candidates()
    identity = dict(script_sha256=sha256(Path(__file__)), refiner_sha256=REFINER_SHA256,
                    annotation_sha256=provenance["annotation_sha256"])
    manifest = dict(experiment="v10_frozen_quality_rescue", **identity, source=provenance,
                    versions=versions, source_pins=SOURCE_PINS, raw_cache_sha256=RAW_CACHE_SHA256,
                    tune_files=sorted(tune), audit_files=sorted(audit), target_rasterization=TARGET,
                    smoke_limit=args.limit, refiner_epochs=0, detector_inference=False,
                    predeclared_candidates=configs, maximum_candidates=15,
                    quality_proxy_definitions=dict(
                        proposal_foreground_probability="qP=mean(p[P]); P original anchor proposal, p resized float32 four-flip mean; zero for empty P",
                        foreground_probability="qF=mean(p[R]); R constrained thresholded refinement BEFORE fallback/cleanup; zero for empty R",
                        retention_fraction="|P intersection R| / |P| before fallback/cleanup; zero for empty P",
                        support_count="Distinct detector views assigned to fixed anchor; not independent votes",
                        anchor_member_ious="Raw mask IoU of each assigned member with fixed strongest anchor, including self=1",
                        quality_rank="V6 consensus*qF only reorders confidence>=.40 masks before fresh cleanup; no new score cutoff",
                        rescue_residual_fraction="Residual pixels / untrimmed refined V8 area after subtracting owned pixels; require >=.90"),
                    combination_gate="Both isolated tune PQ deltas > 1e-12; no audit selection",
                    promotion_gate=dict(min_tune_delta=0.001, min_audit_delta=0.0),
                    selection="tune PQ only, ties within 1e-12 prefer fewer interventions/control",
                    audit_evaluation="Only unchanged control and the frozen tune-selected candidate; no fallback selection",
                    baseline_submission=56026663, baseline_public_score=0.38,
                    audit_caveat=previous["audit_caveat"], full_control_validated=False, promotable=False,
                    requires_full_csv_validation=True, public_score="unmeasured")
    # Written before model creation or any candidate measurement.
    json_write(args.output / "inference_manifest.json", manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is needed only for the frozen crop refiner")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    refiner = base.CropUNet(pretrained=False).to("cuda:0")
    refiner.load_state_dict(torch.load(refiners[0], map_location="cuda:0", weights_only=True))
    refiner.eval()
    totals, diagnostics, source_diagnostics = {}, [], []
    tune_names = [name for name in held_out if name in tune]
    if args.limit:
        tune_names = tune_names[:args.limit]
    isolated = isolated_candidates(configs)
    evaluate_files(base, refiner, tune_names, isolated, images, annotations, split_info,
                   inventory, args.output, identity, totals, diagnostics, source_diagnostics)
    verify_control_counts(diagnostics, baseline_path, tune_names)
    verify_control_counts(source_diagnostics, v8_baseline_path, tune_names, V8_CONTROL, V8_OLD_CONTROL)
    if args.limit:
        write_v10_results(totals, isolated, diagnostics, args.output)
        pd.DataFrame(source_diagnostics).to_csv(args.output / "source_control_image_diagnostics.csv", index=False)
        json_write(args.output / "selected_config.json", dict(candidate=None, smoke=True, promotable=False,
                   full_control_validated=False, reason="Smoke test cannot establish improvement"))
        print("Smoke test complete; partial counts checked, no audit, export, or promotion", flush=True)
        return
    tune_scores = {key: core.pq_value(value["tune"]) for key, value in totals.items()}
    eligible = eligible_candidates(configs, tune_scores)
    combinations = {key: value for key, value in eligible.items() if key not in isolated}
    if combinations:
        evaluate_files(base, refiner, tune_names, combinations, images, annotations, split_info,
                       inventory, args.output, identity, totals, diagnostics, source_diagnostics)
    tune_scores = {key: core.pq_value(value["tune"]) for key, value in totals.items()}
    selected = select_candidate(eligible, tune_scores)
    selection = dict(candidate=selected, config=eligible[selected], selected_on="tune_pq only",
                     unchanged=selected == CONTROL, eligible=list(eligible),
                     excluded_combinations=[key for key in configs if key not in eligible])
    json_write(args.output / "selected_config.json", selection)  # Freeze BEFORE audit evaluation.
    audit_names = [name for name in held_out if name in audit]
    audited = {key: eligible[key] for key in dict.fromkeys((CONTROL, selected))}
    evaluate_files(base, refiner, audit_names, audited, images, annotations, split_info,
                   inventory, args.output, identity, totals, diagnostics, source_diagnostics)
    verify_control_counts(diagnostics, baseline_path, held_out)
    verify_control_counts(source_diagnostics, v8_baseline_path, held_out, V8_CONTROL, V8_OLD_CONTROL)
    pd.DataFrame(source_diagnostics).to_csv(args.output / "source_control_image_diagnostics.csv", index=False)
    write_v10_results(totals, eligible, diagnostics, args.output, audited)
    comparisons = {key: {group: paired_comparison(diagnostics, key, group)
                         for group in (("tune", "audit", "all") if key in audited else ("tune",))}
                   for key in eligible}
    report = dict(**selection, control=CONTROL, comparison=comparisons[selected],
                  ablations=comparisons, public_score="unmeasured", audit_caveat=manifest["audit_caveat"])
    json_write(args.output / "inference_comparison.json", report)
    control_export = export_config(base, refiner, configs[CONTROL], inventory, args.output, identity,
                                   "submission_control.csv")
    verify_control_export(args.output / "submission_control.csv")
    manifest["full_control_validated"] = True
    manifest["promotable"] = (selected != CONTROL and comparisons[selected]["tune"]["delta_pq"] >= 0.001
                              and comparisons[selected]["audit"]["delta_pq"] >= 0.0)
    manifest["selected_candidate"] = selected
    json_write(args.output / "inference_manifest.json", manifest)
    if selected == CONTROL:
        (args.output / "submission.csv").write_bytes((args.output / "submission_control.csv").read_bytes())
        selected_export = dict(control_export, file="submission.csv")
    else:
        selected_export = export_config(base, refiner, eligible[selected], inventory, args.output, identity,
                                        "submission.csv")
    json_write(args.output / "submission_candidates.json", [dict(control_export, candidate=CONTROL),
               dict(selected_export, **selection, promotable=manifest["promotable"],
                    full_control_validated=True, requires_full_csv_validation=True)])
    print(json.dumps(dict(candidate=selected, promotable=manifest["promotable"],
                          exports=[control_export, selected_export]), indent=2), flush=True)


if __name__ == "__main__":
    run_v10()
