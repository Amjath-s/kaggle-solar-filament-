"""Frozen COCO refiner: detector flip proposals and local threshold calibration."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

import optimize_inference as core

REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
CONTROL_SHA256 = "22e36c30e2953cf6f1fec4d9cccc5a2299a8348e4f42ef64a9b4960fb3d91e43"
POLICIES = ("single", "flip_max", "flip_consensus")
LOCAL_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
LOCAL_CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
DEFAULT_DETECTOR_VERSION = 6
EXPECTED_V8_DETECTOR_SHA256 = None
V8_ULTRALYTICS = "8.4.140"


def v8_detector_source(input_root, images, provenance):
    """Resolve the pinned V8 weights and reject a different or leaking split."""
    matches = []
    for path in input_root.rglob("run_manifest.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("experiment") == "v8_single_class_flips_all_annotator_validation":
            matches.append((path, metadata))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one V8 run manifest, found {len(matches)}")
    path, metadata = matches[0]
    train_ids, valid_ids = set(metadata["train_ids"]), set(metadata["validation_all_ids"])
    best_ids = set(metadata["validation_best_ids"])
    if not (train_ids | valid_ids | best_ids) <= set(images):
        raise ValueError("V8 manifest contains unknown annotation-image IDs")
    train_files = {images[i]["file_name"] for i in train_ids}
    valid_files = {images[i]["file_name"] for i in valid_ids}
    if (train_files != set(provenance["training_files"])
            or valid_files != set(provenance["validation_files"])
            or train_files & valid_files
            or {images[i]["file_name"] for i in best_ids} != valid_files):
        raise ValueError("V8 split differs from the accepted evaluation split")
    if valid_ids != {i for i, info in images.items() if info["file_name"] in valid_files}:
        raise ValueError("V8 validation is missing annotator entries")
    if metadata["versions"]["ultralytics"] != V8_ULTRALYTICS:
        raise ValueError("V8 runtime version does not match the pinned inference version")
    checkpoint = path.parent / "yolo_best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not EXPECTED_V8_DETECTOR_SHA256 or digest != EXPECTED_V8_DETECTOR_SHA256:
        raise ValueError("V8 detector checkpoint hash mismatch")
    return checkpoint, dict(kernel="abhimanyu122/solar-filament-yolo-refiner/8",
                           sha256=digest, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           training_files=len(train_files), validation_files=len(valid_files),
                           training_ultralytics=V8_ULTRALYTICS)


def flip_array(array, axes):
    """Each view is its own inverse; copy removes negative strides for inference."""
    return np.ascontiguousarray(np.flip(array, axis=axes)) if axes else array.copy()


def cluster_views(views, policy, match_iou=0.5):
    """Keep the strongest mask per cross-view group; never merge same-view instances.

    Match each new mask to a fixed highest-confidence anchor. This avoids chaining
    two neighboring filaments through a broad intermediate prediction. Consensus
    confidence includes zero votes from missing views, including empty views.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown proposal policy: {policy}")
    if policy == "single":
        return views[0] if views else ([], [])
    if not 0 < match_iou <= 1:
        raise ValueError("Matching IoU must be in (0, 1]")
    proposals = []
    for view_id, (rles, scores) in enumerate(views):
        if len(rles) != len(scores):
            raise ValueError("Every proposal needs a score")
        proposals.extend((rle, float(score), view_id) for rle, score in zip(rles, scores))
    proposals.sort(key=lambda item: -item[1])
    groups = []
    for rle, score, view_id in proposals:
        eligible = [i for i, group in enumerate(groups) if view_id not in group["views"]]
        match = None
        if eligible:
            overlaps = mask_utils.iou([rle], [groups[i]["rle"] for i in eligible],
                                     [0] * len(eligible))[0]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= match_iou:
                match = eligible[best]
        if match is None:
            groups.append(dict(rle=rle, scores=[score], views={view_id}))
        else:
            groups[match]["scores"].append(score)
            groups[match]["views"].add(view_id)
    scores = [max(g["scores"]) if policy == "flip_max" else sum(g["scores"]) / len(views)
              for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -scores[i])
    return [groups[i]["rle"] for i in order], [scores[i] for i in order]


def proposal_views(base, detector, image_path):
    """Cache all four views at a fixed confidence before any candidate selection."""
    cache_id = hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
    cache = base.CFG.proposal_cache / f"{cache_id}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt") as stream:
            serialized = json.load(stream)
        return [([dict(size=r["size"], counts=r["counts"].encode()) for r in view["rles"]],
                 view["scores"]) for view in serialized]
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        # Use the original path on the identity view to reproduce the accepted run.
        result = detector.predict(str(image_path) if not axes else flip_array(image, axes),
                                  imgsz=1024, conf=0.1, agnostic_nms=True,
                                  retina_masks=True, verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != image.shape[:2]:
                    raise ValueError("Detector masks must be in original image coordinates")
                rles.append(core.encode(restored))
        views.append((rles, scores))
    with gzip.open(cache, "wt") as stream:
        json.dump([dict(rles=[dict(size=r["size"], counts=r["counts"].decode()) for r in rles],
                        scores=scores) for rles, scores in views], stream)
    return views


def tta_prediction_variants(base, detector, refiner, image_path, resolution, selected=None):
    if resolution != 1024:
        raise ValueError("This ablation fixes detector resolution at 1024")
    policy = selected["detector_policy"] if selected is not None else base.CFG.detector_policy
    rles, scores = cluster_views(proposal_views(base, detector, image_path), policy)
    masks = [mask_utils.decode(rle) for rle in rles]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {image_path}")
    bounds, _, averaged = core.crop_probabilities(base, refiner, gray, masks, True)
    thresholds = (selected["threshold"],) if selected is not None else LOCAL_THRESHOLDS
    for threshold in thresholds:
        cleaned, kept_scores = core.clean_refined_predictions(
            masks, scores, bounds, averaged, threshold, "refined_constrained")
        yield "refined_constrained", threshold, True, cleaned, kept_scores


def combine_policy_results(output, policies):
    candidates, diagnostics = [], []
    for policy in policies:
        for filename, frames in (("inference_candidates.csv", candidates),
                                 ("inference_image_diagnostics.csv", diagnostics)):
            frame = pd.read_csv(output / policy / filename)
            frame["candidate"] = policy + "_" + frame.candidate
            frame["detector_policy"] = policy
            frames.append(frame)
    candidates = pd.concat(candidates, ignore_index=True).sort_values(
        ["tune_pq", "candidate"], ascending=[False, True])
    candidates.to_csv(output / "inference_candidates.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        output / "inference_image_diagnostics.csv", index=False)
    return candidates


def verify_control_counts(output, baseline):
    """Require unchanged predictions/evaluation before accepting an apparent gain."""
    name = "r1024_refined_constrained_t0.50_tta1_c0.40"
    actual = pd.read_csv(output / "single/inference_image_diagnostics.csv")
    expected = pd.read_csv(baseline / "inference_image_diagnostics.csv")
    fields = ["file_name", "image_id", "group", "target", "iou_sum", "tp", "fp", "fn"]
    a = actual[actual.candidate == name][fields].sort_values(fields[:4]).reset_index(drop=True)
    b = expected[expected.candidate == name][fields].sort_values(fields[:4]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-10)


def run_experiment():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detector-version", type=int, choices=(6, 8), default=DEFAULT_DETECTOR_VERSION)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    sources = list(Path("/kaggle/input").rglob("v6_provenance.json"))
    refiners = list(Path("/kaggle/input").rglob("best_refiner_coco.pt"))
    if len(sources) != 1 or len(refiners) != 1:
        raise ValueError(f"Expected one V6 source and one COCO refiner: {sources}, {refiners}")
    source, baseline = sources[0].parent, refiners[0].parent
    provenance = json.loads(sources[0].read_text())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    if hashlib.sha256(refiners[0].read_bytes()).hexdigest() != REFINER_SHA256:
        raise ValueError("Refiner differs from accepted 0.36 checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    base_path = source / "solar_base.py"
    if args.detector_version == 8:
        # Keep V8's training package version; do not silently downgrade its loader.
        helper = base_path.read_text()
        if helper.count("ultralytics==8.4.138") != 1:
            raise ValueError("Unexpected shared helper package pin")
        base_path = args.output / "solar_base_v8.py"
        base_path.write_text(helper.replace("ultralytics==8.4.138", f"ultralytics=={V8_ULTRALYTICS}"))
    base = core.load_base(base_path)
    base.seed_everything(42)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    _, images, annotations = base.load_coco()
    held_out, tune, audit, best_ids, _ = core.validation_split(base, images, annotations)
    core.verify_split(provenance, held_out, images)
    previous = json.loads((baseline / "inference_manifest.json").read_text())
    target = "COCO polygons; rounded OpenCV only for legacy comparison"
    for field, expected in (("tune_files", sorted(tune)), ("audit_files", sorted(audit)),
                            ("target_rasterization", target)):
        if previous[field] != expected:
            raise ValueError(f"Baseline mismatch: {field}")
    if hashlib.sha256(base.CFG.train_json.read_bytes()).hexdigest() != provenance["annotation_sha256"]:
        raise ValueError("Annotation source changed")
    detector_path = source / "yolo_best.pt"
    detector_provenance = dict(kernel=provenance["kernel"], sha256=provenance["sha256"]["yolo_best.pt"])
    policies, confidences = POLICIES, LOCAL_CONFIDENCES
    if args.detector_version == 8:
        detector_path, detector_provenance = v8_detector_source(Path("/kaggle/input"), images, provenance)
        policies = ("flip_consensus",)
        confidences = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
    manifest = dict(experiment=f"v{args.detector_version}_coco_refiner_detector_flip_tta", source=provenance,
                    detector=detector_provenance, detector_version=args.detector_version,
                    helper_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    refiner_checkpoint_sha256=REFINER_SHA256,
                    baseline_submission=56026663 if args.detector_version == 8 else 56025722,
                    baseline_public_score=0.38 if args.detector_version == 8 else 0.36,
                    target_rasterization=target,
                    tune_files=sorted(tune), audit_files=sorted(audit), smoke_limit=args.limit,
                    refiner_epochs=0, detector_policies=policies, detector_confidence_floor=0.1,
                    thresholds=LOCAL_THRESHOLDS, confidences=confidences,
                    resolutions=[1024], mask_match_iou=0.5,
                    audit_caveat="Excluded from inference tuning; used in V6 checkpoint selection",
                    versions={name: base.version(name) for name in
                              ("ultralytics", "torch", "torchvision", "pycocotools", "numpy")})
    (args.output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base.CFG.proposal_cache = args.output / "detector_proposal_cache" / detector_provenance["sha256"][:16]
    base.CFG.proposal_cache.mkdir(parents=True, exist_ok=True)
    detector = base.YOLO(str(detector_path))
    refiner = base.CropUNet(pretrained=False).to("cuda:0")
    refiner.load_state_dict(torch.load(refiners[0], map_location="cuda:0", weights_only=True))
    refiner.eval()
    core.prediction_variants = tta_prediction_variants
    core.CONFIDENCES = confidences
    for policy in policies:
        base.CFG.detector_policy = policy
        output = args.output / policy
        output.mkdir(exist_ok=True)
        core.sweep(base, detector, refiner, images, annotations, output, [1024], args.limit)
        if policy == "single" and not args.limit:
            verify_control_counts(args.output, baseline)
            print("Accepted control per-image counts reproduced", flush=True)
    candidates = combine_policy_results(args.output, policies)
    if args.limit:
        print("Smoke test finished; no submission exported", flush=True)
        return
    selected = candidates.iloc[0]
    export_rows = [("submission.csv", selected)]
    if args.detector_version == 6:
        control_key = "single_r1024_refined_constrained_t0.50_tta1_c0.40"
        control = candidates[candidates.candidate == control_key].iloc[0]
        export_rows.insert(0, ("submission_control.csv", control))
    exports = []
    for filename, row in export_rows:
        fields = ("resolution", "mode", "threshold", "tta", "confidence", "detector_policy")
        config = {field: row[field].item() if isinstance(row[field], np.generic) else row[field]
                  for field in fields}
        result = core.export_submission(base, detector, refiner, config, args.output / filename)
        result.update(candidate=row.candidate, tune_pq=row.tune_pq, audit_pq=row.audit_pq,
                      all_pq=row.all_pq, sha256=hashlib.sha256((args.output / filename).read_bytes()).hexdigest())
        if filename == "submission_control.csv" and result["sha256"] != CONTROL_SHA256:
            raise ValueError("Control submission differs from accepted 0.36 CSV")
        exports.append(result)
        (args.output / "submission_candidates.json").write_text(json.dumps(exports, indent=2) + "\n")
    print(json.dumps(exports, indent=2), flush=True)


if __name__ == "__main__":
    run_experiment()
