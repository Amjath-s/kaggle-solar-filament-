"""Geometry, duplicate suppression and baseline reproduction without GPU imports."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils

SOURCE = Path(__file__).resolve().parents[1] / "solar/detector_tta.py"
tree = ast.parse(SOURCE.read_text())
functions = {"flip_array", "cluster_views", "verify_control_counts", "tta_prediction_variants", "v8_detector_source"}
helpers = ast.Module(body=[node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name in functions],
                     type_ignores=[])
ns = {"np": np, "pd": pd, "mask_utils": mask_utils, "json": json, "hashlib": hashlib,
      "POLICIES": ("single", "flip_max", "flip_consensus")}
exec(compile(helpers, str(SOURCE), "exec"), ns)


def rectangle(x0, x1):
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:15, x0:x1] = 1
    return mask_utils.encode(np.asfortranarray(mask))


@pytest.mark.parametrize("axes", [(), (0,), (1,), (0, 1)])
def test_view_inverse_preserves_asymmetric_pixels_and_channels(axes):
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    flipped = ns["flip_array"](image, axes)
    assert flipped.flags.c_contiguous
    np.testing.assert_array_equal(ns["flip_array"](flipped, axes), image)
    mask = image[..., 0]
    np.testing.assert_array_equal(ns["flip_array"](ns["flip_array"](mask, axes), axes), mask)


def test_cross_view_duplicates_keep_strongest_shape_and_account_for_empty_views():
    a, b = rectangle(2, 12), rectangle(3, 13)
    views = [([a], [0.8]), ([b], [0.6]), ([], []), ([], [])]
    rles, scores = ns["cluster_views"](views, "flip_max")
    assert rles == [a]
    assert scores == [0.8]
    rles, scores = ns["cluster_views"](views, "flip_consensus")
    assert rles == [a]
    np.testing.assert_allclose(scores, [0.35])
    assert ns["cluster_views"](views, "single") == views[0]


def test_same_view_neighbors_cannot_merge_or_double_vote():
    a, b = rectangle(2, 12), rectangle(3, 13)
    views = [([a, b], [0.9, 0.8]), ([a], [0.7]), ([], []), ([], [])]
    masks, scores = ns["cluster_views"](views, "flip_consensus")
    assert len(masks) == 2
    np.testing.assert_allclose(scores, [0.4, 0.2])


def test_grouping_cannot_chain_neighboring_filaments():
    a, b, c = rectangle(2, 12), rectangle(5, 15), rectangle(8, 18)
    views = [([a], [0.9]), ([b], [0.8]), ([c], [0.7]), ([], [])]
    masks, scores = ns["cluster_views"](views, "flip_max")
    assert masks == [a, c]
    assert scores == [0.9, 0.7]
    assert ns["cluster_views"]([( [], [] )] * 4, "flip_consensus") == ([], [])


def test_baseline_verification_rejects_prediction_drift(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (tmp_path / "single").mkdir()
    data = pd.DataFrame([dict(candidate="r1024_refined_constrained_t0.50_tta1_c0.40",
                             file_name="test.jpeg", image_id=12, group="tune", target="coco",
                             iou_sum=0.7, tp=1, fp=0, fn=2)])
    data.to_csv(baseline / "inference_image_diagnostics.csv", index=False)
    data.to_csv(tmp_path / "single/inference_image_diagnostics.csv", index=False)
    ns["verify_control_counts"](tmp_path, baseline)
    data.loc[0, "tp"] = 2
    data.to_csv(tmp_path / "single/inference_image_diagnostics.csv", index=False)
    with pytest.raises(AssertionError):
        ns["verify_control_counts"](tmp_path, baseline)


def test_export_uses_selected_policy_but_keeps_fixed_proposal_floor(monkeypatch):
    views = [([rectangle(2, 12)], [0.8]), ([], []), ([], []), ([], [])]
    observed = []
    monkeypatch.setitem(ns, "proposal_views", lambda *args: views)
    monkeypatch.setitem(ns, "cv2", SimpleNamespace(IMREAD_GRAYSCALE=0, imread=lambda *a: np.zeros((24, 30))))
    def clean(masks, scores, bounds, probabilities, threshold, mode):
        observed.append((scores, threshold, mode))
        return [rectangle(2, 12)], np.asarray(scores)
    monkeypatch.setitem(ns, "core", SimpleNamespace(
        crop_probabilities=lambda *a: ([], [], []), clean_refined_predictions=clean))
    base = SimpleNamespace(CFG=SimpleNamespace(detector_policy="single"))
    selected = dict(detector_policy="flip_consensus", threshold=0.45, confidence=0.5)
    results = list(ns["tta_prediction_variants"](base, None, None, Path("test.jpeg"), 1024, selected))
    assert observed == [([0.2], 0.45, "refined_constrained")]
    assert results[0][-1].tolist() == [0.2]  # Final confidence cut is applied by export_submission.


@pytest.fixture
def v8_source(tmp_path, monkeypatch):
    checkpoint = tmp_path / "yolo_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = dict(experiment="v8_single_class_flips_all_annotator_validation",
                    train_ids=["train"], validation_best_ids=["valid1"],
                    validation_all_ids=["valid1", "valid2"], versions={"ultralytics": "8.4.140"})
    (tmp_path / "run_manifest.json").write_text(json.dumps(metadata))
    images = {"train": {"file_name": "train.jpeg"}, "valid1": {"file_name": "valid.jpeg"},
              "valid2": {"file_name": "valid.jpeg"}}
    provenance = dict(training_files=["train.jpeg"], validation_files=["valid.jpeg"])
    monkeypatch.setitem(ns, "EXPECTED_V8_DETECTOR_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setitem(ns, "V8_ULTRALYTICS", "8.4.140")
    return tmp_path, images, provenance, metadata


def test_v8_source_preserves_file_split_and_checkpoint_identity(v8_source):
    root, images, provenance, _ = v8_source
    path, result = ns["v8_detector_source"](root, images, provenance)
    assert path == root / "yolo_best.pt"
    assert result["training_files"] == 1
    assert result["validation_files"] == 1
    assert result["sha256"] == ns["EXPECTED_V8_DETECTOR_SHA256"]


@pytest.mark.parametrize("change", ["leak", "missing_annotator", "version", "checkpoint", "unknown_id"])
def test_v8_source_rejects_incompatible_or_changed_inputs(v8_source, change):
    root, images, provenance, metadata = v8_source
    if change == "leak":
        metadata["train_ids"].append("valid1")
    elif change == "missing_annotator":
        metadata["validation_all_ids"] = ["valid1"]
    elif change == "version":
        metadata["versions"]["ultralytics"] = "8.4.138"
    elif change == "checkpoint":
        (root / "yolo_best.pt").write_bytes(b"changed")
    elif change == "unknown_id":
        metadata["train_ids"] = ["unknown"]
    (root / "run_manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        ns["v8_detector_source"](root, images, provenance)
