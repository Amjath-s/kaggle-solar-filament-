"""Check sweep correctness without installing the GPU training dependencies."""
import ast
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils

SOURCE = Path(__file__).resolve().parents[1] / "solar/optimize_inference.py"
NAMES = {"encode", "annotation_rle", "rounded_legacy_rle", "clean_predictions", "pq_counts", "pq_value", "variant_key",
         "select_candidates", "materialize_masks", "verify_split", "encode_crop",
         "clean_refined_predictions", "sweep", "write_results"}
tree = ast.parse(SOURCE.read_text())
helpers = ast.Module(body=[node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name in NAMES],
                     type_ignores=[])
ns = {"np": np, "cv2": cv2, "mask_utils": mask_utils, "MIN_AREA": 20,
      "pd": pd, "gzip": gzip, "json": json, "Path": Path, "time": time, "CONFIDENCES": (0.4,)}
exec(compile(helpers, str(SOURCE), "exec"), ns)


def test_canonical_targets_and_legacy_targets_remain_separate():
    info = {"height": 32, "width": 32}
    annotation = {"segmentation": [[2, 2, 12, 2, 12, 7, 2, 7]]}
    assert int(mask_utils.area(ns["annotation_rle"](info, annotation))) == 50
    assert int(mask_utils.area(ns["rounded_legacy_rle"](info, annotation))) == 66


def test_sweep_counts_tiny_ground_truth_and_writes_cache(tmp_path, monkeypatch):
    name = "tiny.jpeg"
    images = {"ann": {"height": 32, "width": 32, "file_name": name}}
    annotations = {"ann": [{"segmentation": [[2, 2, 4, 2, 4, 4, 2, 4]]}]}
    monkeypatch.setitem(ns, "validation_split", lambda *args:
                        ([name], {name}, set(), {name: "ann"}, {name: ["ann"]}))
    monkeypatch.setitem(ns, "prediction_variants", lambda *args:
                        iter([("yolo_only", 0.0, False, [], np.asarray([]))]))
    base = SimpleNamespace(CFG=SimpleNamespace(train_dir=tmp_path))
    totals, _ = ns["sweep"](base, None, None, images, annotations, tmp_path, [1024], 0)
    counts = next(iter(totals.values()))
    assert counts["all"][3] == 1  # A 4-pixel annotation remains a false negative.
    assert counts["legacy"][3] == 0  # Historical filter is retained only for comparison.
    assert (tmp_path / "prediction_cache/r1024_tiny.json.gz").exists()


@pytest.mark.parametrize("bounds", [(0, 0, 40, 32), (3, 5, 20, 28), (39, 31, 40, 32)])
def test_sparse_crop_rle_matches_full_mask(bounds):
    x0, y0, x1, y1 = bounds
    rng = np.random.default_rng(19)
    for probability in (0.0, 0.4, 1.0):
        crop = rng.random((y1 - y0, x1 - x0)) < probability
        full = np.zeros((32, 40), dtype=np.uint8)
        full[y0:y1, x0:x1] = crop
        rle = ns["encode_crop"](crop, bounds, full.shape)
        np.testing.assert_array_equal(mask_utils.decode(rle), full)


@pytest.mark.parametrize("mode", ["refined_free", "refined_constrained"])
def test_cropped_postprocess_exactly_matches_dense_reference(mode):
    rng = np.random.default_rng(7)
    bounds = [(2, 4, 24, 30), (10, 12, 35, 38), (30, 20, 45, 40)]
    masks, probabilities = [], []
    for x0, y0, x1, y1 in bounds:
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[y0 + 4:y1 - 4, x0 + 4:x1 - 4] = 1
        masks.append(mask)
        probabilities.append(rng.random((16, 16)).astype(np.float32))
    probabilities[-1][:] = 0  # Exercise proposal fallback after empty refinement.
    scores = [0.4, 0.9, 0.7]
    full = ns["materialize_masks"](masks, bounds, probabilities, 0.5, mode)
    expected, expected_scores = ns["clean_predictions"](full, scores)
    actual, actual_scores = ns["clean_refined_predictions"](
        masks, scores, bounds, probabilities, 0.5, mode)
    np.testing.assert_array_equal(actual_scores, expected_scores)
    np.testing.assert_array_equal(mask_utils.decode(actual), mask_utils.decode(expected))


def test_cached_confidence_sweep_equals_independent_postprocessing():
    masks = []
    for x in (5, 8, 30):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:25, x:x + 15] = 1
        masks.append(mask)
    scores = np.array([0.2, 0.9, 0.5])
    cached, kept_scores = ns["clean_predictions"](masks, scores)
    for cutoff in (0.1, 0.3, 0.6, 0.95):
        selected = [rle for rle, score in zip(cached, kept_scores) if score >= cutoff]
        direct, _ = ns["clean_predictions"](
            [mask for mask, score in zip(masks, scores) if score >= cutoff],
            scores[scores >= cutoff])
        assert selected == direct
    decoded = mask_utils.decode(cached)
    assert decoded.sum(axis=2).max() == 1
    assert ns["clean_predictions"]([], [])[0] == []
    with pytest.raises(ValueError):
        ns["clean_predictions"](masks, [0.5])


def test_empty_and_strict_pq_matching():
    assert ns["pq_counts"](np.zeros((2, 0)), 2, 0) == (0.0, 0, 0, 2)
    assert ns["pq_counts"](np.zeros((0, 3)), 0, 3) == (0.0, 0, 3, 0)
    iou = np.array([[0.5, 0.0], [0.0, 0.75]])
    assert ns["pq_counts"](iou, 2, 2) == (0.75, 1, 1, 1)
    assert ns["pq_value"]((0.75, 1, 1, 1)) == 0.375


def test_candidate_selection_cannot_use_audit_scores():
    totals = {"a": {"tune": (0.7, 1, 0, 0), "audit": (0.0, 0, 5, 5)},
              "b": {"tune": (0.6, 1, 0, 0), "audit": (1.0, 1, 0, 0)}}
    configs = {key: {"resolution": 1024} for key in totals}
    assert ns["select_candidates"](totals, configs) == ["a"]


def test_constrained_refinement_and_empty_crop_fallback():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[25:35, 25:35] = 1
    args = ([mask], [(0, 0, 64, 64)])
    result = ns["materialize_masks"](*args, [np.ones((16, 16))], 0.5,
                                      "refined_constrained")[0]
    assert result.sum() > mask.sum()
    assert result[0, 0] == 0
    fallback = ns["materialize_masks"](*args, [np.zeros((16, 16))], 0.5,
                                        "refined_free")[0]
    np.testing.assert_array_equal(fallback, mask)


def test_provenance_rejects_training_leakage():
    images = {1: {"file_name": "train"}, 2: {"file_name": "valid"}}
    valid = {"training_files": ["train"], "validation_files": ["valid"]}
    ns["verify_split"](valid, ["valid"], images)
    with pytest.raises(ValueError):
        ns["verify_split"]({**valid, "training_files": ["valid"]}, ["valid"], images)
    with pytest.raises(ValueError):
        ns["verify_split"](valid, ["train"], images)
