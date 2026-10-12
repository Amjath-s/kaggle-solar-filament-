"""V73: the clamp-knob copy equals optimize_inference at k=7, and the pre-registered gate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solar"))
import train_v73_refiner as V  # noqa: E402

core = V.core


def test_clean_refined_k7_matches_core():
    rng = np.random.default_rng(0)
    masks, bounds, probs = [], [], []
    for x0, y0 in ((10, 10), (30, 25), (70, 60)):
        m = np.zeros((128, 128), np.uint8)
        m[y0:y0 + 20, x0:x0 + 6] = 1
        masks.append(m)
        bounds.append((x0 - 8, y0 - 8, x0 + 24, y0 + 24))
        probs.append(rng.random((16, 16)).astype(np.float32))
    scores = [0.9, 0.5, 0.7]
    a, sa = V.clean_refined(masks, scores, bounds, probs, 0.45, 7)
    b, sb = core.clean_refined_predictions(masks, scores, bounds, probs, 0.45,
                                           "refined_constrained")
    assert [r["counts"] for r in a] == [r["counts"] for r in b] and np.array_equal(sa, sb)
    wide, _ = V.clean_refined(masks, scores, bounds, probs, 0.45, 15)
    assert len(wide) == len(a)


def test_gate():
    c = {"delta": 0.012, "p_gt_0": 0.95, "ci95": [0.001, 0.02], "folds_positive": 4}
    assert V.verdict(c) == "CONTINUE"
    assert V.verdict({**c, "folds_positive": 3}) == "INCONCLUSIVE"
    assert V.verdict({**c, "ci95": [-0.001, 0.02]}) == "INCONCLUSIVE"
    assert V.verdict({**c, "delta": 0.004}) == "KILL"
