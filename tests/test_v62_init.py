"""Tests for V62 head-weight remapping (key logic + toy state dicts + real-model check)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "solar"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("v62_init", SCRIPTS / "v62_init.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_backbone_and_neck_keys_unchanged():
    for k in ("model.0.conv.weight", "model.22.cv2.conv.weight",
              "model.10.m.0.attn.qkv.conv.weight"):
        assert M.remap_key(k, "p2last") == k and M.remap_key(k, "p2first") == k


def test_head_moves_from_23_to_26():
    assert M.remap_key("model.23.proto.cv1.conv.weight", "p2last") == \
        "model.26.proto.cv1.conv.weight"
    assert M.remap_key("model.23.dfl.conv.weight", "p2first") == "model.26.dfl.conv.weight"


def test_scale_index_shift_only_for_p2first():
    k = "model.23.cv3.1.2.weight"
    assert M.remap_key(k, "p2last") == "model.26.cv3.1.2.weight"       # P3,P4,P5 stay 0,1,2
    assert M.remap_key(k, "p2first") == "model.26.cv3.2.2.weight"      # shifted by new P2 at 0
    assert M.remap_key("model.23.cv4.0.0.conv.weight", "p2first") == \
        "model.26.cv4.1.0.conv.weight"
    assert M.remap_key("model.23.one2one_cv2.2.1.conv.weight", "p2first") == \
        "model.26.one2one_cv2.3.1.conv.weight"


def test_proto_is_not_scale_indexed():
    k = "model.23.proto.cv2.conv.weight"
    assert M.remap_key(k, "p2first") == "model.26.proto.cv2.conv.weight"


def test_unknown_layer_index_above_head_is_unmapped():
    assert M.remap_key("model.24.whatever", "p2last") is None


def test_remap_state_dict_copies_same_shape_and_reports_mismatch():
    torch = pytest.importorskip("torch")
    coco = {"model.0.w": torch.ones(2), "model.23.cv3.0.2.weight": torch.full((80, 4), 7.0),
            "model.23.cv2.0.0.weight": torch.full((3, 3), 5.0)}
    new = {"model.0.w": torch.zeros(2), "model.26.cv3.0.2.weight": torch.zeros(1, 4),
           "model.26.cv2.0.0.weight": torch.zeros(3, 3), "model.24.extra": torch.zeros(1)}
    merged, rep = M.remap_state_dict(coco, new, "p2last")
    assert torch.equal(merged["model.0.w"], torch.ones(2))
    assert torch.equal(merged["model.26.cv2.0.0.weight"], torch.full((3, 3), 5.0))
    assert torch.equal(merged["model.26.cv3.0.2.weight"], torch.zeros(1, 4))   # shape differs
    assert len(rep["shape_mismatch"]) == 1
    assert rep["shape_mismatch"][0][1].endswith("cv3.0.2.weight")
    assert rep["copied"] == 2


def test_real_models_remap_puts_coco_values_in_shared_scales():
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    import v62_arch
    from ultralytics import YOLO

    v62_arch.write_all()
    sd0 = YOLO("yolo11s-seg.pt").model.state_dict()
    for variant in ("p2last", "p2first"):
        model = YOLO(str(v62_arch.CONFIGS / M.CFG[variant]))
        merged, rep = M.remap_state_dict(sd0, model.model.state_dict(), variant)
        shift = 1 if variant == "p2first" else 0
        a = sd0["model.23.cv2.1.0.conv.weight"]                  # COCO P4 box branch
        b = merged[f"model.26.cv2.{1 + shift}.0.conv.weight"]
        assert torch.equal(a, b)
        new_idx = 0 if variant == "p2first" else 3               # the brand-new P2 scale
        assert f"model.26.cv2.{new_idx}.0.conv.weight" in rep["new_only"]
        assert rep["copied"] > 500
