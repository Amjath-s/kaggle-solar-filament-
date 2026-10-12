"""Tests for V62 architecture YAML generation (pure text transform + Ultralytics graph parse)."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "solar"
spec = importlib.util.spec_from_file_location("v62_arch", SCRIPTS / "v62_arch.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

BASE = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5)

"""


def test_build_yaml_appends_layers_and_rewires_segment():
    for variant, order in (("p2first", "[[25, 16, 19, 22]"), ("p2last", "[[16, 19, 22, 25]")):
        out = V.build_yaml(BASE, variant)
        assert "[16, 1, nn.Upsample" in out and "[[-1, 2], 1, Concat" in out
        assert "C3k2, [128, False]" in out
        assert order in out
        assert "[[16, 19, 22], 1, Segment" not in out        # old 3-input head is gone
        assert out.rindex("C3k2, [128, False]") < out.index("Segment")   # new layers precede head


def test_build_yaml_rejects_unknown_base():
    with pytest.raises(ValueError):
        V.build_yaml("head:\n  - [[1,2,3], 1, Segment, [nc]]\n", "p2first")


def test_existing_layers_untouched():
    out = V.build_yaml(BASE, "p2first")
    assert out.startswith("head:\n  - [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]\n"
                          "  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)\n")


def test_real_yaml_builds_valid_graphs_with_four_inputs():
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import SegmentationModel

    text = V.base_yaml_path().read_text()
    for variant in ("p2first", "p2last"):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(V.build_yaml(text, variant))
        net = SegmentationModel(f.name, ch=3, nc=1, verbose=False)
        seg = net.model[-1]
        assert seg.nl == 4 and len(net.model) == 27
        assert sorted(float(s) for s in seg.stride) == [4.0, 8.0, 16.0, 32.0]
        expected_first = 4.0 if variant == "p2first" else 8.0   # x[0] feeds the prototypes
        assert float(seg.stride[0]) == expected_first
        net.train()
        with torch.no_grad():
            out = net(torch.zeros(1, 3, 128, 128))
        assert out["proto"].shape[-1] == (64 if variant == "p2first" else 32)
