# ruff: noqa: E501  (long explanatory docstring)
"""V62 initialisation: transfer COCO yolo11s-seg weights into the 4-level (P2) architectures.

Problem found by `v62_arch.py probe`: the new layers (23-25) are inserted BEFORE the Segment head,
so the head moves from `model.23.*` (COCO) to `model.26.*`. Ultralytics transfers by exact key+shape,
so ALL 183 head tensors are silently skipped (378/639 tensors transfer: backbone+neck
only). The baseline keeps its COCO-pretrained head, so an un-remapped V62 would differ from the baseline in
head initialisation as well as in the P2 level - a confound.

This module remaps the head explicitly, so that everything that exists in both models starts from
the SAME COCO values and only genuinely new tensors are random:

  p2last  (inputs [P3,P4,P5,P2]): cv2/cv3/cv4 index i -> i for the 3 shared scales;
                                   proto.* unchanged
                                   (Proto still reads P3, 128 ch). New/random: layers 23-25 and
                                   cv*.3 (the P2 level).
  p2first (inputs [P2,P3,P4,P5]): cv2/cv3/cv4 index i -> i+1 (shared scales shift by
                                   one);
                                   proto.cv1 reads P2 (64 ch, COCO reads 128): random; proto.upsample,
                                   proto.cv2, proto.cv3 keep COCO values (same shapes). New/random:
                                   layers 23-25, cv*.0 (P2 level), proto.cv1.

Tensors whose shape differs (the 1-class classifier convs are NOT reshaped here; Ultralytics
re-initialises them when nc changes, exactly as for the baseline) are reported, not forced.

  python scripts/v62_init.py build --variant p2last --out outputs/v62_init/p2last_init.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

OLD_HEAD = 23           # Segment layer index in yolo11-seg.yaml
NEW_HEAD = 26           # Segment layer index in the V62 graphs
CFG = {"p2last": "yolo11s-p2last-seg.yaml", "p2first": "yolo11s-p2first-seg.yaml"}
_HEAD_SCALE = re.compile(r"^(cv2|cv3|cv4|one2one_cv2|one2one_cv3|one2one_cv4)\.(\d+)\.(.+)$")


def remap_key(key: str, variant: str) -> str | None:
    """COCO key -> V62 key (None if the tensor has no counterpart). Layers 0-22 are unchanged."""
    m = re.match(r"^model\.(\d+)\.(.*)$", key)
    if not m:
        return key
    idx, rest = int(m.group(1)), m.group(2)
    if idx < OLD_HEAD:
        return key
    if idx != OLD_HEAD:
        return None
    s = _HEAD_SCALE.match(rest)
    if s:
        i = int(s.group(2)) + (1 if variant == "p2first" else 0)
        rest = f"{s.group(1)}.{i}.{s.group(3)}"
    return f"model.{NEW_HEAD}.{rest}"


def remap_state_dict(coco_sd: dict, new_sd: dict, variant: str):
    """-> (merged state dict for the new model, report). Only same-shape tensors are copied."""
    merged = dict(new_sd)
    copied, shape_mismatch, unmapped = [], [], []
    for k, v in coco_sd.items():
        nk = remap_key(k, variant)
        if nk is None:
            unmapped.append(k)
        elif nk not in new_sd:
            unmapped.append(k)
        elif new_sd[nk].shape != v.shape:
            shape_mismatch.append((k, nk, tuple(v.shape), tuple(new_sd[nk].shape)))
        else:
            merged[nk] = v.clone()
            copied.append(nk)
    untouched = [k for k in new_sd if k not in set(copied)]
    return merged, {"copied": len(copied), "shape_mismatch": shape_mismatch,
                    "coco_unmapped": unmapped, "new_only": untouched,
                    "total_new_tensors": len(new_sd)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(variant: str, out: Path, coco: str = "yolo11s-seg.pt", cfg_name: str | None = None) -> dict:
    import v62_arch
    from ultralytics import YOLO

    cfg = v62_arch.CONFIGS / (cfg_name or CFG[variant])
    if not cfg.exists():
        v62_arch.write_all()
    base = YOLO(coco)
    model = YOLO(str(cfg))
    merged, rep = remap_state_dict(base.model.state_dict(), model.model.state_dict(), variant)
    model.model.load_state_dict(merged, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    rep.update({"variant": variant, "out": str(out), "sha256": sha256(out),
                "coco_sha16": sha256(Path(coco))[:16] if Path(coco).exists() else None})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--variant", choices=sorted(CFG), required=True)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rep = build(args.variant, args.out)
    shown = {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}
    print(json.dumps(shown, indent=2))
    print("shape mismatches:", [(a, c, d) for a, _b, c, d in rep["shape_mismatch"]][:12])
    print("new-only (random init) sample:", rep["new_only"][:6], "...")


if __name__ == "__main__":
    main()
