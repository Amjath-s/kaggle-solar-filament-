"""Strip a refiner training checkpoint down to an inference-only file that fits in the repo.

train_local_refiner.py saves model_state + optimizer_state (AdamW keeps two moment tensors per
parameter), so best.pt is ~3x the model (172 MB for ResNet18-UNet). Dropping optimizer_state leaves
the fp32 weights untouched (~58 MB), so predictions are bit-identical and under GitHub's 100 MB limit.

Usage:
    python scripts/export_refiner_weights.py
    python scripts/export_refiner_weights.py --src experiments/v16/r18aug_refiner/best.pt \
        --dst experiments/v16/r18aug_refiner/best_slim.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.unet import build_model
from src.utils.config import resolve_path

MAX_MB = 95.0  # GitHub hard limit is 100 MB per file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v16/r18aug_refiner/best.pt")
    ap.add_argument("--dst", default="experiments/v16/r18aug_refiner/best_slim.pt")
    args = ap.parse_args()
    src, dst = resolve_path(ROOT, args.src), resolve_path(ROOT, args.dst)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise KeyError(f"{src} has no 'model_state' (keys: {sorted(ckpt)})")
    slim = {k: v for k, v in ckpt.items() if k != "optimizer_state"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)

    # Verify: same tensors, and the config's architecture loads them strictly.
    back = torch.load(dst, map_location="cpu", weights_only=False)
    ref = ckpt["model_state"]
    assert back["model_state"].keys() == ref.keys(), "state_dict keys changed"
    for k, v in ref.items():
        if not torch.equal(back["model_state"][k], v):
            raise AssertionError(f"tensor mismatch: {k}")
    model_cfg = (back.get("config") or {}).get("model", {})
    model = build_model(
        model_cfg.get("name", "resnet18_unet"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=1,
        pretrained=False,
    )
    model.load_state_dict(back["model_state"], strict=True)

    src_mb, dst_mb = src.stat().st_size / 2**20, dst.stat().st_size / 2**20
    n_params = sum(v.numel() for v in ref.values() if v.is_floating_point())
    print(f"{args.src}: {src_mb:.1f} MB -> {args.dst}: {dst_mb:.1f} MB  "
          f"({n_params / 1e6:.2f}M float params, epoch={back.get('epoch')}, val_dice={back.get('val_dice')})")
    if dst_mb > MAX_MB:
        raise SystemExit(f"{dst.name} is {dst_mb:.1f} MB, over the {MAX_MB} MB budget for GitHub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
