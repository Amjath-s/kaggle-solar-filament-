"""Train V6-A local U-Net refiner on GT crops.

Usage:
    python scripts/train_local_refiner.py --config experiments/v6/configs/local_unet.yaml
    python scripts/train_local_refiner.py --config experiments/v6/configs/local_unet.yaml --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cached_local_crop_dataset import CachedLocalCropDataset
from src.evaluation.metrics import dice_iou_from_logits
from src.models.unet import build_model
from src.training.losses import BCEDiceLoss, FocalDiceLoss
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp: bool, epoch: int):
    model.train()
    total = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()
    for bi, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
        if bi == 1 or bi % 200 == 0 or bi == n_batches:
            print(
                f"  epoch {epoch} batch {bi}/{n_batches}  "
                f"loss={total / max(n, 1):.4f}  elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, threshold: float, use_amp: bool):
    model.eval()
    total_loss = 0.0
    n = 0
    dice_sum = 0.0
    iou_sum = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        dice, iou = dice_iou_from_logits(logits.float(), masks.float(), threshold=threshold)
        bs = images.size(0)
        total_loss += loss.item() * bs
        dice_sum += dice * bs
        iou_sum += iou * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "dice": dice_sum / max(n, 1),
        "iou": iou_sum / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/v6/configs/local_unet.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume (loads weights+optimizer; continues after ckpt epoch)",
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Load model_state only (fresh optimizer/epoch). Mutually exclusive with --resume.",
    )
    args = parser.parse_args()
    if args.resume and args.init_weights:
        raise SystemExit("Use only one of --resume or --init-weights")

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = resolve_path(ROOT, cfg["data"]["images_dir"])
    annotations = resolve_path(ROOT, cfg["data"]["annotations"])
    splits_dir = resolve_path(ROOT, cfg["data"]["splits_dir"])
    image_size = int(cfg.get("image_size", 512))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 2))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 12))
    lr = float(cfg.get("learning_rate", 1e-4))
    threshold = float(cfg.get("threshold", 0.5))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    pad_frac = float(cfg.get("pad_frac", 0.20))
    aug_cfg = cfg.get("augmentation", {})

    crop_root_cfg = cfg.get("data", {}).get("crop_root", "experiments/v6/crops")
    crop_root = resolve_path(ROOT, crop_root_cfg)
    if not (crop_root / "train" / "index.json").exists():
        raise FileNotFoundError(
            f"Crop cache missing at {crop_root}. "
            "Run: python scripts/build_local_crops.py  OR  python scripts/build_yolo_crops.py"
        )
    print(f"crop_root={crop_root}", flush=True)
    train_ds = CachedLocalCropDataset(
        crop_root,
        "train",
        augment=True,
        aug_cfg=aug_cfg,
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = CachedLocalCropDataset(
        crop_root,
        "val",
        augment=False,
        seed=int(cfg.get("seed", 42)) + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model_cfg = cfg.get("model", {})
    model = build_model(
        model_cfg.get("name", "unet"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        out_channels=int(cfg.get("num_classes", 1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    if cfg.get("loss", "bce_dice") == "focal_dice":
        loss_fn = FocalDiceLoss(
            focal_weight=float(cfg.get("focal_weight", 0.5)),
            dice_weight=float(cfg.get("dice_weight", 0.5)),
            gamma=float(cfg.get("focal_gamma", 2.0)),
            alpha=float(cfg.get("focal_alpha", 0.25)),
        )
    else:
        loss_fn = BCEDiceLoss(
            bce_weight=float(cfg.get("bce_weight", 0.5)),
            dice_weight=float(cfg.get("dice_weight", 0.5)),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_cfg = cfg.get("output", {})
    ckpt_dir = resolve_path(ROOT, out_cfg.get("checkpoint_dir", "experiments/v6/local_refiner"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_path = resolve_path(ROOT, out_cfg.get("dir", "experiments/v6")) / "train_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} train_crops={len(train_ds)} val_crops={len(val_ds)} "
        f"batch={batch_size} size={image_size} pad_frac={pad_frac} amp={use_amp}"
    )

    history = []
    best_dice = -1.0
    bad_epochs = 0
    start_epoch = 1
    t0 = time.time()

    if args.resume:
        resume_path = resolve_path(ROOT, args.resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("val_dice", -1.0))
        if hist_path.exists():
            prev = json.loads(hist_path.read_text(encoding="utf-8"))
            history = list(prev.get("history", []))
            best_dice = max(best_dice, float(prev.get("best_dice", best_dice)))
        print(f"resumed from {resume_path}  next_epoch={start_epoch}  best_dice={best_dice:.4f}")
    elif args.init_weights:
        init_path = resolve_path(ROOT, args.init_weights)
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = 1
        best_dice = -1.0
        print(f"init-weights from {init_path}  fresh optimizer  epochs={epochs}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        try:
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler, use_amp, epoch
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            print(f"OOM at batch={batch_size}; retrying with batch=1")
            torch.cuda.empty_cache()
            batch_size = 1
            train_loader = DataLoader(
                train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler, use_amp, epoch
            )

        val = validate(model, val_loader, loss_fn, device, threshold, use_amp)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "batch_size": batch_size,
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs}  loss={tr_loss:.4f}  "
            f"val_dice={val['dice']:.4f}  val_iou={val['iou']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "config": cfg,
            "batch_size": batch_size,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            bad_epochs = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  saved best.pt  dice={best_dice:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch} (patience={patience})")
                break

        with hist_path.open("w", encoding="utf-8") as f:
            json.dump({"history": history, "best_dice": best_dice}, f, indent=2)

    meta = {
        "best_dice": best_dice,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "batch_size": batch_size,
        "weights": str(ckpt_dir / "best.pt"),
        "pad_frac": pad_frac,
        "image_size": image_size,
    }
    with (ckpt_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_dice": best_dice}, f, indent=2)
    print(f"done. best_dice={best_dice:.4f}  weights={ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
