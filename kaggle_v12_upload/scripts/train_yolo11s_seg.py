"""Train YOLO11s-seg for V12 detector (Kaggle GPU).

Friend V17 recipe: imgsz=1024, no train flips (flipud=0, fliplr=0).
Never silently substitutes another model family. On OOM: reduce batch,
then gradient accumulation; keep imgsz=1024.

hyp: extra Ultralytics train() arguments from the config (v17: flips, rotation, overlap_mask=False,
nbs, cos_lr, ...); they override the defaults above. Without an explicit nbs, Ultralytics accumulates
gradients up to 64 images, i.e. only ~9 optimizer steps per epoch at batch 2 on 566 images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def train_once(
    model_path: str,
    data_yaml: str,
    *,
    imgsz: int,
    epochs: int,
    patience: int,
    batch: int,
    amp: bool,
    device: str,
    workers: int,
    project: Path,
    name: str,
    seed: int,
    mask_ratio: int,
    flipud: float,
    fliplr: float,
    accumulate: int = 1,
    hyp: dict | None = None,
):
    import torch
    from ultralytics import YOLO

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(model_path)
    kwargs = dict(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        batch=batch,
        amp=amp,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        pretrained=True,
        seed=seed,
        plots=True,
        verbose=True,
        mask_ratio=mask_ratio,
        flipud=float(flipud),
        fliplr=float(fliplr),
    )
    kwargs.update(hyp or {})
    if accumulate > 1:
        kwargs["nbs"] = batch * accumulate
    return model.train(**kwargs)


def train_yolo11s(
    *,
    model_path: str | Path,
    data_yaml: str | Path,
    out_dir: Path,
    imgsz: int = 1024,
    epochs: int = 80,
    patience: int = 20,
    batch: int = 4,
    amp: bool = True,
    device: str = "0",
    workers: int = 2,
    seed: int = 42,
    mask_ratio: int = 1,
    flipud: float = 0.0,
    fliplr: float = 0.0,
    hyp: dict | None = None,
) -> dict:
    model_path = str(model_path)
    data_yaml = str(Path(data_yaml).resolve())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project = out_dir.parent
    name = out_dir.name

    # The model comes from the config by name; only refuse non-segmentation weights.
    if "-seg" not in Path(model_path).name.lower():
        raise RuntimeError(f"Refusing to train a non-segmentation model: {model_path}")

    batch_used = batch
    accumulate = 1
    imgsz_used = imgsz
    last_err: BaseException | None = None

    attempts = [
        (batch, 1),
        (max(1, batch // 2), 1),
        (1, max(2, batch)),
        (1, 1),
    ]
    uniq = []
    for a in attempts:
        if a not in uniq:
            uniq.append(a)

    for b, acc in uniq:
        try:
            print(
                f"Training YOLO11s-seg imgsz={imgsz} batch={b} accumulate={acc} "
                f"epochs={epochs} flipud={flipud} fliplr={fliplr} model={model_path} hyp={hyp or {}}"
            )
            train_once(
                model_path,
                data_yaml,
                imgsz=imgsz,
                epochs=epochs,
                patience=patience,
                batch=b,
                amp=amp,
                device=device,
                workers=workers,
                project=project,
                name=name,
                seed=seed,
                mask_ratio=mask_ratio,
                flipud=flipud,
                fliplr=fliplr,
                accumulate=acc,
                hyp=hyp,
            )
            batch_used = b
            accumulate = acc
            last_err = None
            break
        except Exception as e:
            last_err = e
            if _is_oom(e):
                import torch

                if torch.cuda.is_available():
                    try:
                        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        print(f"OOM at imgsz={imgsz} batch={b}: GPU mem ≈ {mem:.2f} GiB")
                    except Exception:
                        print(f"OOM at imgsz={imgsz} batch={b}")
                    torch.cuda.empty_cache()
                continue
            raise

    if last_err is not None:
        raise RuntimeError(
            f"YOLO11s-seg could not train at imgsz={imgsz}. Last error: {last_err}"
        ) from last_err

    weights_best = out_dir / "weights" / "best.pt"
    if not weights_best.is_file():
        alt = list(out_dir.rglob("best.pt"))
        if alt:
            weights_best = alt[0]
        else:
            raise FileNotFoundError(f"Missing best.pt under {out_dir}")

    flat = out_dir / "best.pt"
    if weights_best.resolve() != flat.resolve():
        shutil.copy2(weights_best, flat)

    meta = {
        "model": model_path,
        "data": data_yaml,
        "imgsz": imgsz_used,
        "epochs": epochs,
        "patience": patience,
        "batch": batch_used,
        "accumulate": accumulate,
        "mask_ratio": mask_ratio,
        "flipud": flipud,
        "fliplr": fliplr,
        "hyp": hyp or {},
        "amp": amp,
        "device": device,
        "seed": seed,
        "weights": str(weights_best.resolve()),
        "weights_flat": str(flat.resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done. imgsz={imgsz_used} batch={batch_used} weights={weights_best}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg (V12)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=int, default=1)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    args = parser.parse_args()
    train_yolo11s(
        model_path=args.model,
        data_yaml=args.data,
        out_dir=Path(args.out_dir),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        amp=args.amp,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        mask_ratio=args.mask_ratio,
        flipud=args.flipud,
        fliplr=args.fliplr,
    )


if __name__ == "__main__":
    main()
