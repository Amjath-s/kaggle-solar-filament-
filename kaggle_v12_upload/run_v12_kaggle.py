#!/usr/bin/env python3
"""V12 Kaggle entry: all-annotator YOLO export + YOLO11s-seg@1024 train (no flips).

Usage:
  python run_v12_kaggle.py
  python run_v12_kaggle.py --dry-run
  python run_v12_kaggle.py --skip-train   # pack existing best.pt if present
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))
SCRIPTS = BUNDLE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.data.split import read_split_list
from src.utils.paths import resolve_roots


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _unzip_bundle_zips(bundle: Path) -> None:
    """Kaggle --dir-mode zip leaves configs.zip etc.; expand if folders missing."""
    for name in ("configs", "scripts", "splits", "src"):
        zpath = bundle / f"{name}.zip"
        dest = bundle / name
        if not zpath.is_file():
            continue
        if dest.is_dir() and any(dest.iterdir()):
            continue
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)
        print(f"  unzipped {zpath.name} -> {dest}")


def step1_env() -> dict:
    print("=" * 60)
    print("STEP 1 — Environment verification")
    print("=" * 60)
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics_error"] = str(e)
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def step2_find_data(args):
    print("=" * 60)
    print("STEP 2 — Find competition + bundle")
    print("=" * 60)
    roots = resolve_roots(
        bundle_root=args.bundle_root,
        competition_root=args.competition_root,
        work_root=args.work_root,
    )
    for k, v in roots.as_dict().items():
        print(f"  {k}: {v}")
    n_jpeg = len(list(roots.images_dir.glob("*.jpeg"))) + len(
        list(roots.images_dir.glob("*.jpg"))
    )
    print(f"  images_count: {n_jpeg}")
    if n_jpeg < 100:
        raise FileNotFoundError(f"Too few images in {roots.images_dir}: {n_jpeg}")
    if not roots.annotations_path.is_file():
        raise FileNotFoundError(f"Annotations missing: {roots.annotations_path}")
    return roots


def step3_split(roots) -> tuple[list[str], list[str]]:
    print("=" * 60)
    print("STEP 3 — Load seed-42 split")
    print("=" * 60)
    train = read_split_list(roots.splits_dir / "train.txt")
    val = read_split_list(roots.splits_dir / "val.txt")
    print(f"  train={len(train)} val={len(val)}")
    if len(train) != 566 or len(val) != 141:
        print("  WARNING: expected train=566 val=141 (seed-42); continuing anyway")
    return train, val


def step4_export(roots, work: Path) -> dict:
    print("=" * 60)
    print("STEP 4 — Export ALL-annotator YOLO-seg labels")
    print("=" * 60)
    from export_yolo_seg_all import export_yolo_dataset_all

    out_dir = work / "yolo_filament_all"
    summary = export_yolo_dataset_all(
        images_dir=roots.images_dir,
        annotations_path=roots.annotations_path,
        splits_dir=roots.splits_dir,
        out_dir=out_dir,
    )
    (work / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resolve_pretrained(work: Path, model_name: str) -> Path:
    """Prefer cached/attached weights; else let Ultralytics download into work."""
    candidates = [
        work / "v12_weights" / model_name,
        work / model_name,
        Path("/kaggle/input") / model_name if Path("/kaggle/input").is_dir() else None,
    ]
    if Path("/kaggle/input").is_dir():
        for child in Path("/kaggle/input").rglob(model_name):
            candidates.append(child)
    for c in candidates:
        if c is not None and c.is_file():
            print(f"  using pretrained: {c}")
            return c
    # Ultralytics will download by name; copy into work cache after first use
    cache_dir = work / "v12_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(model_name)


def step5_train(roots, work: Path, export_summary: dict, cfg: dict, skip_train: bool) -> dict:
    print("=" * 60)
    print("STEP 5 — Train YOLO11s-seg @1024 (no flips)")
    print("=" * 60)
    out_dir = work / cfg.get("project_subdir", "v12_outputs/yolo11s")
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = work / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best = out_dir / "weights" / "best.pt"
    flat = out_dir / "best.pt"
    if skip_train:
        if best.is_file() or flat.is_file():
            print(f"  --skip-train: using existing weights under {out_dir}")
            return {"weights": str((best if best.is_file() else flat).resolve()), "skipped": True}
        raise FileNotFoundError(f"--skip-train but no best.pt under {out_dir}")

    from train_yolo11s_seg import train_yolo11s

    model_name = str(cfg.get("model", "yolo11s-seg.pt"))
    model_path = _resolve_pretrained(work, model_name)
    data_yaml = export_summary["data_yaml"]

    meta = train_yolo11s(
        model_path=model_path,
        data_yaml=data_yaml,
        out_dir=out_dir,
        imgsz=int(cfg.get("imgsz", 1024)),
        epochs=int(cfg.get("epochs", 80)),
        patience=int(cfg.get("patience", 20)),
        batch=int(cfg.get("batch", 4)),
        amp=bool(cfg.get("amp", True)),
        device=str(cfg.get("device", "0")),
        workers=int(cfg.get("workers", 2)),
        seed=int(cfg.get("seed", 42)),
        mask_ratio=int(cfg.get("mask_ratio", 1)),
        flipud=float(cfg.get("flipud", 0.0)),
        fliplr=float(cfg.get("fliplr", 0.0)),
    )
    return meta


def step6_export_zip(work: Path, train_meta: dict) -> Path:
    print("=" * 60)
    print("STEP 6 — Pack downloadable export")
    print("=" * 60)
    export_dir = work / "v12_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    src_w = Path(train_meta.get("weights_flat") or train_meta.get("weights", ""))
    if not src_w.is_file():
        # search
        hits = list((work / "v12_outputs").rglob("best.pt")) if (work / "v12_outputs").is_dir() else []
        if not hits:
            raise FileNotFoundError("No best.pt found to export")
        src_w = hits[0]

    dest_pt = export_dir / "yolo11s_best.pt"
    shutil.copy2(src_w, dest_pt)
    print(f"  copied {src_w} -> {dest_pt}")

    # optional companions
    for name in ("train_meta.json", "args.yaml", "results.csv"):
        for hit in Path(src_w).parents[1].rglob(name):
            shutil.copy2(hit, export_dir / hit.name)
            break

    readme = export_dir / "README_DOWNLOAD.txt"
    readme.write_text(
        "\n".join(
            [
                "V12 YOLO11s-seg trained weights",
                "",
                "On your PC, copy yolo11s_best.pt to:",
                "  solar-filament-segmentation/experiments/v12/yolo11s/weights/best.pt",
                "",
                "Next (local Agent): consensus TTA + ResNet18 refiner @256 + PQ eval.",
                "You do NOT need to download training images or the full Ultralytics run folder.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (export_dir / "train_meta.json").write_text(
        json.dumps(train_meta, indent=2), encoding="utf-8"
    )

    zip_path = work / "v12_export.zip"
    if zip_path.is_file():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in export_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(export_dir)))
    print(f"  wrote {zip_path}")
    print("  DOWNLOAD THIS FILE from the Kaggle notebook Output panel.")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="V12 Kaggle YOLO11s-seg train")
    parser.add_argument("--bundle-root", type=str, default=None)
    parser.add_argument("--competition-root", type=str, default=None)
    parser.add_argument("--work-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    env = step1_env()
    # Expand configs.zip etc. before path resolve (splits must be a real dir)
    try:
        br_early = Path(__file__).resolve().parent
        _unzip_bundle_zips(br_early)
    except Exception as e:
        print(f"  early unzip note: {e}")
    roots = step2_find_data(args)
    _unzip_bundle_zips(roots.bundle_root)
    train_names, val_names = step3_split(roots)

    cfg_path = roots.bundle_root / "configs" / "v12_train.yaml"
    cfg = _load_yaml(cfg_path) if cfg_path.is_file() else {}
    print(f"config: {cfg_path}")
    print(json.dumps(cfg, indent=2))

    work = roots.work_root
    work.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("=" * 60)
        print("DRY-RUN OK — env/data/split verified; skipping export/train")
        print("=" * 60)
        print(json.dumps({"env": env, "train": len(train_names), "val": len(val_names)}, indent=2))
        return 0

    export_summary = step4_export(roots, work)
    train_meta = step5_train(roots, work, export_summary, cfg, skip_train=args.skip_train)
    zip_path = step6_export_zip(work, train_meta)

    print("")
    print("=" * 60)
    print("V12 KAGGLE TRAIN DONE")
    print("=" * 60)
    print(f"weights: {train_meta.get('weights')}")
    print(f"download: {zip_path}")
    print("On PC → experiments/v12/yolo11s/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
