"""V61 fold detector: train the V17 detector recipe on 4/5 of ALL 707 files, hold out fold K.

Baseline recipe == the V21 refit of V17 (train_v21_full_detector.base_args + continuation_args):
YOLO11s-seg from the pinned COCO checkpoint, 80 epochs mask_ratio=2, then 20 epochs
mask_ratio=1 (AdamW 1e-4), imgsz 1024, batch 4, single class, final (fixed-duration) checkpoints.
Only the training population changes (fold-complement instead of all 707), so this is the V17
recipe measured out-of-fold, not a new model.

Held-out files are never written into the YOLO dataset. `val=False`; data.yaml's val points at
train only to satisfy parsing (same as V21).

  python solar/train_v61_fold.py --fold 99 --recipe p2first --work work/v61 --device 0
  python solar/train_v61_fold.py --fold 0 --smoke          # 1 epoch on 2% of data per stage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v21_full_detector as v21  # noqa: E402

ROOT = SCRIPTS.parent
DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = DATA_ROOT / "train/train_images"
INIT_WEIGHTS = ROOT / "yolo11s-seg.pt"
INIT_SHA16 = "1caa81c0195412ef"            # same COCO checkpoint V21 started from
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fold(folds_path: Path, fold: int):
    m = json.loads(folds_path.read_text())
    if m["annotation_sha256"] != v21.EXPECTED_ANNOTATION_SHA:
        raise RuntimeError("fold manifest built for a different annotation file")
    held = set(m["folds"][str(fold)])
    train = {n for k, f in m["folds"].items() if int(k) != fold for n in f}
    if held & train:
        raise RuntimeError("fold leakage: held-out file also in train set")
    return sorted(train), sorted(held)


def prepare_fold_dataset(fold: int, train_files, work: Path) -> dict:
    """YOLO dataset containing ONLY the training files' records (symlinked images)."""
    data = work / f"yolo_fold{fold}"
    manifest = data / "manifest.json"
    coco = json.loads(ANNOTATIONS.read_text())
    images = {i["id"]: i for i in coco["images"]}
    anns: dict = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    keep = set(train_files)
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["train_files"] == len(keep):
            return saved
        raise RuntimeError(f"refusing to overwrite differing dataset {data}")
    img_out, lab_out = data / "images/train", data / "labels/train"
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)
    n_rec = n_lab = 0
    for image_id in sorted(images, key=str):
        info = images[image_id]
        if info["file_name"] not in keep:
            continue
        src = IMAGE_DIR / info["file_name"]
        p = Path(info["file_name"])
        target = f"{p.stem}_ann{image_id}{p.suffix}"
        (img_out / target).symlink_to(src)
        lines = [v21.polygon_line(poly, info["width"], info["height"])
                 for a in anns.get(image_id, []) for poly in a["segmentation"] if len(poly) >= 6]
        (lab_out / f"{Path(target).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_rec += 1
        n_lab += len(lines)
    (data / "data.yaml").write_text(
        f"path: {data}\ntrain: images/train\nval: images/train\nnames:\n  0: filament\n")
    saved = {"fold": fold, "train_files": len(keep), "records": n_rec, "label_lines": n_lab,
             "annotation_sha256": v21.EXPECTED_ANNOTATION_SHA}
    manifest.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


RECIPES = ("baseline", "p2first")  # final repo: experiment-only recipes removed
ALL_DATA_FOLD = 99        # label for the final all-707-file model (no held-out set)


def stage_args(recipe: str, stage: str, smoke: bool) -> dict:
    """Training arguments: EXACTLY the V21/V17 recipe for both `baseline` and `p2first`."""
    if recipe not in RECIPES:
        raise NotImplementedError(f"recipe {recipe!r} not defined")
    return v21.base_args(smoke) if stage == "base" else v21.continuation_args(smoke)


def train_stage(name, source: Path, args: dict, data_yaml: Path, out: Path, device: str):
    import pandas as pd
    import torch
    from ultralytics import YOLO

    stage = out / name
    final, marker = stage / "final.pt", stage / "stage_result.json"
    if final.exists() and marker.exists():
        return json.loads(marker.read_text())
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model = YOLO(str(source))
    model.train(data=str(data_yaml), device=device, project=str(stage), name="run",
                exist_ok=False, **args)
    last = Path(model.trainer.last)
    shutil.copy2(last, final)
    results = pd.read_csv(Path(model.trainer.save_dir) / "results.csv")
    rec = {"stage": name, "elapsed_seconds": time.monotonic() - t0,
           "epochs_completed": len(results), "checkpoint": str(final),
           "checkpoint_sha256": sha256(final), "source": str(source), "args": args}
    marker.write_text(json.dumps(rec, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True, choices=list(range(5)) + [ALL_DATA_FOLD],
                    help="0-4 = OOF fold; 99 = ALL 707 files (final model, no held-out)")
    ap.add_argument("--folds", type=Path, default=ROOT / "configs/v61_folds.json")
    ap.add_argument("--recipe", default="baseline")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--device", default="0,1,2,3")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if sha256(ANNOTATIONS) != v21.EXPECTED_ANNOTATION_SHA:
        raise RuntimeError("annotation identity changed")
    if sha256(INIT_WEIGHTS)[:16] != INIT_SHA16:
        raise RuntimeError("COCO initial checkpoint changed")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    np.random.seed(SEED)

    if args.fold == ALL_DATA_FOLD:
        m = json.loads(args.folds.read_text())
        train_files = sorted(n for f in m["folds"].values() for n in f)
        held = []
        if len(train_files) != 707 or len(set(train_files)) != 707:
            raise RuntimeError("all-data mode must cover exactly 707 distinct files")
    else:
        train_files, held = load_fold(args.folds, args.fold)
    loss_info = None
    init_source = INIT_WEIGHTS
    if args.recipe == "p2first":
        if len([d for d in args.device.split(",") if d]) != 1:
            raise RuntimeError("V62 patches the validator in-process: use exactly one GPU")
        import v62_init
        import v62_patch
        init_source = ROOT / "outputs/v62_init/p2first_init.pt"
        if not init_source.exists():
            v62_init.build("p2first", init_source)
        v62_patch.install(2)
    out = args.work / (f"smoke_{args.recipe}_fold{args.fold}" if args.smoke
                       else f"{args.recipe}_fold{args.fold}")
    ds = prepare_fold_dataset(args.fold, train_files, args.work)
    data_yaml = args.work / f"yolo_fold{args.fold}" / "data.yaml"
    base = train_stage("base", init_source, stage_args(args.recipe, "base", args.smoke),
                       data_yaml, out, args.device)
    cont = train_stage("mask_ratio1", Path(base["checkpoint"]),
                       stage_args(args.recipe, "mask_ratio1", args.smoke), data_yaml, out,
                       args.device)
    result = {"status": "complete", "fold": args.fold, "recipe": args.recipe,
              "loss_patch": loss_info, "init_source": str(init_source),
              "init_sha256": sha256(Path(init_source)),
              "train_files": len(train_files), "held_out_files": len(held), "dataset": ds,
              "base": base, "mask_ratio1": cont, "checkpoint": cont["checkpoint"],
              "checkpoint_sha256": cont["checkpoint_sha256"], "reads_held_out": False}
    (out / "fold_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "fold", "recipe", "checkpoint",
                                             "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
