"""V21 full-data detector refit: frozen V6 base schedule then V16 mask_ratio=1.

Uses all 707 approved physical files as 1,154 independent annotator records with
8,199 one-class annotations. Since all labels are used for fitting, there is no
held-out model selection: both fixed-duration stages export their final checkpoint.
No test inference or Kaggle submission occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get(
    "V21_DATA_ROOT",
    str(ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
))
WORK = Path(os.environ.get("V21_WORK", "/mnt/solar-v21-full-refit"))
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_DIR = DATA_ROOT / "train/train_images"
YOLO_DATA = WORK / "yolo_all"
OUTPUT = WORK / "detector"
INIT_WEIGHTS = WORK / "inputs/yolo11s-seg.pt"
EXPECTED_ANNOTATION_SHA = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
EXPECTED = {"files": 707, "records": 1154, "annotations": 8199}
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def polygon_line(polygon, width: int, height: int) -> str:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("polygon has fewer than three points")
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.ravel())


def prepare_dataset() -> dict:
    if sha256(ANNOTATIONS) != EXPECTED_ANNOTATION_SHA:
        raise RuntimeError("approved annotation file identity changed")
    coco = json.loads(ANNOTATIONS.read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    actual = {
        "files": len({image["file_name"] for image in images.values()}),
        "records": len(images), "annotations": len(coco["annotations"]),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"all-data population mismatch: {actual} != {EXPECTED}")
    if YOLO_DATA.exists():
        manifest = YOLO_DATA / "manifest.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text())
            if saved.get("population") == EXPECTED and saved.get("annotation_sha256") == EXPECTED_ANNOTATION_SHA:
                verify_dataset()
                return saved
        raise RuntimeError(f"refusing to overwrite unverified dataset {YOLO_DATA}")

    image_out = YOLO_DATA / "images/train"
    label_out = YOLO_DATA / "labels/train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    line_count = 0
    for image_id in sorted(images, key=str):
        info = images[image_id]
        source = IMAGE_DIR / info["file_name"]
        if not source.is_file():
            raise RuntimeError(f"missing image {source}")
        source_path = Path(info["file_name"])
        target_name = f"{source_path.stem}_ann{image_id}{source_path.suffix}"
        (image_out / target_name).symlink_to(source)
        lines = [
            polygon_line(polygon, info["width"], info["height"])
            for annotation in annotations.get(image_id, [])
            for polygon in annotation["segmentation"]
            if len(polygon) >= 6
        ]
        line_count += len(lines)
        (label_out / f"{Path(target_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if line_count != EXPECTED["annotations"]:
        raise RuntimeError(f"expected 8199 polygon labels, wrote {line_count}")
    # val points to train only to satisfy dataset parsing; both training stages
    # explicitly set val=False and never use this for model selection.
    (YOLO_DATA / "data.yaml").write_text(
        f"path: {YOLO_DATA}\ntrain: images/train\nval: images/train\nnames:\n  0: filament\n"
    )
    manifest = {
        "population": EXPECTED, "annotation_sha256": EXPECTED_ANNOTATION_SHA,
        "records_written": len(images), "label_lines": line_count,
        "selection_scope": "all approved labels used for fitting; no local holdout",
    }
    write_json(YOLO_DATA / "manifest.json", manifest)
    verify_dataset()
    return manifest


def verify_dataset() -> None:
    images = list((YOLO_DATA / "images/train").glob("*.jpeg"))
    labels = list((YOLO_DATA / "labels/train").glob("*.txt"))
    if len(images) != EXPECTED["records"] or len(labels) != EXPECTED["records"]:
        raise RuntimeError(f"dataset record mismatch {len(images)}/{len(labels)}")
    if {p.stem for p in images} != {p.stem for p in labels}:
        raise RuntimeError("image/label stems differ")
    instances = sum(len([line for line in path.read_text().splitlines() if line.strip()]) for path in labels)
    if instances != EXPECTED["annotations"]:
        raise RuntimeError(f"dataset annotation mismatch {instances}")


def base_args(smoke: bool = False) -> dict:
    args = dict(
        epochs=80, patience=0, batch=4, imgsz=1024, workers=8,
        pretrained=True, optimizer="auto", seed=SEED, deterministic=True,
        single_cls=True, rect=False, cos_lr=False, close_mosaic=5, resume=False,
        amp=True, fraction=1.0, freeze=None, multi_scale=0.0,
        overlap_mask=False, mask_ratio=2, dropout=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
        scale=0.3, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        bgr=0.0, mosaic=0.3, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        save_period=20, val=False, plots=False,
    )
    if smoke:
        args.update(epochs=1, fraction=0.02, close_mosaic=0, save_period=-1)
    return args


def continuation_args(smoke: bool = False) -> dict:
    args = dict(
        epochs=20, patience=0, batch=4, imgsz=1024, workers=8,
        resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1, momentum=0.9,
        weight_decay=5e-4, cos_lr=False, warmup_epochs=1.0,
        warmup_bias_lr=0.0, nbs=64, seed=SEED, deterministic=True,
        single_cls=True, mask_ratio=1, overlap_mask=False, amp=True, rect=False,
        multi_scale=0.0, fraction=1.0, freeze=None,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1,
        scale=0.3, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0,
        bgr=0.0, mosaic=0.3, close_mosaic=5, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, save_period=5, val=False, plots=False,
    )
    if smoke:
        args.update(epochs=1, fraction=0.02, close_mosaic=0, save_period=-1)
    return args


def train_stage(name: str, source: Path, args: dict, device: str) -> dict:
    import pandas as pd
    import torch
    from ultralytics import YOLO

    stage = OUTPUT / name
    marker = stage / "stage_result.json"
    final = stage / "final.pt"
    signature = {
        "stage": name, "source_sha256": sha256(source), "args": args,
        "dataset_manifest": json.loads((YOLO_DATA / "manifest.json").read_text()),
        "device": device,
    }
    if marker.exists() and final.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or sha256(final) != saved["checkpoint_sha256"]:
            raise RuntimeError(f"cannot resume changed stage {name}")
        return saved
    stage.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = YOLO(str(source))
    model.train(
        data=str(YOLO_DATA / "data.yaml"), device=device,
        project=str(stage), name="run", exist_ok=False, **args,
    )
    elapsed = time.monotonic() - started
    save_dir = Path(model.trainer.save_dir)
    last = Path(model.trainer.last)
    shutil.copy2(last, final)
    results_path = save_dir / "results.csv"
    results = pd.read_csv(results_path)
    record = {
        "signature": signature, "elapsed_seconds": elapsed,
        "epochs_completed": len(results), "checkpoint_policy": "fixed-duration final checkpoint",
        "checkpoint_sha256": sha256(final), "checkpoint": str(final),
        "results_sha256": sha256(results_path), "save_dir": str(save_dir),
    }
    write_json(marker, record)
    del model
    torch.cuda.empty_cache()
    return record


def smoke(device: str) -> None:
    prepare_dataset()
    smoke_root = OUTPUT / "smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    original = globals()["OUTPUT"]
    globals()["OUTPUT"] = smoke_root
    base = train_stage("base", INIT_WEIGHTS, base_args(True), device)
    continuation = train_stage("mask_ratio1", Path(base["checkpoint"]), continuation_args(True), device)
    result = {"base": base, "continuation": continuation, "passed": True}
    write_json(WORK / "detector_smoke_result.json", result)
    globals()["OUTPUT"] = original
    print(json.dumps(result, indent=2))


def train(device: str) -> None:
    manifest = prepare_dataset()
    if not INIT_WEIGHTS.is_file():
        raise RuntimeError(f"missing pinned initialization {INIT_WEIGHTS}")
    base = train_stage("base", INIT_WEIGHTS, base_args(False), device)
    continuation = train_stage("mask_ratio1", Path(base["checkpoint"]), continuation_args(False), device)
    result = {
        "status": "complete", "population": manifest["population"],
        "base": base, "mask_ratio1": continuation,
        "selected_checkpoint": continuation["checkpoint"],
        "selected_checkpoint_sha256": continuation["checkpoint_sha256"],
        "validation": "none: all approved labels used for fitting",
    }
    write_json(WORK / "detector_full_result.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "train"))
    parser.add_argument("--device", default="0,1,2,3")
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED)
    if args.mode == "prepare":
        print(json.dumps(prepare_dataset(), indent=2))
    elif args.mode == "smoke":
        smoke(args.device)
    else:
        train(args.device)


if __name__ == "__main__":
    main()
