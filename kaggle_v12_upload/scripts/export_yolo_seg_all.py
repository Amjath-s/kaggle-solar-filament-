"""Export MAGFiLO COCO polygons to YOLO-seg labels (ALL annotators, single class).

Unlike V7 primary-only export, this merges every annotator record per file_name
(friend V17 recipe). Creates images/{train,val} via hardlink/symlink/copy and
writes data.yaml.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from src.data.annotations import build_file_to_anns, load_coco
from src.data.split import read_split_list


def polygon_to_yolo_line(poly: list[float], w: int, h: int, cls: int = 0) -> str | None:
    if not poly or len(poly) < 6:
        return None
    xs = poly[0::2]
    ys = poly[1::2]
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    coords = []
    for x, y in zip(xs, ys):
        coords.append(f"{max(0.0, min(1.0, float(x) / w)):.6f}")
        coords.append(f"{max(0.0, min(1.0, float(y) / h)):.6f}")
    return f"{cls} " + " ".join(coords)


def export_split(
    names: list[str],
    file_to_anns: dict,
    images_dir: Path,
    labels_dir: Path,
    list_path: Path,
) -> tuple[int, int]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_img = 0
    n_inst = 0
    lines_out = []
    for name in names:
        img_path = images_dir / name
        if not img_path.is_file():
            continue
        anns = file_to_anns.get(name, [])
        w = h = 2048
        label_lines = []
        for ann in anns:
            for poly in ann.get("segmentation", []):
                line = polygon_to_yolo_line(poly, w, h, cls=0)
                if line:
                    label_lines.append(line)
                    n_inst += 1
        stem = Path(name).stem
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
        )
        lines_out.append(str(img_path.resolve()).replace("\\", "/"))
        n_img += 1
    list_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return n_img, n_inst


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def export_yolo_dataset_all(
    *,
    images_dir: Path,
    annotations_path: Path,
    splits_dir: Path,
    out_dir: Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(Path(annotations_path))
    file_to_anns = build_file_to_anns(coco)  # ALL annotators
    train_names = read_split_list(Path(splits_dir) / "train.txt")
    val_names = read_split_list(Path(splits_dir) / "val.txt")
    images_dir = Path(images_dir)

    n_tr, i_tr = export_split(
        train_names, file_to_anns, images_dir, out_dir / "labels" / "train", out_dir / "train.txt"
    )
    n_va, i_va = export_split(
        val_names, file_to_anns, images_dir, out_dir / "labels" / "val", out_dir / "val.txt"
    )

    for split, names in ("train", train_names), ("val", val_names):
        img_split = out_dir / "images" / split
        img_split.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = images_dir / name
            if not src.is_file():
                continue
            link_or_copy(src, img_split / name)

    yaml_text = f"""# YOLO-seg filament (single class) — ALL annotators
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: filament
"""
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {
        "out_dir": str(out_dir.resolve()),
        "data_yaml": str((out_dir / "data.yaml").resolve()),
        "annotators": "all",
        "n_train": n_tr,
        "n_val": n_va,
        "n_train_instances": i_tr,
        "n_val_instances": i_va,
    }
    print(f"ALL-annotator export: train images={n_tr} instances={i_tr}")
    print(f"ALL-annotator export: val images={n_va} instances={i_va}")
    print(f"wrote {out_dir / 'data.yaml'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default=str(BUNDLE / "splits"))
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    export_yolo_dataset_all(
        images_dir=Path(args.images_dir),
        annotations_path=Path(args.annotations),
        splits_dir=Path(args.splits_dir),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
