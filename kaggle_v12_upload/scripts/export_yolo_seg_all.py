"""Export MAGFiLO COCO polygons to YOLO-seg labels (ALL annotators, single class).

Unlike V7 primary-only export, this merges every annotator record per file_name
(friend V17 recipe). Creates images/{train,val} via hardlink/symlink/copy and
writes data.yaml.

per_annotator=True (v17): instead of stacking 2-3 annotators' near-duplicate polygons in one label
file (which, with overlap_mask, paints the duplicates over each other into slivers), every annotator
becomes its own training sample: <stem>__a<k>.jpeg -> link to the same image. Val stays one sample
per image (primary annotator) so val mAP is not inflated by duplicates.
train_all=True: train on train+val (final fit); val is still exported for monitoring only.
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

from src.data.annotations import build_file_to_ann_sets, build_file_to_anns, load_coco
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
    file_to_samples: dict[str, list[list[dict]]],
    images_dir: Path,
    labels_dir: Path,
    list_path: Path,
) -> tuple[int, int, list[tuple[str, str]]]:
    """file_to_samples: file_name -> one annotation list per training sample.

    A file with one sample keeps its name; with several, sample k is written as <stem>__a<k><suffix>.
    Returns (n_samples, n_instances, [(image file name, sample file name)]).
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    img_root = list_path.parent / "images" / labels_dir.name
    n_img = 0
    n_inst = 0
    lines_out = []
    links = []
    for name in names:
        img_path = images_dir / name
        if not img_path.is_file():
            continue
        samples = file_to_samples.get(name) or [[]]
        w = h = 2048
        for k, anns in enumerate(samples):
            label_lines = []
            for ann in anns:
                for poly in ann.get("segmentation", []):
                    line = polygon_to_yolo_line(poly, w, h, cls=0)
                    if line:
                        label_lines.append(line)
                        n_inst += 1
            sample = name if len(samples) == 1 else f"{Path(name).stem}__a{k}{Path(name).suffix}"
            (labels_dir / f"{Path(sample).stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )
            links.append((name, sample))
            lines_out.append((img_root / sample).resolve().as_posix())
            n_img += 1
    list_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return n_img, n_inst, links


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
    per_annotator: bool = False,
    train_all: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(Path(annotations_path))
    train_names = read_split_list(Path(splits_dir) / "train.txt")
    val_names = read_split_list(Path(splits_dir) / "val.txt")
    if train_all:
        train_names = train_names + [n for n in val_names if n not in set(train_names)]
    images_dir = Path(images_dir)
    if per_annotator:
        sets = build_file_to_ann_sets(coco)
        train_samples = {n: [s for s in v if s] or [[]] for n, v in sets.items()}
        val_samples = {n: [max(v, key=len)] for n, v in sets.items()}  # primary annotator
    else:
        merged = build_file_to_anns(coco)  # ALL annotators stacked in one label file
        train_samples = val_samples = {n: [a] for n, a in merged.items()}

    n_tr, i_tr, links_tr = export_split(
        train_names, train_samples, images_dir, out_dir / "labels" / "train", out_dir / "train.txt"
    )
    n_va, i_va, links_va = export_split(
        val_names, val_samples, images_dir, out_dir / "labels" / "val", out_dir / "val.txt"
    )

    for split, links in ("train", links_tr), ("val", links_va):
        img_split = out_dir / "images" / split
        img_split.mkdir(parents=True, exist_ok=True)
        for name, sample in links:
            link_or_copy(images_dir / name, img_split / sample)

    yaml_text = f"""# YOLO-seg filament (single class) — ALL annotators{" (one sample per annotator)" if per_annotator else ""}
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
        "per_annotator": per_annotator,
        "train_all": train_all,
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
    parser.add_argument("--per-annotator", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    args = parser.parse_args()
    export_yolo_dataset_all(
        images_dir=Path(args.images_dir),
        annotations_path=Path(args.annotations),
        splits_dir=Path(args.splits_dir),
        out_dir=Path(args.out_dir),
        per_annotator=args.per_annotator,
        train_all=args.train_all,
    )


if __name__ == "__main__":
    main()
