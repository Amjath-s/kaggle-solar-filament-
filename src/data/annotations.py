"""COCO-like annotation loading and polygon → binary mask.

Rasterization (`raster`):
  pil  : PIL polygon fill + outline (legacy; every local score up to v16b used this). Includes the
         boundary pixels, so thin filaments come out ~1 px fatter than the official masks.
  coco : pycocotools frPyObjects/decode, the conversion Kaggle documents for the ground truth and
         the submission format. Use this for anything compared against the leaderboard.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RASTERS = ("pil", "coco")


def load_coco(annotations_path: Path) -> dict:
    with annotations_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_file_to_anns(coco: dict) -> dict[str, list[dict]]:
    """Map unique file_name -> list of annotation dicts.

    Multiple COCO image rows can share the same file_name; we merge all
    annotations linked to any of those image ids.
    """
    file_to_ids: dict[str, set] = defaultdict(set)
    for im in coco["images"]:
        file_to_ids[im["file_name"]].add(im["id"])

    id_to_file: dict = {}
    for file_name, ids in file_to_ids.items():
        for i in ids:
            id_to_file[i] = file_name

    file_to_anns: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        file_name = id_to_file.get(ann["image_id"])
        if file_name is not None:
            file_to_anns[file_name].append(ann)
    return dict(file_to_anns)


def build_file_to_anns_primary(coco: dict) -> dict[str, list[dict]]:
    """Map file_name -> anns from a single image_id (primary annotator).

    Prefer the image_id that has the most annotations for that file.
    Avoids double-counting multi-annotator duplicates for instance metrics.
    """
    file_to_ids: dict[str, list] = defaultdict(list)
    for im in coco["images"]:
        file_to_ids[im["file_name"]].append(im["id"])

    id_to_anns: dict = defaultdict(list)
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)

    out: dict[str, list[dict]] = {}
    for file_name, ids in file_to_ids.items():
        best_id = max(ids, key=lambda i: len(id_to_anns.get(i, [])))
        out[file_name] = list(id_to_anns.get(best_id, []))
    return out


def build_file_to_ann_sets(coco: dict) -> dict[str, list[list[dict]]]:
    """Map file_name -> one annotation list per annotator (COCO image row), in JSON order."""
    id_to_anns: dict = defaultdict(list)
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)
    out: dict[str, list[list[dict]]] = defaultdict(list)
    for im in coco["images"]:
        out[im["file_name"]].append(list(id_to_anns.get(im["id"], [])))
    return dict(out)


def _polys(anns: list[dict]) -> list[list[float]]:
    return [p for a in anns for p in a.get("segmentation", []) if p and len(p) >= 6]


def _raster(polys: list[list[float]], height: int, width: int, raster: str) -> np.ndarray:
    if raster == "coco":
        if not polys:
            return np.zeros((height, width), dtype=np.uint8)
        from pycocotools import mask as mask_utils

        rle = mask_utils.merge(mask_utils.frPyObjects(polys, height, width))
        return np.ascontiguousarray(mask_utils.decode(rle), dtype=np.uint8)
    if raster != "pil":
        raise ValueError(f"raster must be one of {RASTERS}, got {raster!r}")
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polys:
        draw.polygon(list(zip(poly[0::2], poly[1::2])), outline=1, fill=1)
    return np.asarray(mask_img, dtype=np.uint8)


def ann_to_instance_mask(ann: dict, height: int, width: int, raster: str = "pil") -> np.ndarray:
    """Single annotation polygon(s) → binary uint8 mask."""
    return _raster(_polys([ann]), height, width, raster)


def anns_to_instance_masks(anns: list[dict], height: int, width: int, raster: str = "pil") -> list[np.ndarray]:
    """One mask per annotation; skip empty fills."""
    masks = []
    for ann in anns:
        m = ann_to_instance_mask(ann, height, width, raster)
        if m.sum() > 0:
            masks.append(m)
    return masks


def polygons_to_mask(
    anns: list[dict],
    height: int,
    width: int,
    raster: str = "pil",
) -> np.ndarray:
    """Binary uint8 mask (H, W) with 0/1 from filled polygons."""
    return _raster(_polys(anns), height, width, raster)
