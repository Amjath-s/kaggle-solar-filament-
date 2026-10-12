"""Decode and validate a filament submission before spending a Kaggle submission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def validate_submission(path, test_stems, shape=(2048, 2048), min_area=20):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("Unexpected submission columns")
    if frame.filament_id.duplicated().any() or frame.empty:
        raise ValueError("Submission is empty or has duplicate instance IDs")
    parsed = frame.filament_id.str.rsplit("_", n=1, expand=True)
    if parsed.shape[1] != 2 or not parsed[1].str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Malformed filament IDs")
    unknown = set(parsed[0]) - set(test_stems)
    if unknown:
        raise ValueError(f"Unknown test images: {sorted(unknown)}")
    frame["image"] = parsed[0]
    areas = []
    for stem, rows in frame.groupby("image"):
        occupied = np.zeros(shape, dtype=bool)
        for row in rows.itertuples(index=False):
            mask = mask_utils.decode({"size": list(shape),
                                      "counts": row.segmentation_rle.encode("utf-8")}).astype(bool)
            area = int(mask.sum())
            if area < min_area:
                raise ValueError(f"Undersized mask: {row.filament_id}, area={area}")
            if np.any(mask & occupied):
                raise ValueError(f"Overlapping instances in {stem}: {row.filament_id}")
            occupied |= mask
            areas.append(area)
    counts = frame.groupby("image").size()
    return {"file": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "rows": len(frame), "populated_images": len(counts),
            "missing_images": sorted(set(test_stems) - set(counts.index)),
            "median_instances_populated": float(counts.median()),
            "min_mask_area": min(areas), "median_mask_area": float(np.median(areas)),
            "non_overlap": True, "full_rle_decode": True}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    test_dir = args.test_dir or next((root / "data/raw").rglob("test_images"))
    stems = {path.stem for path in test_dir.glob("*.jpeg")}
    if len(stems) != 180:
        raise ValueError(f"Expected 180 test images, found {len(stems)}")
    report = validate_submission(args.submission, stems)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
