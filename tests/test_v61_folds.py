"""Tests for the V61 5-fold split (synthetic COCO-like document, no data needed)."""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "solar"
spec = importlib.util.spec_from_file_location("make_v61_folds", SCRIPTS / "make_v61_folds.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def _doc(n_files=60, records_per_file=(1, 2, 3), anns_per_record=(2, 5, 9)):
    images, anns = [], []
    iid = 0
    d0 = date(2015, 1, 1)
    for i in range(n_files):
        d = d0 + timedelta(days=i // 2)                  # two files per calendar day
        name = f"{d:%Y%m%d}{i % 2:02d}0000Lh.jpeg"
        for r in range(records_per_file[i % 3]):
            iid += 1
            images.append({"id": iid, "file_name": name, "width": 2048, "height": 2048})
            for _ in range(anns_per_record[(i + r) % 3]):
                anns.append({"image_id": iid, "segmentation": [[0, 0, 1, 0, 1, 1]]})
    return {"images": images, "annotations": anns}


def test_partition_covers_every_file_exactly_once():
    names, by_file, _, _, folds = F.build(_doc(), 0)
    flat = [n for f in folds for n in f]
    assert sorted(flat) == sorted(names) and len(flat) == len(set(flat))
    assert len(folds) == F.N_FOLDS and all(folds)


def test_records_of_a_file_never_split():
    doc = _doc()
    names, by_file, _, _, folds = F.build(doc, 0)
    where = {n: k for k, f in enumerate(folds) for n in f}
    for info in doc["images"]:
        assert info["file_name"] in where            # every record's file has exactly one fold


def test_deterministic():
    a = F.build(_doc(), 0)[4]
    b = F.build(_doc(), 0)[4]
    assert a == b


def test_balanced_on_annotation_count():
    _, by_file, ann_count, _, folds = F.build(_doc(200), 0)
    loads = [sum(ann_count[n] for n in f) for f in folds]
    assert (max(loads) - min(loads)) / (sum(loads) / len(loads)) < 0.05


def test_group_days_keeps_near_dates_together():
    doc = _doc(60)
    names, _, _, groups, folds = F.build(doc, 3)
    where = {n: k for k, f in enumerate(folds) for n in f}
    for g in groups:
        assert len({where[n] for n in g}) == 1
    days = sorted({F.file_date(n) for n in names})
    assert len(groups) < len(days)                   # consecutive days were chained


def test_zero_group_days_is_per_file():
    names, _, _, groups, _ = F.build(_doc(), 0)
    assert all(len(g) == 1 for g in groups) and len(groups) == len(names)
