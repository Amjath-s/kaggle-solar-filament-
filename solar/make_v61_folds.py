"""V61: deterministic 5-fold split over ALL 707 physical files, grouped so no near-duplicate leaks.

Why grouped: GONG frames from the same calendar day show the same filaments. If one frame is in
the training folds and another from that day is held out, the OOF score is inflated. File names
are YYYYMMDDHHMMSS<site>.jpeg, so files sharing a calendar date are kept in ONE fold
(`--group-days 1`); `--group-days 3` additionally chains dates within 2 days of each other.

All annotator records of a physical file stay together (a file is the unit, never a record).
Folds are balanced on annotation count (not file count): greedy, largest group first, into the
currently lightest fold, with a seeded shuffle for tie-breaking. Fully deterministic.

  python scripts/make_v61_folds.py --annotations <ann.json> --out configs/v61_folds.json
  python scripts/make_v61_folds.py --annotations <ann.json> --report-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
N_FOLDS = 5
SEED = 61


def file_date(name: str) -> date:
    stem = Path(name).stem
    return date(int(stem[0:4]), int(stem[4:6]), int(stem[6:8]))


def group_files(names, group_days: int):
    """Union-find over files: same group if dates differ by < group_days days (chained)."""
    order = sorted(names, key=lambda n: (file_date(n), n))
    groups, current, last = [], [], None
    for n in order:
        d = file_date(n)
        if last is not None and (d - last).days >= group_days:
            groups.append(current)
            current = []
        current.append(n)
        last = d
    if current:
        groups.append(current)
    return groups


def assign_folds(groups, weight, n_folds=N_FOLDS, seed=SEED):
    """Greedy balance: biggest group first (seeded shuffle breaks ties), lightest fold wins."""
    rng = random.Random(seed)
    idx = list(range(len(groups)))
    rng.shuffle(idx)
    idx.sort(key=lambda i: -sum(weight[n] for n in groups[i]))
    load = [0] * n_folds
    folds = [[] for _ in range(n_folds)]
    for i in idx:
        k = min(range(n_folds), key=lambda f: (load[f], f))
        folds[k].extend(groups[i])
        load[k] += sum(weight[n] for n in groups[i])
    return [sorted(f) for f in folds]


def build(document, group_days: int = 0):
    by_file = defaultdict(list)
    for info in document["images"]:
        by_file[info["file_name"]].append(str(info["id"]))
    ann_count = defaultdict(int)
    id_to_file = {str(i["id"]): i["file_name"] for i in document["images"]}
    for a in document["annotations"]:
        ann_count[id_to_file[str(a["image_id"])]] += 1
    names = sorted(by_file)
    groups = group_files(names, group_days)
    weight = {n: max(ann_count[n], 1) for n in names}
    folds = assign_folds(groups, weight)
    return names, by_file, ann_count, groups, folds


def report(names, by_file, ann_count, groups, folds):
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {
        "files": len(names), "records": sum(len(v) for v in by_file.values()),
        "annotations": sum(ann_count.values()), "groups": len(groups),
        "largest_groups": sizes[:8], "files_in_multi_file_groups":
            sum(len(g) for g in groups if len(g) > 1),
        "fold_files": [len(f) for f in folds],
        "fold_records": [sum(len(by_file[n]) for n in f) for f in folds],
        "fold_annotations": [sum(ann_count[n] for n in f) for f in folds],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--group-days", type=int, default=0, dest="group_days",
                    help="0 = every physical file is its own group (matches how the test set "
                         "sits relative to train); N>0 keeps files < N days apart together")
    ap.add_argument("--report-only", action="store_true", dest="report_only")
    args = ap.parse_args()
    digest = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if digest != ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    doc = json.loads(args.annotations.read_text())
    names, by_file, ann_count, groups, folds = build(doc, args.group_days)
    rep = report(names, by_file, ann_count, groups, folds)
    print(json.dumps(rep, indent=2))
    if args.report_only:
        return
    manifest = {"annotation_sha256": digest, "seed": SEED, "group_days": args.group_days,
                "n_folds": N_FOLDS, "report": rep,
                "folds": {str(k): f for k, f in enumerate(folds)}}
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
