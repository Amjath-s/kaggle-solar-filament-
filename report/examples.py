"""Example-segmentation figure for the report (CPU, train files only, OOF predictions).

Panels per row: H-alpha crop | annotator records (one colour per record) | V84 OOF prediction.
Files are chosen deterministically: the 3-record files sorted by stem, taking the 3 with the median
number of GT polygons (not cherry-picked for score). Predictions come from outputs/v87/fold*_v84.csv,
the leak-free 5-fold OOF export in submission format.
  CUDA_VISIBLE_DEVICES= .venv/bin/python report/examples.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pycocotools import mask as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
OUT = ROOT / "report/figures"
H = W = 2048
CROP = 640
csv.field_size_limit(sys.maxsize)

ann = json.load(open(RAW / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
by_stem = defaultdict(list)  # stem -> [(record_id, [rle, ...])]
polys = defaultdict(list)
for a in ann["annotations"]:
    polys[a["image_id"]].append(a["segmentation"])
for im in ann["images"]:
    rid = im["id"]
    stem = im["file_name"].rsplit(".", 1)[0]
    rles = []
    for seg in polys.get(rid, []):
        r = mu.merge(mu.frPyObjects(seg, H, W))
        if mu.area(r) > 0:
            rles.append(r)
    by_stem[stem].append((rid, rles))

pred = defaultdict(list)
for p in sorted((ROOT / "outputs/v87").glob("fold*_v84.csv")):
    for row in csv.DictReader(open(p)):
        stem = row["filament_id"].rsplit("_", 1)[0]
        pred[stem].append({"size": [H, W], "counts": row["segmentation_rle"]})

three = sorted(s for s, r in by_stem.items() if len(r) == 3 and s in pred)
assert len(three) > 10, len(three)
three.sort(key=lambda s: (sum(len(r) for _, r in by_stem[s]), s))
mid = len(three) // 2
picks = three[mid - 1:mid + 2]
print("picked", picks)

COLORS = [(0.95, 0.3, 0.3), (0.3, 0.8, 0.3), (0.3, 0.5, 1.0)]
fig, axes = plt.subplots(len(picks), 3, figsize=(7.0, 2.35 * len(picks)))
for i, stem in enumerate(picks):
    gray = cv2.imread(str(RAW / "train_images" / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    recs = by_stem[stem]
    union = np.zeros((H, W), bool)
    for _, rles in recs:
        for r in rles:
            union |= mu.decode(r).astype(bool)
    ys, xs = np.nonzero(union)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - CROP // 2, 0, H - CROP)); x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    for j in range(3):
        ax = axes[i, j]
        ax.imshow(gray[sl], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
    for k, (_, rles) in enumerate(recs):
        for r in rles:
            m = mu.decode(r)[sl]
            if m.any():
                axes[i, 1].contour(m, levels=[0.5], colors=[COLORS[k]], linewidths=0.6)
    n_pred = 0
    for r in pred[stem]:
        m = mu.decode(r)[sl]
        if m.any():
            n_pred += 1
            axes[i, 2].contour(m, levels=[0.5], colors=["yellow"], linewidths=0.6)
    axes[i, 0].set_ylabel(stem, fontsize=6)
    axes[i, 1].set_title(f"{len(recs)} annotator records ({', '.join(str(len(r)) for _, r in recs)} polygons)", fontsize=7)
    axes[i, 2].set_title(f"V84 OOF prediction ({n_pred} masks in crop)", fontsize=7)
axes[0, 0].set_title(f"H$\\alpha$ crop ({CROP} px of 2048)", fontsize=7)
fig.tight_layout(h_pad=0.4, w_pad=0.3)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"examples.{ext}", dpi=220)
json.dump({"picks": picks}, open(OUT / "examples.json", "w"))
print("wrote", OUT / "examples.png")
