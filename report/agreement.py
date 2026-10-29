"""Report figure (CPU): where V84's errors sit relative to annotator agreement, on 5-fold OOF (707 training files).

Reads the V85 per-GT / per-FP rows (outputs/v85_analysis/rows.pkl, written by solar/v85_analyze.py from the
V84 bundle's kept set, PQ 0.45898). (a) Recall of annotated filaments by how many of the image's n records drew
the same filament (k of n; agreement = other records' polygons matched at IoU > 0.5). (b) Record-level false
positives by class, split by whether another record of the same image drew the predicted filament.
  CUDA_VISIBLE_DEVICES= .venv/bin/python report/agreement.py
"""
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
rows = pickle.load(open(ROOT / "outputs/v85_analysis/rows.pkl", "rb"))
gts, fps = rows["gts"], rows["fps"]
assert len(gts) == 8199 and len(fps) == 1881, (len(gts), len(fps))

groups = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]  # (k, n): drawn by k of n records
rec = {}
for k, n in groups:
    sel = [g["tp"] for g in gts if g["n_rec"] == n and g["agree"] + 1 == k]
    rec[f"{k}/{n}"] = (float(np.mean(sel)), len(sel))
assert sum(v[1] for v in rec.values()) == len(gts)

cats = ["no_overlap", "nearmiss_0.3-0.5", "fragment(inside GT)", "weak_overlap", "merge(>=2 GT)"]  # V85 labels
names = dict(zip(cats, ["no overlap", "near-miss\n(IoU .3-.5)", "fragment", "weak", "merge"]))
other = Counter(f["cat"] for f in fps if f["other_rec"])
own = Counter(f["cat"] for f in fps if not f["other_rec"])
assert set(other) | set(own) <= set(cats), set(other) | set(own)

plt.rcParams.update({"font.size": 7})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(groups))
col = ["grey", "indianred", "seagreen", "indianred", "goldenrod", "seagreen"]
vals = [rec[f"{k}/{n}"][0] for k, n in groups]
a.bar(x, vals, color=col, edgecolor="black", lw=0.3)
for i, (k, n) in enumerate(groups):
    a.text(i, vals[i] + 0.02, f"{rec[f'{k}/{n}'][1]}", ha="center", fontsize=5)
a.set_xticks(x)
a.set_xticklabels([f"{k}/{n}" for k, n in groups])
a.set_xlabel("drawn by k of n records")
a.set_ylabel("recall (IoU > 0.5)")
a.set_ylim(0, 1.05)
a.set_title("(a) recall vs agreement", fontsize=7)
a.grid(axis="y", ls="--", lw=0.3)

y = np.arange(len(cats))
b.barh(y, [own.get(c, 0) for c in cats], color="steelblue", edgecolor="black", lw=0.3, label="no record drew it")
b.barh(y, [other.get(c, 0) for c in cats], left=[own.get(c, 0) for c in cats], color="darkorange",
       edgecolor="black", lw=0.3, label="another record's TP")
b.set_yticks(y)
b.set_yticklabels([names[c] for c in cats], fontsize=5.5)
b.invert_yaxis()
b.set_xlabel("record-level FP")
b.set_title("(b) FP by class", fontsize=7)
b.legend(fontsize=5, loc="lower right")
b.grid(axis="x", ls="--", lw=0.3)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"agreement.{ext}", dpi=220)
stats = {"recall_by_k_of_n": rec, "fp_other_record": dict(other), "fp_no_record": dict(own),
         "fp_other_total": sum(other.values()), "fp_total": len(fps)}
(OUT / "agreement_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats))
