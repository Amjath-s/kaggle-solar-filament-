"""Report figures (CPU): host self-evaluation plots (IoU, Dice, 1:n / n:1) for the V84 pipeline on
5-fold OOF predictions (707 training files only). Reuses the cached V84 extract (outputs/v84/extract_soft.pkl,
per-record full IoU matrices of every refined candidate) and the V84 bundle's cross-fitted demote rule,
so the kept set is exactly the scored V84 OOF bundle (asserted PQ 0.45898).
Host logic mirrored from references/notebooks/self_evaluation (plot_distribution, plot_m2n_counts):
pairs with IoU > 0 only; Dice = 2 IoU / (1 + IoU) for the same pair (exact identity); records without GT
are absent from the host GT frame, so they are skipped.
  CUDA_VISIBLE_DEVICES= .venv/bin/python report/plots.py
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solar"))
import v84_bundle as B  # noqa: E402

OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)

res = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]
A = B.arrays(res)
counts, keep, _ = B.run_arm(res, A, B.crossfit(A), 1, 1)
pq = B.V.pq(counts.sum(0))
assert abs(pq - 0.45898) < 5e-5, pq

iou_pairs, gt_deg, pred_deg = [], [], []
start = np.searchsorted(A["fi"], np.arange(len(res)))  # rows are grouped by file in order
tp_iou = []
for f, r in enumerate(res):
    k = keep[start[f]:start[f] + len(r["rows"])]
    for m in r["iou"]:
        if m.shape[0] == 0:
            continue
        m = m[:, k]
        iou_pairs.append(m[m > 0])
        tp_iou.append(m[m > 0.5])
        hit = m > 0
        gt_deg += hit.sum(1).tolist()
        pred_deg += hit.sum(0).tolist()
iou = np.concatenate(iou_pairs)
dice = 2 * iou / (1 + iou)
tp_iou = np.concatenate(tp_iou)
tp_dice = 2 * tp_iou / (1 + tp_iou)

plt.rcParams.update({"font.size": 8})


def dist(ax, s, tp, label):  # host plot_distribution: 50 bins on [0,1], probability, mean line
    bins = np.linspace(0, 1, 51)
    c, e = np.histogram(s, bins)
    ax.bar((e[:-1] + e[1:]) / 2, c / c.sum(), width=0.02, edgecolor="black", lw=0.3, alpha=0.7, color="navy")
    ax.axvline(s.mean(), color="crimson", ls="--", lw=1.2, label=f"mean (IoU>0 pairs) = {s.mean():.3f}")
    ax.axvline(tp.mean(), color="darkorange", ls=":", lw=1.2, label=f"mean (TP, IoU>0.5) = {tp.mean():.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Probability")
    ax.grid(axis="y", ls="--", lw=0.4)
    ax.legend(fontsize=6.5)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
dist(axes[0], iou, tp_iou, "IoU")
axes[0].set_title("(a) IoU distribution")
dist(axes[1], dice, tp_dice, "Dice")
axes[1].set_title("(b) Dice distribution")

ax = axes[2]  # host plot_m2n_counts
g, p = Counter(gt_deg), Counter(pred_deg)
for d, c in g.items():
    ax.bar(d + 0.5, c, 0.8, color="crimson" if d == 0 else "forestgreen", edgecolor="black", lw=0.3, alpha=0.7)
for d, c in p.items():
    ax.bar(-(d + 0.5), c, 0.8, color="darkorange" if d == 0 else "steelblue", edgecolor="black", lw=0.3, alpha=0.7)
pos = sorted([d + 0.5 for d in g] + [-(d + 0.5) for d in p])
ax.set_xticks(pos)
ax.set_xticklabels([f"{int(round(-v - .5))}:1" if v < 0 else f"1:{int(round(v - .5))}" for v in pos], rotation=60, fontsize=6)
ax.axvline(0, color="black", lw=0.6)
ax.set_yscale("log")
ax.set_ylabel("Frequency (log)")
ax.set_title("(c) n:1 (pred) | 1:n (GT)")
ax.set_ylim(5, 3e5)
ax.grid(axis="y", ls="--", lw=0.4)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color="forestgreen", label=f"GT hit {sum(c for d, c in g.items() if d)}"),
                   Patch(color="crimson", label=f"GT miss {g.get(0, 0)}"),
                   Patch(color="darkorange", label=f"pred miss {p.get(0, 0)}"),
                   Patch(color="steelblue", label=f"pred hit {sum(c for d, c in p.items() if d)}")], fontsize=5.5, ncol=2, loc="upper center")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"host_plots.{ext}", dpi=200)

stats = dict(pq=pq, tp_fp_fn=counts.sum(0)[1:].tolist(), kept=int(keep.sum()), n_pairs=int(iou.size),
             iou_mean=float(iou.mean()), dice_mean=float(dice.mean()), tp_iou_mean=float(tp_iou.mean()),
             tp_dice_mean=float(tp_dice.mean()), gt_degree=dict(sorted(g.items())), pred_degree=dict(sorted(p.items())),
             gt_split_ge2=sum(c for d, c in g.items() if d >= 2), pred_merge_ge2=sum(c for d, c in p.items() if d >= 2))
(OUT / "host_plots_stats.json").write_text(json.dumps(stats, indent=1, default=int))
print(json.dumps(stats, default=int))
