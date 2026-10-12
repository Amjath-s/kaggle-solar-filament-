"""V84 stage 2 (CPU): score the polish bundle, its leave-one-out and single-component arms.
Reads outputs/v84/extract_{base,soft}.pkl, writes outputs/v84/result.json.
Pre-registration: docs/v84_polish_bundle_preregistration_2026-10-11.md.
Demote rule = V83 "demote" (same features/LightGBM params), refit cross-fitted per configuration.
  python scripts/v84_bundle.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v83_rule as V  # noqa: E402  (pq, file_counts, fit, boot, FEATS)

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v84"
CUT, MINA = 0.30, 200
GATE = dict(delta=0.005, p=0.90, folds=4)
# name: (refiner, t, min200, demote)
ARMS = {"baseline": ("base", 0.45, 0, 0), "bundle": ("soft", 0.50, 1, 1),
        "loo_soft": ("base", 0.50, 1, 1), "loo_t050": ("soft", 0.45, 1, 1),
        "loo_min200": ("soft", 0.50, 0, 1), "loo_demote": ("soft", 0.50, 1, 0),
        "only_soft": ("soft", 0.45, 0, 0), "only_t050": ("base", 0.50, 0, 0),
        "only_min200": ("base", 0.45, 1, 0), "only_demote": ("base", 0.45, 0, 1)}


def arrays(res):
    X, y, w, fi, fold, sc, area = [], [], [], [], [], [], []
    for f, r in enumerate(res):  # same target/weight as v83_rule.load
        R = len(r["n_gt"])
        for c, row in enumerate(r["rows"]):
            X.append([row[k] for k in V.FEATS])
            y.append(min(1.0, sum(float(m[:, c][m[:, c] > 0.5].sum()) for m in r["iou"] if m.size) / R))  # >1 only if a candidate matches 2 overlapping GTs of one record
            w.append(R)
            fi.append(f)
            fold.append(r["fold"])
            sc.append(r["scores"][c])
            area.append(row["area"])
    return dict(X=np.asarray(X, float), y=np.asarray(y), w=np.asarray(w, float), fi=np.asarray(fi),
                fold=np.asarray(fold), sc=np.asarray(sc), area=np.asarray(area))


def crossfit(A):
    """Outer OOF predictions + inner (fold-k-free) predictions, as v83_rule.main."""
    X, y, w, fold = A["X"], A["y"], A["w"], A["fold"]
    yhat, inner = np.zeros(len(y)), {}
    for k in range(5):
        tr = fold != k
        yhat[fold == k] = V.fit(X[tr], y[tr], w[tr]).predict(X[fold == k])
        yin = np.zeros(len(y))
        for j in range(5):
            if j != k:
                t2 = (fold != k) & (fold != j)
                yin[fold == j] = V.fit(X[t2], y[t2], w[t2]).predict(X[fold == j])
        inner[k] = yin
    return yhat, inner


def lam_fix(res, base_keep, yh, fi, files):
    rows = files[fi]
    lam = V.pq(V.file_counts(res, base_keep & rows, fi)[files].sum(0))
    for _ in range(50):
        new = V.pq(V.file_counts(res, base_keep & (yh > lam / 2) & rows, fi)[files].sum(0))
        if abs(new - lam) < 1e-6:
            break
        lam = new
    return lam


def run_arm(res, A, pred, min200, demote):
    keep = A["sc"] >= CUT
    if min200:
        keep &= A["area"] >= MINA
    lams = []
    if demote:
        yhat, inner = pred
        ffold = np.asarray([r["fold"] for r in res])
        new = keep.copy()
        for k in range(5):
            lam = lam_fix(res, keep, inner[k], A["fi"], ffold != k)
            lams.append(lam)
            te = A["fold"] == k
            new[te] = keep[te] & (yhat[te] > lam / 2)
        keep = new
    return V.file_counts(res, keep, A["fi"]), keep, lams


def main():
    data = {}
    for fam in ("base", "soft"):
        for t, res in pickle.load(open(OUT / f"extract_{fam}.pkl", "rb")).items():
            data[(fam, round(t, 2))] = res
    names = [r["name"] for r in data[("base", 0.45)]]
    assert all([r["name"] for r in v] == names for v in data.values())
    ffold = np.asarray([r["fold"] for r in data[("base", 0.45)]])
    A = {k: arrays(v) for k, v in data.items()}
    preds = {k: crossfit(A[k]) for k in A}
    print("crossfit done", flush=True)
    counts, out = {}, {"n_files": len(names), "arms": {}}
    for name, (fam, t, m2, dm) in ARMS.items():
        k = (fam, t)
        c, keep, lams = run_arm(data[k], A[k], preds[k], m2, dm)
        counts[name] = c
        tt = c.sum(0)
        out["arms"][name] = dict(refiner=fam, t=t, min200=m2, demote=dm, pq=V.pq(tt), sq=float(tt[0] / max(tt[1], 1)),
                                 tp=int(tt[1]), fp=int(tt[2]), fn=int(tt[3]), kept=int(keep.sum()), lambdas=lams)
    b = counts["baseline"]
    assert len(names) < 707 or abs(V.pq(b.sum(0)) - 0.45393) < 5e-5, V.pq(b.sum(0))
    for name, c in counts.items():
        if name == "baseline":
            continue
        a = out["arms"][name]
        a["boot"] = V.boot(b, c, seed=84)
        a["fold_delta"] = [V.pq(c[ffold == k].sum(0)) - V.pq(b[ffold == k].sum(0)) for k in range(5)]
        print(name, round(a["pq"], 5), json.dumps(a["boot"]), np.round(a["fold_delta"], 4).tolist(), flush=True)
    a = out["arms"]["bundle"]
    npos = sum(d > 0 for d in a["fold_delta"])
    out["verdict"] = ("CONTINUE" if a["boot"]["delta"] >= GATE["delta"] and a["boot"]["p_gt_0"] >= GATE["p"]
                      and npos >= GATE["folds"] else "KEEP CURRENT (+MIN200)")
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print("baseline", out["arms"]["baseline"]["pq"], "verdict", out["verdict"])


if __name__ == "__main__":
    main()
