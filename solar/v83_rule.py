"""V83 stage 2 (CPU): cross-fitted q*s regressor + Dinkelbach keep rule on the refined output.
  python scripts/v83_rule.py   (reads outputs/v83/extract.pkl, writes outputs/v83/result.json)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
import lightgbm as lgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "outputs/v83"
CUT = 0.30
FEATS = ["score", "n_views", "mean_member", "max_member", "min_member", "rank",
         "p_in", "p_max", "p_seed", "margin", "fallback", "seed_area", "ratio_pre_seed", "retain",
         "iou_seed_post", "log_area", "elong", "width", "rdist", "contrast", "dark",
         "disk_mean", "disk_std", "lowfreq_cv", "n_nbr", "nbr_max_score", "n_file_030", "n_file_cand"]
PARAMS = dict(objective="cross_entropy", num_leaves=31, learning_rate=0.05, min_data_in_leaf=50,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, seed=83, num_threads=8,
              verbose=-1)
GATE = dict(delta=0.010, p=0.90, folds=4, kill=0.005)


def pq(t):
    d = t[1] + 0.5 * (t[2] + t[3])
    return float(t[0] / d) if d else 0.0


def load():
    res = pickle.load(open(OUT / (sys.argv[1] if len(sys.argv) > 1 else "extract.pkl"), "rb"))
    X, y, w, fi, fold, sc = [], [], [], [], [], []
    for f, r in enumerate(res):
        R = len(r["n_gt"])
        for c, row in enumerate(r["rows"]):
            X.append([row[k] for k in FEATS])
            s = 0.0
            for m in r["iou"]:
                col = m[:, c] if m.size else np.zeros(0)
                s += float(col[col > 0.5].sum())
            y.append(s / R)
            w.append(R)
            fi.append(f)
            fold.append(r["fold"])
            sc.append(r["scores"][c])
    return res, *(np.asarray(a, float) for a in (X, y, w)), np.asarray(fi), np.asarray(fold), np.asarray(sc)


def file_counts(res, keep, fi):
    """Exact per-file (iou, tp, fp, fn) with the kept candidate columns (pq_counts semantics)."""
    out = np.zeros((len(res), 4))
    for f, r in enumerate(res):
        k = keep[fi == f]
        for m, n in zip(r["iou"], r["n_gt"]):
            sub = m[:, k] if m.size else np.zeros((n, int(k.sum())))
            if not n or not k.sum():
                out[f] += (0.0, 0, int(k.sum()), n)
                continue
            h = sub > 0.5
            out[f] += (sub[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum())
    return out


def fit(X, y, w):
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w), num_boost_round=300)


def arm_keep(arm, yhat, lam, sc):
    elig = {"full": sc >= 0, "demote": sc >= CUT, "promote": sc < CUT}[arm]
    forced = (sc >= CUT) if arm == "promote" else np.zeros_like(elig)
    return forced | (elig & (yhat > lam / 2))


def dinkelbach(res, arm, yhat, sc, fi, mask_files):
    """lambda fixed point on the files in mask_files (bool per file)."""
    rows = mask_files[fi]
    base = file_counts(res, (sc >= CUT) & rows, fi)[mask_files].sum(0)
    lam = pq(base)
    for _ in range(50):
        new = pq(file_counts(res, arm_keep(arm, yhat, lam, sc) & rows, fi)[mask_files].sum(0))
        if abs(new - lam) < 1e-6:
            break
        lam = new
    return lam


def boot(a, b, seed=83):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        d.append(pq(b[i].sum(0)) - pq(a[i].sum(0)))
    d = np.asarray(d)
    return dict(delta=pq(b.sum(0)) - pq(a.sum(0)), ci95=[float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5))], p_gt_0=float((d > 0).mean()))


def summ(c):
    t = c.sum(0)
    return dict(pq=pq(t), sq=float(t[0] / max(t[1], 1)), tp=int(t[1]), fp=int(t[2]), fn=int(t[3]))


def main():
    res, X, y, w, fi, fold, sc = load()
    ffold = np.asarray([r["fold"] for r in res])
    yhat, inner, imp = np.zeros(len(y)), {}, np.zeros(len(FEATS))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = fit(X[tr], y[tr], w[tr])
        yhat[te] = m.predict(X[te])
        imp += m.feature_importance("gain")
        yin = np.zeros(len(y))
        for j in range(5):
            if j != k:
                t2 = (fold != k) & (fold != j)
                yin[fold == j] = fit(X[t2], y[t2], w[t2]).predict(X[fold == j])
        inner[k] = yin
        print("fold", k, "done", flush=True)
    base = file_counts(res, sc >= CUT, fi)
    assert len(res) < 707 or abs(pq(base.sum(0)) - 0.4539) < 5e-4, pq(base.sum(0))
    out = dict(n_cand=int(len(y)), n_base=int((sc >= CUT).sum()), baseline=summ(base),
               spearman=dict(yhat=float(spearmanr(yhat, y)[0]), score=float(spearmanr(sc, y)[0])),
               importance=dict(sorted(zip(FEATS, (imp / imp.sum()).round(4).tolist()), key=lambda t: -t[1])),
               arms={})
    for arm in ("full", "demote", "promote"):
        keep, lams = np.zeros(len(y), bool), []
        for k in range(5):
            lam = dinkelbach(res, arm, inner[k], sc, fi, ffold != k)
            lams.append(lam)
            keep[fold == k] = arm_keep(arm, yhat, lam, sc)[fold == k]
        c = file_counts(res, keep, fi)
        out["arms"][arm] = dict(**summ(c), lambdas=lams, boot=boot(base, c),
                                promoted=int((keep & (sc < CUT)).sum()), demoted=int((~keep & (sc >= CUT)).sum()),
                                fold_delta=[pq(c[ffold == k].sum(0)) - pq(base[ffold == k].sum(0)) for k in range(5)])
        print(arm, json.dumps(out["arms"][arm]), flush=True)
    lam = dinkelbach(res, "full", y, sc, fi, np.ones(len(res), bool))
    oc = file_counts(res, arm_keep("full", y, lam, sc), fi)
    out["oracle"] = dict(**summ(oc), lam=lam, delta=pq(oc.sum(0)) - pq(base.sum(0)))
    a = out["arms"]["full"]
    folds_pos = sum(d > 0 for d in a["fold_delta"])
    d, p = a["boot"]["delta"], a["boot"]["p_gt_0"]
    out["verdict"] = ("CONTINUE" if d >= GATE["delta"] and p >= GATE["p"] and folds_pos >= GATE["folds"]
                      else "KILL" if d < GATE["kill"] else "INCONCLUSIVE")
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))


if __name__ == "__main__":
    main()
