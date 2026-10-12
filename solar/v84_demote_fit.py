"""V84 export: fit the demote rule (V83 LightGBM, V84 targets clipped at 1) on all 707 soft/t0.50 OOF candidates.
lambda = Dinkelbach fixed point on the pooled 707 cross-fitted OOF predictions (MIN200 base set), as in V84.
Also writes the per-fold cross-fitted models/lambdas and the bundle reference used by the export parity check.
  python scripts/v84_demote_fit.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v83_rule as V  # noqa: E402
import v84_bundle as B  # noqa: E402

OUT = B.OUT / "export"


def save(model, lam, path, **extra):
    model.save_model(str(path.with_suffix(".txt")))
    sha = hashlib.sha256(path.with_suffix(".txt").read_bytes()).hexdigest()
    meta = dict(model=path.with_suffix(".txt").name, model_sha256=sha, **{"lambda": lam}, features=V.FEATS,
                cut=B.CUT, min_area=B.MINA, floor=0.15, mask_threshold=0.50, refiner="soft", **extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return sha


def main():
    res = pickle.load(open(B.OUT / "extract_soft.pkl", "rb"))[0.5]
    A = B.arrays(res)
    yhat, inner = B.crossfit(A)
    ffold = np.asarray([r["fold"] for r in res])
    base_keep = (A["sc"] >= B.CUT) & (A["area"] >= B.MINA)
    allf = np.ones(len(res), bool)
    lam = B.lam_fix(res, base_keep, yhat, A["fi"], allf)
    sha = save(V.fit(A["X"], A["y"], A["w"]), lam, OUT / "demote_all", n_candidates=int(len(A["y"])),
               lambda_source="Dinkelbach on pooled 707 cross-fitted OOF predictions (V84 bundle base set)")
    # parity material: fold models (same fits as B.crossfit), fold lambdas (as B.run_arm), bundle keep
    P = OUT / "parity"
    P.mkdir(parents=True, exist_ok=True)
    keep = base_keep.copy()
    for k in range(5):
        tr, te = A["fold"] != k, A["fold"] == k
        m = V.fit(A["X"][tr], A["y"][tr], A["w"][tr])
        assert np.abs(m.predict(A["X"][te]) - yhat[te]).max() < 1e-12, "LightGBM refit not deterministic"
        lk = B.lam_fix(res, base_keep, inner[k], A["fi"], ffold != k)
        save(m, lk, P / f"demote_fold{k}", fold=k)
        keep[te] = base_keep[te] & (yhat[te] > lk / 2)
    c = V.file_counts(res, keep, A["fi"])
    pqb = V.pq(c.sum(0))
    want = json.loads((B.OUT / "result.json").read_text())["arms"]["bundle"]["pq"]
    assert abs(pqb - want) < 1e-12, (pqb, want)  # reproduces the V84 bundle exactly
    ref = {r["name"]: dict(fold=r["fold"], counts=c[f].tolist(), yhat=yhat[A["fi"] == f].tolist(),
                           scores=A["sc"][A["fi"] == f].tolist(), area=A["area"][A["fi"] == f].tolist(),
                           keep=keep[A["fi"] == f].tolist()) for f, r in enumerate(res)}
    pickle.dump(ref, open(P / "bundle_ref.pkl", "wb"))
    print(json.dumps(dict(lam=lam, thresh=lam / 2, model_sha256=sha, bundle_pq=pqb,
                          kept=int(keep.sum()), base=int(base_keep.sum()))))


if __name__ == "__main__":
    main()
