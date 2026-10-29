"""V85 error analysis, stage 2 (CPU): decompose the V84 OOF PQ loss.
Final kept set = V84 bundle (soft t0.50, cut 0.30, MIN200, cross-fitted demote), asserted PQ 0.45898.
Reads outputs/v85_analysis/extract.pkl (v85_extract) + outputs/v84/extract_soft.pkl; writes report.json.
  .venv/bin/python solar/v85_analyze.py [extract.pkl]
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solar"))
import v84_bundle as B  # noqa: E402

OUT = ROOT / "outputs/v85_analysis"
ref = pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]
A = B.arrays(ref)
counts, keep, _ = B.run_arm(ref, A, B.crossfit(A), 1, 1)
PQ0 = B.V.pq(counts.sum(0))
assert abs(PQ0 - 0.45898) < 5e-5, PQ0
start = np.searchsorted(A["fi"], np.arange(len(ref)))
keep_by = {r["name"]: keep[start[f]:start[f] + len(r["rows"])] for f, r in enumerate(ref)}
sc_by = {r["name"]: np.asarray(r["scores"]) for r in ref}
area_by = {r["name"]: np.asarray([row["area"] for row in r["rows"]]) for r in ref}
res = pickle.load(open(OUT / (sys.argv[1] if len(sys.argv) > 1 else "extract.pkl"), "rb"))
res = [r for r in res if r["name"] in keep_by]


def pq(t):
    return B.V.pq(np.asarray(t, float))


def inter(iou, a, b):  # |A∩B| from IoU and areas
    return iou * (a + b) / (1 + iou)


def bins(x, edges):
    return int(np.searchsorted(edges, x, side="right"))


SIZE = [500, 1000, 2000, 4000, 8000]
SIZE_L = ["<=500", "500-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]
RN = [0.5, 0.7, 0.85, 0.95]
RN_L = ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">0.95"]
EL = [10, 20, 40, 80]
EL_L = ["<10", "10-20", "20-40", "40-80", ">80"]  # elongation = skeleton length / width = length^2/area

gts, fps, files = [], [], []
T = np.zeros(4)
for r in res:
    nm = r["name"]
    kc = keep_by[nm]
    kept = [r["cand"][c] for c in np.flatnonzero(kc)]  # alive indices kept
    cand_set = {a: c for c, a in enumerate(r["cand"])}
    R = len(r["n_gt"])
    pa = r["a_area"]
    ft = np.zeros(4)
    # per kept pred: does it match (IoU>0.5) a GT in each record
    match_rec = np.zeros((R, len(kept)), bool)
    for ri in range(R):
        if r["n_gt"][ri] and kept:
            match_rec[ri] = (r["I_post"][ri][:, kept] > 0.5).any(0)
    for ri in range(R):
        n = r["n_gt"][ri]
        Ik = r["I_post"][ri][:, kept] if n else np.zeros((0, len(kept)))
        h = Ik > 0.5
        c = np.array([Ik[h].sum(), h.sum(), (~h.any(0)).sum() if len(kept) else 0, (~h.any(1)).sum() if n else 0])
        if not n:
            c = np.array([0, 0, len(kept), 0])
        ft += c
        ag = np.asarray([g["area"] for g in r["gt"][ri]], float)
        IN = inter(Ik, ag[:, None], pa[kept][None]) if n and kept else np.zeros((n, len(kept)))
        # ---- GT rows
        for j in range(n):
            g = dict(r["gt"][ri][j], file=nm, rec=ri, n_rec=R, ann=r["ids"][ri].split("-")[0])
            g["elong"] = g["length"] / max(g["width"], 1e-6)
            best = float(Ik[j].max()) if kept else 0.0
            g["tp"] = best > 0.5
            g["iou"] = best
            if not g["tp"]:
                post = r["I_post"][ri][j]
                alive_hit = [a for a in range(len(post)) if post[a] > 0.5]
                cat = None
                for a in alive_hit:  # a TP-quality refined mask exists but was not kept
                    c_ = cand_set.get(a)
                    s = r["g_score"][r["alive"][a]]
                    if s < 0.30:
                        cat = cat or "drop_cut<0.30"
                    elif c_ is not None and area_by[nm][c_] < 200:
                        cat = cat or "drop_min200"
                    elif c_ is not None:
                        cat = cat or "drop_demote"
                g["drop_iou"] = max([post[a] for a in alive_hit], default=0.0)
                g["drop_alive"] = alive_hit
                if cat is None and len(r["I_pre"][ri][j]) and r["I_pre"][ri][j].max() > 0.5:
                    cat = "drop_ownership"
                if cat is None and len(r["I_rep"][ri][j]) and r["I_rep"][ri][j].max() > 0.5:
                    cat = "drop_refiner(seed ok)"
                if cat is None and len(r["I_raw"][ri][j]) and r["I_raw"][ri][j].max() > 0.5:
                    cat = "drop_consensus(view ok)"
                if cat is None and best >= 0.3:
                    cat = "nearmiss_kept_0.3-0.5"
                anym = max([float(m[j].max()) for m in (r["I_post"][ri], r["I_pre"][ri], r["I_rep"][ri], r["I_raw"][ri])
                            if m.shape[1]] + [0.0])
                if cat is None and anym >= 0.3:
                    cat = "nearmiss_pool_0.3-0.5"
                if cat is None and anym >= 0.05:
                    cat = "weak_0.05-0.3"
                g["cat"] = cat or "no_candidate"
                g["any_iou"] = anym
                # split: >=2 kept preds mostly inside this GT, each covering >=10%
                frag = [k for k in range(len(kept)) if IN[j, k] / pa[kept[k]] >= 0.5 and IN[j, k] / ag[j] >= 0.1]
                g["split"] = len(frag) >= 2
                # swallowed: a kept pred covers >=50% of it and also covers >=30% of another GT
                g["merged"] = any(IN[j, k] / ag[j] >= 0.5 and any(IN[o, k] / ag[o] >= 0.3 for o in range(n) if o != j)
                                  for k in range(len(kept)))
            gts.append(g)
        # ---- FP rows (kept preds unmatched in this record)
        for k in range(len(kept)):
            if n and h[:, k].any():
                continue
            f = dict(file=nm, k=k, rec=ri, n_rec=R, area=float(pa[kept[k]]), score=float(r["g_score"][r["alive"][kept[k]]]))
            f["other_rec"] = bool(match_rec[np.arange(R) != ri, k].any()) if R > 1 else False
            f["any_rec"] = f["other_rec"]
            col = Ik[:, k] if n else np.zeros(0)
            cov = IN[:, k] / ag if n else np.zeros(0)
            ins = IN[:, k] / pa[kept[k]] if n else np.zeros(0)
            if n and col.max() >= 0.3:
                cat = "nearmiss_0.3-0.5"
            elif n and (cov >= 0.3).sum() >= 2:
                cat = "merge(>=2 GT)"
            elif n and ins.max() >= 0.5:
                cat = "fragment(inside GT)"
            elif n and col.max() > 0.05:
                cat = "weak_overlap"
            else:
                cat = "no_overlap"
            f["cat"] = cat
            f["frag_of_matched"] = bool(cat == "fragment(inside GT)" and (Ik[int(ins.argmax())] > 0.5).any())
            fps.append(f)
    T += ft
    files.append(dict(name=nm, n_rec=R, c=ft, n_gt=sum(r["n_gt"]), fold=r["fold"],
                      anns=[i.split("-")[0] for i in r["ids"]]))

assert abs(pq(T) - PQ0) < 5e-5 or len(res) < 707, (pq(T), PQ0)
S, TP, FP, FN = T
SQ = S / TP
out = dict(pq=pq(T), sq=SQ, tp=int(TP), fp=int(FP), fn=int(FN), n_files=len(res), n_records=int(sum(f["n_rec"] for f in files)))


def fn_gain(rows, iou=None):  # convert FN rows to TP at their recoverable IoU (or SQ), holding all else
    s = sum((g.get(iou) if iou and g.get(iou, 0) > 0.5 else SQ) for g in rows)
    return pq((S + s, TP + len(rows), FP, FN - len(rows))) - pq(T)


def fp_gain(n):
    return pq((S, TP, FP - n, FN)) - pq(T)


fn_rows = [g for g in gts if not g["tp"]]
tp_rows = [g for g in gts if g["tp"]]
out["fn_cat"] = {k: dict(n=v, share=v / FN, pq_if_fixed=fn_gain([g for g in fn_rows if g["cat"] == k], "drop_iou"))
                 for k, v in Counter(g["cat"] for g in fn_rows).most_common()}


def table(rows_all, key, edges, labels):
    d = {}
    for i, lab in enumerate(labels):
        sel = [g for g in rows_all if bins(g[key], edges) == i]
        fn = [g for g in sel if not g["tp"]]
        tp = [g for g in sel if g["tp"]]
        d[lab] = dict(n_gt=len(sel), recall=len(tp) / max(len(sel), 1), fn=len(fn), fn_share=len(fn) / FN,
                      tp_iou=float(np.mean([g["iou"] for g in tp])) if tp else None, pq_if_all_fn_fixed=fn_gain(fn))
    return d


multi = [g for g in gts if g["n_rec"] > 1]
for g in gts:
    g["agree_frac"] = (g["agree"] + 1) / g["n_rec"]
out["fn_by_agree_multi"] = {}
for nr in (2, 3, 4):
    for k in range(nr):
        sel = [g for g in multi if g["n_rec"] == nr and g["agree"] == k]
        if sel:
            fn = [g for g in sel if not g["tp"]]
            out["fn_by_agree_multi"][f"{k + 1}of{nr}"] = dict(n_gt=len(sel), recall=1 - len(fn) / len(sel), fn=len(fn),
                                                              fn_share=len(fn) / FN, pq_if_all_fn_fixed=fn_gain(fn))
out["fn_by_size"] = table(gts, "area", SIZE, SIZE_L)
out["fn_by_rnorm"] = table(gts, "rnorm", RN, RN_L)
out["fn_by_elong"] = table(gts, "elong", EL, EL_L)
q = np.quantile([g["contrast"] for g in gts], [0.2, 0.4, 0.6, 0.8]).tolist()
out["fn_by_contrast_quintile"] = table(gts, "contrast", q, ["q1(low)", "q2", "q3", "q4", "q5(high)"])
out["fn_by_nrec"] = {k: dict(n_gt=len(s), recall=1 - sum(not g["tp"] for g in s) / len(s))
                     for k in (1, 2, 3, 4) for s in [[g for g in gts if g["n_rec"] == k]] if s}
# cat x agreement (single-annotator share per cat, multi-record files)
out["fn_cat_single_annot_share_multi"] = {
    c: float(np.mean([g["agree"] == 0 for g in multi if not g["tp"] and g["cat"] == c]))
    for c in out["fn_cat"] if any(not g["tp"] and g["cat"] == c for g in multi)}

# ---- FP
out["fp_cat"] = {}
for k, v in Counter(f["cat"] for f in fps).most_common():
    sel = [f for f in fps if f["cat"] == k]
    out["fp_cat"][k] = dict(n=v, share=v / FP, other_rec_match=sum(f["other_rec"] for f in sel),
                            frag_of_matched=sum(f["frag_of_matched"] for f in sel),
                            pq_if_removed_this_record_only=fp_gain(v),
                            pq_if_removed_no_other_rec=fp_gain(sum(not f["other_rec"] for f in sel)))
out["fp_other_rec"] = dict(n=sum(f["other_rec"] for f in fps), share=sum(f["other_rec"] for f in fps) / FP)
out["fp_by_size"] = {lab: dict(n=c, share=c / FP) for i, lab in enumerate(SIZE_L)
                     for c in [sum(bins(f["area"], SIZE) == i for f in fps)]}
# exact: drop preds that match no GT in any record (pure hallucinations, removed from every record)
# per pred: matched in no record of its file
pred_rec = defaultdict(list)
for f in fps:
    pred_rec[(f["file"], f["k"])].append(f)
nrec = {f["name"]: f["n_rec"] for f in files}
pure = [v for k, v in pred_rec.items() if len(v) == nrec[k[0]]]
out["fp_pure_preds"] = dict(n_preds=len(pure), fp_rows=sum(len(v) for v in pure),
                            pq_if_removed=fp_gain(sum(len(v) for v in pure)),
                            by_cat=dict(Counter(v[0]["cat"] for v in pure)),
                            by_size={lab: sum(bins(v[0]["area"], SIZE) == i for v in pure) for i, lab in enumerate(SIZE_L)})

# ---- SQ
out["sq_by_size"] = {lab: dict(n_tp=len(s), mean_iou=float(np.mean([g["iou"] for g in s])) if s else None)
                     for i, lab in enumerate(SIZE_L) for s in [[g for g in tp_rows if bins(g["area"], SIZE) == i]]}
out["sq_plus_005"] = pq((S + 0.05 * TP, TP, FP, FN)) - pq(T)
out["tp_iou_hist"] = np.histogram([g["iou"] for g in tp_rows], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0].tolist()
out["tp_iou_050_055"] = dict(n=sum(g["iou"] < 0.55 for g in tp_rows))

# ---- split / merge (final kept set)
spl = [g for g in fn_rows if g["split"]]
mrg = [g for g in fn_rows if g["merged"]]
n_merge_fp = sum(f["cat"] == "merge(>=2 GT)" for f in fps)
out["split_merge"] = dict(
    gt_split_fn=len(spl), gt_swallowed_fn=len(mrg), merge_fp=n_merge_fp,
    frag_fp=out["fp_cat"].get("fragment(inside GT)", {}).get("n", 0),
    pq_perfect_split_fix=pq((S + SQ * len(spl), TP + len(spl), FP - sum(1 for f in fps if f["cat"] == "fragment(inside GT)" and not f["frag_of_matched"]), FN - len(spl))) - pq(T),
    pq_perfect_merge_fix=pq((S + SQ * len(mrg), TP + len(mrg), FP - n_merge_fp, FN - len(mrg))) - pq(T))

# ---- per file / records
fpq = np.array([pq(f["c"]) if f["c"][1:].sum() else np.nan for f in files])
C = np.array([f["c"] for f in files])
out["per_file"] = dict(quantiles={str(p): float(np.nanquantile(fpq, p)) for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
                       n_zero=int(np.sum(fpq == 0)), mean_file_pq=float(np.nanmean(fpq)))
# loss concentration: gain from making file perfect (FP=FN=0, S=TP+FN at IoU SQ)
gain = []
for i, f in enumerate(files):
    c = f["c"].copy()
    n = c[1] + c[3]
    t2 = T - c + np.array([c[0] + SQ * c[3], n, 0, 0])
    gain.append(pq(t2) - pq(T))
gain = np.array(gain)
o = np.argsort(-gain)
cs = np.cumsum(gain[o]) / gain.sum()
out["per_file"]["loss_share_worst"] = {f"{p}%": float(cs[int(len(o) * p / 100) - 1]) for p in (5, 10, 20, 50)}
out["by_nrec"] = {}
for k in (1, 2, 3, 4):
    m = np.array([f["n_rec"] == k for f in files])
    if m.any():
        t = C[m].sum(0)
        out["by_nrec"][k] = dict(files=int(m.sum()), records=int(m.sum() * k), pq=pq(t), tp=int(t[1]), fp=int(t[2]), fn=int(t[3]),
                                 fp_per_rec=float(t[2] / m.sum() / k), fn_per_rec=float(t[3] / m.sum() / k))
# per annotator code
ann = defaultdict(lambda: np.zeros(4))
rec_c = defaultdict(lambda: np.zeros(4))
for f in fps:
    rec_c[(f["file"], f["rec"])][2] += 1
for g in gts:
    k = (g["file"], g["rec"])
    if g["tp"]:
        rec_c[k][0] += g["iou"]; rec_c[k][1] += 1
    else:
        rec_c[k][3] += 1
annof = {}
for g in gts:
    annof[(g["file"], g["rec"])] = g["ann"]
for f in files:
    for ri, a in enumerate(f["anns"]):
        annof.setdefault((f["name"], ri), a)
for k, c in rec_c.items():
    ann[annof[k]] += c
ann_tab = {a: dict(records=sum(1 for k in rec_c if annof[k] == a), pq=pq(c), gt_per_rec=float((c[1] + c[3]) / max(1, sum(1 for k in rec_c if annof[k] == a))),
                   fp=int(c[2]), fn=int(c[3]), loss_if_avg=None) for a, c in ann.items()}
for a, c in ann.items():  # PQ if this annotator's records scored at the global PQ (i.e. remove their excess loss)
    ann_tab[a]["pq_without"] = pq(T - c)
out["by_annotator"] = dict(sorted(ann_tab.items(), key=lambda kv: -kv[1]["records"]))

# ---- public LB resampling (90 files)
rng = np.random.default_rng(85)
for n in (60, 90, 120):
    draws = np.array([pq(C[rng.choice(len(C), n, replace=False)].sum(0)) for _ in range(20000)])
    out[f"resample_{n}"] = dict(mean=float(draws.mean()), std=float(draws.std()),
                                p05=float(np.quantile(draws, 0.05)), p95=float(np.quantile(draws, 0.95)),
                                p_le_0395=float((draws <= 0.395).mean()), p_ge_041=float((draws >= 0.41).mean()))
# record-mix sensitivity: PQ if test record-count mix is p1/p2/p3 (files), using per-n_rec pooled counts per file
avg = {k: C[[f["n_rec"] == k for f in files]].mean(0) for k in (1, 2, 3) if any(f["n_rec"] == k for f in files)}
mix = {}
for w in [(1, 0, 0), (0.58, 0.21, 0.21), (0.5, 0.25, 0.25), (0.8, 0.1, 0.1), (0.3, 0.35, 0.35), (0, 0, 1), (0, 1, 0)]:
    t = sum(wi * avg[k] for wi, k in zip(w, (1, 2, 3)))
    mix[str(w)] = pq(t)
out["record_mix_sensitivity"] = mix
out["record_mix_train"] = dict(Counter(f["n_rec"] for f in files))

(OUT / "report.json").write_text(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
pickle.dump(dict(gts=gts, fps=fps, files=files), open(OUT / "rows.pkl", "wb"))
print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
