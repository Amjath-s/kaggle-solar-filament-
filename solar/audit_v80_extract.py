"""V80 forensics, stage 1 (CPU only): instrumented real pipeline on all 707 OOF files.

p2first OOF 4-flip views -> flip_consensus groups (match 0.5, score = sum/4) -> V73 `base` fold
refiner (leak-free, k7 clamp, t0.45) with per-proposal pre/post-ownership masks -> cut 0.30.
Per GT instance (per annotator record) records the IoU at every stage + geometry/contrast;
per refined prediction records its delta to pooled counts (for cut sweeps).
Asserts per-file counts vs outputs/v73/base_fold*.json (k7, t0.45_c0.30).
  CUDA_VISIBLE_DEVICES= python scripts/audit_v80_extract.py [limit]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v73_refiner as V73  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v80_forensics"
T, KS, CUT = 0.45, 7, 0.30
RT, MODELS = None, {}


def groups_with_ids(views):
    """detector_tta.cluster_views (flip_consensus, 0.5) keeping raw member ids."""
    from pycocotools import mask as mu
    props = [(r, float(s), v, (v, k)) for v, (rl, sl) in enumerate(views)
             for k, (r, s) in enumerate(zip(rl, sl))]
    props.sort(key=lambda x: -x[1])
    groups = []
    for rle, s, v, rid in props:
        elig = [i for i, g in enumerate(groups) if v not in g["views"]]
        match = None
        if elig:
            ov = mu.iou([rle], [groups[i]["rle"] for i in elig], [0] * len(elig))[0]
            b = int(np.argmax(ov))
            if ov[b] >= 0.5:
                match = elig[b]
        if match is None:
            groups.append(dict(rle=rle, scores=[s], views={v}, ids=[rid]))
        else:
            groups[match]["scores"].append(s)
            groups[match]["views"].add(v)
            groups[match]["ids"].append(rid)
    for g in groups:
        g["score"] = sum(g["scores"]) / len(views)
    order = sorted(range(len(groups)), key=lambda i: -groups[i]["score"])
    return [groups[i] for i in order]


def refine_instrumented(masks, scores, bounds, probs):
    """V73.clean_refined(t, k7) but returns pre-ownership and post-ownership masks per proposal."""
    import cv2
    core = RT["core"]
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KS, KS))
    pre, post = [None] * len(masks), [None] * len(masks)
    for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
        x0, y0, x1, y1 = bounds[i]
        proposal = masks[i][y0:y1, x0:x1]
        crop = cv2.resize(probs[i], (x1 - x0, y1 - y0)) >= T
        crop &= cv2.dilate(proposal, kernel) > 0
        if not crop.any():
            crop = proposal.astype(bool)
        pre[i] = core.encode_crop(crop, bounds[i], occupied.shape)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < core.MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        post[i] = core.encode_crop(crop, bounds[i], occupied.shape)
    return pre, post


def iou_mat(a, b):
    from pycocotools import mask as mu
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))


def disk_fit(gray):
    import cv2
    b = cv2.GaussianBlur(gray, (0, 0), 4)
    _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(th)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    cx, cy = cen[k]
    r = float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi))
    disk = lab == k
    ys, xs = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    inner = disk & ((xs - cx) ** 2 + (ys - cy) ** 2 < (0.9 * r) ** 2)
    vals = gray[inner].astype(np.float32)
    big = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 40)[inner]
    return dict(cx=float(cx), cy=float(cy), r=r, disk_mean=float(vals.mean()),
                disk_std=float(vals.std()), disk_p5=float(np.percentile(vals, 5)),
                lowfreq_cv=float(big.std() / max(big.mean(), 1)), disk_frac=float(disk.mean()))


def gt_feats(m, gray, disk):
    import cv2
    from skimage.morphology import skeletonize
    ys, xs = np.where(m)
    y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 25, m.shape[0])
    x0, x1 = max(xs.min() - 24, 0), min(xs.max() + 25, m.shape[1])
    g, im = m[y0:y1, x0:x1].astype(np.uint8), gray[y0:y1, x0:x1].astype(np.float32)
    sk = int(skeletonize(g > 0).sum())
    ring = (cv2.dilate(g, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(g, np.ones((7, 7), np.uint8)) > 0)
    fg, bg = float(im[g > 0].mean()), float(im[ring].mean()) if ring.any() else float("nan")
    cy, cx = ys.mean(), xs.mean()
    return dict(area=int(m.sum()), length=sk, width=float(m.sum() / max(sk, 1)),
                contrast=(bg - fg) / max(bg, 1), fg=fg, bg=bg,
                rnorm=float(np.hypot(cx - disk["cx"], cy - disk["cy"]) / disk["r"]))


def init():
    import torch
    torch.set_num_threads(1)


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = RT, RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    raw = {(v, k): r for v, (rl, _) in enumerate(views) for k, r in enumerate(rl)}
    raw_ids = list(raw)
    raw_sc = {(v, k): s for v, (_, sl) in enumerate(views) for k, s in enumerate(sl)}
    groups = groups_with_ids(views)
    props = [g["rle"] for g in groups]
    scores = [g["score"] for g in groups]
    gid_of = {rid: gi for gi, g in enumerate(groups) for rid in g["ids"]}
    if props:
        masks = [mu.decode(r) for r in props]
        bounds, _, probs = core.crop_probabilities(rt["base"], MODELS[fold], gray, masks, True)
        pre, post = refine_instrumented(masks, scores, bounds, probs)
    else:
        pre, post = [], []
    alive = [i for i in range(len(groups)) if post[i] is not None]
    post_l = [post[i] for i in alive]
    keep = [i for i in alive if scores[i] >= CUT]
    keep_l = [post[i] for i in keep]
    disk = disk_fit(gray)
    # GT agreement (other records with an IoU>0.5 GT)
    R = len(recs)
    cross = {(a, b): iou_mat(recs[a], recs[b]) for a in range(R) for b in range(R) if a != b}
    agree = [[int(sum((cross[(a, b)][i] > 0.5).any() for b in range(R) if b != a))
              for i in range(len(recs[a]))] for a in range(R)]
    gt_rows, tot = [], np.zeros(4)
    pa = {i: float(mu.area(post[i])) for i in alive}
    # per refined (post-ownership) prediction: per-record delta counts if it is added
    pred_rows = [dict(gi=i, score=scores[i], n_views=len(groups[i]["views"]),
                      mean_member=float(np.mean(groups[i]["scores"])), area=pa[i],
                      iou_sum=0.0, tp=0, fp=0, fn=0, m_agreed=0, m_single=0) for i in alive]
    I_post = [iou_mat(g, post_l) for g in recs]
    for ri, g in enumerate(recs):
        ip = I_post[ri]
        for c, pr in enumerate(pred_rows):
            col = ip[:, c] if g else np.zeros(0)
            if g and col.max() > 0.5:
                j = int(col.argmax())
                pr["iou_sum"] += float(col[j]); pr["tp"] += 1; pr["fn"] -= 1
                pr["m_agreed" if agree[ri][j] else "m_single"] += 1
            else:
                pr["fp"] += 1
    kidx = [alive.index(i) for i in keep]
    I_raw = [iou_mat(g, [raw[r] for r in raw_ids]) for g in recs]
    I_rep = [iou_mat(g, props) for g in recs]
    I_pre = [iou_mat(g, pre) for g in recs]
    for ri, g in enumerate(recs):
        ik = I_post[ri][:, kidx] if kidx else np.zeros((len(g), 0))
        tot += core.pq_counts(ik, len(g), len(keep))
        matched_pred = (ik > 0.5).any(0) if g else np.zeros(len(keep), bool)
        ag = np.array([mu.area(r) for r in g], float)
        ka = np.array([pa[i] for i in keep], float)
        inter = ik * (ag[:, None] + ka[None]) / (1 + ik) if len(keep) and g else ik
        for gi in range(len(g)):
            row = dict(file=name, fold=fold, rec=ri, gi=gi, n_rec=R, agree=agree[ri][gi])
            row["final"] = float(ik[gi].max()) if len(keep) else 0.0
            row["matched"] = row["final"] > 0.5
            if not row["matched"]:
                ir = I_raw[ri][gi]
                row["raw"] = float(ir.max()) if len(ir) else 0.0
                row["raw_n50"] = int((ir > 0.5).sum())
                # group with good rep: prefer highest score; else group of best raw member
                rep = I_rep[ri][gi]
                good = [i for i in range(len(groups)) if rep[i] > 0.5]
                if good:
                    gb = max(good, key=lambda i: scores[i])
                elif row["raw"] > 0.5:
                    gb = gid_of[raw_ids[int(ir.argmax())]]
                else:
                    gb = None
                if gb is not None:
                    G = groups[gb]
                    row.update(g_rep=float(rep[gb]), g_score=G["score"], g_views=len(G["views"]),
                               g_mean=float(np.mean(G["scores"])), g_max=float(max(G["scores"])),
                               g_pre=float(I_pre[ri][gi][gb]),
                               g_post=float(I_post[ri][gi][alive.index(gb)]) if gb in alive else -1.0,
                               g_best_raw_sc=float(raw_sc[raw_ids[int(ir.argmax())]]))
                # any refined (post) mask >0.5 incl. below cut
                row["post_any"] = float(I_post[ri][gi].max()) if alive else 0.0
                row["pre_any"] = float(I_pre[ri][gi].max()) if pre else 0.0
                # fragment / merge on final preds
                if len(keep):
                    gm = mu.decode(g[gi]).astype(bool)
                    cov = inter[gi] / ag[gi]
                    inside = inter[gi] / ka
                    frag = [j for j in np.argsort(-inter[gi])[:3] if inside[j] >= 0.5 and inter[gi][j] > 0]
                    best_u, u = 0.0, np.zeros_like(gm)
                    for j in frag:
                        u |= mu.decode(keep_l[j]).astype(bool)
                        best_u = max(best_u, float((u & gm).sum() / (u | gm).sum()))
                    row["frag_n"], row["frag_union"] = len(frag), best_u
                    big = [j for j in range(len(keep)) if cov[j] >= 0.5 and ik[gi, j] <= 0.5]
                    others = [o for o in range(len(g)) if o != gi]
                    row["merge_nbr"] = any(any(inter[o, j] / ag[o] >= 0.3 for o in others) for j in big)
                    row["oversize"] = len(big) > 0
                    row["absorb_fp"] = int(sum(1 for j in frag if not matched_pred[j]))
                    jb = int(ik[gi].argmax())
                    row["best_unmatched"] = bool(not matched_pred[jb]) and ik[gi, jb] >= 0.3
            row.update(gt_feats(mu.decode(g[gi]).astype(bool), gray, disk))
            gt_rows.append(row)
    return dict(name=name, fold=fold, tot=tot.tolist(), gt=gt_rows, preds=pred_rows, disk=disk,
                n_raw=len(raw_ids), n_groups=len(groups))


def main():
    global RT
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v73/base_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    if len(sys.argv) > 1:
        names = names[:: max(1, len(names) // int(sys.argv[1]))][: int(sys.argv[1])]
    ref = {}
    for f in range(5):
        ref.update(json.loads((ROOT / f"outputs/v73/base_fold{f}.json").read_text())["per_file"])
    with torch.no_grad(), Pool(16, initializer=init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    diffs = [max(abs(a - b) for a, b in zip(r["tot"], ref[r["name"]]["k7"]["t0.45_c0.30"])) for r in res]
    tot = np.sum([r["tot"] for r in res], 0)
    reft = np.sum([ref[r["name"]]["k7"]["t0.45_c0.30"] for r in res], 0)
    print("files", len(res), "max file diff", max(diffs), "n_files_diff", sum(d > 1e-6 for d in diffs),
          "pq", V73.R.K.pq_value(tot), "ref pq", V73.R.K.pq_value(reft), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / ("extract.pkl" if len(sys.argv) == 1 else "extract_smoke.pkl"), "wb"))


if __name__ == "__main__":
    main()
