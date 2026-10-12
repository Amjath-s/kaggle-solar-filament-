"""V83 stage 1 (CPU): real refined pipeline (as audit_v80_extract) + per-candidate features and
per-record IoU matrices. Asserts the cut-0.30 counts reproduce outputs/v73/base_fold*.json.
  CUDA_VISIBLE_DEVICES= python scripts/v83_extract.py [limit]
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
import audit_v80_extract as E  # noqa: E402
import train_v73_refiner as V73  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v83"
T, KS, CUT, FLOOR = 0.45, 7, 0.30, 0.15


def refine_feats(masks, scores, bounds, probs, gray, disk):
    """E.refine_instrumented + refiner-internal / shape / photometry features per proposal."""
    import cv2
    core = E.RT["core"]
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KS, KS))
    k5 = np.ones((5, 5), np.uint8)
    post, feats = [None] * len(masks), [None] * len(masks)
    for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
        x0, y0, x1, y1 = bounds[i]
        seed = masks[i][y0:y1, x0:x1]
        P = cv2.resize(probs[i], (x1 - x0, y1 - y0))
        crop = P >= T
        crop &= cv2.dilate(seed, kernel) > 0
        fallback = not crop.any()
        if fallback:
            crop = seed.astype(bool)
        a_pre = int(crop.sum())
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < core.MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        post[i] = core.encode_crop(crop, bounds[i], occupied.shape)
        c8 = crop.astype(np.uint8)
        a = int(crop.sum())
        sb = seed.astype(bool)
        inner = crop & ~(cv2.erode(c8, k5) > 0)
        outer = (cv2.dilate(c8, k5) > 0) & ~crop
        pts = np.column_stack(np.where(crop))[:, ::-1].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        L = max(w, h, 1.0)
        g = gray[y0:y1, x0:x1].astype(np.float32)
        ring = (cv2.dilate(c8, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(c8, np.ones((7, 7), np.uint8)) > 0)
        fg = float(g[crop].mean())
        bg = float(g[ring].mean()) if ring.any() else fg
        ys, xs = np.where(crop)
        feats[i] = dict(
            p_in=float(P[crop].mean()), p_max=float(P[crop].max()),
            p_seed=float(P[sb].mean()) if sb.any() else 0.0,
            margin=float(P[inner].mean() - (P[outer].mean() if outer.any() else 0.0)),
            fallback=float(fallback), seed_area=float(sb.sum()),
            ratio_pre_seed=a_pre / max(sb.sum(), 1), retain=a / max(a_pre, 1),
            iou_seed_post=float((sb & crop).sum() / max((sb | crop).sum(), 1)),
            log_area=float(np.log(a)), elong=L * L / a, width=a / L,
            rdist=float(np.hypot(xs.mean() + x0 - disk["cx"], ys.mean() + y0 - disk["cy"]) / disk["r"]),
            contrast=(bg - fg) / max(bg, 1), dark=fg / max(disk["disk_mean"], 1),
            disk_mean=disk["disk_mean"], disk_std=disk["disk_std"], lowfreq_cv=disk["lowfreq_cv"],
            bbox=(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
    return post, feats


def candidate_rows(groups, scores, alive, feats):
    """Candidates (alive indices with score >= FLOOR) and their feature rows (group, rank, neighbour,
    file features + refine_feats). Shared by v83/v84 extract and the V84 export demote rule."""
    cand = [c for c, i in enumerate(alive) if scores[i] >= FLOOR]
    rows = []
    bb = [feats[alive[c]]["bbox"] for c in cand]
    n030 = sum(scores[alive[c]] >= CUT for c in cand)
    for k, c in enumerate(cand):
        i = alive[c]
        G = groups[i]
        f = {key: v for key, v in feats[i].items() if key != "bbox"}
        x0, y0, x1, y1 = bb[k]
        nb = [m for m in range(len(cand)) if m != k and bb[m][0] < x1 + 25 and bb[m][2] > x0 - 25
              and bb[m][1] < y1 + 25 and bb[m][3] > y0 - 25]
        f.update(score=scores[i], n_views=len(G["views"]), mean_member=float(np.mean(G["scores"])),
                 max_member=float(max(G["scores"])), min_member=float(min(G["scores"])),
                 rank=float(sum(scores[alive[m]] > scores[i] for m in cand)),
                 n_nbr=float(len(nb)), nbr_max_score=max([scores[alive[m]] for m in nb], default=0.0),
                 n_file_030=float(n030), n_file_cand=float(len(cand)))
        rows.append(f)
    return cand, rows


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    groups = E.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    disk = E.disk_fit(gray)
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
        post, feats = refine_feats(masks, scores, bounds, probs, gray, disk)
    else:
        post, feats = [], []
    alive = [i for i in range(len(groups)) if post[i] is not None]
    # reproduction check at the pipeline cut (all alive incl. < FLOOR)
    keep = [c for c, i in enumerate(alive) if scores[i] >= CUT]
    I_all = [E.iou_mat(g, [post[i] for i in alive]) for g in recs]
    tot = np.zeros(4)
    for ri, g in enumerate(recs):
        tot += core.pq_counts(I_all[ri][:, keep] if g else np.zeros((0, len(keep))), len(g), len(keep))
    cand, rows = candidate_rows(groups, scores, alive, feats)
    iou = [I_all[ri][:, cand] if g else np.zeros((0, len(cand))) for ri, g in enumerate(recs)]
    return dict(name=name, fold=fold, tot=tot.tolist(), rows=rows, iou=iou,
                n_gt=[len(g) for g in recs], scores=[scores[alive[c]] for c in cand])


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v73/base_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    if len(sys.argv) > 1:
        names = names[:: max(1, len(names) // int(sys.argv[1]))][: int(sys.argv[1])]
    ref = {}
    for f in range(5):
        ref.update(json.loads((ROOT / f"outputs/v73/base_fold{f}.json").read_text())["per_file"])
    with torch.no_grad(), Pool(20, initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    diffs = [max(abs(a - b) for a, b in zip(r["tot"], ref[r["name"]]["k7"]["t0.45_c0.30"])) for r in res]
    tot = np.sum([r["tot"] for r in res], 0)
    print("files", len(res), "max file diff", max(diffs), "pq", V73.R.K.pq_value(tot),
          "cands", sum(len(r["rows"]) for r in res), flush=True)
    assert max(diffs) <= 1e-3, "pipeline reproduction failed"
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / ("extract.pkl" if len(sys.argv) == 1 else "extract_smoke.pkl"), "wb"))


if __name__ == "__main__":
    main()
