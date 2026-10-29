"""V85 error analysis, stage 1 (CPU): V84 soft refiner, t0.50, full per-stage IoU per GT.
Same pipeline as v84_extract (soft family, t=0.50) plus: raw view masks, consensus reps, pre-ownership refined
masks, all alive post masks (incl. < FLOOR), GT geometry/contrast, cross-record GT agreement.
Asserts the candidate IoU columns equal outputs/v84/extract_soft.pkl[0.5].
  CUDA_VISIBLE_DEVICES= .venv/bin/python solar/v85_extract.py [limit]
"""
from __future__ import annotations

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
import v83_extract as X  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v85_analysis"
T = 0.50


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold = job
    rt, core = E.RT, E.RT["core"]
    X.T = T
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs, ids = [], []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
        ids.append(str(image_id))
    views = V73.R.K.read_cache_views(V73.R.O.cache_path(V73.RECIPE, fold, name))
    raw = [(r, float(s)) for rl, sl in views for r, s in zip(rl, sl)]
    groups = E.groups_with_ids(views)
    scores = [g["score"] for g in groups]
    disk = E.disk_fit(gray)
    pre = []
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
        post, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (X.KS, X.KS))
        for i, m in enumerate(masks):  # pre-ownership refined mask (refine_feats without the occupied clip)
            x0, y0, x1, y1 = bounds[i]
            seed = m[y0:y1, x0:x1]
            crop = (cv2.resize(probs[i], (x1 - x0, y1 - y0)) >= T) & (cv2.dilate(seed, kern) > 0)
            if not crop.any():
                crop = seed.astype(bool)
            pre.append(core.encode_crop(crop, bounds[i], m.shape))
    else:
        post, feats = [], []
    alive = [i for i in range(len(groups)) if post[i] is not None]
    cand, _ = X.candidate_rows(groups, scores, alive, feats) if alive else ([], [])
    R = len(recs)
    cross = {(a, b): E.iou_mat(recs[a], recs[b]) for a in range(R) for b in range(R) if a != b}
    gt = []
    for a, g in enumerate(recs):
        rows = []
        for i, r in enumerate(g):
            d = E.gt_feats(mu.decode(r).astype(bool), gray, disk)
            d["agree"] = int(sum((cross[(a, b)][i] > 0.5).any() for b in range(R) if b != a))
            d["best_other"] = max([float(cross[(a, b)][i].max()) for b in range(R) if b != a and len(recs[b])],
                                  default=0.0)
            rows.append(d)
        gt.append(rows)
    post_l = [post[i] for i in alive]
    return dict(
        name=name, fold=fold, ids=ids, n_gt=[len(g) for g in recs], gt=gt, disk=disk,
        g_score=np.asarray(scores), g_views=[len(g["views"]) for g in groups], alive=alive, cand=cand,
        a_area=np.asarray([float(mu.area(p)) for p in post_l]),
        raw_score=np.asarray([s for _, s in raw]),
        I_post=[E.iou_mat(g, post_l) for g in recs], I_pre=[E.iou_mat(g, pre) for g in recs],
        I_rep=[E.iou_mat(g, [x["rle"] for x in groups]) for g in recs],
        I_raw=[E.iou_mat(g, [r for r, _ in raw]) for g in recs],
        P_post=E.iou_mat(post_l, post_l) if post_l else np.zeros((0, 0)))


def main():
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v84/soft_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    with torch.no_grad(), Pool(20, initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    ref = {r["name"]: r for r in pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5]}
    worst = 0.0
    for r in res:
        q = ref[r["name"]]
        assert len(q["iou"]) == len(r["I_post"])
        for m, ip in zip(q["iou"], r["I_post"]):
            if m.size:
                worst = max(worst, float(np.abs(ip[:, r["cand"]] - m).max()))
    print("files", len(res), "max |iou - v84 extract|", worst, flush=True)
    assert worst < 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / ("extract.pkl" if not lim else "extract_smoke.pkl"), "wb"))


if __name__ == "__main__":
    main()
