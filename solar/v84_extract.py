"""V84 stage 1 (CPU): V83 extract (real refined pipeline + per-candidate features + per-record IoU) for one
refiner family, saving t0.45 and t0.50 from the same crop probabilities.
Asserts per-file cut-0.30 counts vs the family's own fold jsons (V73 base: outputs/v73/base_fold*.json;
soft: outputs/v84/soft_fold*.json).
  CUDA_VISIBLE_DEVICES= python scripts/v84_extract.py base|soft [limit]
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
import v83_extract as X  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v84"
TS = (0.45, 0.50)
CKPT = {"base": "outputs/v73/base_fold{}", "soft": "outputs/v84/soft_fold{}"}


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
    out = {}
    if groups:
        masks = [mu.decode(g["rle"]) for g in groups]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
    for t in TS:
        X.T = t  # refine_feats reads the module-level threshold
        post, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk) if groups else ([], [])
        alive = [i for i in range(len(groups)) if post[i] is not None]
        keep = [c for c, i in enumerate(alive) if scores[i] >= X.CUT]
        I_all = [E.iou_mat(g, [post[i] for i in alive]) for g in recs]
        tot = np.zeros(4)
        for ri, g in enumerate(recs):
            tot += core.pq_counts(I_all[ri][:, keep] if g else np.zeros((0, len(keep))), len(g), len(keep))
        cand, rows = X.candidate_rows(groups, scores, alive, feats)
        for c, f in zip(cand, rows):
            f["area"] = float(mu.area(post[alive[c]]))
        iou = [I_all[ri][:, cand] if g else np.zeros((0, len(cand))) for ri, g in enumerate(recs)]
        out[t] = dict(name=name, fold=fold, tot=tot.tolist(), rows=rows, iou=iou,
                      n_gt=[len(g) for g in recs], scores=[scores[alive[c]] for c in cand])
    return out


def main():
    import torch
    fam = sys.argv[1]
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / (CKPT[fam].format(f) + "_refiner.pt"), map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    fmap = V73.R.fold_of_file()
    names = sorted(fmap)
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if lim:
        names = names[:: max(1, len(names) // lim)][:lim]
    ref = {}
    for f in range(5):
        ref.update(json.loads((ROOT / (CKPT[fam].format(f) + ".json")).read_text())["per_file"])
    with torch.no_grad(), Pool(22, initializer=E.init) as p:
        res = p.map(work, [(n, fmap[n]) for n in names], chunksize=1)
    by_t = {t: [r[t] for r in res] for t in TS}
    for t, rs in by_t.items():
        key = f"t{t:.2f}_c0.30"
        diffs = [max(abs(a - b) for a, b in zip(r["tot"], ref[r["name"]]["k7"][key])) for r in rs]
        tot = np.sum([r["tot"] for r in rs], 0)
        print(fam, key, "files", len(rs), "max file diff", max(diffs), "pq", V73.R.K.pq_value(tot),
              "cands", sum(len(r["rows"]) for r in rs), flush=True)
        assert max(diffs) <= 1e-3, f"pipeline reproduction failed {fam} {key}"
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(by_t, open(OUT / f"extract_{fam}{'_smoke' if lim else ''}.pkl", "wb"))


if __name__ == "__main__":
    main()
