"""V73: 5-fold OOF crop refiner (all 707 files) + one-change upgrade arms. Copied from V63.

Pre-registration: docs/v73_refiner_preregistration_2026-10-10.md. Never reads test, never submits.

For V61 fold K: train the refiner on p2first OOF proposals (and GT) of the files in the OTHER four
folds, then refine and score the fold-K files with p2first's fold-K OOF proposals. Pooling K=0..4
scores the full real pipeline (flip_consensus -> refiner -> refined_constrained) on 707 files.

Leakage: the canonical refiner (r3's init) was trained on the 566 canonical FIT files, so an
r3-initialised refiner has seen the labels of every non-canonical file it scores. Arms therefore
start from ImageNet and re-run the lineage per fold (stage A: GT-centred crops, V6-like; stage B:
the r3 recipe on OOF proposals). `r3c` (canonical init, exactly r3) is kept as a control; only its
141 canonical files are clean.

Arms (ONE change vs `base`):
  base     ImageNet ResNet18, 256 crops, 7x7 clamp     (leak-free r3 baseline)
  crop512  as base, 512 crops (train + inference)
  rn34     as base, ImageNet ResNet34 encoder (same channel widths, same decoder)
  dil15    base's models, 15x15 clamp at inference     (scored inside every fold job, no training)
  r3c      canonical init, stage B only = r3 exactly   (control: reproduction + leakage size)

  CUDA_VISIBLE_DEVICES=N python scripts/train_v73_refiner.py fold --arm base --fold 0
  python scripts/train_v73_refiner.py compare            # needs all arm x fold JSONs
  CUDA_VISIBLE_DEVICES= python scripts/train_v73_refiner.py parity --limit 4   # CPU harness check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "kaggle_inference"))
import eval_v62_refiner as R  # noqa: E402
import optimize_inference as core  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v63_refiner as V63  # noqa: E402  (fit_records reused unchanged)
import v11_refiner as v11  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v73"
RECIPE = "p2first"
ARMS = {  # init, encoder, crop size
    "base": ("imagenet", "resnet18", 256),
    "crop512": ("imagenet", "resnet18", 512),
    "rn34": ("imagenet", "resnet34", 256),
    "r3c": ("canonical", "resnet18", 256),
}
KERNELS = (7, 15)  # clamp: cv2 ellipse size (7 = optimize_inference's refined_constrained)
CELL = "t0.45_c0.30"
STAGE_A = (8, 3e-4)  # V6 refiner: ImageNet init, 8 epochs, lr 3e-4, GT-centred crops
STAGE_B = (12, 1e-4)  # r3: 12 epochs, lr 1e-4, OOF proposal crops
GATE = dict(delta=0.010, p=0.90, folds=4, kill=0.005)
# primary contrasts: (arm label, row, baseline row); rows are "<arm>_k<kernel>"
PRIMARY = [("crop512", "crop512_k7", "base_k7"), ("rn34", "rn34_k7", "base_k7"),
           ("dil15", "base_k15", "base_k7")]
DEV = None


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def runtime():
    """v50.runtime_state() minus its CUDA guard and model load (models are built per arm)."""
    import torch
    guard, build = torch.cuda.is_available, v11.build_model
    torch.cuda.is_available, v11.build_model = (lambda: True), (lambda *a, **k: None)
    try:
        rt = v50.runtime_state()
    finally:
        torch.cuda.is_available, v11.build_model = guard, build
    if DEV == "cpu":  # runtime_state pins 2 threads; CPU smoke/parity needs more
        torch.set_num_threads(16)
    rt["by_file"] = {}
    for image_id, info in rt["images"].items():
        rt["by_file"].setdefault(info["file_name"], []).append(image_id)
    return rt


def build_model(base, init, encoder):
    import torch
    import torchvision

    model = base.CropUNet(pretrained=False)
    if init == "canonical":
        model.load_state_dict(torch.load(v50.CANONICAL_REFINER, map_location="cpu",
                                         weights_only=True))
    else:  # ResNet18/34 share stage widths 64/128/256/512, so CropUNet's decoder is unchanged
        enc = getattr(torchvision.models, encoder)(weights="DEFAULT")
        model.stem = torch.nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        model.pool, model.layer1, model.layer2 = enc.maxpool, enc.layer1, enc.layer2
        model.layer3, model.layer4 = enc.layer3, enc.layer4
    return model.to(DEV)


def gt_records(rt, names):
    """Stage A: one GT-centred record per non-empty annotation (seed = target)."""
    from pycocotools import mask as mask_utils

    core, base, records = rt["core"], rt["base"], []
    for name in names:
        for image_id in sorted(rt["by_file"][name], key=str):
            info = rt["images"][image_id]
            for a in rt["annotations"].get(image_id, []):
                rle = core.annotation_rle(info, a)
                if mask_utils.area(rle) > 0:
                    records.append({"name": name, "seed": rle, "target": rle,
                                    "bounds": tuple(base.square_bounds(mask_utils.decode(rle)))})
    return records


class CropDataset(V63.CropDataset):
    """V63 dataset with a crop-size knob (v11.crop_triplet's size default is bound at def time)."""

    def __init__(self, records, train_dir, epoch, seed, size):
        super().__init__(records, train_dir, epoch, seed)
        self.size = size

    def __getitem__(self, index):
        import cv2
        import torch
        from pycocotools import mask as mask_utils

        r = self.records[index]
        if r["name"] not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[r["name"]] = cv2.imread(str(self.train_dir / r["name"]),
                                               cv2.IMREAD_GRAYSCALE)
        gray = self.cache[r["name"]]
        seed, target = mask_utils.decode(r["seed"]), mask_utils.decode(r["target"])
        rng = np.random.default_rng(self.seed * 7919 + self.epoch * 1_000_003 + index)
        jit = v11.jitter_bounds(r["bounds"], gray.shape, rng)
        trip = v11.crop_triplet(gray, seed, target, jit, self.size)
        if not trip[1].any() or not trip[2].any():
            trip = v11.crop_triplet(gray, seed, target, r["bounds"], self.size)
        image, _, tgt = v11.augment_triplet(*trip, rng)
        return (torch.from_numpy(np.repeat(image[None], 3, axis=0)),
                torch.from_numpy(tgt.astype(np.float32)[None]))


def train(rt, model, records, epochs, lr, seed, size, stage):
    """V63.train's loop (AdamW wd 1e-4, cosine, 0.5 BCE + 0.5 Dice, batch 16, fp32)."""
    import torch

    base = rt["base"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        ds = CropDataset(records, base.CFG.train_dir, epoch, seed, size)
        order = np.random.default_rng(seed + epoch).permutation(len(records)).tolist()
        loader = torch.utils.data.DataLoader(ds, batch_size=v11.BATCH_SIZE, sampler=order,
                                             num_workers=8, pin_memory=DEV == "cuda")
        model.train()
        total = n = 0
        t0 = time.monotonic()
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                    + 0.5 * base.soft_dice_loss(logits, y))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss {stage} epoch {epoch}")
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
        sched.step()
        history.append({"stage": stage, "epoch": epoch, "mean_loss": total / n,
                        "seconds": time.monotonic() - t0})
        if DEV == "cuda":
            history[-1]["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps(history[-1]), flush=True)
    return history


def clean_refined(masks, scores, bounds, probs, threshold, ksize):
    """optimize_inference.clean_refined_predictions(mode=refined_constrained), clamp kernel knob."""
    import cv2

    if not masks:
        return [], np.asarray([])
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    rles, kept = [], []
    for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
        x0, y0, x1, y1 = bounds[i]
        proposal = masks[i][y0:y1, x0:x1]
        crop = cv2.resize(probs[i], (x1 - x0, y1 - y0)) >= threshold
        crop &= cv2.dilate(proposal, kernel) > 0
        if not crop.any():
            crop = proposal.astype(bool)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < core.MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        rles.append(core.encode_crop(crop, bounds[i], occupied.shape))
        kept.append(scores[i])
    return rles, np.asarray(kept)


def file_counts(rt, model, fold, name, kernels=KERNELS):
    """R.file_counts with a kernel loop -> {k: {cell: [iou, tp, fp, fn]}}."""
    import cv2
    from pycocotools import mask as mask_utils

    core, tta = rt["core"], rt["tta"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    gts = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rles = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        gts.append([r for r in rles if mask_utils.area(r) > 0])
    views = R.K.read_cache_views(R.O.cache_path(RECIPE, fold, name))
    proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
    masks = [mask_utils.decode(r) for r in proposals]
    if masks:
        bounds, _, probs = core.crop_probabilities(rt["base"], model, gray, masks, True)
    out = {}
    for k in kernels:
        cells = out[f"k{k}"] = {}
        for t in v50.THRESHOLDS:
            rles, fs = (clean_refined(masks, scores, bounds, probs, t, k) if masks
                        else ([], np.asarray([])))
            for c in v50.CONFIDENCES:
                kept = [r for r, s in zip(rles, fs, strict=True) if s >= c]
                total = np.zeros(4)
                for gt in gts:
                    iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                           else np.zeros((len(gt), len(kept))))
                    total += core.pq_counts(iou, len(gt), len(kept))
                cells[f"t{t:.2f}_c{c:.2f}"] = total.tolist()
    return out


def cmd_fold(args):
    import torch

    init, encoder, size = ARMS[args.arm]
    tag = f"{args.arm}_all" if args.train_all else f"{args.arm}_fold{args.fold}"
    result = args.out / f"{tag}.json"
    if result.exists():
        print(f"{result} exists; skipping")
        return
    t0 = time.monotonic()
    rt = runtime()
    rt["base"].CFG.crop_size = size  # crop_probabilities reads it at inference
    fmap = R.fold_of_file()
    if len(fmap) != 707:
        raise RuntimeError("V61 folds no longer cover 707 files")
    test = sorted(n for n, k in fmap.items() if k == args.fold)
    fit = sorted(n for n, k in fmap.items() if k != args.fold)
    if set(test) & set(fit):
        raise RuntimeError("fold leak")
    if args.train_all:  # export refiner (V84): fit on all 707 files, nothing held out, nothing scored
        fit, test = sorted(fmap), []
    if args.limit:
        fit, test = fit[: args.limit], test[: args.limit]
    canon = set(rt["split"][0])
    rep = {"arm": args.arm, "fold": args.fold, "init": init, "encoder": encoder, "crop": size,
           "kernels": list(KERNELS), "recipe": RECIPE, "reads_test": False, "seed": args.seed,
           "fit_files": len(fit), "scored_files": len(test),
           "scored_canonical_files": len(canon & set(test)), "device": DEV, "history": []}
    rt["base"].seed_everything(args.seed)
    model = build_model(rt["base"], init, encoder)
    if init == "imagenet":
        recs = gt_records(rt, fit)
        rep["records_a"] = len(recs)
        rep["history"] += train(rt, model, recs, args.epochs_a, STAGE_A[1], args.seed, size, "A")
    recs = V63.fit_records(rt, RECIPE, fit, fmap)
    rep["records_b"] = len(recs)
    rep["history"] += train(rt, model, recs, args.epochs_b, STAGE_B[1], args.seed, size, "B")
    torch.save({"model": model.state_dict(), "arm": args.arm, "fold": args.fold},
               args.out / f"{tag}_refiner.pt")
    if args.train_all:  # bare CropUNet state_dict, the format export_v74.refiner_runtime loads
        torch.save(model.state_dict(), args.out / f"{tag}_refiner_state.pt")
        rep.update(fold=None, elapsed_seconds=time.monotonic() - t0)
        result.write_text(json.dumps(rep) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=2))
        return
    model.eval()
    t1 = time.monotonic()
    rep["per_file"] = {}
    for i, name in enumerate(test, 1):
        rep["per_file"][name] = file_counts(rt, model, args.fold, name)
        if i % 20 == 0:
            print(f"score {i}/{len(test)}", flush=True)
    rep["score_seconds"] = time.monotonic() - t1
    rep["elapsed_seconds"] = time.monotonic() - t0
    for k in KERNELS:
        tot = np.sum([v[f"k{k}"][CELL] for v in rep["per_file"].values()], axis=0)
        rep[f"k{k}_{CELL}"] = {"pq": R.K.pq_value(tot), "tp": int(tot[1]), "fp": int(tot[2]),
                               "fn": int(tot[3])}
    result.write_text(json.dumps(rep) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_file", "history")},
                     indent=2))


def pq(t):
    d = t[1] + 0.5 * (t[2] + t[3])
    return float(t[0] / d) if d else 0.0


def paired(a, b, seed=73, n=4000):
    """b - a, file bootstrap; a, b: (F, 4) count arrays over the same files."""
    rng = np.random.default_rng(seed)
    d = np.asarray([pq(b[i].sum(0)) - pq(a[i].sum(0))
                    for i in (rng.integers(0, len(a), len(a)) for _ in range(n))])
    return {"delta": pq(b.sum(0)) - pq(a.sum(0)), "p_gt_0": float(np.mean(d > 0)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def verdict(c):
    if c["delta"] < GATE["kill"]:
        return "KILL"
    if (c["delta"] >= GATE["delta"] and c["p_gt_0"] >= GATE["p"] and c["ci95"][0] > 0
            and c["folds_positive"] >= GATE["folds"]):
        return "CONTINUE"
    return "INCONCLUSIVE"


def cmd_compare(args):
    rows, fold_of = {}, {}
    for arm in ARMS:
        parts = [args.out / f"{arm}_fold{k}.json" for k in range(5)]
        if not all(p.exists() for p in parts):
            print(f"{arm}: missing folds {[p.name for p in parts if not p.exists()]}")
            continue
        per = {}
        for k, p in enumerate(parts):
            for n, v in json.loads(p.read_text())["per_file"].items():
                per[n], fold_of[n] = v, k
        for kk in KERNELS:
            rows[f"{arm}_k{kk}"] = per
    names = sorted(fold_of)
    arr = {r: np.asarray([per[n][f"k{r.rsplit('_k', 1)[1]}"][CELL] for n in names])
           for r, per in rows.items() if sorted(per) == names}
    canon = set(runtime_split_canonical())
    rep = {"files": len(names), "cell": CELL, "gate": GATE,
           "rows": {r: {"pq": pq(a.sum(0)), **dict(zip(("tp", "fp", "fn"),
                                                        map(int, a.sum(0)[1:])))}
                    for r, a in arr.items()}, "primary": {}, "reported": {}}
    folds = np.asarray([fold_of[n] for n in names])
    for label, b, a in PRIMARY:
        if a in arr and b in arr:
            c = {"a": a, "b": b, **paired(arr[a], arr[b])}
            c["fold_deltas"] = [pq(arr[b][folds == k].sum(0)) - pq(arr[a][folds == k].sum(0))
                                for k in range(5)]
            c["folds_positive"] = int(sum(d > 0 for d in c["fold_deltas"]))
            c["verdict"] = verdict(c)
            rep["primary"][label] = c
    # reported only: reproduction of r3 on its 141 clean files, and leakage size of canonical init
    cm = np.asarray([n in canon for n in names])
    if "r3c_k7" in arr and "base_k7" in arr:
        rep["reported"]["r3c_minus_base_canonical141"] = paired(arr["base_k7"][cm],
                                                                 arr["r3c_k7"][cm])
        rep["reported"]["r3c_minus_base_other566_LEAKY"] = paired(arr["base_k7"][~cm],
                                                                   arr["r3c_k7"][~cm])
    r3 = ROOT / "outputs/v63/r3.json"
    if "r3c_k7" in arr and r3.exists():
        old = json.loads(r3.read_text())["per_file"]
        cn = [i for i, n in enumerate(names) if n in old]
        a = np.asarray([old[names[i]][CELL] for i in cn])
        rep["reported"]["r3c_5fold_vs_v63_r3_on_141"] = {"files": len(cn),
                                                         **paired(a, arr["r3c_k7"][cn])}
    (args.out / "compare.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


def runtime_split_canonical():
    """The 141 canonical held-out names without loading the refiner runtime."""
    return json.loads((v50.V6_SOURCE / "v6_provenance.json").read_text())["validation_files"]


def cmd_parity(args):
    """CPU: our scorer + outputs/v63/r3_refiner.pt must reproduce r3.json per-file counts."""
    import torch

    rt = runtime()
    fmap = R.fold_of_file()
    model = rt["base"].CropUNet(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "outputs/v63/r3_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
    model.to(DEV).eval()
    old = json.loads((ROOT / "outputs/v63/r3.json").read_text())["per_file"]
    worst = 0.0
    for name in sorted(old)[: args.limit]:
        new = file_counts(rt, model, fmap[name], name, kernels=(7,))["k7"]
        diff = max(abs(a - b) for cell in new for a, b in zip(new[cell], old[name][cell]))
        worst = max(worst, diff)
        print(name, f"fold{fmap[name]}", "max|diff|", diff, flush=True)
    print(json.dumps({"files": args.limit, "max_abs_count_diff": worst,
                      "parity": worst < 1e-6}))


def main():
    global DEV
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--arm", choices=sorted(ARMS), required=True)
    f.add_argument("--fold", type=int, choices=range(5), required=True)
    f.add_argument("--epochs-a", type=int, default=STAGE_A[0], dest="epochs_a")
    f.add_argument("--epochs-b", type=int, default=STAGE_B[0], dest="epochs_b")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--limit", type=int, default=0, help="smoke: first N fit and scored files")
    f.add_argument("--train-all", action="store_true", dest="train_all",
                   help="export refiner: fit on all 707 files (--fold ignored), no scoring")
    p = sub.add_parser("parity")
    p.add_argument("--limit", type=int, default=4)
    sub.add_parser("compare")
    for s in (f, p, sub.choices["compare"]):
        s.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    DEV = device()
    {"fold": cmd_fold, "parity": cmd_parity, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
