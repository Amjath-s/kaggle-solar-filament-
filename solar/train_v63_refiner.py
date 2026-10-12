"""V63: re-fit the V11 crop refiner on OOF detector proposals, then score it on the 141 files.

V11 = canonical refiner continued 4 epochs (AdamW 5e-5, cosine, 0.5 BCE + 0.5 Dice, batch 16,
proposal-centred 256 crops) on V6-detector proposals of the 566 canonical FIT files. V63 keeps
that recipe and changes only the proposal source: the V61/V62 OOF cache of `--recipe`, so the
refiner trains on the proposal distribution it will see. The 141 canonical held-out files are
never read for training (their OOF proposals, images and labels are only scored).

  --no-train : score the frozen V11 refiner (the control row, per-file counts saved).

Output JSON keeps per-file counts at every (t, c) cell so `compare_v63.py` can pair arms.

  CUDA_VISIBLE_DEVICES=1 python scripts/train_v63_refiner.py --recipe p2first --tag r1 \
      --out outputs/v63
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
import eval_v61_oof as O  # noqa: E402
import eval_v62_refiner as R  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import v11_refiner as v11  # noqa: E402
import v61_toolkit as K  # noqa: E402


def fit_records(rt, recipe, names, fmap):
    """Matched (seed proposal, GT target) pairs from OOF proposals, V11 matching policy."""
    from pycocotools import mask as mask_utils

    base, core, tta = rt["base"], rt["core"], rt["tta"]
    by_file = {}  # split[4] covers only the held-out files
    for image_id, info in rt["images"].items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    records = []
    for i, name in enumerate(names, 1):
        views = K.read_cache_views(O.cache_path(recipe, fmap[name], name))
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        for image_id in sorted(by_file.get(name, []), key=str):
            info = rt["images"][image_id]
            targets = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
            targets = [t for t in targets if mask_utils.area(t) > 0]
            if not targets or not proposals:
                continue
            iou = mask_utils.iou(targets, proposals, [0] * len(proposals))
            for ti, pi in v11.match_positive_proposals(iou, scores):
                seed = mask_utils.decode(proposals[pi])
                records.append({"name": name, "seed": proposals[pi], "target": targets[ti],
                                "bounds": tuple(base.square_bounds(seed))})
        if i % 50 == 0:
            print(f"cohort {i}/{len(names)} records={len(records)}", flush=True)
    if not records:
        raise RuntimeError("empty training cohort")
    return records


class CropDataset:
    def __init__(self, records, train_dir, epoch, seed):
        self.records, self.train_dir, self.epoch, self.seed = records, train_dir, epoch, seed
        self.cache = {}

    def __len__(self):
        return len(self.records)

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
        trip = v11.crop_triplet(gray, seed, target, v11.jitter_bounds(r["bounds"], gray.shape, rng))
        if not trip[1].any() or not trip[2].any():  # thin mask vanished: use unjittered crop
            trip = v11.crop_triplet(gray, seed, target, r["bounds"])
        image, _, tgt = v11.augment_triplet(*trip, rng)
        return (torch.from_numpy(np.repeat(image[None], 3, axis=0)),
                torch.from_numpy(tgt.astype(np.float32)[None]))


def train(rt, records, epochs, lr, seed, out):
    import torch

    base = rt["base"]
    base.seed_everything(seed)
    model = v11.build_model(base, v50.CANONICAL_REFINER)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        ds = CropDataset(records, base.CFG.train_dir, epoch, seed)
        order = np.random.default_rng(seed + epoch).permutation(len(records)).tolist()
        loader = torch.utils.data.DataLoader(ds, batch_size=v11.BATCH_SIZE, sampler=order,
                                             num_workers=8, pin_memory=True)
        model.train()
        total = n = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                    + 0.5 * base.soft_dice_loss(logits, y))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss epoch {epoch}")
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
        sched.step()
        history.append({"epoch": epoch, "mean_loss": total / n})
        print(json.dumps(history[-1]), flush=True)
    torch.save({"model": model.state_dict()}, out)
    return model.eval(), history


def score(rt, recipe, fmap, limit=0):
    """Per-file counts at every (t, c) on the 141 held-out files with rt['model']."""
    import cv2
    from pycocotools import mask as mask_utils

    core, by_file = rt["core"], rt["split"][4]
    canon = sorted(rt["split"][0])
    if len(canon) != 141:
        raise RuntimeError("canonical held-out population changed")
    per = {}
    for i, name in enumerate(canon[:limit] if limit else canon, 1):
        gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gt = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            rles = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
            gt[image_id] = [r for r in rles if mask_utils.area(r) > 0]
        counts = R.file_counts(rt, recipe, fmap[name], name, gray, gt)
        per[name] = {f"t{t:.2f}_c{c:.2f}": v.tolist() for (t, c), v in counts.items()}
        if i % 20 == 0:
            print(f"score {i}/{len(canon)}", flush=True)
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True, help="OOF proposal source (train AND score)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=v11.EPOCHS)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-train", action="store_true", dest="no_train")
    ap.add_argument("--limit", type=int, default=0, help="smoke: few fit/score files")
    ap.add_argument("--train-all", action="store_true", dest="train_all",
                    help="export refiner: train on all 707 OOF proposal sets, no scoring")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = args.out / f"{args.tag}.json"
    if result.exists():
        print(f"{result} exists; skipping")
        return
    t0 = time.monotonic()
    rt = v50.runtime_state()
    fmap = R.fold_of_file()
    held = set(rt["split"][0])
    fit = sorted(n for n in fmap if n not in held)
    if len(fit) != 566 or held & set(fit):
        raise RuntimeError("fit/held-out partition changed")
    rep = {"tag": args.tag, "recipe": args.recipe, "reads_test": False,
           "trained": not args.no_train}
    if args.train_all:  # final refiner for test export; nothing held out, so nothing scored
        import torch

        records = fit_records(rt, args.recipe, sorted(fmap), fmap)
        model, rep["history"] = train(rt, records, args.epochs, args.lr, args.seed,
                                      args.out / f"{args.tag}_refiner.pt")
        torch.save(model.state_dict(), args.out / f"{args.tag}_refiner_state.pt")
        rep.update(records=len(records), files=len(fmap), epochs=args.epochs, lr=args.lr,
                   elapsed_seconds=time.monotonic() - t0)
        result.write_text(json.dumps(rep) + "\n")
        print(json.dumps(rep, indent=2))
        return
    if not args.no_train:
        records = fit_records(rt, args.recipe, fit[: args.limit] if args.limit else fit, fmap)
        rep.update(records=len(records), epochs=args.epochs, lr=args.lr, seed=args.seed)
        rt["model"], rep["history"] = train(rt, records, args.epochs, args.lr, args.seed,
                                            args.out / f"{args.tag}_refiner.pt")
    rep["per_file"] = score(rt, args.recipe, fmap, args.limit)
    tot = np.sum([v[f"t{R.V17_CELL[0]:.2f}_c{R.V17_CELL[1]:.2f}"]
                  for v in rep["per_file"].values()], axis=0)
    rep["v17_cell"] = {"pq": K.pq_value(tot), "tp": int(tot[1]), "fp": int(tot[2]),
                       "fn": int(tot[3])}
    rep["elapsed_seconds"] = time.monotonic() - t0
    result.write_text(json.dumps(rep) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_file"}, indent=2))


if __name__ == "__main__":
    main()
