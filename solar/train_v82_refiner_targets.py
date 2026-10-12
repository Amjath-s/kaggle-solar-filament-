"""V82: agreement-aware stage-B targets for the V73 `base` fold refiner. Only the target changes.

Pre-registration: docs/v82_refiner_targets_preregistration_2026-10-11.md. Never reads test.
Runs train_v73_refiner.cmd_fold unchanged (same files, stage A/B epochs, lr, seed, crops,
scoring) with three hooks: fit_records gains per-proposal agreement fields, the dataset builds
the arm's target, and the loss takes a per-crop weight. Stage A (GT-centred crops) is untouched.

Per stage-B sample (record r, OOF proposal p, GT g matched by V11's matcher, exactly V73's list):
  base     target g                                   (V73 base; w=1)
  soft     mean over the image's R records of the GT each record matched to p (0 if none)
  union    pixel max over the matched GTs of all records
  qweight  target g, loss weight q_p = (#records matching p) / R, normalised by sum(w) per batch
  CUDA_VISIBLE_DEVICES=0 python scripts/train_v82_refiner_targets.py fold --arm soft --fold 0 --out outputs/v82
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v73_refiner as V73  # noqa: E402

V63, v11, v50, R = V73.V63, V73.v11, V73.v50, V73.R
NEW = ("soft", "union", "qweight")
for a in NEW:
    V73.ARMS[a] = V73.ARMS["base"]
ARM = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "base"


def fit_records(rt, recipe, names, fmap):
    """V63.fit_records (same loop, same order) + per-proposal `agree` (matched GT per record)."""
    from pycocotools import mask as mask_utils

    base, core, tta = rt["base"], rt["core"], rt["tta"]
    records = []
    for i, name in enumerate(names, 1):
        views = R.K.read_cache_views(R.O.cache_path(recipe, fmap[name], name))
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        ids = sorted(rt["by_file"].get(name, []), key=str)
        file_recs, matched = [], {}  # matched[pi] = [GT rle per record that matched pi]
        for image_id in ids:
            info = rt["images"][image_id]
            targets = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
            targets = [t for t in targets if mask_utils.area(t) > 0]
            if not targets or not proposals:
                continue
            iou = mask_utils.iou(targets, proposals, [0] * len(proposals))
            for ti, pi in v11.match_positive_proposals(iou, scores):
                seed = mask_utils.decode(proposals[pi])
                file_recs.append({"name": name, "seed": proposals[pi], "target": targets[ti],
                                  "bounds": tuple(base.square_bounds(seed)), "pi": pi})
                matched.setdefault(pi, []).append(targets[ti])
        for r in file_recs:  # V11 matcher: <= 1 GT per proposal per record
            r["agree"], r["q"] = matched[r["pi"]], len(matched[r["pi"]]) / len(ids)
        records += file_recs
        if i % 50 == 0:
            print(f"cohort {i}/{len(names)} records={len(records)}", flush=True)
    if not records:
        raise RuntimeError("empty training cohort")
    return records


def target_u8(r, arm):
    """Arm target as uint8 0..255 (crop_triplet resizes with INTER_NEAREST, keeping levels)."""
    from pycocotools import mask as mask_utils

    if arm in ("soft", "union") and "agree" in r:
        stack = np.stack([mask_utils.decode(t) for t in r["agree"]]).astype(np.float32)
        n = len(r["agree"]) / r["q"]  # R records; non-matching records contribute 0
        m = stack.sum(0) / n if arm == "soft" else stack.max(0)
        return np.rint(m * 255).astype(np.uint8)
    return mask_utils.decode(r["target"]) * np.uint8(255)


class CropDataset(V73.CropDataset):
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
        seed, target = mask_utils.decode(r["seed"]), target_u8(r, ARM)
        rng = np.random.default_rng(self.seed * 7919 + self.epoch * 1_000_003 + index)
        jit = v11.jitter_bounds(r["bounds"], gray.shape, rng)
        trip = v11.crop_triplet(gray, seed, target, jit, self.size)
        if not trip[1].any() or not trip[2].any():
            trip = v11.crop_triplet(gray, seed, target, r["bounds"], self.size)
        image, _, tgt = v11.augment_triplet(*trip, rng)
        w = r.get("q", 1.0) if ARM == "qweight" else 1.0
        return (torch.from_numpy(np.repeat(image[None], 3, axis=0)),
                torch.from_numpy(tgt.astype(np.float32)[None] / 255.0),
                torch.tensor(w, dtype=torch.float32))


def train(rt, model, records, epochs, lr, seed, size, stage):
    """V73.train with a per-crop weighted loss (w=1 -> identical to V73's 0.5 BCE + 0.5 Dice)."""
    import json
    import time

    import torch
    F = torch.nn.functional

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        ds = CropDataset(records, rt["base"].CFG.train_dir, epoch, seed, size)
        order = np.random.default_rng(seed + epoch).permutation(len(records)).tolist()
        loader = torch.utils.data.DataLoader(ds, batch_size=v11.BATCH_SIZE, sampler=order,
                                             num_workers=8, pin_memory=V73.DEV == "cuda")
        model.train()
        total = n = 0
        t0 = time.monotonic()
        for x, y, w in loader:
            x, y, w = x.to(V73.DEV), y.to(V73.DEV), w.to(V73.DEV)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean((1, 2, 3))
            p = logits.sigmoid()
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = ((0.5 * bce + 0.5 * dice) * w).sum() / w.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss {stage} epoch {epoch}")
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
        sched.step()
        history.append({"stage": stage, "epoch": epoch, "mean_loss": total / n,
                        "seconds": time.monotonic() - t0})
        if V73.DEV == "cuda":
            history[-1]["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps(history[-1]), flush=True)
    return history


def demo():
    """CPU self-check of the target/weight logic on toy RLEs."""
    from pycocotools import mask as mu
    a = np.zeros((8, 8), np.uint8, order="F"); a[2:6, 2:6] = 1
    b = np.zeros((8, 8), np.uint8, order="F"); b[2:7, 2:7] = 1
    ra, rb = mu.encode(a), mu.encode(b)
    r = {"target": ra, "agree": [ra, rb], "q": 2 / 3}  # 2 of 3 records matched
    s, u, base = target_u8(r, "soft"), target_u8(r, "union"), target_u8(r, "base")
    assert s[3, 3] == 170 and s[6, 6] == 85 and s[0, 0] == 0, s
    assert u[6, 6] == 255 and u.sum() == 255 * 25 and (base == a * 255).all()


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo(); print("demo ok"); sys.exit(0)
    V73.V63.fit_records, V73.CropDataset, V73.train = fit_records, CropDataset, train
    V73.main()
