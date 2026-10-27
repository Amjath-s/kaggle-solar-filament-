"""V13 inference-only sweep on the locked V13 stack (no training).

Axes (full product of the values passed on the CLI):
  refiner   : hard | hard_tta | v12 | v12_tta | ens | ens_tta   (#1 TTA, #2 ensemble)
  conf      : YOLO conf for consensus candidates                  (#4)
  thr       : refiner threshold                                   (#4)
  supp_iou  : whole-instance duplicate suppression IoU            (#4)
  rescue    : keep single-view dets with conf >= rescue (inf=off) (#3)
  post      : none | fill | fill_speck                            (#6)
  score     : conf | conf_prob  (suppress + exclusive ordering)   (#5)

YOLO (4 flips @ conf_min) and the refiners run once per image; every variant is
scored from that cache. Pipeline per variant matches
make_submission_v12_indep_iou050_excl.predict_nonoverlap:
consensus → refine+clamp → independent min_area → iou suppress → exclusive → min_area.
PQ matching is identical to src.evaluation.pq.compute_pq (greedy, IoU > 0.5), computed
on bbox crops for speed; aggregate PQ = (ΣIoU/TP) * RQ like v13_eval_refiner_hard_pq.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_v12_pq as v12
import make_submission_v12_indep_iou050_excl as pipe
from collections import defaultdict

from src.data.annotations import anns_to_instance_masks, build_file_to_anns_primary, load_coco
from src.data.split import read_split_list
from src.inference.local_crops import expand_xyxy, resize_image_mask
from src.inference.v12_consensus import (
    FLIP_VIEWS,
    box_iou_matrix,
    dilate_mask,
    flip_image,
    inverse_flip_boxes,
    inverse_flip_mask,
    yolo_seg_predict,
)
from src.utils.config import load_config, resolve_path

REFINERS = {
    "hard": ("experiments/v13/refiner_hard/best.pt", "experiments/v13/configs/resnet18_refiner_hard.yaml"),
    "v12": ("experiments/v12/resnet18_refiner/best.pt", "experiments/v12/configs/resnet18_refiner.yaml"),
    "v14": ("experiments/v14/resnet18_refiner/best.pt", "experiments/v14/configs/resnet18_refiner.yaml"),
    "v14hard": ("experiments/v14/refiner_hard/best.pt", "experiments/v14/configs/resnet18_refiner_hard.yaml"),
    "r34": ("experiments/v15/resnet34_refiner/best.pt", "experiments/v15/configs/resnet34_refiner.yaml"),
    "r34hard": ("experiments/v15/resnet34_hard/best.pt", "experiments/v15/configs/resnet34_refiner_hard.yaml"),
    "r18aug": ("experiments/v16/r18aug_refiner/best.pt", "experiments/v16/configs/r18aug_refiner.yaml"),
    "r18aughard": ("experiments/v16/r18aug_hard/best.pt", "experiments/v16/configs/r18aug_refiner_hard.yaml"),
    "r18focal": ("experiments/v19/r18aug_focal/best.pt", "experiments/v19/configs/r18aug_focal.yaml"),
}
BASELINE = {"pq": 0.43892, "tp": 667, "fp": 283, "fn": 478}


# ---------------------------------------------------------------- instances as crops
class Inst:
    __slots__ = ("y0", "x0", "m", "area")

    def __init__(self, y0: int, x0: int, m: np.ndarray):
        ys, xs = np.nonzero(m)
        if len(ys):
            m = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            y0, x0 = y0 + int(ys.min()), x0 + int(xs.min())
        self.y0, self.x0, self.m = y0, x0, m.astype(bool)
        self.area = int(self.m.sum())


def inter(a: Inst, b: Inst) -> int:
    y0, x0 = max(a.y0, b.y0), max(a.x0, b.x0)
    y1 = min(a.y0 + a.m.shape[0], b.y0 + b.m.shape[0])
    x1 = min(a.x0 + a.m.shape[1], b.x0 + b.m.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    sa = a.m[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0]
    sb = b.m[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0]
    return int(np.logical_and(sa, sb).sum())


def iou(a: Inst, b: Inst) -> float:
    i = inter(a, b)
    return i / max(a.area + b.area - i, 1) if i else 0.0


def pq_counts(gts: list[Inst], preds: list[Inst], thr: float = 0.5) -> tuple[int, int, int, float]:
    flat = []
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            v = (inter(g, p) + 1e-6) / (g.area + p.area - inter(g, p) + 1e-6)
            if v > thr:
                flat.append((v, i, j))
    flat.sort(reverse=True)
    ug, up, s = set(), set(), 0.0
    for v, i, j in flat:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        s += v
    tp = len(ug)
    return tp, len(preds) - tp, len(gts) - tp, s


# ---------------------------------------------------------------- detection
def detect_views(yolo, img, imgsz, conf_min, device, iou=0.7):
    h, w = img.shape[:2]
    boxes, confs, views, seeds = [], [], [], []
    for vi, view in enumerate(FLIP_VIEWS):
        b, c, m = yolo_seg_predict(yolo, flip_image(img, view), imgsz=imgsz, conf=conf_min, iou=iou, device=device)
        b = inverse_flip_boxes(b, w, h, view)
        for k in range(len(b)):
            boxes.append(b[k])
            confs.append(float(c[k]))
            views.append(vi)
            seeds.append(inverse_flip_mask(m[k], view))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), []
    return np.stack(boxes).astype(np.float32), np.asarray(confs, np.float32), np.asarray(views), seeds


def seed_iou_matrix(seeds: list[np.ndarray], biou: np.ndarray) -> np.ndarray:
    """Pairwise mask IoU of the YOLO seeds (only pairs whose boxes touch; others are 0)."""
    insts = [Inst(0, 0, s > 0) for s in seeds]
    n = len(insts)
    out = np.eye(n, dtype=np.float32)
    for i, j in zip(*np.nonzero(np.triu(biou > 0, k=1))):
        out[i, j] = out[j, i] = iou(insts[i], insts[j])
    return out


def select(boxes, confs, views, biou, conf, rescue, match_iou=0.5, nms_iou=0.5, det="single",
           agree_mat=None, nms_mat=None, partner_conf=None, clusters=False):
    """consensus_merge on candidates with conf>=conf, plus single-view rescue >= rescue.

    det (two-YOLO mode; views >= 4 come from the second detector):
      single : first detector only (original behaviour)
      second : second detector only
      union  : both detectors, any cross-view match counts
      both   : both detectors, a box must be matched by a view of the OTHER detector

    agree_mat / nms_mat: pairwise overlap used for the cross-view match and for NMS (default: box IoU;
    pass seed_iou_matrix for mask IoU, which does not confuse thin neighbours sharing a box).
    partner_conf: a candidate (conf >= conf) may be confirmed by another view down to this conf.
    clusters=True also returns, per kept index, the pool indices it would suppress (nms_mat > nms_iou),
    i.e. its own other-view copies, so callers can fuse their seeds. Use with a mask nms_mat: with box
    overlap a cluster can swallow a thin neighbour.
    """
    agree_mat = biou if agree_mat is None else agree_mat
    nms_mat = biou if nms_mat is None else nms_mat
    pool = np.nonzero(confs >= (conf if partner_conf is None else min(conf, partner_conf)))[0]
    src = views >= len(FLIP_VIEWS)
    if det == "single":
        pool = pool[~src[pool]]
    elif det == "second":
        pool = pool[src[pool]]
    cand = pool[confs[pool] >= conf]
    keep = []
    for i in cand:
        if det == "both":
            matched = any(src[j] != src[i] and agree_mat[i, j] >= match_iou for j in pool)
        else:
            matched = any(j != i and views[j] != views[i] and agree_mat[i, j] >= match_iou for j in pool)
        if matched or confs[i] >= rescue:
            keep.append(int(i))
    if not keep:
        return ([], {}) if clusters else []
    keep = np.asarray(keep)
    order = keep[np.argsort(-confs[keep], kind="stable")]
    kept, members = [], {}
    while order.size:
        c = int(order[0])
        kept.append(c)
        rest = order[1:]
        order = rest[nms_mat[c, rest] <= nms_iou]
        if clusters:
            members[c] = sorted({c, *pool[nms_mat[c, pool] > nms_iou].tolist()})
    return (kept, members) if clusters else kept


# ---------------------------------------------------------------- refine
@torch.no_grad()
def refine_probs(models: dict, img, box, image_size, pad_frac, device):
    h, w = img.shape[:2]
    exp = expand_xyxy(box, pad_frac, w, h)
    crop = img[exp[1] : exp[3], exp[0] : exp[2]]
    img_r, _ = resize_image_mask(crop, None, size=image_size)
    x = torch.from_numpy(img_r)
    batch = torch.stack([x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)])[:, None].to(device)
    out = {}
    for name, model in models.items():
        p = torch.sigmoid(model(batch))[:, 0].float()
        p = torch.stack([p[0], p[1].flip(-1), p[2].flip(-2), p[3].flip(-1).flip(-2)])
        out[name] = p[0].cpu().numpy()
        out[name + "_tta"] = p.mean(0).cpu().numpy()
    if "hard" in models and "v12" in models:
        out["ens"] = (out["hard"] + out["v12"]) / 2
        out["ens_tta"] = (out["hard_tta"] + out["v12_tta"]) / 2
    return exp, out


def largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def build_mask(prob, thr, exp, seed_allow, post, upsample="nearest") -> tuple[Inst, float]:
    """post: none | fill | fill_speck | fill_largest (one piece, no holes, as MAGFiLO labels are).
    upsample: nearest (threshold at refiner res, then resize) | bilinear (resize probs, then threshold)."""
    pred = (prob >= thr).astype(np.uint8)
    cw, ch = int(exp[2] - exp[0]), int(exp[3] - exp[1])
    if upsample == "bilinear":
        m = np.asarray(Image.fromarray(prob.astype(np.float32), mode="F").resize((cw, ch), Image.BILINEAR)) >= thr
    else:
        m = np.asarray(Image.fromarray(pred * 255, mode="L").resize((cw, ch), Image.NEAREST)) > 127
    m = np.logical_and(m, seed_allow)
    if post in ("fill", "fill_speck", "fill_largest") and m.any():
        m = ndi.binary_fill_holes(m)
    if post == "fill_speck" and m.any():
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
            ok = np.nonzero(sizes >= 0.10 * sizes.max())[0] + 1
            m = np.isin(lab, ok)
    if post == "fill_largest" and m.any():
        m = largest_component(m)
    mp = float(prob[pred > 0].mean()) if pred.any() else 0.0
    return Inst(int(exp[1]), int(exp[0]), m), mp


# ---------------------------------------------------------------- post pipeline
def merge_fragments(insts: list[Inst], scores: np.ndarray, gap: int) -> tuple[list[Inst], np.ndarray]:
    """Join instances whose masks come within `gap` px of each other (union-find); score = max of group."""
    n = len(insts)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    st = np.ones((3, 3), bool)
    grown = []
    for a in insts:
        m = np.pad(a.m, gap)
        grown.append(Inst.__new__(Inst))
        g = grown[-1]
        g.y0, g.x0, g.m = a.y0 - gap, a.x0 - gap, ndi.binary_dilation(m, st, iterations=gap)
        g.area = int(g.m.sum())
    for i in range(n):
        for j in range(i + 1, n):
            if inter(grown[i], insts[j]):
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    out, out_sc = [], []
    for idx in groups.values():
        if len(idx) == 1:
            out.append(insts[idx[0]])
            out_sc.append(scores[idx[0]])
            continue
        y0 = min(insts[i].y0 for i in idx)
        x0 = min(insts[i].x0 for i in idx)
        y1 = max(insts[i].y0 + insts[i].m.shape[0] for i in idx)
        x1 = max(insts[i].x0 + insts[i].m.shape[1] for i in idx)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        for i in idx:
            a = insts[i]
            m[a.y0 - y0 : a.y0 - y0 + a.m.shape[0], a.x0 - x0 : a.x0 - x0 + a.m.shape[1]] |= a.m
        out.append(Inst(y0, x0, m))
        out_sc.append(max(scores[i] for i in idx))
    return out, np.asarray(out_sc, np.float32)


def finish(insts: list[Inst], scores: np.ndarray, min_area: int, supp_iou: float, h: int, w: int,
           merge_gap: int = 0, greedy: bool = False, min_keep_frac: float = 0.0,
           one_piece: bool = False) -> list[Inst]:
    """min_area -> duplicate suppression -> optional fragment merge -> score-ordered exclusive painting.

    greedy=False (legacy): an instance is dropped if ANY higher-scoring instance overlaps it, even one
    that was itself dropped (chain suppression). greedy=True: standard NMS, only kept instances suppress.
    min_keep_frac: after painting, drop instances that kept less than this fraction of their area
    (leftover slivers of a duplicate). one_piece: keep only the largest component of what was kept.
    """
    idx = [i for i, m in enumerate(insts) if m.area >= min_area]
    insts = [insts[i] for i in idx]
    scores = scores[idx]
    n = len(insts)
    keep = []
    if greedy:
        for j in np.argsort(-scores, kind="stable"):
            if all(iou(insts[int(j)], insts[k]) < supp_iou for k in keep):
                keep.append(int(j))
        keep.sort()
    else:
        for j in range(n):
            best = 0.0
            for i in range(n):
                if i == j:
                    continue
                higher = scores[i] > scores[j] + 1e-12 or (abs(float(scores[i] - scores[j])) <= 1e-12 and i < j)
                if higher:
                    best = max(best, iou(insts[j], insts[i]))
            if best < supp_iou:
                keep.append(j)
    insts = [insts[i] for i in keep]
    scores = scores[keep]
    if merge_gap > 0 and len(insts) > 1:
        insts, scores = merge_fragments(insts, scores, merge_gap)
    claimed = np.zeros((h, w), dtype=bool)
    out = []
    for k in np.argsort(-scores, kind="stable"):
        a = insts[int(k)]
        reg = claimed[a.y0 : a.y0 + a.m.shape[0], a.x0 : a.x0 + a.m.shape[1]]
        kept = a.m & ~reg
        if one_piece and kept.any():
            kept = largest_component(kept)
        if kept.any() and kept.sum() >= min_keep_frac * a.area:
            reg |= kept
            b = Inst(a.y0, a.x0, kept)
            if b.area >= min_area:
                out.append(b)
    return out


def parse_list(s: str, typ):
    return [typ(v) for v in s.split(",")]


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="hard,hard_tta,v12,v12_tta,ens,ens_tta")
    ap.add_argument("--conf", default="0.35,0.40,0.45")
    ap.add_argument("--thr", default="0.40,0.45,0.50")
    ap.add_argument("--supp-iou", default="0.40,0.50,0.60")
    ap.add_argument("--rescue", default="inf")
    ap.add_argument("--post", default="none")
    ap.add_argument("--score", default="conf")
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--yolo-weights", default="experiments/v12/yolo11s/weights/best.pt")
    ap.add_argument("--yolo-weights2", default="", help="second detector for --det second/union/both")
    ap.add_argument("--det", default="single", help="single,second,union,both (see select)")
    ap.add_argument("--merge-gap", default="0", help="fragment merge gap in px; 0 = off")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/v13/inference_sweep/pass1")
    ap.add_argument("--gt", choices=["primary", "all"], default="primary",
                    help="primary: one annotator per image (old). all: every annotator set scored as its own image")
    args = ap.parse_args()

    axes = {
        "refiner": parse_list(args.refiner, str),
        "conf": parse_list(args.conf, float),
        "thr": parse_list(args.thr, float),
        "supp_iou": parse_list(args.supp_iou, float),
        "rescue": parse_list(args.rescue, float),
        "post": parse_list(args.post, str),
        "score": parse_list(args.score, str),
        "det": parse_list(args.det, str),
        "merge_gap": parse_list(args.merge_gap, int),
    }
    if set(axes["det"]) - {"single"} and not args.yolo_weights2:
        ap.error("--det other than single needs --yolo-weights2")
    combos = list(itertools.product(*axes.values()))
    det_keys = sorted({(c, r, d) for c in axes["conf"] for r in axes["rescue"] for d in axes["det"]})
    mask_keys = sorted({(rf, t, p) for rf in axes["refiner"] for t in axes["thr"] for p in axes["post"]})
    print(f"variants={len(combos)} det_sets={len(det_keys)} mask_sets={len(mask_keys)}", flush=True)

    cfg = load_config(ROOT / REFINERS["hard"][1])
    infer = cfg.get("infer", {})
    image_size = int(cfg.get("image_size", 256))
    pad_frac = float(cfg.get("pad_frac", 0.20))
    seed_dilate = int(infer.get("seed_dilate", 7))
    images_dir = resolve_path(ROOT, cfg["data"]["images_dir"])
    names = read_split_list(resolve_path(ROOT, cfg["data"]["splits_dir"]) / "val.txt")
    if args.limit:
        names = names[: args.limit]
    coco = load_coco(resolve_path(ROOT, cfg["data"]["annotations"]))
    file_to_anns = build_file_to_anns_primary(coco)
    file_to_sets: dict[str, list[list[dict]]] = defaultdict(list)
    if args.gt == "all":
        id_to_anns = defaultdict(list)
        for a in coco["annotations"]:
            id_to_anns[a["image_id"]].append(a)
        for im in coco["images"]:
            file_to_sets[im["file_name"]].append(id_to_anns.get(im["id"], []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(resolve_path(ROOT, args.yolo_weights)))
    yolo2 = YOLO(str(resolve_path(ROOT, args.yolo_weights2))) if args.yolo_weights2 else None
    need = {r.replace("_tta", "") for r in axes["refiner"]}
    if need & {"ens"}:
        need = (need - {"ens"}) | {"hard", "v12"}
    models = {}
    for n in need:
        ck, cf = REFINERS[n]
        models[n] = pipe.load_refiner(ROOT / ck, load_config(ROOT / cf), device)
    conf_min = min(axes["conf"])

    acc = {c: [0, 0, 0, 0.0, 0] for c in combos}
    t0 = time.time()
    for name in tqdm(names, desc="sweep"):
        img, _, gt_full, _, _ = v12.load_image_and_gt(images_dir / name, file_to_anns.get(name, []), out_size=2048)
        h, w = img.shape[:2]
        if args.gt == "all":
            gt_sets = [[Inst(0, 0, g > 0) for g in anns_to_instance_masks(a, h, w)] for a in file_to_sets[name]]
        else:
            gt_sets = [[Inst(0, 0, g > 0) for g in gt_full]]
        boxes, confs, views, seeds = detect_views(yolo, img, 1024, conf_min, device)
        if yolo2 is not None:
            b2, c2, v2, s2 = detect_views(yolo2, img, 1024, conf_min, device)
            if len(b2):
                boxes = np.concatenate([boxes, b2]) if len(boxes) else b2
                confs = np.concatenate([confs, c2])
                views = np.concatenate([views, v2 + len(FLIP_VIEWS)])
                seeds = seeds + s2
        biou = box_iou_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
        sel = {k: select(boxes, confs, views, biou, k[0], k[1], det=k[2]) for k in det_keys}
        used = sorted({i for v in sel.values() for i in v})
        cache = {}
        for i in used:
            exp, probs = refine_probs(models, img, boxes[i], image_size, pad_frac, device)
            sc = seeds[i][exp[1] : exp[3], exp[0] : exp[2]]
            allow = dilate_mask((sc > 0).astype(np.uint8), seed_dilate).astype(bool)
            for rf, t, p in mask_keys:
                cache[(i, rf, t, p)] = build_mask(probs[rf], t, exp, allow, p)
        del seeds
        for combo in combos:
            rf, c, t, su, rs, p, sc, dm, mg = combo
            idx = sel[(c, rs, dm)]
            insts = [cache[(i, rf, t, p)][0] for i in idx]
            cf = confs[idx] if idx else np.zeros(0, np.float32)
            if sc == "conf_prob":
                cf = cf * np.asarray([cache[(i, rf, t, p)][1] for i in idx], np.float32)
            preds = finish(insts, np.asarray(cf, np.float32), args.min_area, su, h, w, merge_gap=mg)
            a = acc[combo]
            for gts in gt_sets:
                tp, fp, fn, s = pq_counts(gts, preds)
                a[0] += tp; a[1] += fp; a[2] += fn; a[3] += s; a[4] += len(preds)

    rows = []
    for combo, (tp, fp, fn, s, npred) in acc.items():
        sq = s / tp if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        rows.append(dict(zip(axes.keys(), combo)) | {
            "pq": round(sq * rq, 6), "sq": round(sq, 6), "rq": round(rq, 6),
            "tp": tp, "fp": fp, "fn": fn, "n_pred": npred,
            "d_pq": round(sq * rq - BASELINE["pq"], 6),
        })
    rows.sort(key=lambda r: -r["pq"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "meta.json").write_text(json.dumps({"axes": axes, "gt": args.gt, "n_val": len(names), "baseline": BASELINE,
                                               "elapsed_s": time.time() - t0}, indent=2), encoding="utf-8")
    base = [r for r in rows if r["refiner"] == "hard" and r["conf"] == 0.40 and r["thr"] == 0.45
            and r["supp_iou"] == 0.50 and r["rescue"] == float("inf") and r["post"] == "none" and r["score"] == "conf"]
    if base:
        print("BASELINE_REPRO", base[0], flush=True)
    for r in rows[:25]:
        print(r, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
