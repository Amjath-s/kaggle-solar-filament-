from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from importlib.metadata import version
from dataclasses import dataclass
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.138"], check=True)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm
from ultralytics import YOLO


class CFG:
    root_candidates = [
        Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    ]
    root = next((candidate for candidate in root_candidates if candidate.exists()), root_candidates[0])
    train_dir = root / "train" / "train_images"
    test_dir = root / "test" / "test_images"
    train_json = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    yolo_dir = Path("/kaggle/working/yolo_dataset")
    seed = 42
    imgsz = 1024
    batch = 2
    epochs = 60
    patience = 15
    model = "yolo11m-seg.pt"
    mask_ratio = 2
    crop_size = 256
    crop_context = 1.8
    min_crop = 96
    refiner_epochs = 8
    refiner_threshold = 0.65
    min_component_size = 20
    overlap_dilate = 3
    use_all_train_annotators = True
    single_class = True
    fliplr = 0.5
    flipud = 0.5
    top_submission_candidates = 5
    public_hedge_margin = 0.03


@dataclass(frozen=True)
class Candidate:
    pq: float
    conf: float
    mode: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco():
    with CFG.train_json.open() as file:
        coco = json.load(file)
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    return coco, images, annotations


def best_annotator_ids(images, annotations):
    file_to_ids = {}
    for image_id, info in images.items():
        file_to_ids.setdefault(info["file_name"], []).append(image_id)

    def completeness(image_id):
        anns = annotations.get(image_id, [])
        return len(anns), sum(ann.get("area", 0) for ann in anns)

    return {file_name: max(ids, key=completeness) for file_name, ids in file_to_ids.items()}


def polygon_to_yolo_line(polygon, width, height, class_id):
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in points.flatten())


def write_yolo_split(split_name, image_ids, images, annotations, cat_id_to_yolo, unique_names=False):
    image_output = CFG.yolo_dir / "images" / split_name
    label_output = CFG.yolo_dir / "labels" / split_name
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        info = images[image_id]
        source = CFG.train_dir / info["file_name"]
        source_path = Path(info["file_name"])
        target_name = (
            f"{source_path.stem}_ann{image_id}{source_path.suffix}" if unique_names else info["file_name"]
        )
        target = image_output / target_name
        if not target.exists():
            os.symlink(source, target)

        lines = []
        for annotation in annotations.get(image_id, []):
            class_id = cat_id_to_yolo[annotation["category_id"]] if cat_id_to_yolo else 0
            for polygon in annotation["segmentation"]:
                if len(polygon) >= 6:
                    lines.append(
                        polygon_to_yolo_line(polygon, info["width"], info["height"], class_id)
                    )
        (label_output / f"{Path(target_name).stem}.txt").write_text("\n".join(lines))


def prepare_yolo_data(coco, images, annotations):
    if CFG.yolo_dir.exists():
        shutil.rmtree(CFG.yolo_dir)
    file_to_ids = {}
    for image_id, info in images.items():
        file_to_ids.setdefault(info["file_name"], []).append(image_id)
    file_to_best_id = best_annotator_ids(images, annotations)
    files = sorted(file_to_best_id)
    random.Random(CFG.seed).shuffle(files)
    n_valid = round(len(files) * 0.2)
    valid_files = files[:n_valid]
    train_files = files[n_valid:]
    if CFG.use_all_train_annotators:
        train_ids = [image_id for file_name in train_files for image_id in file_to_ids[file_name]]
    else:
        train_ids = [file_to_best_id[file_name] for file_name in train_files]
    valid_ids = [file_to_best_id[file_name] for file_name in valid_files]

    if CFG.single_class:
        categories = [{"id": 0, "name": "filament"}]
        cat_id_to_yolo = None
    else:
        categories = sorted(coco["categories"], key=lambda category: category["id"])
        cat_id_to_yolo = {category["id"]: index for index, category in enumerate(categories)}

    write_yolo_split(
        "train",
        train_ids,
        images,
        annotations,
        cat_id_to_yolo,
        unique_names=CFG.use_all_train_annotators,
    )
    write_yolo_split("val", valid_ids, images, annotations, cat_id_to_yolo)

    names = "\n".join(f"  {index}: {category['name']}" for index, category in enumerate(categories))
    (CFG.yolo_dir / "data.yaml").write_text(
        f"path: {CFG.yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    return train_ids, valid_ids


def square_bounds(mask, context=CFG.crop_context, minimum=CFG.min_crop):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    cx = (xs.min() + xs.max() + 1) / 2
    cy = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CropUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))


def annotation_mask(image_shape, annotation):
    height, width = image_shape
    polygons = annotation["segmentation"]
    if not polygons:
        return np.zeros(image_shape, dtype=np.uint8)
    rles = mask_utils.frPyObjects(polygons, height, width)
    return mask_utils.decode(mask_utils.merge(rles))


def build_crop_arrays(image_ids, images, annotations):
    crop_images, crop_masks = [], []
    for image_id in tqdm(image_ids, desc="building refiner crops"):
        info = images[image_id]
        image = cv2.imread(str(CFG.train_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        for annotation in annotations.get(image_id, []):
            mask = annotation_mask(image.shape, annotation)
            if mask.sum() == 0:
                continue
            x0, y0, x1, y1 = square_bounds(mask)
            crop_images.append(
                cv2.resize(
                    image[y0:y1, x0:x1],
                    (CFG.crop_size, CFG.crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            crop_masks.append(
                cv2.resize(
                    mask[y0:y1, x0:x1],
                    (CFG.crop_size, CFG.crop_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
    return np.stack(crop_images), np.stack(crop_masks)


class CropDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).float() / 255
        mask = torch.from_numpy(self.masks[index].copy()).float().unsqueeze(0)
        if self.augment:
            rotations = random.randrange(4)
            image = torch.rot90(image, rotations, (0, 1))
            mask = torch.rot90(mask, rotations, (1, 2))
            if random.random() < 0.5:
                image = torch.flip(image, (1,))
                mask = torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.2)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.9, 1.1), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask


def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train_refiner(train_ids, valid_ids, images, annotations):
    train_images, train_masks = build_crop_arrays(train_ids, images, annotations)
    valid_images, valid_masks = build_crop_arrays(valid_ids, images, annotations)
    train_loader = DataLoader(
        CropDataset(train_images, train_masks, augment=True),
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        CropDataset(valid_images, valid_masks),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CropUNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.refiner_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_dice = -1.0

    for epoch in range(1, CFG.refiner_epochs + 1):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"refiner {epoch}/{CFG.refiner_epochs}", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                    logits, masks
                ) + 0.5 * soft_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        dice_total = 0.0
        n = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs = imgs.to(device)
                masks = masks.to(device).bool()
                pred = model(imgs).sigmoid() >= CFG.refiner_threshold
                inter = (pred & masks).sum((1, 2, 3)).float()
                denom = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_total += ((2 * inter + 1e-7) / (denom + 1e-7)).sum().item()
                n += len(imgs)
        val_dice = dice_total / n
        print(f"refiner epoch={epoch} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "/kaggle/working/best_refiner.pt")
        scheduler.step()

    model.load_state_dict(torch.load("/kaggle/working/best_refiner.pt", map_location=device))
    model.eval()
    return model


@torch.no_grad()
def refine_masks(
    image_gray,
    masks,
    refiner,
    threshold=CFG.refiner_threshold,
    constrain_to_yolo=True,
    dilate_radius=CFG.overlap_dilate,
):
    device = next(refiner.parameters()).device
    refined = []
    for mask in masks:
        if mask.sum() == 0:
            refined.append(mask)
            continue
        x0, y0, x1, y1 = square_bounds(mask)
        crop = cv2.resize(
            image_gray[y0:y1, x0:x1],
            (CFG.crop_size, CFG.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = (
            torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            / 255
        )
        prob = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
        crop_mask = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= threshold
        if constrain_to_yolo:
            yolo_crop = mask[y0:y1, x0:x1].astype(np.uint8)
            if dilate_radius > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * dilate_radius + 1, 2 * dilate_radius + 1),
                )
                yolo_crop = cv2.dilate(yolo_crop, kernel)
            crop_mask = crop_mask & (yolo_crop > 0)
        if crop_mask.any():
            full = np.zeros_like(mask, dtype=np.uint8)
            full[y0:y1, x0:x1] = crop_mask
            refined.append(full)
        else:
            refined.append(mask)
    return refined


def make_non_overlapping(scored_masks):
    kept = []
    occupied = None
    for score, mask in sorted(scored_masks, key=lambda item: item[0], reverse=True):
        clean = mask.astype(bool)
        if occupied is None:
            occupied = np.zeros_like(clean, dtype=bool)
        clean = clean & ~occupied
        if clean.sum() < CFG.min_component_size:
            continue
        occupied |= clean
        kept.append((score, clean.astype(np.uint8)))
    return kept


def predict_yolo(model, image_path, conf):
    result = model.predict(
        str(image_path),
        imgsz=CFG.imgsz,
        conf=conf,
        agnostic_nms=True,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.masks is None:
        return [], []
    masks = [mask.cpu().numpy().astype(np.uint8) for mask in result.masks.data]
    scores = result.boxes.conf.detach().cpu().numpy().tolist()
    return masks, scores


def predict_instances(model, image_path, conf, refiner, mode="refined_constrained"):
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    masks, scores = predict_yolo(model, image_path, conf)
    if not masks:
        return []
    if mode == "yolo_only":
        pred_masks = masks
    elif mode == "refined_free":
        pred_masks = refine_masks(image_gray, masks, refiner, constrain_to_yolo=False)
    else:
        pred_masks = refine_masks(image_gray, masks, refiner, constrain_to_yolo=True)
    scored = [(score, mask) for score, mask in zip(scores, pred_masks) if mask.sum() >= CFG.min_component_size]
    return [mask for _, mask in make_non_overlapping(scored)]


def pairwise_iou(gt_masks, pred_masks):
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt = [mask_utils.encode(np.asfortranarray(mask.astype(bool).astype(np.uint8))) for mask in gt_masks]
    pred = [mask_utils.encode(np.asfortranarray(mask.astype(bool).astype(np.uint8))) for mask in pred_masks]
    return mask_utils.iou(pred, gt, [0] * len(gt)).T


def all_validation_annotator_ids(valid_ids, images):
    valid_files = {images[image_id]["file_name"] for image_id in valid_ids}
    return [image_id for image_id, info in images.items() if info["file_name"] in valid_files]


def pq_for_image(gt_masks, pred_masks, threshold=0.5):
    iou = pairwise_iou(gt_masks, pred_masks)
    if iou.size == 0:
        tp = 0
        fp = len(pred_masks)
        fn = len(gt_masks)
        return 0.0, tp, fp, fn
    hits = iou > threshold
    tp_scores = iou[hits]
    fp = int((hits.sum(axis=0) == 0).sum())
    fn = int((hits.sum(axis=1) == 0).sum())
    denom = len(tp_scores) + 0.5 * fp + 0.5 * fn
    return (float(tp_scores.sum() / denom) if denom else 0.0), len(tp_scores), fp, fn


def validation_sweep(yolo_model, refiner, valid_ids, images, annotations):
    confs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    modes = ["yolo_only", "refined_constrained", "refined_free"]
    best = Candidate(0.0, 0.25, "refined_constrained", 0, 0, 0)
    candidates = []
    lines = []
    for mode in modes:
        for conf in confs:
            iou_sum = 0.0
            tp_total = fp_total = fn_total = 0
            for image_id in tqdm(valid_ids, desc=f"val {mode} conf {conf}"):
                info = images[image_id]
                image_path = CFG.train_dir / info["file_name"]
                gt_masks = []
                for ann in annotations.get(image_id, []):
                    mask = annotation_mask((info["height"], info["width"]), ann)
                    if mask.any():
                        gt_masks.append(mask)
                pred_masks = [
                    mask
                    for mask in predict_instances(yolo_model, image_path, conf, refiner, mode=mode)
                    if mask.sum() >= CFG.min_component_size
                ]
                pq, tp, fp, fn = pq_for_image(gt_masks, pred_masks)
                denom = tp + 0.5 * fp + 0.5 * fn
                iou_sum += pq * denom
                tp_total += tp
                fp_total += fp
                fn_total += fn
            denominator = tp_total + 0.5 * fp_total + 0.5 * fn_total
            final_pq = iou_sum / denominator if denominator else 0.0
            candidate = Candidate(final_pq, conf, mode, tp_total, fp_total, fn_total)
            candidates.append(candidate)
            line = (
                f"mode={mode} conf={conf:.2f} val_pq={final_pq:.6f} "
                f"tp={tp_total} fp={fp_total} fn={fn_total} "
                f"precision={candidate.precision:.6f} recall={candidate.recall:.6f}"
            )
            print(line)
            lines.append(line)
            if final_pq > best.pq:
                best = candidate
    Path("/kaggle/working/validation_variants.txt").write_text("\n".join(lines) + "\n")
    ranked = sorted(candidates, key=lambda item: item.pq, reverse=True)
    pd.DataFrame(
        [
            {
                "rank": index + 1,
                "mode": candidate.mode,
                "conf": candidate.conf,
                "val_pq": candidate.pq,
                "tp": candidate.tp,
                "fp": candidate.fp,
                "fn": candidate.fn,
                "precision": candidate.precision,
                "recall": candidate.recall,
            }
            for index, candidate in enumerate(ranked)
        ]
    ).to_csv("/kaggle/working/validation_candidates.csv", index=False)
    return best, ranked


def write_validation_image_diagnostics(yolo_model, refiner, candidate, valid_ids, images, annotations):
    rows = []
    for image_id in tqdm(valid_ids, desc="validation image diagnostics"):
        info = images[image_id]
        image_path = CFG.train_dir / info["file_name"]
        gt_masks = []
        for ann in annotations.get(image_id, []):
            mask = annotation_mask((info["height"], info["width"]), ann)
            if mask.any():
                gt_masks.append(mask)
        pred_masks = [
            mask
            for mask in predict_instances(yolo_model, image_path, candidate.conf, refiner, mode=candidate.mode)
            if mask.sum() >= CFG.min_component_size
        ]
        pq, tp, fp, fn = pq_for_image(gt_masks, pred_masks)
        rows.append(
            {
                "image_id": image_id,
                "file_name": info["file_name"],
                "pq": pq,
                "gt_count": len(gt_masks),
                "pred_count": len(pred_masks),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    pd.DataFrame(rows).sort_values(["pq", "fn", "fp"], ascending=[True, False, False]).to_csv(
        "/kaggle/working/validation_image_diagnostics.csv",
        index=False,
    )


def choose_submission_candidates(ranked):
    chosen = []
    seen = set()

    def add(candidate):
        key = (candidate.mode, candidate.conf)
        if key not in seen:
            chosen.append(candidate)
            seen.add(key)

    for candidate in ranked:
        add(candidate)
        if len(chosen) >= CFG.top_submission_candidates:
            break

    best = ranked[0]
    near_best = [
        candidate
        for candidate in ranked
        if best.pq - candidate.pq <= CFG.public_hedge_margin
    ]
    add(max(near_best, key=lambda item: (item.recall, item.pq)))
    add(max(near_best, key=lambda item: (item.precision, item.pq)))

    yolo_only = [candidate for candidate in ranked if candidate.mode == "yolo_only"]
    if yolo_only:
        add(yolo_only[0])
    return chosen


def write_submission(yolo_model, refiner, conf, mode, output_name="submission.csv"):
    rows = []
    per_image_counts = []
    test_files = sorted(CFG.test_dir.glob("*.jpeg"))
    for image_path in tqdm(test_files, desc="test inference"):
        counter = 1
        image_count = 0
        for mask in predict_instances(yolo_model, image_path, conf, refiner, mode=mode):
            if mask.sum() < CFG.min_component_size:
                continue
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rows.append(
                {
                    "filament_id": f"{image_path.stem}_{counter}",
                    "segmentation_rle": rle["counts"].decode("utf-8"),
                }
            )
            counter += 1
            image_count += 1
        per_image_counts.append(image_count)
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    output_path = Path("/kaggle/working") / output_name
    submission.to_csv(output_path, index=False)
    counts = np.array(per_image_counts, dtype=np.int32)
    diagnostics_name = output_path.with_suffix("").name + "_diagnostics.txt"
    Path("/kaggle/working", diagnostics_name).write_text(
        "\n".join(
            [
                f"file={output_name}",
                f"rows={len(submission)}",
                f"test_images={len(test_files)}",
                f"populated_images={int((counts > 0).sum())}",
                f"missing_images={int((counts == 0).sum())}",
                f"mean_instances={float(counts.mean()) if len(counts) else 0:.4f}",
                f"median_instances={float(np.median(counts)) if len(counts) else 0:.4f}",
                f"max_instances={int(counts.max()) if len(counts) else 0}",
                f"conf={conf}",
                f"mode={mode}",
            ]
        )
        + "\n"
    )
    print(submission.head())
    print(f"{output_name} rows={len(submission)} conf={conf} mode={mode}")


def main():
    seed_everything(CFG.seed)
    coco, images, annotations = load_coco()
    train_ids, valid_ids = prepare_yolo_data(coco, images, annotations)
    evaluation_ids = all_validation_annotator_ids(valid_ids, images)
    assert not (
        {images[image_id]["file_name"] for image_id in train_ids}
        & {images[image_id]["file_name"] for image_id in evaluation_ids}
    ), "Training and validation files overlap"
    Path("/kaggle/working/run_manifest.json").write_text(json.dumps({
        "experiment": "single_class_flips_coco_targets_all_annotator_validation",
        "config": {key: value for key, value in vars(CFG).items() if not key.startswith("_")},
        "versions": {name: version(name) for name in ["ultralytics", "torch", "torchvision", "pycocotools", "numpy"]},
        "train_ids": train_ids,
        "validation_best_ids": valid_ids,
        "validation_all_ids": evaluation_ids,
        "selection_metric": "all_annotator_global_pq",
        "target_rasterization": "COCO polygons via frPyObjects/merge/decode",
    }, indent=2, default=str))
    device = "0,1" if torch.cuda.device_count() > 1 else (0 if torch.cuda.is_available() else "cpu")
    print(f"torch_cuda={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()} device={device}")

    yolo_model = YOLO(CFG.model)
    yolo_model.train(
        data=str(CFG.yolo_dir / "data.yaml"),
        epochs=CFG.epochs,
        imgsz=CFG.imgsz,
        batch=CFG.batch,
        device=device,
        patience=CFG.patience,
        seed=CFG.seed,
        single_cls=CFG.single_class,
        mask_ratio=CFG.mask_ratio,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        fliplr=CFG.fliplr,
        flipud=CFG.flipud,
        mosaic=0.3,
        degrees=10,
        scale=0.3,
        hsv_v=0.3,
        close_mosaic=5,
        project="/kaggle/working/runs_filament",
        name="yolo_seg_optimized",
    )
    best_weights = Path(yolo_model.trainer.best)
    shutil.copy(best_weights, "/kaggle/working/yolo_best.pt")
    yolo_model = YOLO("/kaggle/working/yolo_best.pt")

    refiner = train_refiner(train_ids, valid_ids, images, annotations)
    best, ranked_candidates = validation_sweep(yolo_model, refiner, evaluation_ids, images, annotations)
    Path("/kaggle/working/validation_summary.txt").write_text(
        "\n".join(
            [
                f"best_conf={best.conf}",
                f"best_pq={best.pq}",
                f"best_mode={best.mode}",
                f"best_tp={best.tp}",
                f"best_fp={best.fp}",
                f"best_fn={best.fn}",
                f"best_precision={best.precision}",
                f"best_recall={best.recall}",
                "evaluation=all_annotators",
                f"validation_files={len(valid_ids)}",
                f"validation_annotator_entries={len(evaluation_ids)}",
            ]
        )
        + "\n"
    )
    write_validation_image_diagnostics(yolo_model, refiner, best, evaluation_ids, images, annotations)
    for index, candidate in enumerate(choose_submission_candidates(ranked_candidates), start=1):
        name = "submission.csv" if index == 1 else f"submission_candidate_{index}.csv"
        write_submission(yolo_model, refiner, candidate.conf, candidate.mode, output_name=name)


if __name__ == "__main__":
    main()
