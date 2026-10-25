"""On-the-fly GT local crop dataset for V6-A local U-Net."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.annotations import (
    ann_to_instance_mask,
    build_file_to_anns_primary,
    load_coco,
    polygons_to_mask,
)
from src.data.split import read_split_list
from src.inference.local_crops import expand_xyxy, mask_to_xyxy, resize_image_mask


def _apply_augs(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    hflip: bool,
    vflip: bool,
    rotate_deg: float,
    translate_frac: float,
    scale_range: tuple[float, float],
    p: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Physically mild augs on float image [0,1] and float mask {0,1}."""
    img = image.copy()
    m = mask.copy()
    h, w = img.shape[:2]

    if hflip and rng.random() < p:
        img = np.ascontiguousarray(np.fliplr(img))
        m = np.ascontiguousarray(np.fliplr(m))
    if vflip and rng.random() < p:
        img = np.ascontiguousarray(np.flipud(img))
        m = np.ascontiguousarray(np.flipud(m))

    angle = 0.0
    if rotate_deg > 0 and rng.random() < p:
        angle = rng.uniform(-rotate_deg, rotate_deg)

    scale = 1.0
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        if rng.random() < p:
            scale = rng.uniform(scale_range[0], scale_range[1])

    tx = ty = 0.0
    if translate_frac > 0 and rng.random() < p:
        tx = rng.uniform(-translate_frac, translate_frac) * w
        ty = rng.uniform(-translate_frac, translate_frac) * h

    if abs(angle) > 1e-3 or abs(scale - 1.0) > 1e-3 or abs(tx) > 1e-3 or abs(ty) > 1e-3:
        # PIL affine: expand=False keeps size
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        m_u8 = (m > 0.5).astype(np.uint8) * 255
        pil_i = Image.fromarray(img_u8, mode="L")
        pil_m = Image.fromarray(m_u8, mode="L")
        # Center-ish affine via rotate+resize+paste for simplicity
        if abs(angle) > 1e-3:
            pil_i = pil_i.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            pil_m = pil_m.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        if abs(scale - 1.0) > 1e-3:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            pil_i = pil_i.resize((nw, nh), Image.BILINEAR)
            pil_m = pil_m.resize((nw, nh), Image.NEAREST)
            canvas_i = Image.new("L", (w, h), 0)
            canvas_m = Image.new("L", (w, h), 0)
            ox = (w - nw) // 2 + int(round(tx))
            oy = (h - nh) // 2 + int(round(ty))
            canvas_i.paste(pil_i, (ox, oy))
            canvas_m.paste(pil_m, (ox, oy))
            pil_i, pil_m = canvas_i, canvas_m
        elif abs(tx) > 1e-3 or abs(ty) > 1e-3:
            canvas_i = Image.new("L", (w, h), 0)
            canvas_m = Image.new("L", (w, h), 0)
            canvas_i.paste(pil_i, (int(round(tx)), int(round(ty))))
            canvas_m.paste(pil_m, (int(round(tx)), int(round(ty))))
            pil_i, pil_m = canvas_i, canvas_m
        img = np.asarray(pil_i, dtype=np.float32) / 255.0
        m = (np.asarray(pil_m) > 127).astype(np.float32)

    return img, m


class LocalCropDataset(Dataset):
    """Positive GT-instance crops + matched negative background crops."""

    def __init__(
        self,
        file_names: list[str],
        images_dir: Path,
        annotations_path: Path,
        *,
        image_size: int = 512,
        pad_frac: float = 0.20,
        neg_ratio: float = 1.0,
        neg_max_fg_frac: float = 0.01,
        min_pos_area: int = 32,
        augment: bool = False,
        aug_cfg: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.image_size = int(image_size)
        self.pad_frac = float(pad_frac)
        self.neg_ratio = float(neg_ratio)
        self.neg_max_fg_frac = float(neg_max_fg_frac)
        self.min_pos_area = int(min_pos_area)
        self.augment = bool(augment)
        self.aug_cfg = aug_cfg or {}
        self.rng = random.Random(int(seed))

        coco = load_coco(Path(annotations_path))
        self.file_to_anns = build_file_to_anns_primary(coco)

        # Build crop index: list of (file_name, kind, ann_idx_or_-1)
        self.items: list[tuple[str, str, int]] = []
        for name in file_names:
            anns = self.file_to_anns.get(name, [])
            n_pos = 0
            for i, ann in enumerate(anns):
                # cheap area filter via bbox if present
                bbox = ann.get("bbox")  # COCO xywh
                if bbox is not None and len(bbox) >= 4:
                    if float(bbox[2]) * float(bbox[3]) < self.min_pos_area:
                        continue
                self.items.append((name, "pos", i))
                n_pos += 1
            n_neg = int(round(n_pos * self.neg_ratio))
            for _ in range(n_neg):
                self.items.append((name, "neg", -1))

        self._cache_img: dict[str, np.ndarray] = {}
        self._cache_union: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, name: str) -> np.ndarray:
        if name not in self._cache_img:
            arr = np.asarray(Image.open(self.images_dir / name).convert("L"), dtype=np.uint8)
            self._cache_img[name] = arr
            # Limit cache size roughly
            if len(self._cache_img) > 8:
                oldest = next(iter(self._cache_img))
                if oldest != name:
                    self._cache_img.pop(oldest, None)
                    self._cache_union.pop(oldest, None)
        return self._cache_img[name]

    def _union_mask(self, name: str, h: int, w: int) -> np.ndarray:
        if name not in self._cache_union:
            anns = self.file_to_anns.get(name, [])
            self._cache_union[name] = polygons_to_mask(anns, height=h, width=w)
        return self._cache_union[name]

    def _sample_neg_box(self, union: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w = union.shape
        # Sample square-ish crop size similar to median filament scale
        side = int(rng.uniform(64, 256))
        for _ in range(40):
            x1 = rng.randint(0, max(0, w - side))
            y1 = rng.randint(0, max(0, h - side))
            x2, y2 = min(w, x1 + side), min(h, y1 + side)
            patch = union[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            if float(patch.mean()) <= self.neg_max_fg_frac:
                return np.array([x1, y1, x2, y2], dtype=np.int32)
        # fallback: top-left
        return np.array([0, 0, min(side, w), min(side, h)], dtype=np.int32)

    def __getitem__(self, idx: int) -> dict:
        name, kind, ann_idx = self.items[idx]
        img = self._load_image(name)
        h, w = img.shape[:2]
        anns = self.file_to_anns.get(name, [])
        union = self._union_mask(name, h, w)

        # per-worker deterministic-ish RNG
        rng = random.Random((self.rng.randint(0, 10**9) + idx) & 0xFFFFFFFF)

        if kind == "pos":
            ann = anns[ann_idx]
            inst = ann_to_instance_mask(ann, height=h, width=w)
            box0 = mask_to_xyxy(inst)
            if box0 is None:
                # empty — treat as neg
                box = self._sample_neg_box(union, rng)
                crop_img = img[box[1] : box[3], box[0] : box[2]]
                crop_mask = np.zeros_like(crop_img, dtype=np.uint8)
            else:
                box = expand_xyxy(box0, self.pad_frac, w, h)
                crop_img = img[box[1] : box[3], box[0] : box[2]]
                crop_mask = inst[box[1] : box[3], box[0] : box[2]]
        else:
            box = self._sample_neg_box(union, rng)
            crop_img = img[box[1] : box[3], box[0] : box[2]]
            crop_mask = np.zeros_like(crop_img, dtype=np.uint8)

        img_r, mask_r = resize_image_mask(crop_img, crop_mask, size=self.image_size)
        assert mask_r is not None

        if self.augment:
            img_r, mask_r = _apply_augs(
                img_r,
                mask_r,
                hflip=bool(self.aug_cfg.get("hflip", True)),
                vflip=bool(self.aug_cfg.get("vflip", True)),
                rotate_deg=float(self.aug_cfg.get("rotate_deg", 15)),
                translate_frac=float(self.aug_cfg.get("translate_frac", 0.05)),
                scale_range=tuple(self.aug_cfg.get("scale_range", [0.9, 1.1])),
                p=float(self.aug_cfg.get("p", 0.5)),
                rng=rng,
            )

        image_t = torch.from_numpy(img_r.astype(np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(mask_r.astype(np.float32)).unsqueeze(0)
        return {
            "image": image_t,
            "mask": mask_t,
            "file_name": name,
            "kind": kind,
        }


def local_crop_dataset_from_split(
    split: str,
    *,
    images_dir: Path,
    annotations_path: Path,
    splits_dir: Path,
    image_size: int = 512,
    pad_frac: float = 0.20,
    neg_ratio: float = 1.0,
    neg_max_fg_frac: float = 0.01,
    min_pos_area: int = 32,
    augment: bool = False,
    aug_cfg: dict | None = None,
    seed: int = 42,
) -> LocalCropDataset:
    names = read_split_list(Path(splits_dir) / f"{split}.txt")
    return LocalCropDataset(
        file_names=names,
        images_dir=images_dir,
        annotations_path=annotations_path,
        image_size=image_size,
        pad_frac=pad_frac,
        neg_ratio=neg_ratio,
        neg_max_fg_frac=neg_max_fg_frac,
        min_pos_area=min_pos_area,
        augment=augment and split == "train",
        aug_cfg=aug_cfg,
        seed=seed,
    )
