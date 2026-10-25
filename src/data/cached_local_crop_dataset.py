"""Cached 512 local-crop dataset for V6-A (reads experiments/v6/crops/)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
    img = image.copy()
    m = mask.copy()
    h, w = img.shape[:2]

    if hflip and rng.random() < p:
        img = np.ascontiguousarray(np.fliplr(img))
        m = np.ascontiguousarray(np.fliplr(m))
    if vflip and rng.random() < p:
        img = np.ascontiguousarray(np.flipud(img))
        m = np.ascontiguousarray(np.flipud(m))

    # Rotation only (skip slow scale/translate paste most of the time)
    if rotate_deg > 0 and rng.random() < p:
        angle = rng.uniform(-rotate_deg, rotate_deg)
        if abs(angle) > 1e-3:
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            m_u8 = (m > 0.5).astype(np.uint8) * 255
            pil_i = Image.fromarray(img_u8, mode="L").rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            pil_m = Image.fromarray(m_u8, mode="L").rotate(angle, resample=Image.NEAREST, fillcolor=0)
            img = np.asarray(pil_i, dtype=np.float32) / 255.0
            m = (np.asarray(pil_m) > 127).astype(np.float32)

    if scale_range != (1.0, 1.0) and rng.random() < (p * 0.5):
        scale = rng.uniform(scale_range[0], scale_range[1])
        if abs(scale - 1.0) > 1e-3:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            pil_i = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L").resize(
                (nw, nh), Image.BILINEAR
            )
            pil_m = Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L").resize((nw, nh), Image.NEAREST)
            canvas_i = Image.new("L", (w, h), 0)
            canvas_m = Image.new("L", (w, h), 0)
            ox, oy = (w - nw) // 2, (h - nh) // 2
            if translate_frac > 0 and rng.random() < p:
                ox += int(round(rng.uniform(-translate_frac, translate_frac) * w))
                oy += int(round(rng.uniform(-translate_frac, translate_frac) * h))
            canvas_i.paste(pil_i, (ox, oy))
            canvas_m.paste(pil_m, (ox, oy))
            img = np.asarray(canvas_i, dtype=np.float32) / 255.0
            m = (np.asarray(canvas_m) > 127).astype(np.float32)

    return img, m


class CachedLocalCropDataset(Dataset):
    def __init__(
        self,
        crop_root: Path,
        split: str,
        *,
        augment: bool = False,
        aug_cfg: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.split_dir = Path(crop_root) / split
        index_path = self.split_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing crop cache {index_path}. Run: python scripts/build_local_crops.py"
            )
        with index_path.open(encoding="utf-8") as f:
            data = json.load(f)
        self.items = data["items"]
        self.img_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        self.augment = bool(augment)
        self.aug_cfg = aug_cfg or {}
        self.seed = int(seed)
        self._draws = 0  # per-process counter; workers each get a copy, offset by idx so still varied

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        stem = rec["id"]
        # Optional "dir": crop lives in another cache split (used by k-fold indexes that mix train/val crops)
        base = Path(rec["dir"]) if "dir" in rec else self.split_dir
        img = np.asarray(Image.open(base / "images" / f"{stem}.png").convert("L"), dtype=np.float32) / 255.0
        mask = (np.asarray(Image.open(base / "masks" / f"{stem}.png").convert("L")) > 127).astype(np.float32)

        if self.augment:
            # Seed must change across epochs; (seed, idx) alone replays one fixed augmentation per crop forever.
            self._draws += 1
            rng = random.Random((self.seed * 1000003 + idx * 9973 + self._draws * 7919) & 0xFFFFFFFF)
            img, mask = _apply_augs(
                img,
                mask,
                hflip=bool(self.aug_cfg.get("hflip", True)),
                vflip=bool(self.aug_cfg.get("vflip", True)),
                rotate_deg=float(self.aug_cfg.get("rotate_deg", 15)),
                translate_frac=float(self.aug_cfg.get("translate_frac", 0.05)),
                scale_range=tuple(self.aug_cfg.get("scale_range", [0.9, 1.1])),
                p=float(self.aug_cfg.get("p", 0.5)),
                rng=rng,
            )

        return {
            "image": torch.from_numpy(img.astype(np.float32)).unsqueeze(0),
            "mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "file_name": rec.get("file_name", ""),
            "kind": rec.get("kind", ""),
        }
