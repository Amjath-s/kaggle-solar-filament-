"""Online 1152-tile dataset with filament-biased sampling for V8-A."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.annotations import build_file_to_anns, load_coco, polygons_to_mask


def _boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological edge of binary mask (uint8 0/1) → float32 0/1."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    edge = (m - eroded).astype(np.float32)
    return edge


def _edt_mean_width(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return float(dist[m > 0].mean() * 2.0)


def _tile_origins(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(y, x) for y in ys for x in xs]


def soft_tile_weights(tile: int, overlap: int) -> np.ndarray:
    """Center-weighted 2D blend weights (raised-cosine in overlap bands)."""
    w = np.ones((tile, tile), dtype=np.float32)
    if overlap <= 0:
        return w
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32)))
    for i, r in enumerate(ramp):
        w[i, :] *= r
        w[-(i + 1), :] *= r
        w[:, i] *= r
        w[:, -(i + 1)] *= r
    return w


class V8TileDataset(Dataset):
    """Sample 1152 crops from native 2048 images with filament-biased mix."""

    BUCKETS = ("fg", "boundary", "thin_small", "hard_neg")

    def __init__(
        self,
        file_names: list[str],
        images_dir: Path,
        annotations_path: Path,
        tile_size: int = 1152,
        overlap: int = 128,
        samples_per_epoch: int = 800,
        mix: dict | None = None,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.samples_per_epoch = int(samples_per_epoch)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.mix = mix or {"fg": 0.50, "boundary": 0.20, "thin_small": 0.15, "hard_neg": 0.15}

        coco = load_coco(Path(annotations_path))
        file_to_anns = build_file_to_anns(coco)
        self.file_names = [f for f in file_names if f in file_to_anns]
        self.file_to_anns = {f: file_to_anns[f] for f in self.file_names}

        # Precompute instance stats for thin/small targeting (primary anns per file).
        self._inst_meta: dict[str, list[dict]] = {}
        widths: list[float] = []
        areas: list[float] = []
        for fn in self.file_names:
            anns = self.file_to_anns[fn]
            # Use full mask size from first load lazily; store ann-level area from COCO.
            metas = []
            for ann in anns:
                area = float(ann.get("area", 0.0))
                metas.append({"ann": ann, "area": area, "width": None})
                if area > 0:
                    areas.append(area)
            self._inst_meta[fn] = metas
        self._area_t1 = float(np.percentile(areas, 33)) if areas else 200.0
        self._area_t2 = float(np.percentile(areas, 67)) if areas else 800.0

        # Cache full masks lightly? Too big for RAM — load per sample.
        # Build FG / empty image lists for hard-neg.
        self.fg_files: list[str] = []
        self.empty_files: list[str] = []
        for fn in self.file_names:
            if any(float(a.get("area", 0)) > 0 for a in self.file_to_anns[fn]):
                self.fg_files.append(fn)
            else:
                self.empty_files.append(fn)
        if not self.fg_files:
            self.fg_files = list(self.file_names)
        if not self.empty_files:
            self.empty_files = list(self.file_names)

        self._logged_mix = {
            "fg": self.mix["fg"],
            "boundary": self.mix["boundary"],
            "thin_small": self.mix["thin_small"],
            "hard_neg": self.mix["hard_neg"],
            "n_fg_files": len(self.fg_files),
            "n_empty_files": len(self.empty_files),
            "samples_per_epoch": self.samples_per_epoch,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "area_tertile_33": self._area_t1,
            "area_tertile_67": self._area_t2,
        }

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_image_mask(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.images_dir / file_name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        mask = polygons_to_mask(self.file_to_anns[file_name], h, w).astype(np.float32)
        return img.astype(np.float32) / 255.0, mask

    def _pick_bucket(self) -> str:
        keys = list(self.BUCKETS)
        p = np.array([self.mix[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        return str(self.rng.choice(keys, p=p))

    def _crop(self, img: np.ndarray, mask: np.ndarray, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        t = self.tile_size
        h, w = img.shape[:2]
        y0 = int(np.clip(y0, 0, max(h - t, 0)))
        x0 = int(np.clip(x0, 0, max(w - t, 0)))
        return img[y0 : y0 + t, x0 : x0 + t], mask[y0 : y0 + t, x0 : x0 + t]

    def _sample_fg_tile(self, prefer_boundary: bool = False, prefer_thin: bool = False):
        fn = str(self.rng.choice(self.fg_files))
        img, mask = self._load_image_mask(fn)
        h, w = img.shape[:2]
        t = self.tile_size
        ys, xs = np.where(mask > 0.5)
        if len(ys) == 0:
            y0 = int(self.rng.integers(0, max(h - t, 1)))
            x0 = int(self.rng.integers(0, max(w - t, 1)))
            return self._crop(img, mask, y0, x0), "fg_fallback"

        if prefer_thin:
            # Bias toward small-area instance centroids via COCO bbox.
            anns = self.file_to_anns[fn]
            small = [a for a in anns if float(a.get("area", 0)) <= self._area_t1 and a.get("bbox")]
            if small:
                ann = small[int(self.rng.integers(0, len(small)))]
                bx, by, bw, bh = ann["bbox"]
                cy = int(by + bh / 2)
                cx = int(bx + bw / 2)
                y0 = cy - t // 2
                x0 = cx - t // 2
                tile_img, tile_mask = self._crop(img, mask, y0, x0)
                # Prefer thin EDT if available.
                if tile_mask.sum() > 0 and _edt_mean_width(tile_mask) > 12 and len(small) > 1:
                    pass
                return (tile_img, tile_mask), "thin_small"

        if prefer_boundary:
            edge = _boundary_from_mask(mask)
            eys, exs = np.where(edge > 0.5)
            if len(eys) > 0:
                i = int(self.rng.integers(0, len(eys)))
                y0 = int(eys[i] - t // 2)
                x0 = int(exs[i] - t // 2)
                return self._crop(img, mask, y0, x0), "boundary"

        i = int(self.rng.integers(0, len(ys)))
        y0 = int(ys[i] - t // 2)
        x0 = int(xs[i] - t // 2)
        return self._crop(img, mask, y0, x0), "fg"

    def _sample_hard_neg(self):
        # Dark background tile with little/no FG; prefer empty images or low-mask crops.
        if self.rng.random() < 0.5 and self.empty_files:
            fn = str(self.rng.choice(self.empty_files))
        else:
            fn = str(self.rng.choice(self.file_names))
        img, mask = self._load_image_mask(fn)
        h, w = img.shape[:2]
        t = self.tile_size
        best = None
        for _ in range(12):
            y0 = int(self.rng.integers(0, max(h - t, 1)))
            x0 = int(self.rng.integers(0, max(w - t, 1)))
            crop_i, crop_m = self._crop(img, mask, y0, x0)
            fg_frac = float(crop_m.mean())
            mean_i = float(crop_i.mean())
            score = fg_frac * 10.0 + mean_i  # prefer dark + low FG
            if best is None or score < best[0]:
                best = (score, crop_i, crop_m)
        assert best is not None
        return (best[1], best[2]), "hard_neg"

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.rng.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if self.rng.random() < 0.5:
            angle = float(self.rng.uniform(-15.0, 15.0))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # brightness/contrast ±10%
        alpha = float(self.rng.uniform(0.90, 1.10))
        beta = float(self.rng.uniform(-0.10, 0.10))
        img = np.clip(img * alpha + beta, 0.0, 1.0)
        if self.rng.random() < 0.5:
            noise = self.rng.normal(0.0, 0.02, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
        return img.astype(np.float32), (mask > 0.5).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        bucket = self._pick_bucket()
        if bucket == "hard_neg":
            (img, mask), tag = self._sample_hard_neg()
        elif bucket == "boundary":
            (img, mask), tag = self._sample_fg_tile(prefer_boundary=True)
        elif bucket == "thin_small":
            (img, mask), tag = self._sample_fg_tile(prefer_thin=True)
        else:
            (img, mask), tag = self._sample_fg_tile()

        if self.augment:
            img, mask = self._augment(img, mask)
        else:
            mask = (mask > 0.5).astype(np.float32)

        boundary = _boundary_from_mask(mask)
        return {
            "image": torch.from_numpy(img[None, ...]).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "boundary": torch.from_numpy(boundary[None, ...]).float(),
            "bucket": tag,
        }


class V8FullImageDataset(Dataset):
    """Full native-resolution images for stitch validation / PQ."""

    def __init__(
        self,
        file_names: list[str],
        images_dir: Path,
        annotations_path: Path,
        max_images: int | None = None,
    ) -> None:
        coco = load_coco(Path(annotations_path))
        file_to_anns = build_file_to_anns(coco)
        names = [f for f in file_names if f in file_to_anns]
        if max_images is not None:
            names = names[: int(max_images)]
        self.file_names = names
        self.images_dir = Path(images_dir)
        self.file_to_anns = {f: file_to_anns[f] for f in names}

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        fn = self.file_names[idx]
        path = self.images_dir / fn
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        mask = polygons_to_mask(self.file_to_anns[fn], h, w).astype(np.float32)
        return {
            "image": torch.from_numpy((img.astype(np.float32) / 255.0)[None, ...]).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "file_name": fn,
            "height": h,
            "width": w,
        }


def load_split_names(splits_dir: Path, split: str) -> list[str]:
    path = Path(splits_dir) / f"{split}.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def dump_mix_log(path: Path, dataset: V8TileDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset._logged_mix, indent=2), encoding="utf-8")
