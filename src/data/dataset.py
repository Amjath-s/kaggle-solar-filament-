"""PyTorch dataset for filament binary segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.annotations import build_file_to_anns, load_coco, polygons_to_mask
from src.data.split import read_split_list


class FilamentSegDataset(Dataset):
    def __init__(
        self,
        file_names: list[str],
        images_dir: Path,
        annotations_path: Path,
        image_size: int = 512,
    ) -> None:
        self.file_names = list(file_names)
        self.images_dir = Path(images_dir)
        self.image_size = int(image_size)
        coco = load_coco(Path(annotations_path))
        self.file_to_anns = build_file_to_anns(coco)

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        name = self.file_names[idx]
        path = self.images_dir / name
        img = Image.open(path).convert("L")
        w, h = img.size
        anns = self.file_to_anns.get(name, [])
        mask = polygons_to_mask(anns, height=h, width=w)

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_img = Image.fromarray(mask * 255, mode="L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        mask = (np.asarray(mask_img) > 127).astype(np.float32)

        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(img_arr).unsqueeze(0)  # 1,H,W
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # 1,H,W
        return {"image": image_t, "mask": mask_t, "file_name": name}


def dataset_from_split(
    split: str,
    *,
    images_dir: Path,
    annotations_path: Path,
    splits_dir: Path,
    image_size: int = 512,
) -> FilamentSegDataset:
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    names = read_split_list(Path(splits_dir) / f"{split}.txt")
    return FilamentSegDataset(
        file_names=names,
        images_dir=images_dir,
        annotations_path=annotations_path,
        image_size=image_size,
    )
