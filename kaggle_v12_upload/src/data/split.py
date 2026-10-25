"""Train/validation split helpers for unique on-disk training images.

The Kaggle test set under data/raw/.../test/ is never used here.
Splits are based on unique JPEG file names in train/train_images (707 files),
not on the duplicated rows in the COCO images list (1154 entries).
"""

from __future__ import annotations

import random
from pathlib import Path

DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.2


def project_root() -> Path:
    """solar-filament-segmentation/ (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def default_train_images_dir(root: Path | None = None) -> Path:
    root = root or project_root()
    return (
        root
        / "data"
        / "raw"
        / "MAGFiLO_1.0_Kaggle_2026"
        / "train"
        / "train_images"
    )


def default_splits_dir(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "data" / "splits"


def list_unique_train_images(train_images_dir: Path) -> list[str]:
    """Sorted unique JPEG file names present on disk."""
    if not train_images_dir.is_dir():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")
    names = sorted({p.name for p in train_images_dir.glob("*.jpeg")})
    if not names:
        # some dumps use .jpg
        names = sorted({p.name for p in train_images_dir.glob("*.jpg")})
    if not names:
        raise FileNotFoundError(f"No JPEG/JPG images in {train_images_dir}")
    return names


def make_train_val_split(
    file_names: list[str],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Return (train_names, val_names) with a reproducible shuffle."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    names = list(file_names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * val_fraction)))
    if n_val >= len(names):
        raise ValueError("val_fraction leaves no training images")
    val_names = sorted(names[:n_val])
    train_names = sorted(names[n_val:])
    return train_names, val_names


def write_split_list(path: Path, file_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(file_names) + ("\n" if file_names else ""), encoding="utf-8")


def read_split_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            lines.append(name)
    return lines


def create_and_save_split(
    *,
    root: Path | None = None,
    train_images_dir: Path | None = None,
    splits_dir: Path | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Create train/val lists and write train.txt / val.txt. Returns summary dict."""
    root = root or project_root()
    train_images_dir = train_images_dir or default_train_images_dir(root)
    splits_dir = splits_dir or default_splits_dir(root)

    all_names = list_unique_train_images(train_images_dir)
    train_names, val_names = make_train_val_split(
        all_names, val_fraction=val_fraction, seed=seed
    )

    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"
    write_split_list(train_path, train_names)
    write_split_list(val_path, val_names)

    # overlap check
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise RuntimeError(f"Train/val overlap ({len(overlap)} files)")

    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "n_total": len(all_names),
        "n_train": len(train_names),
        "n_val": len(val_names),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_images_dir": str(train_images_dir),
    }


def load_train_val_splits(splits_dir: Path | None = None) -> tuple[list[str], list[str]]:
    splits_dir = splits_dir or default_splits_dir()
    train = read_split_list(splits_dir / "train.txt")
    val = read_split_list(splits_dir / "val.txt")
    return train, val
