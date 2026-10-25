"""Bundle / competition / work root detection for local + Kaggle (V12)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RUNNER_NAME = "run_v12_kaggle.py"


@dataclass
class PathRoots:
    bundle_root: Path
    competition_root: Path
    work_root: Path
    images_dir: Path
    annotations_path: Path
    splits_dir: Path

    def as_dict(self) -> dict:
        return {
            "BUNDLE_ROOT": str(self.bundle_root),
            "COMPETITION_ROOT": str(self.competition_root),
            "WORK_ROOT": str(self.work_root),
            "images_dir": str(self.images_dir),
            "annotations": str(self.annotations_path),
            "splits_dir": str(self.splits_dir),
        }


def _is_kaggle() -> bool:
    return Path("/kaggle/input").is_dir() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def find_bundle_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if (p / RUNNER_NAME).is_file():
            return p.resolve()
        raise FileNotFoundError(f"BUNDLE_ROOT missing {RUNNER_NAME}: {p}")

    env = os.environ.get("BUNDLE_ROOT")
    if env:
        p = Path(env)
        if (p / RUNNER_NAME).is_file():
            return p.resolve()

    # This file lives at bundle/src/utils/paths.py → parents[2] = bundle
    here = Path(__file__).resolve().parents[2]
    if (here / RUNNER_NAME).is_file():
        return here

    if Path("/kaggle/input").is_dir():
        for child in sorted(Path("/kaggle/input").iterdir()):
            if not child.is_dir():
                continue
            candidates = [child, *sorted(child.iterdir())]
            for c in candidates:
                if c.is_dir() and (c / RUNNER_NAME).is_file():
                    return c.resolve()
                if c.is_file() and c.name == RUNNER_NAME:
                    return c.parent.resolve()

    raise FileNotFoundError(
        f"Could not locate V12 bundle ({RUNNER_NAME}). "
        "Set BUNDLE_ROOT or attach the solar-filament-v12-code dataset."
    )


def _looks_like_competition(root: Path) -> tuple[Path, Path] | None:
    """Return (images_dir, annotations_json) if MAGFiLO layout is found under root."""
    patterns_img = [
        root / "train" / "train_images",
        root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images",
        root / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images",
        root / "train_images",
    ]
    patterns_ann = [
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        "annotations_train.json",
        "train_annotations.json",
    ]

    images_dir = None
    for p in patterns_img:
        if p.is_dir() and (any(p.glob("*.jpeg")) or any(p.glob("*.jpg"))):
            images_dir = p
            break
    if images_dir is None:
        for p in root.rglob("train_images"):
            if p.is_dir() and (any(p.glob("*.jpeg")) or any(p.glob("*.jpg"))):
                images_dir = p
                break
    if images_dir is None:
        return None

    ann = None
    search_roots = [images_dir.parent, root]
    for sr in search_roots:
        for name in patterns_ann:
            cand = sr / name
            if cand.is_file():
                ann = cand
                break
        if ann is None:
            for cand in sr.glob("*.json"):
                if "annot" in cand.name.lower() or "magfilo" in cand.name.lower():
                    ann = cand
                    break
        if ann is not None:
            break
    if ann is None:
        for cand in root.rglob("*Annotations*train*.json"):
            ann = cand
            break
        if ann is None:
            for cand in root.rglob("*annot*train*.json"):
                ann = cand
                break
    if ann is None or not ann.is_file():
        return None
    return images_dir, ann


def find_competition_root(
    bundle_root: Path,
    explicit: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return (competition_root, images_dir, annotations_path)."""
    if explicit is not None:
        root = Path(explicit)
        found = _looks_like_competition(root)
        if found is None:
            raise FileNotFoundError(f"COMPETITION_ROOT has no MAGFiLO images/annotations: {root}")
        images_dir, ann = found
        return root.resolve(), images_dir.resolve(), ann.resolve()

    env = os.environ.get("COMPETITION_ROOT")
    if env:
        root = Path(env)
        found = _looks_like_competition(root)
        if found is None:
            raise FileNotFoundError(f"COMPETITION_ROOT invalid: {root}")
        images_dir, ann = found
        return root.resolve(), images_dir.resolve(), ann.resolve()

    candidates: list[Path] = []
    if Path("/kaggle/input").is_dir():
        for child in sorted(Path("/kaggle/input").iterdir()):
            if child.is_dir() and child.resolve() != bundle_root.resolve():
                if (child / RUNNER_NAME).is_file():
                    continue
                candidates.append(child)
                for nested in child.iterdir():
                    if nested.is_dir():
                        candidates.append(nested)

    # Local project MAGFiLO (bundle lives under solar-filament-segmentation/)
    local = bundle_root.parent / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    if local.is_dir():
        candidates.insert(0, local)

    for root in candidates:
        found = _looks_like_competition(root)
        if found is not None:
            images_dir, ann = found
            return root.resolve(), images_dir.resolve(), ann.resolve()

    raise FileNotFoundError(
        "Could not find Solar Filament competition data under /kaggle/input or local data/raw. "
        "Attach the competition dataset and set COMPETITION_ROOT if needed."
    )


def find_work_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    env = os.environ.get("WORK_ROOT")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    if _is_kaggle() or Path("/kaggle/working").is_dir():
        p = Path("/kaggle/working")
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    p = Path(__file__).resolve().parents[2].parent / "experiments" / "v12_kaggle_work"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def resolve_roots(
    *,
    bundle_root: str | Path | None = None,
    competition_root: str | Path | None = None,
    work_root: str | Path | None = None,
) -> PathRoots:
    br = find_bundle_root(bundle_root)
    cr, images_dir, ann = find_competition_root(br, competition_root)
    wr = find_work_root(work_root)
    splits = br / "splits"
    if not splits.is_dir():
        raise FileNotFoundError(f"Missing splits dir: {splits}")
    return PathRoots(
        bundle_root=br,
        competition_root=cr,
        work_root=wr,
        images_dir=images_dir,
        annotations_path=ann,
        splits_dir=splits,
    )
