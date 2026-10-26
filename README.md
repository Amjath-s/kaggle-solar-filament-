# Solar Filament Segmentation (Kaggle 2026)

Code for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

**Final version: v16b** — Kaggle public score **0.38**, local val PQ **0.4586**.

## Pipeline

1. **YOLO11s-seg @1024** (trained on Kaggle GPU, all annotators, 80 epochs) proposes filament boxes + masks.
2. **4-flip consensus TTA** on the detector, conf 0.35.
3. **ResNet18 U-Net refiner @256** (trained locally, augmentation on) re-segments each box crop,
   4-flip TTA, threshold 0.50, clamped to the 7x-dilated YOLO seed.
4. Duplicate suppression (IoU 0.25, score = conf x mean refiner prob), non-overlapping masks, min area 80.

Full details: `experiments/v12/V12_ARCHITECTURE.md`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Data (not in this repo)

The competition data is not redistributed here. Put your Kaggle API key at
`C:\Users\<you>\.kaggle\kaggle.json` (Kaggle → Settings → Create New Token), accept the
competition rules on Kaggle, then:

```powershell
kaggle competitions download -c filament-segmentation-2026 -p data/raw
Expand-Archive data/raw/filament-segmentation-2026.zip -DestinationPath data/raw
```

Expected layout:

```
data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images/
data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/
```

The train/val split used for every local score is in `data/splits/train.txt` and `data/splits/val.txt`
(seed 42, 566 / 141 images).

## Weights

- YOLO detector: `experiments/v12/yolo11s/weights/best.pt` (20 MB, included).
- Refiner: `experiments/v16/r18aug_refiner/best_slim.pt` (~55 MB, included). This is the training
  checkpoint `best.pt` (172 MB) with the AdamW optimizer state removed; the fp32 weights are unchanged,
  so predictions are identical. To regenerate it from a (re)trained `best.pt`:

```powershell
python -u scripts/export_refiner_weights.py   # best.pt -> best_slim.pt, verifies tensors + size
```

## Train

**Detector (Kaggle GPU):** see `kaggle_v12_upload/README_KAGGLE_V12.md` — upload that folder as a
Kaggle dataset, attach the competition data, run `kaggle_v12_train.ipynb`, then copy the exported
`best.pt` to `experiments/v12/yolo11s/weights/best.pt`.

**Refiner (local GPU):** trains on YOLO proposal crops in `experiments/v12/crops`
(built from the detector, see `experiments/v12/V12_ARCHITECTURE.md`):

```powershell
python -u scripts/train_local_refiner.py --config experiments/v16/configs/r18aug_refiner.yaml
python -u scripts/export_refiner_weights.py
```

## Make the submission

```powershell
python -u scripts/make_submission_v16b.py             # val PQ check + test CSV
python -u scripts/make_submission_v16b.py --skip-val  # test CSV only
```

Output: `submissions/submission_v16b_v12yolo_r18aug_c035_t050_s025.csv`. All submissions and scores
are logged in `submissions/submission_log.csv`.
