# V12 Kaggle upload — train YOLO11s-seg (all annotators, no flips)

Same workflow as V6-B / V7: upload **code dataset** → attach **competition MAGFiLO** → GPU notebook → download weights.

This job trains **only** the detector. Refiner / TTA come later on your PC.

## What it trains

| Knob | Value |
|------|------:|
| Model | **YOLO11s-seg** |
| imgsz | **1024** |
| Labels | **all annotators** (single class) |
| Train flips | **off** (`flipud=0`, `fliplr=0`) |
| Epochs | **80** (patience **20**) |
| Split | seed-42 **566 / 141** |

## Step 1 — Create private Kaggle Dataset from this folder

Edit `dataset-metadata.json`: keep or set `id` to `YOUR_USERNAME/solar-filament-v12-code`.

From a machine with the [Kaggle API](https://github.com/Kaggle/kaggle-api) configured:

```bash
cd solar-filament-segmentation
kaggle datasets create -p kaggle_v12_upload
```

Or zip `kaggle_v12_upload` and upload via **Kaggle → Datasets → New Dataset**.

Do **not** pack MAGFiLO JPEGs into this dataset.

## Step 2 — Create a Kaggle Notebook

Code → New Notebook. Prefer cells from **`kaggle_v12_train.ipynb`**.

## Step 3 — Add inputs

1. Official **Solar Filament Segmentation** competition / MAGFiLO train data  
2. This code dataset: **`solar-filament-v12-code`**

## Step 4 — Enable GPU

Settings → Accelerator → **GPU** (T4 or better).

## Step 5 — Run

```bash
# Optional
pip install -q -r /kaggle/input/solar-filament-v12-code/requirements.txt

# /kaggle/input is read-only — copy code to working
cp -r /kaggle/input/solar-filament-v12-code /kaggle/working/v12_code
cd /kaggle/working/v12_code
python run_v12_kaggle.py
```

Or run all cells in `kaggle_v12_train.ipynb` (they do the same).

Pretrained `yolo11s-seg.pt` is downloaded by Ultralytics into `/kaggle/working`. If download fails, attach `yolo11s-seg.pt` as an input dataset and re-run.

## Step 6 — What to download

After the run finishes, **Save** the notebook version.

From **Output** / `/kaggle/working` download:

| File | Why |
|------|-----|
| **`v12_export.zip`** | Preferred — contains `yolo11s_best.pt` + short README |

You do **not** need training images, labels, or the full Ultralytics run folder.

### Put weights on your PC

Unzip `v12_export.zip` and copy:

```text
yolo11s_best.pt
  → solar-filament-segmentation/experiments/v12/yolo11s/weights/best.pt
```

## Step 7 — Next (after download, on PC)

1. Confirm `experiments/v12/yolo11s/weights/best.pt` exists  
2. Agent next: consensus 4-flip TTA + ResNet18 refiner @256 + local PQ vs V7 0.379  
3. Submit CSV only if local PQ is strong (~0.42+)

## Local smoke test (this machine)

```bash
cd solar-filament-segmentation/kaggle_v12_upload
python run_v12_kaggle.py --dry-run
```

Full training is for Kaggle GPU, not a 4GB laptop.

## Folder layout

```text
kaggle_v12_upload/
├── dataset-metadata.json
├── README_KAGGLE_V12.md
├── kaggle_v12_train.ipynb
├── run_v12_kaggle.py
├── requirements.txt
├── configs/v12_train.yaml
├── scripts/export_yolo_seg_all.py
├── scripts/train_yolo11s_seg.py
├── splits/train.txt
├── splits/val.txt
└── src/...
```
