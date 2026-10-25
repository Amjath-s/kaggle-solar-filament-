# Solar Filament Segmentation

Local ML research project for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Status

- Env: **ready** (CUDA torch + Jupyter kernel `Python (solar-filament)`)
- Data: **downloaded** under `data/raw/MAGFiLO_1.0_Kaggle_2026/`
- V0: guided data understanding — use `notebooks/01_dataset_exploration.ipynb`
- Final documentation: deferred

## Installation

```bash
cd solar-filament-segmentation
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kernel for notebooks: select  
`solar-filament-segmentation\.venv\Scripts\python.exe`

## Create split

```bash
python scripts/create_split.py --val-fraction 0.2 --seed 42
```

## Train V1 baseline

```bash
python scripts/train.py --config configs/v1.yaml
```

Checkpoints: `models/v1_baseline/best.pt`  
Logs: `experiments/v1/metrics.json`

## Train V2 (768)

```bash
python scripts/train.py --config configs/v2.yaml
```

## Make Kaggle submission (V2)

```bash
python scripts/make_submission.py --checkpoint models/v2_res768/best.pt --config configs/v2.yaml
```

Writes `submissions/submission_v2.csv` (`filament_id`, `segmentation_rle`). Upload manually on Kaggle — this repo does not auto-submit.

