# Dataset

Status: local files present under `data/raw/`. Guided V0 understanding continues with you.

## Source

- Competition: Solar Filament Segmentation Challenge 2026
- Kaggle slug: `filament-segmentation-2026`
- Local path: `data/raw/MAGFiLO_1.0_Kaggle_2026/`
- Zip: `data/raw/filament-segmentation-2026.zip` (~671 MB downloaded)

## Inventory (from local files only)

| Item | Value |
|------|--------|
| Train images on disk | **707** (`.jpeg` under `train/train_images/`) |
| Test images on disk | **180** (`.jpeg` under `test/test_images/`) |
| Sample image size / mode | **2048 × 2048**, mode `L` (grayscale) — one sample checked |
| Train annotations file | `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` (~48.6 MB) |
| Annotation top-level keys | `info`, `licenses`, `categories`, `images`, `annotations` (COCO-like) |
| `images` entries in JSON | **1154** |
| `annotations` entries in JSON | **8199** |
| `categories` length | **4** |
| Sample submission on disk | Not in zip; format from Kaggle overview |
| Official metric | Mean Dice + Panoptic Quality style (overlap matching of instances) |

## Kaggle submission format

CSV columns:

- `filament_id` — e.g. `20150125172714Mh_1` (`{image_stem}_{k}`)
- `segmentation_rle` — COCO RLE **counts only** (size fixed 2048×2048)

Build locally (V2):

```bash
python scripts/make_submission.py --checkpoint models/v2_res768/best.pt --config configs/v2.yaml
```

Output: `submissions/submission_v2.csv` (upload manually on Kaggle).

## Train / validation split (local)

Created with `python scripts/create_split.py --val-fraction 0.2 --seed 42`.

| Split | Count | File |
|-------|-------|------|
| Train | **566** | `data/splits/train.txt` |
| Val | **141** | `data/splits/val.txt` |
| Total unique train images | **707** | |
| Kaggle test | **180** | untouched (`test/test_images`) |

Logic: unique on-disk `file_name`s only (not the 1154 duplicated JSON `images` rows).  
Helpers: `src/data/split.py`, runner: `scripts/create_split.py`.

## Open questions for guided V0

- Exact polygon / RLE fields for **submission** (train polys are clear)
- Sample submission CSV schema
- Official evaluation metric definition
- Semantic vs instance formulation from categories + evaluation

## Notes from guided V0

(Add discoveries as we go under your direction.)
Categories: {1: 'Left', 2: 'Right', 3: 'Unidentifiable', 4: 'Ambiguous'}

Total annotations (filaments): 8199
Category counts:
  1 (Left): 2535
  2 (Right): 2590
  3 (Unidentifiable): 3074

Area (from JSON field 'area'):
  min=9.0  median=1228.0  mean=2119.8  max=37739.0

Filaments per image_id (annotation side):
  images_with_anns=1154  min=1  median=7.0  mean=7.10  max=26

JSON images entries: 1154
Unique file_name in JSON: 707
JPEGs on disk: 707
On disk & in JSON: 707
In JSON but missing on disk: 0
On disk but not in JSON: 0
Duplicate file_name rows in JSON: 447

Annotations linked to on-disk images: 8199

**Resolved 1154 vs 707:** JSON has 447 duplicate `file_name` rows (same JPEG listed more than once with different `id`s). All 707 disk files are present; nothing missing. Use **unique file names** for the dataset/split (done above). Category **Ambiguous (4)** did not appear in train annotation counts.