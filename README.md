# Solar filament instance segmentation (Kaggle `filament-segmentation-2026`)

Instance masks of solar filaments on 2048x2048 full-disk H-alpha images (MAGFiLO, GONG network).
The pipeline is a YOLO11s-seg detector with a high-resolution mask head, followed by a crop
U-Net refiner and a score-ordered pixel-ownership step, so the output masks never overlap.

`notebooks/reproduce_submission.ipynb` regenerates **both selected submissions** from the
180 test images and the weights in `weights/`. It verifies the weight hashes and compares the
output with the CSVs that were submitted (`submissions/`).

## Results

| submission | detector | refiner | mask threshold | min area | demote | rows | public LB | OOF refined PQ (707 files, 5-fold) |
|---|---|---|---|---|---|---|---|---|
| **V65** `submission_v65_p2first_r3all.csv` | p2first, all 707 files | r3all | 0.45 | 20 | off | 1224 | 0.39 | 0.4539 ¹ |
| **V84** `submission_v84.csv` | same | soft_all | 0.50 | 200 | LightGBM | 1112 | 0.39 | 0.4590 |

¹ The V65 refiner (r3) starts from a checkpoint that saw 566 of the 707 train files, so its own 5-fold
OOF would leak. 0.4539 is the leak-free re-run of the same recipe (ImageNet → GT crops → OOF proposals,
`train_v73_refiner.py --arm base`), which is the comparable number for V84 (same harness).
The public board shows 2 decimals. The V84 OOF gain (+0.0050) is about the size of the fold noise.

**Report.** `report/report.tex` describes the method, the host evaluation plots (IoU/Dice distributions, 1:n and
n:1 counts), the annotator-agreement analysis and every pre-registered experiment that did not beat V84
(how to regenerate the figures: `report/README.md`). In short: one annotator scored against the others reaches
PQ 0.34, most remaining errors are filaments only some annotators drew, and 17 further experiments after V84
(larger detectors, Mask R-CNN, a whole-disk U-Net, agreement-only labels, metric-aligned losses, fragment merging,
a pixel quality head, per-instance thresholds) changed OOF PQ by at most +0.002.

## Pipeline (inference, one image at a time)

1. **Detector** (`weights/p2first_all707.pt`). YOLO11s-seg plus a P2/4 level, with mask prototypes read
   from P2 at stride 2 (`configs/yolo11s-p2first-seg.yaml`, `solar/v62_arch.py`). It runs on 4 flip
   views at 1024 px: box IoU 0.3, retina masks, agnostic NMS, score floor 0.1 (`solar/v61_toolkit.generate_views`).
2. **Flip consensus.** The views are clustered at mask IoU 0.5. Group score = sum of the member scores / 4
   (`solar/detector_tta.cluster_views`).
3. **Crop refiner.** A ResNet18 U-Net (`solar/solar_base.CropUNet`) runs on 256 px proposal-centred crops
   with context 1.8 and 4-flip TTA (`solar/optimize_inference.crop_probabilities`).
4. **refined_constrained.** The refined probability is thresholded and clamped to the 7x7-dilated proposal.
   Pixels are owned in descending score order, so there are no overlaps. Then the confidence cut is 0.30
   (`optimize_inference.clean_refined_predictions`).
5. **V84 only.** Masks below 200 px are dropped. Then the demote rule runs: a LightGBM model on 28
   per-candidate features (`solar/v83_extract.refine_feats` / `candidate_rows`) drops a kept mask when
   ŷ ≤ λ/2, with λ = 0.458722 (`solar/export_v74.Demote`).
6. COCO RLE, then the CSV, then `tools/validate_submission.py` (format, 180 ids, non-overlap, min area).

Entry point: `scripts/export.sh` → `solar/export_v74.py` (one process per GPU shard).

## How to run

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install nbconvert==7.17.1 ipykernel==7.3.0       # only to execute the notebook
# test images: data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images (180 .jpeg)
SOLAR_GPUS="0 1 2" .venv/bin/jupyter nbconvert --to notebook --execute notebooks/reproduce_submission.ipynb
```

Or call the script directly:
```bash
PY=.venv/bin/python GPUS="0" MIN_AREA=20 MASK_T=0.45 REFINER=weights/r3all_refiner_state.pt \
  NAME=submission_v65 bash scripts/export.sh
PY=.venv/bin/python GPUS="0" MIN_AREA=200 MASK_T=0.50 REFINER=weights/soft_all_refiner_state.pt \
  DEMOTE=weights/demote_all.json NAME=submission_v84 bash scripts/export.sh
```
`TEST_DIR=<dir>` points the script at another image directory. On Kaggle, use the competition's `test_images` path.

**Data.** `data/` is not in the repo (it is in `.gitignore`). Download the competition data with
`kaggle competitions download -c filament-segmentation-2026` and unzip it into `data/raw/filament-segmentation-2026/`.

**Acceptance (2026-10-12).** A fresh venv was built from `requirements.txt` only, and the notebook was run on
3x T4. Both CSVs are **byte-identical** to the submitted files (V65 sha `530e89ea…`, 1224 rows;
V84 sha `2925e2a2…`, 1112 rows). Per GPU process, inference takes about 1.8 s per image for V65 and 2.5 s per image for V84.
See `ACCEPTANCE.md` for the full table.

## Environment

The submissions were produced in this environment, and the acceptance run reproduced them in it:
- Python 3.10.12, torch 2.4.1+cu121 (CUDA 12.1, cuDNN 9.1.0), ultralytics 8.4.138, numpy 2.2.6,
  opencv-python 5.0.0.93, lightgbm 4.5.0.
- 4x NVIDIA Tesla T4 (16 GB), driver 610.43.02, Linux. Inference used 1 GPU per shard.

`solar/v62_patch.py` monkeypatches Ultralytics validator internals during training. Keep ultralytics==8.4.138.

## Training (documented CLI)

All commands run from the repo root. They write to `outputs/` and `work/` (both gitignored). Each one is the
script that produced the shipped weight, copied unchanged except for repo-relative paths. In
`train_v61_fold.py`, the experiment-only recipes were removed and the p2first branch was kept.
The detector's COCO initialisation is `yolo11s-seg.pt` (Ultralytics release, sha256 prefix `1caa81c0195412ef`),
placed in the repo root. The refiners' stage A uses torchvision's ImageNet ResNet18. Both are downloaded, not trained.

1. **Folds.** `configs/v61_folds.json` holds 5 folds by file, with dates grouped
   (`python solar/make_v61_folds.py --annotations <train json> --out configs/v61_folds.json`).
2. **Detector, 5 OOF folds + all data** (fold 99 = all 707 files gives `weights/p2first_all707.pt`). The recipe is
   80 epochs at mask_ratio 2, then 20 epochs at mask_ratio 1 (AdamW 1e-4), imgsz 1024, batch 4, seed 42, on one GPU.
   ```bash
   for K in 0 1 2 3 4 99; do python solar/train_v61_fold.py --fold $K --recipe p2first --work work/v61 --device 0; done
   for K in 0 1 2 3 4; do python solar/eval_v61_oof.py infer --fold $K --recipe p2first \
       --weights work/v61/p2first_fold$K/mask_ratio1/final.pt; done      # OOF views -> outputs/v61_oof_cache
   ```
3. **V84 soft refiner** (leak-free, two stages: ImageNet → stage A on GT crops for 8 epochs at 3e-4 → stage B on OOF
   proposals for 12 epochs at 1e-4, soft targets):
   ```bash
   for K in 0 1 2 3 4; do python solar/train_v82_refiner_targets.py fold --arm soft --fold $K --out outputs/v84; done
   python solar/train_v82_refiner_targets.py fold --arm soft --fold 0 --train-all --out outputs/v84/export
   ```
   The last command writes `outputs/v84/export/soft_all_refiner_state.pt`.
4. **V84 demote model.** It needs the V73 `base` fold refiners as well (the comparison arm of the bundle):
   ```bash
   for K in 0 1 2 3 4; do python solar/train_v73_refiner.py fold --arm base --fold $K; done   # -> outputs/v73
   python solar/v84_extract.py base && python solar/v84_extract.py soft   # CPU, per-candidate features
   python solar/v84_bundle.py                                            # -> outputs/v84/result.json
   python solar/v84_demote_fit.py                                        # -> outputs/v84/export/demote_all.{txt,json}
   ```
5. **V65 r3all refiner.** It continues from `weights/training/best_refiner_coco.pt` on all 707 OOF proposal sets
   for 12 epochs at AdamW 1e-4:
   ```bash
   python solar/train_v63_refiner.py --recipe p2first --tag r3all --train-all --epochs 12 --lr 1e-4 --out outputs/v65/export
   ```
   **Lineage gap (V65 only).** `best_refiner_coco.pt` (sha `8d008053…`) was produced by two Kaggle-kernel runs that
   were not re-run for this repo:
   - (a) `solar/train_and_submit.py` (V6). It trains the V6 refiner, an ImageNet ResNet18 U-Net, for 8 epochs on GT
     crops of the 566 canonical FIT files.
   - (b) `solar/optimize_inference.py --source <V6 output> --refiner-epochs 4 --resolutions 1024`. This is the
     COCO-mask fine-tune: AdamW 5e-5, cosine, with the epoch picked by crop soft Dice on the 94 tune files.

   The scripts are included. The intermediate V6 refiner checkpoint is not, so the checkpoint is shipped as a
   training input. V84 has no such gap, because its refiner starts from ImageNet.

The refiner scripts call `reproduce_v17_from_cache.runtime_state()`. It checks the train annotation sha and the
canonical split in `solar/v6_provenance.json`, and it needs `weights/training/best_refiner_coco.pt`.

## Data use

- Training uses only the competition's train images and train annotations (707 files, 1154 annotator records).
- Inference uses only the 180 test H-alpha images, one image at a time. It reads no test labels and no other images.
- No training or inference step uses test ground truth (the public MAGFiLO release is not used, and it is not in
  this repo). The pipeline uses no external labelled data (for example SWEFil) and no other ground truth.
- **Disclosure.** During development we tried training-only experiments on nearby GONG archive frames. These
  were label transfer from a labelled train file to frames of the same site close in time, and pseudo-labels.
  The host allowed this in discussion thread 748045. None of it improved OOF PQ, and **none of it is in the
  final pipeline or in this repo**.

## Layout

```
solar/          pipeline modules (copied from the development repo; module names kept so the provenance is traceable)
scripts/export.sh   sharded test export -> merged, validated CSV
tools/validate_submission.py
notebooks/reproduce_submission.ipynb
configs/        detector graph yaml, fold manifest
weights/        deployed weights + SHA256SUMS (training/ = training-only init)
submissions/    the two submitted CSVs (acceptance reference)
report/         technical report (LaTeX), figure scripts and figures
tests/          unit tests (pytest)
```

## License

AGPL-3.0 (see `LICENSE`). It is OSI-approved and does not restrict commercial use. We chose it because the
detector is built on Ultralytics YOLO, which is AGPL-3.0.
