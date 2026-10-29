# Final report outline (4 pages, Overleaf template)

Skeleton for the technical report due **Nov 15, 2026**. Every number below is taken from this repo
(`README.md`, `ACCEPTANCE.md`) or from the experiment log of the project; fill prose around them.
Rubric reminder: 70 % quantitative (PQ, Dice/IoU distributions, one-to-many / many-to-one counts),
30 % qualitative (pipeline description, apparent mask morphology, code quality).

## 1. Problem and data (~0.4 page)
- Task: instance masks of solar filaments in 2048x2048 GONG H-alpha images; metric Panoptic Quality,
  masks must not overlap.
- Data: 707 train images, 8,199 polygons in 1,154 annotator records (1–3 annotators per image), 180 test
  images. Images are NSO display JPEGs, already exposure- and size-normalised with limb darkening removed,
  so no further photometric preprocessing is applied (grayscale, scaled to [0, 1]).
- Label noise: the MAGFiLO paper's Cohen's κ 0.47 (0.66 ignoring "unidentifiable") measures agreement on
  the chirality labels, not on the masks; it reports no segmentation agreement. We measure it directly: one
  annotator scored against the others under the competition metric gets PQ 0.340 (host-exact scorer, V69).
  Annotators skipped small, faint and near-limb (> ~70°) filaments and disagree on one-vs-many splits.
- Rules compliance: only the competition's ground truth is used (no public MAGFiLO release, no other
  labelled data); pretrained weights are the public COCO (YOLO11s-seg) and ImageNet (ResNet18) releases.

## 2. Method (~1.4 pages, one pipeline figure)
Figure 1: detector → flip consensus → crop refiner → ownership → demote, with an example image.
1. **Detector** (`weights/p2first_all707.pt`): YOLO11s-seg with an extra P2/4 level and mask prototypes read
   at stride 2 (`configs/yolo11s-p2first-seg.yaml`). 4 flip views at 1024 px, box IoU 0.3, retina masks,
   score floor 0.1. Training: 80 epochs at mask_ratio 2 + 20 at mask_ratio 1, AdamW 1e-4, imgsz 1024,
   batch 4, seed 42; 5 date-grouped folds plus an all-707 fit.
2. **Flip consensus**: views clustered at mask IoU 0.5; group score = sum of member scores / 4.
3. **Crop refiner** (`weights/soft_all_refiner_state.pt`): ResNet18 U-Net on 256 px proposal-centred crops,
   context 1.8, 4-flip TTA. Two-stage leak-free training: ImageNet → GT crops (8 epochs, 3e-4) → out-of-fold
   proposals (12 epochs, 1e-4) with soft targets.
4. **Ownership**: refined probability thresholded at 0.50, clamped to the 7x7-dilated proposal, pixels owned
   in descending score order (no overlaps), confidence cut 0.30, masks < 200 px dropped.
5. **Demote** (`weights/demote_all.json`): LightGBM on 28 per-candidate features drops a kept mask when
   ŷ ≤ λ/2, λ = 0.458722 (the PQ-optimal Dinkelbach threshold).

## 3. Results (~1.2 pages, two tables + Dice/IoU histograms)
Table 1 (main):

| submission | OOF refined PQ (707 files, 5-fold) | public LB | rows |
|---|---|---|---|
| V65 (no demote, min area 20) | 0.4539 | 0.39 | 1224 |
| V84 (soft refiner, MIN200, demote) | 0.4590 | 0.39 | 1112 |

- Aggregation: a half-test probe (public 0.31 vs predicted 0.32 pooled / 0.20 per-image) shows the
  leaderboard pools TP/FP/FN across images, matching our OOF scorer.
- Report the per-instance Dice and IoU distributions and one-to-many / many-to-one counts on OOF (the
  quantitative rubric items); include 3–4 qualitative overlays (good, merged, missed, near-limb).
- Runtime: 2.48 s per image per GPU process (V84), 198 s wall for 180 images on 3x T4.

Table 2 (what did not help — shows rigour; each vs the deployed model with fresh-seed controls):
- Union-of-annotators detector labels: −0.018 PQ (extra proposals are FP for records that did not draw them).
- Mask R-CNN (COCO-finetuned, 1024–2048 px): 0.354–0.383 on the held-out split vs 0.449 for the YOLO path.
- Detector union ensemble: union recall +0.026, but the realistic merged pool beats the best single source at
  1 of 7 confidences; 18.7 % of instances are missed by every detector.
- Image-based candidate selector: +0.0005; proposal-geometry mask splitting: −0.0028; 1536 px training: no gain.
- Larger detector and train-time flips (V101, fold 0, 141 files, refined path, seed noise ≈ ±0.002):
  YOLO11m −0.0064 (P 0.13), flips −0.0076 (P 0.05), YOLO11m + flips −0.0100 (95 % CI [−0.023, +0.001], P 0.04).
- Mask R-CNN as a YOLO-first gap filler (V101, same judge): ±0.0001. It found 2,326 instance groups on the
  141 files, but only 1 (score ≥ 0.5) lay where the detector had proposed nothing, and it matched no annotation.
- Whole-disk ConvNeXt-T U-Net (1536 px, pixel Dice 0.685) blended into the refiner's crop probability (V102,
  fold 0): w 0.5 −0.0050 (P 0.04), w 0.3 +0.0021 on the pre-demote path but 0.4540 vs 0.4590 after demote,
  w 0.7 −0.0111. A second shape estimate is not better than the refiner's.
- Detector fine-tuned only on annotator-agreed filaments (≥ max(2, ⌈R/2⌉) records; V103, fold 0): −0.019
  (95 % CI [−0.032, −0.006]); FP 426 → 216 but TP 1,032 → 861. The test labels are raw per-annotator records,
  so filaments drawn by a single annotator must stay positives.
- PQ-aligned refiner losses (V104, fold 0): Lovász hinge −0.0018 (P 0.16); + Skeleton Recall −0.0070 (P 0.00).
- Fragment merge of collinear kept pieces (V105, 5-fold nested): +0.0032 OOF before demote (5/5 folds), but
  0.0 on the deployed path (0.4588 vs 0.4590): MIN200 and demote already remove those fragments.
- Per-instance refiner threshold maximising expected IoU (Nowozin 2014; V107, 707 files, demote refit):
  −0.0007 (95 % CI [−0.0023, +0.0010]); hysteresis 0.60/0.40 −0.0015. The soft refiner's fixed 0.50 cut is
  already at the per-instance optimum, consistent with it being trained toward the annotators' mean mask.
- Instance-quality CNN for the keep decision (V106; ResNet18 on a 5-channel crop: image, mask, refiner
  probability, seed, other candidates; nested cross-fit on all 707 files). Alone it ranks candidates worse
  than the LightGBM on 28 features (Spearman 0.572 vs 0.595); averaged with it, 0.612, and PQ +0.0019
  (95 % CI [+0.0003, +0.0035], 4/5 folds): a real but small gain, below our +0.005 shipping bar. Pixels add
  little beyond the detector score and hand features, as the literature on learned selectors suggests.

## 4. Discussion and limitations (~0.5 page)
- Local-to-public gap (OOF 0.459 → 0.39) is shared by every clean team (0.05–0.08). Five one-knob public probes
  (mask ±1 px, confidence 0.20/0.40, fold-0-only models) moved the public score by the same amount as OOF, and
  the export reproduces OOF bit-for-bit, so the gap is in the test labels, not in the pipeline.
- Ceiling: with non-overlapping predictions PQ = 2·ΣIoU/(P+G), and the Jaccard-distance triangle inequality bounds
  the agreement any fixed prediction can reach with two independent annotators (RQ ≲ 0.81). Human-vs-human PQ
  is 0.34 and our SQ (0.688) already exceeds the human-vs-human SQ (0.63); the practical OOF ceiling is ≈ 0.48–0.50.
- Leaderboard: public scores ≥ 0.55 come from a hard-coded CSV copied by 83 teams and from re-exported public
  MAGFiLO masks (which contain test labels); no clean team publishes code above 0.41 (two Mask2Former repos).
- Misses are not missing proposals: 658 of 1,640 annotated instances on fold 0 match no kept candidate before
  refinement, yet the detector proposes (at low score) almost everywhere a second architecture finds a filament.
  The loss is in selection and mask shape, which V100 showed is bounded by annotator disagreement.
- Ceiling: annotator disagreement on one-vs-many splits and near-limb filaments; instance boundaries are
  ambiguous from H-alpha alone.
- Future work: annotator-aware training, external unlabelled H-alpha for self-supervised pretraining.

## 5. Reproducibility (~0.2 page)
- `notebooks/reproduce_submission.ipynb` regenerates both CSVs from the 180 test images and the committed
  weights; verified byte-identical (V84 sha256 `2925e2a2…`) from a fresh clone on 2026-10-27.
- Environment: Python 3.10.12, torch 2.4.1+cu121, ultralytics 8.4.138; training CLI in `README.md`.
