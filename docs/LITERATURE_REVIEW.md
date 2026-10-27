# Literature Review — Solar Filament Segmentation (Kaggle 2026)

Compiled 2026-10-26 for the `filament-segmentation-2026` entry (IEEE BigData Cup 2026). Every paper below
was read in full text (arXiv HTML / publisher HTML / PDF), plus the authors' code where it existed.
Numbers were spot-checked against the extracted text. Anything not read in a primary source is marked
**[unverified]**.

---

## 0. Short answers

**Which dataset to train on?** Only the **Kaggle competition data** (`MAGFiLO_1.0_Kaggle_2026`) may be used
as ground truth. The public MAGFiLO release on Harvard Dataverse is a superset that *contains the test-set
annotations*; the host stated submissions using it "will be disregarded". The host also refused other
labelled filament datasets (SWEFil) on the principle "we do not allow using more ground-truth data". So no
external labelled dataset (Flat U-Net/BBSO, MORDEN AHAS/MHAS, Zhu 2019, Diercke catalogue) is usable.
What *is* allowed: unlabelled GONG H-alpha frames (pretraining, pseudo-labels, and copying *training*
labels onto frames minutes away), original FITS files, unlabelled images from other observatories, and
co-temporal GONG magnetograms as a training-only signal — always excluding test images and anything
within ±1 h of a test timestamp, and never using extra inputs at inference. See §1.

**Best preprocessing for the Kaggle images?** Very little. The JPEGs are NSO's display product, which is
already exposure-normalised, size-normalised and **limb-darkening-flattened** (wavelet/Fourier filter).
No published paper ablates limb-darkening correction or CLAHE for deep filament segmentation, and the one
competition-specific test found **CLAHE lowered PQ at every clip strength**. Recommended: grayscale →
[0,1] or per-image standardisation, keep native 2048 resolution for mask refinement, mask/suppress
off-disk and far-limb predictions, enforce MAGFiLO's mask rules (one piece, no holes, no overlaps), and
use geometric + mild photometric/seeing augmentations. See §6.1.

**Best preprocessing for external (unlabelled) data?** Make it look like the Kaggle JPEGs: GONG FITS →
the organisers' own `fits2jpeg` converter; other observatories → recentre/rescale the disk to the GONG
radius, divide out a radial-median limb profile, match the intensity histogram to a GONG reference frame,
and drop bad-seeing/cloudy frames. Use them only for self-supervised pretraining or pseudo-labelling.
See §6.2.

---

## 1. Competition constraints (verified on Kaggle, 2026-10-26)

| Topic | Ruling | Source |
|---|---|---|
| External data in general | Allowed if public and free (Rules §2.6) | Rules tab |
| Extra ground truth | "the models may not use any other ground-truth metadata for training" | Overview → Evaluation |
| Public MAGFiLO release | Overlaps the test set; GT "should be limited to what is provided through this competition. Otherwise, the submissions will be disregarded." | Discussion 729396 (host) |
| Other labelled datasets (SWEFil) | Refused: "we do not allow using more ground-truth data … could easily cause data leakage" | Discussion 746318 (host) |
| Unlabelled GONG H-alpha | Allowed for pretraining / training; exclude test images and ±1 h around them | 737215, 744047, 748045 (host) |
| Copying training labels to frames minutes apart | Allowed ("does not violate any rules") | 748045 (host) |
| Unlabelled KSO / Meudon / BBSO | Allowed for self-supervised pretraining; exclude ±1 h of test timestamps | 748498 (host) |
| Original GONG FITS | Allowed; organisers' converter: bitbucket.org/dataresearchlab/mleco-fits2jpegconvertor | 738951 (host) |
| Co-temporal GONG magnetograms | Allowed as training-only auxiliary signal | 743521 (host) |
| Neighbouring frames at inference | **Not allowed** — inference must use only the supplied test image | 743385, 744047 (host) |
| Provided `spine`, `bbox`, `area`, `category_id` | Allowed as supervision (spine and category are manual; bbox/area derived from masks) | 736543 (host) |
| Test-record hand labelling | Forbidden (Foundational Rules 4.b) | Rules tab |
| Unannotated real filaments | Host wants all filaments found, but notes tiny ones may be noise/sunspots; post-processing may clean them | 735256 (host) |

**Metric and judging.** The leaderboard is Panoptic Quality (Kirillov et al. 2019):
PQ = Σ IoU(TP) / (|TP| + ½|FP| + ½|FN|). Final judging is 70 % quantitative (PQ, Dice/IoU distributions,
one-to-many and many-to-one counts) and 30 % qualitative (pipeline description, apparent mask morphology,
code quality). Overlapping predictions are not accepted.

**Final-submission requirements (deadline Nov 15, 2026).** A 4-page report (Overleaf template), a public
Git repo with `requirements.txt`, a Jupyter notebook running the whole pipeline, and **model weights stored
in the repo**. Our refiner checkpoint (172 MB) is currently not in the repo — see §7.

---

## 2. The dataset: MAGFiLO

### 2.1 Ahmadzadeh et al. 2024 — "A dataset of manually annotated filaments from H-alpha observations"

*Scientific Data* 11, 1031. doi:10.1038/s41597-024-03876-y. Dataset: "MAGFiLO: Manually Annotated GONG
Filaments in H-Alpha Observations", Harvard Dataverse doi:10.7910/DVN/J6JNVK (CC BY-NC 4.0).

**Source and sampling.** GONG 2k×2k H-alpha from six sites (Big Bear, Mauna Loa, Learmonth, Udaipur,
El Teide, Cerro Tololo); 60 s cadence per site. Observations were sampled using AAFDCC/HEK filament
reports from 2011–2017 (42,045 filaments in 2,626 observations), keeping those whose total filament area
exceeded the 10th percentile; the final data span 2011–2022.

**Conversion (key for preprocessing).** FITS (0–2¹⁴) were converted to JPEG with "the exact same procedure
adopted by NSO": variable exposure normalises intensity to disk centre; image size is normalised for the
~3 % yearly change in solar diameter; and "large scale variations of intensity (e.g., solar limb
darkening) are removed by applying a digital filter based on a Fourier Transform". Annotators "worked on
the exact same JPEG images as the GONG archive provides, but without the watermarked metadata". The NSO
product page describes the same step as a *wavelet* transform "to reduce limb darkening and 'flatten'
image intensity"; FITS files "have not been processed with the wavelet transform".

**Anomaly filtering.** 8×8-cell intensity test (ANOVA-tuned on 400 labelled images), F1 = 0.88; ~3–5 % of
GONG frames are anomalous. Mildly blurry frames were kept.

**Annotation protocol.** V7 Darwin brush/eraser → dense polygons; 24 groups of 3 independent annotators;
two-stage review (reviewers, then domain experts); 1,066 person-hours; under 30 % acceptance.
Labels: **Left / Right** (chirality from barb orientation), **Unidentifiable** (no visible barbs), and
Ambiguous (relabelled away — "no filaments are labeled as ambiguous" in the final data; our local train
JSON has none either).

**Mask rules that matter for PQ.**
- One piece, no islands, no holes; tight fit to the body; include all visible barbs.
- Ignore filaments that are too small/faint, small clustered groups, and filaments **beyond roughly ±70°
  from the central meridian**.
- Splitting one filament into several was subjective; disagreements were kept rather than merged.

**Statistics.** 10,244 filaments; 1,593 annotator–image pairs; 958 unique observations (548 annotated once,
185 twice, 225 three times). Left 30.5 %, Right 32.0 %, Unidentifiable 37.5 %. Mean mask area 2,253 /
2,416 / 1,683 px vs. mean bbox area 15,714 / 17,523 / 6,709 px — the objects are thin. Up to 26 filaments
per image.

**Agreement.** Cohen's κ = 0.47 for three classes, 0.66 when Unidentifiable disagreements are not
penalised. **No mask-level (IoU) inter-annotator agreement is reported**, so the PQ ceiling is unknown.

**Relation to the Kaggle data.** Kaggle = 707 train + 180 test = 887 unique images; train annotations
8,199 ≈ 80 % of 10,244. A participant reports the remaining annotations belong to the 180 test images
(Discussion 729396); the host did not dispute it. Treat the Dataverse JSON as off-limits.

**Implications.** (1) No further limb-darkening work is needed on the JPEGs. (2) Predictions near the
limb and tiny blobs are likely scored as FP. (3) One-component, hole-free masks match the label
convention. (4) Instance-split ambiguity puts a hard ceiling on PQ.

### 2.2 McDonald et al. 2023 — "A Machine Learning Ecosystem for Filament Analysis – Phase I"

Poster, Space Weather Workshop 2023 (UCAR CPAESS); NSF PAR 10427981. A progress poster with no numbers.
It announced the dataset and planned products: a GONG viewer, a "chirality-aware" augmentation engine, an
MIoU-based segmentation loss, and a segmentation/classification network. Superseded by §2.1; cite §2.1
for any figure. Mirror flips reverse chirality, but our task is class-agnostic, so flips stay safe.

### 2.3 Ahmadzadeh et al. 2021 — "Multiscale IoU: A Metric for Evaluation of Salient Object Detection with Fine Structures"

IEEE ICIP 2021, pp. 684–688, doi:10.1109/ICIP42928.2021.9506337, arXiv:2105.14572.

**Definition.** Downsample the ground-truth and predicted *contours* at scales δ (cells touched = 1),
compute r(δ) = |s(GT) ∩ s(pred)| / |s(GT)|, and integrate over scales (Δ = 2ⁿ, n < 10).
It is boundary-based, multi-scale, normalised by GT only (recall-like), and has no instance matching or
FP penalty.

**Evidence.** Synthetic smoothing tests show only MIoU drops as jaggedness (barbs) is removed. On 2,500
COCO instances it tracks IoU with lower variance. One BBSO filament example shows MIoU preferring a
prediction with barbs.

**Relevance.** The BigData Cup page lists MIoU among the metrics, but the current Kaggle Evaluation section
(Aug 7 update) does not. It is still useful as a local diagnostic for the 30 % "apparent morphology"
judging.

---

## 3. Filament-segmentation methods

### 3.1 Ahmadzadeh et al. 2019 — "Toward Filament Segmentation Using Deep Neural Networks"

IEEE BigData 2019, pp. 4932–4941, arXiv:1912.02743.

- **Data.** BBSO full-disk H-alpha, 2048², already flat/dark/limb-darkening corrected. Ground truth is
  *automatic*: HEK polygons from Bernasconi's AAFDCC. Train 2012, val 2013, test 2014–2016. HEK reports
  matched to images within 3 min (≈0.5 px of solar rotation). Arcsec → pixel conversion via FITS header.
- **Method.** Off-the-shelf Mask R-CNN (ResNet-50-FPN), trained from scratch. No preprocessing,
  augmentation or hyper-parameters reported.
- **Results.** Mask AP@0.5 = 0.568 / 0.626 / 0.743 (2014/15/16), but mask AP@0.75 only 0.135 / 0.130 /
  0.246, while bbox AP@0.75 is 0.387 / 0.425 / 0.531. Box recall is limited (AR100 0.46–0.57).
- **Failure modes.** HEK "artificial bridges" join fragments; spurious HEK detections; fragmentation of
  large filaments; defective frames.
- **Relevance.** This is the same pattern as our pipeline: localisation is easier than precise thin masks
  (our "box found but IoU < 0.5" bucket). The fragment-bridging convention directly drives PQ split/merge
  errors.

### 3.2 Zhu et al. 2019 — "Solar Filament Recognition Based on Deep Learning"

*Solar Physics* 294, 117, arXiv:1909.06580.

- **Data.** BBSO 2013, semi-automatically labelled ("Photoshop + traditional processing", then manually
  corrected). 30 train / 20 val / 10 test, augmented to 6,040 samples. Bulk dataset (~22k pairs) is public
  at sun10.bao.ac.cn, but the authors themselves recommend re-correcting its ground truth.
- **Preprocessing.** Limb darkening removed (method not stated); off-disk set to grey; resize to 512.
  Augmentation: small rotation/shift/shear/zoom, flips.
- **Method.** U-Net (64-channel start), dropout 0.5, nearest upsampling, BCE, Adam 1e-4, batch 2,
  20 epochs.
- **Results.** Mean DSC 0.894, TPR 0.915 on 10 images. However, the threshold was chosen *per image* from
  its own ROC curve (0.30–0.76), which is an oracle choice, so the scores are optimistic.
- **Relevance.** The wide optimal-threshold range is evidence that a fixed 0.5 mask threshold is not
  generally optimal; calibrate on validation PQ.

### 3.3 Diercke et al. 2024 — "A universal method for solar filament detection from Hα observations using semi-supervised deep learning"

*A&A* 686, A213, doi:10.1051/0004-6361/202348314, arXiv:2402.15407. Code: github.com/adiercke/DeepFilamentSegmentation (GPL-3.0).

- **Data.** ChroTel: 955 days with bounding boxes from three observers (681/84/192, the test set a
  contiguous 20 % block per year). GONG: 17,413 frames at 4 h cadence (2010–2021), 16,759 after the
  Jarolim et al. (2020) image-quality filter. KSO used qualitatively.
- **Preprocessing (most detailed of all papers).** Rescale to 1024²; centre-to-limb correction (code: 6th-
  degree polynomial in cos(rπ/2), divide); 36-term Zernike flattening of residual large-scale variation;
  clip to [0.8, 1.3] and map to [−1, 1]; off-limb = −1. None of these steps is ablated.
- **Method.** YOLOv5-L (from scratch, 1024) gives boxes on ChroTel. Pseudo-masks are a threshold at
  mean − 1σ inside each box, keeping components above 1 % of box area. A U-Net is trained on 18,472 images
  (ChroTel + GONG pseudo-labels) with ×10 positive-weighted BCE.
- **Results (box-level, IoU > 0.1 match — not comparable to PQ).**
  - YOLO: accuracy 84.8, precision 74.5, recall 84.9.
  - U-Net: accuracy 92.4, precision 49.8, recall 93.2.
  - Morphological baseline: 89.0 / 55.7 / 88.3.
  - Removing objects under 20 px² (at 1024 px) raises precision above 60 % with recall still about 90 %.
- **Failure modes on GONG.** Small and low-opacity filaments missed; polar-crown filaments missed in bad
  seeing; YOLO split one large filament into three boxes; penumbral filaments and sunspots detected.
- **Relevance.** This is the closest published design to ours (detector → U-Net masks). The full-disk
  semantic U-Net had much higher recall than YOLO, which suggests a second proposal source for our
  "no box" misses. It also shows that self-training on public GONG frames is feasible (and allowed if the
  frames are unlabelled; §1).

### 3.4 Zhu et al. 2025 — "Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk Hα Images"

*ApJ* 980, 176, doi:10.3847/1538-4357/adadff, arXiv:2502.07259. Data: NADC doi:10.12149/101545 (7.8 GB);
code: Zenodo 10.5281/zenodo.14610155 (MIT).

- **Data.** Raw BBSO 2048² FITS with **no limb-darkening correction**, labels from traditional methods
  with quality screening (about 1,000 train/val from 2022–23, about 100 test from 2024). The authors admit
  label errors in both directions.
- **Preprocessing.** Min–max normalisation only; resize to 512; flips and rotation. They deliberately keep
  limb darkening and moderate clouds, but never compare against corrected inputs.
- **Method.** Constant-width encoder–decoder with simplified channel attention (SCA) and channel
  self-attention (CSA) blocks; BCE; Adam with initial LR 1e-3; batch 9.
- **Results.**
  - Classical U-Net (28.95 M parameters): DSC 0.69, precision 0.98, recall 0.53.
  - Flat U-Net C = 32 (0.26 M parameters): DSC 0.79.
  - Block ablation: all-SCA 0.7581 → one CSA in the bottleneck 0.7943 → SCA(1) + CSA(4) 0.8185.
  - Every model is high-precision and low-recall, consistent with threshold-derived labels missing faint
    filaments.
- **Relevance.** Model capacity is not the bottleneck: a 0.26 M model beats a 29 M U-Net. Training on
  uncorrected images works, so limb-darkening correction is optional rather than required.

### 3.5 Solomon et al. 2025 — "EdgeAttNet: Towards Barb-Aware Filament Segmentation"

arXiv:2509.02964 (no venue listed). Code: github.com/dasjar/EdgeAttNet (not runnable as published:
placeholder imports, model file holds an eval script).

- **Data — the only method paper on MAGFiLO.** 1,593 GONG observations → 1,439 with at least one
  filament. Split 1,295 / 45 / 99; the repo's file counts show this is chronological (2011–18 / 2019–20 /
  2021–22). Each annotator's copy is kept as a separate sample. Training is semantic (instances lost).
- **Preprocessing (from FITS).**
  1. Normalise to [0,1].
  2. Hough-circle disk detection and disk mask.
  3. "Radial flattening" limb correction (annular bins → smoothed profile → divide).
  4. Gaussian 3×3 blur, σ = 0.7.
  5. CLAHE inside the disk.
  6. Off-disk pixels = 0.
  - Repo resizes to 512. No augmentation, no pretraining. Nothing ablated.
- **Method.** U-Net [64…512] with two edge-guided multi-head self-attention blocks at the bottleneck. A
  learned 3×3 edge map is added to Q and K. Loss BCE + Dice; Adam 1e-4; 50 epochs.
- **Results (99 test images).**

  | Model | mIoU pairwise | mIoU multiscale |
  |---|---|---|
  | U-Net | 0.5724 | 0.5848 |
  | U-Net+MHSA (no PE) | 0.5856 | 0.6000 |
  | U-Net+MHSA (PE) | 0.6200 | 0.6601 |
  | **EdgeAttNet** | **0.6451** | **0.7032** |

  Parameters: EdgeAttNet 22.7 M vs U-Net 31.0 M.
- **Limitations.** Both metrics ignore missed and spurious objects, so they say nothing about PQ recall.
  Single run; no isolated ablation of the edge prior; no timing despite the "faster" claim.
- **Relevance.** This is the strongest published number on our exact data, but it is not comparable to
  PQ. The edge prior is cheap to add to our refiner, with low evidence. A chronological split is a useful
  second validation view.

### 3.6 Hu et al. 2026 — "A Modern ConvNet for Solar Filament Detection" (MORDEN)

arXiv:2607.24525, accepted in RAA. Code: github.com/irisaltHu/MORDEN (MIT). Data: zenodo.org/records/21582245 (CC BY 4.0).

- **Data.** CHASE (space-based, ~1″/px; no seeing or clouds).
  - HAS: 120 images, k-means labels.
  - **MHAS**: the same 120, manually corrected, including *bridging fragments within one channel*.
  - **AHAS**: 775 automatically labelled (MORDEN + DenseCRF), manually screened.
- **Preprocessing.** None beyond resizing to 1024 and min–max normalisation. Augmentation:
  - vertical flip;
  - rotation ±45°;
  - gamma 0.5–1.5;
  - a down-then-up resampling degradation (mimics seeing).
- **Method.** U-Net variant: 5×5 kernels, InstanceNorm, adaptive pyramid pooling, patch attention gate,
  two heads. Focal loss (α = 0.66, γ = 2); AdamW 2e-4.
- **Ablation (MHAS IoU; full model 0.696 ± 0.005).** Removing each component gives:
  - InstanceNorm: 0.678
  - large kernel: 0.663
  - APP: 0.678
  - PAG: 0.688
  - branch block: 0.688
  - multihead: 0.695

  Single runs on a 20-image test set.
- **Cross-dataset.** IoU 0.789 on the Flat U-Net set vs Flat U-Net's 0.68.
- **Data scaling.** Training on 6× more pseudo-labelled data (AHAS) gave IoU 0.699–0.702 vs 0.696 —
  marginal. Model selection on clean manual validation beat selection on pseudo-labelled validation.
- **Post-processing.** DenseCRF at native 2K (improves edge contrast, not shown to improve IoU); hole
  filling; DBSCAN to group fragments into filaments (expert-judged 84 % correct).
- **Relevance.** These are the only ablation-backed architecture gains in the literature (InstanceNorm,
  5×5 kernels), cheap to try in the refiner. They also suggest expecting small returns from pseudo-labelled
  data, and suggest fragment-grouping post-processing tuned on PQ.

### 3.7 Mais et al. 2024 — "FISBe: … instance segmentation of long-range thin filamentous structures"

CVPR 2024, arXiv:2404.00130. Not solar (3D neuron light microscopy, 101 images, 590 neurons), but the
closest benchmark on *instance* segmentation of thin, elongated objects.

- **Why box methods struggle (their §1).** For thin, elongated objects:
  - boxes are poor proxies: many instances share large, overlapping boxes, so box NMS can suppress real
    neighbours;
  - in three-label models, interior equals boundary;
  - center-cue methods (Cellpose) and StarDist do not apply.
- **Metrics.** Centreline-based clDice matching; avF1 + coverage; explicit false-split (FS) and
  false-merge (FM) counts. They argue pixel IoU/PQ is harsh for thin structures (citing "Panoptic quality
  should be avoided…", Foucart et al. 2023).
- **Baselines.** PatchPerPix S = 0.35 (best), FFN 0.25, colour clustering 0.30. Validation-tuned
  thresholds were needed; adding partly-labelled data improved FFN.
- **Relevance.** Two things transfer:
  - Check whether our box-level NMS (Ultralytics IoU 0.7, consensus NMS 0.5) suppresses neighbouring
    filaments with overlapping boxes.
  - Use FS/FM counts as diagnostics. The judges also score one-to-many and many-to-one relations.

### 3.8 Asensio Ramos et al. 2023 — "Machine learning in solar physics" (review)

*Living Reviews in Solar Physics* 20, doi:10.1007/s41116-023-00038-x, arXiv:2306.15308. This is the
standard survey, but **it does not cover H-alpha filament segmentation**. It cites none of Zhu 2019,
Ahmadzadeh 2019, Liu 2021, Guo 2022 or Diercke 2024, and nothing on GONG or limb darkening.

Useful general points:
- §1.1.2: split by physical unit to avoid leakage (our analogue: same observation, several annotators).
- §5.4: "bag of tricks" — augmentation, regularisation, BatchNorm caveats with small batches.
- §7.1: segmentation of solar images, dominated by coronal holes (U-Net, CHRONNOS). Notably, CHRONNOS
  separates CHs from filament channels using multi-channel input.
- §7.3: always report a simple baseline on the same metric.

Cite it for context only.

### 3.9 Classical pipelines (context for preprocessing)

| Work | Preprocessing | Detection |
|---|---|---|
| Shih & Kowalski 2003, Sol Phys 218, 99 | brightness/area normalisation | local or global threshold + morphology |
| Fuller et al. 2005, Sol Phys 227, 61 | two-stage median-filter background subtraction (limb darkening), dust-line removal | region growing from seeds |
| Qu et al. 2005, Sol Phys 228, 119 | SIDE diffusion enhancement | adaptive threshold + SVM vs sunspots |
| Bernasconi et al. 2005 (AAFDCC → HEK) | histogram standardisation to a good-seeing reference frame | thresholds + morphology + clustering; chirality |
| Yuan et al. 2011, Sol Phys 272, 101 | polynomial surface fit for uneven brightness; Hough disk fit | adaptive threshold, skeleton graph |
| Hao et al. 2013, arXiv:1303.6367 | analytic limb-darkening law + top-hat | global threshold |

All describe limb-darkening or background flattening as necessary *for thresholding*. None provides
evidence that it helps a learned model.

### 3.10 Other relevant work found

- **Zheng et al. 2024** (CHASE, arXiv:2402.14209): U-Net + focal loss, test IoU 0.67, cosine
  limb-darkening fit.
- **Chalmers & Ahmadzadeh 2025** (arXiv:2509.18214): chirality classification on MAGFiLO crops
  (ConvNeXt-Base). Per the abstract, per-class accuracy is 0.69 (Left) and 0.73 (Right).
- **Castelló, Luna & Terradas 2026** (arXiv:2607.01095): a GONG filament-oscillation pipeline with its own
  limb and Zernike flattening.
- **Reche & Cid 2024** (ESA SPAICE; https://spaice.esa.int/?p=1044) **[unverified]**: DETR + U-Net
  instance segmentation plus tracking on GONG.
- **Guo et al. 2022**, Sol Phys 297, 104 **[unverified, cited by Zheng 2024]**: CondInst instance
  segmentation aimed at filament fragmentation — directly relevant to PQ splits.
- **Competitor repo** github.com/biconcavelens/filament-segmentation-2026 (not peer-reviewed):
  - Mask R-CNN + YOLO11m ensemble with crop refinement and flip TTA: LB 0.39–0.40, local PQ 0.44–0.46.
  - CLAHE preprocessing lowered PQ at every clip strength (clip 4.0: 0.4252 → 0.4121).
  - Eight flat-fielding variants failed to increase filament contrast.
  - Plain U-Net: local 0.408, LB 0.32.

---

## 4. Comparison of methods

| Paper | Data (instrument) | Labels | Input size | Model | Preprocessing | Metric (headline) | Instance? |
|---|---|---|---|---|---|---|---|
| Ahmadzadeh 2019 | BBSO | HEK auto polygons | not stated | Mask R-CNN R50-FPN | BBSO corrected product | mask AP@.5 0.57–0.74 | yes |
| Zhu 2019 | BBSO | semi-auto + manual | 512 | U-Net | LD removed, off-disk grey | DSC 0.89 (oracle thr.) | no |
| Diercke 2024 | ChroTel + GONG | boxes → pseudo-masks | 1024 | YOLOv5-L → U-Net | CLV + Zernike, clip, [−1,1] | box recall 93 % (IoU>0.1) | boxes |
| Flat U-Net 2025 | BBSO raw | threshold-derived | 512 | SCA/CSA U-Net (0.26 M) | min–max only | DSC 0.82 | no |
| EdgeAttNet 2025 | **GONG / MAGFiLO** | manual | 512 | U-Net + edge MHSA | radial flatten, blur, CLAHE | mIoU pairwise 0.645 | no |
| MORDEN 2026 | CHASE | manual + auto | 1024 | ConvNet, IN, 5×5 | min–max; seeing-degradation aug | IoU 0.696 (MHAS) | DBSCAN groups |
| **Ours (v16b)** | **GONG / MAGFiLO** | manual | 1024 det + 256 crops | YOLO11s-seg → ResNet18 U-Net | none | **PQ 0.458 local / 0.38 LB** | yes |

No published work reports PQ on MAGFiLO, so the competition leaderboard is the only comparable benchmark.

---

## 5. External dataset candidates and verdict

| Dataset | Instrument | Labels | Public | Allowed here? |
|---|---|---|---|---|
| Kaggle `MAGFiLO_1.0_Kaggle_2026` train | GONG | manual instance polygons + spine + chirality | yes | **Yes — the only allowed ground truth** |
| MAGFiLO v1.0 (Dataverse J6JNVK) | GONG | same, *incl. test images* | yes | **No** — host: submissions "will be disregarded" |
| SWEFil (Reche & Cid 2026) | GONG | manual semantic masks | yes (CC BY 4.0) | **No** — explicitly refused (746318) |
| Flat U-Net set (NADC 101545) | BBSO | threshold semantic | yes | No (extra GT) |
| MORDEN MHAS / AHAS (Zenodo 21582245) | CHASE | manual / auto semantic | yes | No (extra GT) |
| Zhu 2019 filaments-unet-dataset | BBSO | semi-auto semantic | yes | No (extra GT) |
| Diercke GONG catalogue | GONG | pseudo semantic | not found | No (extra GT) |
| GONG H-alpha archive (gong2.nso.edu; doi:10.25668/AS28-7P13) | GONG | none | yes | **Yes**, training only; exclude test ±1 h |
| Training-image neighbours ± minutes | GONG | *our* training labels copied | yes | **Yes** (748045) |
| GONG magnetograms (bzi) | GONG | none | yes | **Yes**, training-only auxiliary signal |
| KSO / Meudon / BBSO / ChroTel unlabelled | other | none | yes | **Yes**, self-supervised pretraining; exclude ±1 h |

**Recommended data strategy, cheapest first.**
1. Train on the Kaggle data, treating each annotator copy as a separate sample (host-endorsed), and keep
   all copies of one observation in the same fold (our split already does this by file name).
2. **Label transfer.** Download GONG frames a few minutes from each *training* observation, from the same
   or other sites, and reuse the training labels. Solar rotation over 3 min is about 0.5 px (Ahmadzadeh
   2019). This gives real seeing, cloud and site variation with correct labels at almost no cost, and the
   host has explicitly allowed it.
3. **Self-training.** Pseudo-label unlabelled GONG frames (outside test ±1 h) with the current model and
   retrain. Expect modest gains: MORDEN saw +0.003–0.006 IoU from 6× pseudo-labelled data.
4. **Self-supervised encoder pretraining** on unlabelled GONG / other observatories. Untested for this
   task.

---

## 6. Preprocessing recommendations

### 6.1 For the Kaggle MAGFiLO JPEGs

| Step | Recommendation | Evidence |
|---|---|---|
| Colour | Read as 1-channel grayscale; replicating to 3 channels for pretrained backbones is allowed | Kaggle Data page; host reply 744544 |
| Limb darkening / flattening | **Don't redo it**: already removed by NSO's Fourier/wavelet filter | MAGFiLO §Data Conversion; NSO product page |
| CLAHE / histogram equalisation | **Avoid** (or test only as an *extra* channel) | Competitor ablation: PQ fell at every clip strength; no paper ablates it positively |
| Intensity scaling | /255, or per-image z-score on the disk | All papers do simple scaling; no ablations |
| Resolution | Detector at ≥1024; masks refined at **native 2048** crops; don't squash long boxes into 256 px (tile, or keep the aspect ratio) | Mean mask ≈ 1.7–2.4 k px (thin); MORDEN edge results at 2K; Ahmadzadeh mask AP@.75 collapse |
| Disk mask | Zero / suppress predictions off-disk; radius is fixed by NSO's remap (check on a few images) | Diercke, EdgeAttNet; NSO remap |
| Limb band | Down-weight or drop predictions beyond about ±70° from the central meridian; tune on validation PQ | MAGFiLO annotation rules |
| Label rasterisation | `pycocotools` polygon → mask on the 2048 canvas | Kaggle Data page |
| Multiple annotators | Separate samples; group folds by observation | Host guidance; EdgeAttNet repo |
| Augmentation | Flips and 90° rotations (masks are class-agnostic, so chirality flips don't matter); rotation ±30–45°; gamma 0.5–1.5; down-up resample blur to mimic seeing; mild brightness | MORDEN Table 1; Zhu 2019. None ablated, so validate each |
| Mask threshold | Calibrate on validation PQ, possibly per size bucket | Zhu 2019 optimal thresholds 0.30–0.76 |
| Post-processing | Largest component only, fill holes, no overlaps, min-area tuned on PQ; optionally merge collinear fragments | MAGFiLO mask rules; Diercke Fig. 6; MORDEN DBSCAN |

### 6.2 For external (unlabelled) data

Goal: match the Kaggle JPEG domain exactly.

1. **GONG FITS.** Convert with the organisers' `mleco-fits2jpegconvertor`, which reproduces the NSO
   wavelet + clip + 8-bit pipeline. Never use raw FITS intensities alongside the JPEGs without this
   conversion.
2. **Other observatories (KSO, BBSO, Meudon, ChroTel).**
   1. Recentre and rescale the disk to the GONG radius (sunpy, or a Hough circle fit).
   2. Remove limb darkening by dividing by an azimuthal-*median* radial profile, so filaments and plage
      don't bias it (Ravindra 2016; Castelló 2026). Optionally add Zernike flattening (Diercke 2024).
   3. Match the intensity histogram to a good-seeing GONG reference frame (Bernasconi via Ahmadzadeh
      2019).
   4. Resample to 2048² and save as 8-bit.
3. **Quality filter.** Drop cloudy or bad-seeing frames, e.g. with the Jarolim et al. 2020 image-quality
   method (Diercke removed 3.8 % of GONG) or MAGFiLO's 8×8-cell intensity test.
4. **Leakage guard.** Drop any frame within ±1 h of a test timestamp (parse `YYYYMMDDHHMMSS` from the
   test filenames).
5. Use the result only for pretraining or pseudo-labelling, and select models only on clean MAGFiLO
   validation data (MORDEN Table 8).

---

## 7. What this means for our pipeline (v16b, LB 0.38)

Ordered by expected value for our error profile: about 60 % of FNs are "no box", plus thin masks with
IoU < 0.5.

1. **Repo compliance (blocking).** The final submission needs weights *in the repo* and a reproducing
   notebook.
   - Save the refiner as an fp16 `state_dict` without optimizer state, which should fit well under GitHub's
     100 MB limit, or use Git LFS.
   - Fix the hard dependency on the missing v14 weights (`scripts/make_submission_v16b.py:124`).
2. **Recall — a second proposal source.** Add a full-disk or tiled semantic U-Net. Turn its connected
   components that no YOLO instance covers into extra instances, then refine them (Diercke: U-Net recall
   93 % vs YOLO 85%). Also check whether box NMS suppresses neighbouring elongated filaments (FISBe).
3. **Label transfer to neighbouring GONG frames** (§5, step 2): more real training images with no rule
   risk.
4. **Thin-mask refinement.**
   - Refine at native resolution with aspect-preserving or tiled crops.
   - Add InstanceNorm and 5×5 kernels (MORDEN), plus focal or clDice loss.
   - Calibrate the threshold per size bucket.
   - Optionally DenseCRF on 2K intensities, accepted only if validation PQ improves.
5. **Post-processing matched to the label rules.** One component, fill holes, limb-band filter, min-area
   tuned, collinear-fragment merge tuned on PQ.
6. **Diagnostics for the report (30 % qualitative).**
   - Local MIoU (boundary quality, barbs) and FS/FM (one-to-many, many-to-one) counts.
   - A chronological validation split as a second view, since EdgeAttNet used one.
7. **Expect a ceiling.** κ = 0.47–0.66 and subjective one-vs-many splits mean PQ cannot approach 1. The
   host values PQ above 0.35 (Aug 20 announcement).

---

## 8. Open questions / not verified

- How the test ground truth handles images annotated by several annotators (Discussion 729487 is
  unanswered). This affects whether to predict "union-like" or single-annotator-like masks.
- The exact PQ implementation (IoU threshold, overlap handling). See the host's self-evaluation notebook
  `azimahmadzadeh/self-evaluation-notebook` (not yet read).
- The Kaggle JPEGs are inferred, not confirmed, to be exactly the NSO wavelet product. Check by plotting
  the azimuthal-median radial profile of a few training images; it should be roughly flat.
- Whether the judges compute MIoU (listed on the BigData Cup page, absent from the Kaggle Evaluation
  section).

---

## 9. References

1. Ahmadzadeh A. et al. (2024). A dataset of manually annotated filaments from H-alpha observations. *Sci. Data* 11, 1031. https://doi.org/10.1038/s41597-024-03876-y — Dataset: https://doi.org/10.7910/DVN/J6JNVK
2. McDonald S. et al. (2023). A Machine Learning Ecosystem for Filament Analysis – Phase I. Space Weather Workshop poster. https://par.nsf.gov/biblio/10427981
3. Ahmadzadeh A., Kempton D.J., Chen Y., Angryk R.A. (2021). Multiscale IoU. *IEEE ICIP*, 684–688. https://arxiv.org/abs/2105.14572
4. Ahmadzadeh A., Mahajan S.S., Kempton D.J., Angryk R.A., Ji S. (2019). Toward Filament Segmentation Using Deep Neural Networks. *IEEE BigData*, 4932–4941. https://arxiv.org/abs/1912.02743
5. Zhu G., Lin G., Wang D., Liu S., Yang X. (2019). Solar Filament Recognition Based on Deep Learning. *Sol. Phys.* 294, 117. https://arxiv.org/abs/1909.06580
6. Diercke A. et al. (2024). A universal method for solar filament detection from Hα observations using semi-supervised deep learning. *A&A* 686, A213. https://arxiv.org/abs/2402.15407
7. Zhu G., Lin G., Yang X., Zeng C. (2025). Flat U-Net. *ApJ* 980, 176. https://arxiv.org/abs/2502.07259
8. Solomon V., Martens P., Liu J., Angryk R. (2025). EdgeAttNet: Towards Barb-Aware Filament Segmentation. https://arxiv.org/abs/2509.02964
9. Hu J.R. et al. (2026). A Modern ConvNet for Solar Filament Detection. RAA (accepted). https://arxiv.org/abs/2607.24525
10. Mais L. et al. (2024). FISBe. *CVPR*. https://arxiv.org/abs/2404.00130
11. Asensio Ramos A., Cheung M.C.M., Chifu I., Gafeira R. (2023). Machine learning in solar physics. *Living Rev. Sol. Phys.* 20. https://arxiv.org/abs/2306.15308
12. Kirillov A. et al. (2019). Panoptic Segmentation. *CVPR*. https://arxiv.org/abs/1801.00868
13. Zheng et al. (2024). CHASE filament detection. https://arxiv.org/abs/2402.14209
14. Chalmers & Ahmadzadeh (2025). Filament chirality classification on MAGFiLO. https://arxiv.org/abs/2509.18214
15. Castelló, Luna & Terradas (2026). Automatic detection of solar filament oscillations I. https://arxiv.org/abs/2607.01095
16. Fuller N., Aboudarham J., Bentley R.D. (2005). *Sol. Phys.* 227, 61. https://doi.org/10.1007/s11207-005-8364-1
17. Hao Q., Fang C., Chen P.F. (2013). https://arxiv.org/abs/1303.6367
18. NSO GONG H-alpha data product description. https://nso.edu/nisp/products/h-alpha/
19. Kaggle competition pages (Overview, Data, Rules, Discussion threads 729396, 736543, 737215, 738951, 743385, 743521, 744047, 746318, 748045, 748498). https://www.kaggle.com/competitions/filament-segmentation-2026
