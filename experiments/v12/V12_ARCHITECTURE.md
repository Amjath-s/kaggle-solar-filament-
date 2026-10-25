# V12 full architecture — how it works (code-level)

This document describes the **V12 solar-filament stack** that reaches local **PQ ≈ 0.425–0.426** on the seed-42 val split (141 images), with locked inference knobs:

| Knob | Winning value | Config default (yaml) |
|------|--------------:|----------------------:|
| Mode | `refined` | — |
| `yolo_conf` | **0.40** | 0.30 |
| `min_area` | **80** (64 ≈ same) | 20 |
| `yolo_imgsz` | 1024 | 1024 |
| `yolo_iou` | 0.7 | 0.7 |
| `match_iou` / `nms_iou` | 0.5 / 0.5 | 0.5 / 0.5 |
| Refiner size / thr / `seed_dilate` / `pad_frac` | 256 / 0.45 / 7 / 0.20 | same |

**Measured result** (`experiments/v12/eval/v12_refined_c040_a080_summary.json`):

- PQ **0.4264**, SQ 0.683, RQ 0.625  
- TP 670, FP 330, FN 475  
- Gate: **MILESTONE** (≥ 0.42)

---

## 1. Big picture

```text
Native MAGFiLO image (gray, ~2048×2048)
        │
        ▼
┌───────────────────────────────────────┐
│ A. YOLO11s-seg @1024                  │  Ultralytics; boxes + coarse masks
│    (optionally ×4 flips)              │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ B. Consensus merge (4-flip TTA)       │  src/inference/v12_consensus.py
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ C. ResNet18-UNet crop refine          │  per box; clamp to dilated YOLO seed
│    scripts/eval_v12_pq.py::refine_one │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ D. Score-order exclusive + min_area   │  make_masks_exclusive_by_score
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│ E. PQ @ mask IoU > 0.5                │  src/evaluation/pq.py
└───────────────────────────────────────┘
```

**Roles**

| Stage | Finds new filaments? | Improves mask shape? |
|-------|---------------------|----------------------|
| YOLO | **Yes** (proposals) | Rough |
| Consensus | Filters unstable proposals | No |
| Refiner | **No** | **Yes** (inside seed) |
| Exclusive / min_area | Drops junk / overlaps | No |

Config hub: `experiments/v12/configs/resnet18_refiner.yaml`  
Eval entry: `scripts/eval_v12_pq.py`  
Submit entry: `scripts/make_submission_v12.py`

---

## 2. Data and split

| Item | Path / value |
|------|----------------|
| Images | `data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images` |
| Annotations | `.../MAGFiLO_1.0_Annotations_kaggle2026_train.json` |
| Split | `data/splits/train.txt` (566) + `val.txt` (141), seed 42 |
| YOLO train labels | All-annotator export (Kaggle `export_yolo_seg_all.py`) |
| PQ GT | Primary-annotator instance masks |

Do **not** shrink val to juice local PQ — breaks comparison to V7 / friend / this 0.425 baseline.

---

## 3. Stage A — YOLO11s-seg detector

### 3.1 What it is

Ultralytics **YOLO11s-seg**: single-class instance segmentation.

- ~**10.08M** parameters, ~203 layers, ~33.2 GFLOPs  
- Backbone (downsample): Conv → C3k2 blocks → SPPF → C2PSA  
- Neck: upsample + Concat + C3k2 (FPN/PAN style)  
- Head: `Segment` — boxes + mask coefficients / prototypes, `nc=1`

At train time ImageNet-style COCO pretrained weights transfer; final head remapped to **1 class**.

### 3.2 How it was trained (Kaggle)

Bundle: `kaggle_v12_upload/` → dataset `solar-filament-v12-code`.

| Param | First (winning weights) | Current retrain upload |
|-------|-------------------------|------------------------|
| `model` | yolo11s-seg.pt | same |
| `imgsz` | 1024 | 1024 |
| `epochs` | **50** (ran all 50) | **80** |
| `patience` | 12 | 20 |
| `batch` | 4 (T4 may drop to 2) | 4 |
| `flipud` / `fliplr` | 0.0 | 0.0 |
| `seed` | 42 | 42 |
| `mask_ratio` | 1 | 1 |

Evidence for the 50-epoch run: repo root `v12_export.zip` → `results.csv`, `train_meta.json`.

- Best box mAP50 ≈ **0.565** (epoch ~48)  
- Best mask mAP50 ≈ **0.430** (epoch ~44)  
- Wall time ≈ 2.5 h on Kaggle T4  

Weights used for PQ 0.425:

`experiments/v12/yolo11s/weights/best.pt`

### 3.3 Inference function — `yolo_seg_predict`

**File:** `src/inference/v12_consensus.py`

```text
yolo_seg_predict(model, image, imgsz, conf, iou, device)
  → boxes (N,4) xyxy, confs (N,), masks [N] full-res uint8
```

What it does:

1. `model.predict(source=image, imgsz=..., conf=..., iou=..., device=..., verbose=False)`  
2. Read `result.boxes.xyxy` and `result.boxes.conf`  
3. Read `result.masks.data`; if not native HxW, nearest-resize to image size; threshold 0.5  
4. If no masks (rare), fill rectangle under each box  
5. Align lengths of boxes / confs / masks  

**`conf`**: drop detections below this score (winning: **0.40**).  
**`iou`**: Ultralytics NMS IoU among overlapping boxes in **one** forward (**0.7**).  
This is **not** the PQ match IoU (0.5) and **not** consensus `nms_iou` (0.5).

### 3.4 Single-view helper — `detect_single`

Thin wrapper: one call to `yolo_seg_predict` (no flips). Used for detector-only mode and rescue stream.

---

## 4. Stage B — 4-flip consensus TTA

**File:** `src/inference/v12_consensus.py`  
**Friend-style rule:** keep a detection only if it matches another flip view; then NMS.

### 4.1 Views

```python
FLIP_VIEWS = ("original", "horizontal", "vertical", "horizontal_vertical")
```

| Function | Role |
|----------|------|
| `flip_image` | Apply geometric flip to HxW array |
| `inverse_flip_boxes` | Map boxes from flipped coords → original |
| `inverse_flip_mask` | Map mask the same way (flips are involutions) |

### 4.2 Orchestrator — `detect_consensus_4flip`

```text
for view in FLIP_VIEWS:
    flipped = flip_image(image, view)
    boxes, confs, masks = yolo_seg_predict(flipped, ...)
    map boxes/masks back to original coords
return consensus_merge(...)
```

**Cost:** 4 YOLO forwards per image (vs 1 for `detect_single`).

### 4.3 Merge — `consensus_merge`

Parameters: `match_iou=0.5`, `nms_iou=0.5`.

1. Flatten all candidates from all views (tag each with view name).  
2. Build pairwise box IoU matrix (`box_iou_matrix`).  
3. **Consensus filter:** keep index `i` only if there exists `j ≠ i` with  
   - `views[j] != views[i]`  
   - `iou[i,j] ≥ match_iou`  
   Unstable one-view-only dets are dropped (some become `consensus_killed` FNs in diagnostics).  
4. **Score-order NMS:** sort by conf descending; suppress others with IoU > `nms_iou`.  
5. Return surviving boxes, confs, masks.

### 4.4 Other helpers in this file

| Function | Role |
|----------|------|
| `box_iou_matrix` | NxM box IoU |
| `dilate_mask` | Binary dilation (iterations = radius); used when clamping refine seeds |

---

## 5. Stage C — ResNet18-UNet refiner

### 5.1 What it does (concept)

For **each** consensus YOLO box + YOLO seed mask:

1. Expand box by `pad_frac`  
2. Crop gray image (+ seed crop)  
3. Resize crop to **256×256**  
4. Run ResNet18-UNet → probability  
5. Threshold → binary  
6. Resize binary back to crop size  
7. **Clamp:** `refined ∩ dilate(yolo_seed, seed_dilate)`  
8. Paste onto full-image canvas  

It **cannot invent a filament with no YOLO box**. It only reshapes inside the allowed seed region.

### 5.2 Network — `ResNet18UNet`

**File:** `src/models/resnet_unet.py`

**Encoder (torchvision ResNet18, grayscale-adapted):**

| Step | Tensor shape (H=W=256) | Notes |
|------|------------------------|-------|
| Input | 1×256×256 | Gray |
| `conv1` + bn + relu | 64×128×128 | `/2`; ImageNet RGB weights averaged → 1ch |
| `maxpool` | 64×64×64 | `/4` |
| `layer1` | 64×64×64 | `/4` |
| `layer2` | 128×32×32 | `/8` |
| `layer3` | 256×16×16 | `/16` |
| `layer4` | 512×8×8 | `/32` |

**Decoder (U-Net skips):**

| Step | Op | Channels |
|------|-----|----------|
| up4 + cat e3 | ConvTranspose2d + `ConvBlock` | → 256 |
| up3 + cat e2 | … | → 128 |
| up2 + cat e1 | … | → 64 |
| up1 + cat x0 | … | → 64 |
| up0 | … | → 32 |
| `head` | Conv 1×1 | → 1 logit map |

`ConvBlock` = Conv3×3–BN–ReLU × 2.

`forward` returns **logits** (sigmoid applied outside).

### 5.3 How the refiner was trained

Config: `experiments/v12/configs/resnet18_refiner.yaml`  
Crops: built from YOLO proposals (`experiments/v12/crops/`)  
Meta: `experiments/v12/resnet18_refiner/train_meta.json`

| Param | Value |
|-------|------:|
| image_size | 256 |
| pad_frac | 0.20 |
| batch_size | 8 |
| epochs / patience | 40 / 12 |
| Actually ran | **15** (early stop) |
| Best val Dice | **≈ 0.688** at epoch **3** |
| optimizer | AdamW lr 1e-4, wd 1e-4 |
| loss | BCE 0.5 + Dice 0.5 |
| amp | true |
| aug | hflip / vflip p=0.5 |

Weights: `experiments/v12/resnet18_refiner/best.pt`

### 5.4 Crop geometry — `expand_xyxy` / `resize_image_mask`

**File:** `src/inference/local_crops.py`

- `expand_xyxy(box, pad_frac, W, H)`: grow box by `pad_frac * width/height` each side; clamp to image; return int `[x1,y1,x2,y2]` for slicing.  
- `resize_image_mask(crop, mask, size)`: bilinear image → `size×size` float `[0,1]`; nearest mask if provided.

### 5.5 Core refine — `refine_one` / `refine_instances`

**File:** `scripts/eval_v12_pq.py`

```text
refine_one(model, img_u8, box, seed_mask, image_size, pad_frac, threshold, seed_dilate, device)
  → full-image uint8 mask for one instance
```

Steps in code:

1. `exp = expand_xyxy(box, pad_frac, w, h)`  
2. `crop = img_u8[y1:y2, x1:x2]`, `seed_crop = seed_mask[...]`  
3. `img_r, _ = resize_image_mask(crop, None, size=image_size)` → 256×256 float  
4. `logits = model(tensor)`; `prob = sigmoid(logits)`  
5. `pred = (prob >= threshold)`  # default 0.45  
6. Nearest-resize `pred` to crop HxW  
7. `allow = dilate_mask(seed_crop, seed_dilate)`  # default 7 iterations  
8. `clamped = pred_crop AND allow`  
9. Paste `clamped` into zeros canvas at `exp`  

`refine_instances`: loops `refine_one` over all boxes/seeds.

**Why clamp:** stops the U-Net from painting filaments far outside YOLO’s belief (cuts some FP bleed).

---

## 6. Stage D — Post-process

### 6.1 `make_masks_exclusive_by_score`

**File:** `scripts/eval_v12_pq.py`

1. Sort instances by YOLO conf descending  
2. Maintain `claimed` pixel map  
3. Keep `mask AND NOT claimed` if area ≥ `min_area`  
4. Update claimed  

Winning `min_area`: **80** (pixels on the eval canvas, typically 2048).

### 6.2 `load_image_and_gt`

Loads gray native image + primary GT masks; optionally rescales image/GT to `out_size=2048` and returns scale factors for boxes.

### 6.3 Optional rescue path (NOT in 0.425 recipe)

Also in `eval_v12_pq.py` / submit — dual-conf stream:

| Function | Role |
|----------|------|
| `filter_rescue_by_main_iou` | Keep low-conf boxes with max IoU vs main &lt; threshold |
| `gate_rescue_refined` | min_area / min_fg_frac / optional topk |

Tried; **HOLD** (FN↓ but FP↑). Documented for completeness only.

---

## 7. Stage E — PQ metric

**File:** `src/evaluation/pq.py`

| Function | Role |
|----------|------|
| `mask_iou` | Pixel IoU of two binary masks |
| `mask_dice` | Dice coefficient |
| `compute_pq` | Greedy one-to-one match if IoU **> 0.5** |

Formulas:

```text
TP = matched pairs
FP = preds with no match
FN = GTs with no match

SQ = mean IoU over TP pairs
RQ = TP / (TP + 0.5*FP + 0.5*FN)
PQ = SQ × RQ
```

Gates in yaml: V7 ref 0.379 · milestone 0.42 · submit-ready 0.45.

---

## 8. Eval script modes (`eval_v12_pq.py`)

| `--mode` | Pipeline |
|----------|----------|
| `detector` | `detect_single` → exclusive (YOLO masks, no refine) |
| `consensus` | `detect_consensus_4flip` → exclusive |
| `refined` | consensus → `refine_instances` → exclusive  ← **PQ 0.425 path** |
| `rescue` | main refined @0.40 + gated low-conf stream (HOLD) |

Typical winning command:

```bash
.venv\Scripts\python.exe -u scripts\eval_v12_pq.py --mode refined --yolo-conf 0.40 --min-area 80
```

---

## 9. Submission path

**File:** `scripts/make_submission_v12.py`

Same refined pipeline on **test** images → RLE strings (`src/inference/rle.py`: `mask_to_rle_counts`).

Example artifact: `submissions/submission_v12_c040_a080.csv` (or similar).

---

## 10. Supporting train / export (Kaggle YOLO)

| File | Role |
|------|------|
| `kaggle_v12_upload/run_v12_kaggle.py` | Env → find MAGFiLO → export labels → train → zip `v12_export.zip` |
| `kaggle_v12_upload/scripts/export_yolo_seg_all.py` | All-annotator YOLO-seg labels |
| `kaggle_v12_upload/scripts/train_yolo11s_seg.py` | Ultralytics train wrapper |
| `kaggle_v12_upload/configs/v12_train.yaml` | YOLO hyperparams |
| `kaggle_v12_upload/src/utils/paths.py` | Bundle / competition / work roots |

After download: copy `yolo11s_best.pt` → `experiments/v12/yolo11s/weights/best.pt`, rebuild crops if boxes change a lot, optionally retrain refiner, re-eval PQ.

---

## 11. Parameter cheat-sheet (locked 0.425 system)

### YOLO (infer)
| Name | Value | Where |
|------|------:|-------|
| weights | `experiments/v12/yolo11s/weights/best.pt` | yaml `infer.yolo_weights` |
| conf | **0.40** | CLI override |
| imgsz | 1024 | `infer.yolo_imgsz` |
| iou (Ultralytics NMS) | 0.7 | `infer.yolo_iou` |

### Consensus
| Name | Value |
|------|------:|
| match_iou | 0.5 |
| nms_iou | 0.5 |
| use_consensus | true |
| use_orphans | false |

### Refiner
| Name | Value |
|------|------:|
| checkpoint | `experiments/v12/resnet18_refiner/best.pt` |
| image_size | 256 |
| pad_frac | 0.20 |
| threshold | 0.45 |
| seed_dilate | 7 |
| min_area (post) | **80** |

### Metric
| Name | Value |
|------|------:|
| PQ IoU threshold | 0.5 |
| out_size | 2048 |

---

## 12. Failure modes (why later ideas failed)

| Stage tag (FN diag) | Meaning | Fixed by refine? |
|---------------------|---------|------------------|
| `no_box` (~60%) | YOLO never proposed | **No** — need better YOLO / rescue / tiles |
| `matched_shape_fail` | Box exists, IoU&lt;0.5 after refine | Partially — better refiner / hard crops |
| `consensus_killed` | Flip-unstable | Looser consensus / single-view rescue |
| `empty_refine` | Refine empty after clamp | Seed / thr issues |

Tried and **HOLD**: dual-conf rescue, thr×dilate sweep, 4-tile miss probe (−43 GT misses only).

---

## 13. File index (quick)

| Path | Purpose |
|------|---------|
| `experiments/v12/configs/resnet18_refiner.yaml` | Master V12 config |
| `experiments/v12/yolo11s/weights/best.pt` | Detector |
| `experiments/v12/resnet18_refiner/best.pt` | Refiner |
| `experiments/v12/resnet18_refiner/train_meta.json` | Refiner train summary |
| `src/inference/v12_consensus.py` | YOLO predict + 4-flip consensus |
| `src/models/resnet_unet.py` | ResNet18-UNet layers |
| `src/inference/local_crops.py` | Box expand / resize / paste |
| `src/evaluation/pq.py` | PQ / SQ / RQ |
| `scripts/eval_v12_pq.py` | Val PQ + refine_one |
| `scripts/make_submission_v12.py` | Test CSV |
| `experiments/v12/eval/v12_refined_c040_a080_summary.json` | Locked score |
| `v12_export.zip` | Original 50-epoch YOLO export |

---

## 14. One-sentence summary

**V12 = all-annotator YOLO11s-seg@1024 proposals → 4-flip consensus → ResNet18-UNet@256 mask refine clamped to dilated YOLO seed → score-exclusive instances → PQ@0.5; best local recipe conf 0.40 + min_area 80 → PQ ≈ 0.426.**
