# Experiments

Document a row when a version finishes. Never invent scores.

| Version | Model | Resolution | Loss | Augmentation | Val Dice | Val IoU | Official Metric (local PQ) | Kaggle Score | Notes |
| ------- | ----- | ---------- | ---- | ------------ | -------- | ------- | -------------------------- | ------------ | ----- |
| V0 | — | — | — | — | — | — | — | — | Data understood; split 566/141 (seed 42); metric TBD |
| V1 | U-Net | 512 | BCE+Dice | none | **0.643** | **0.483** | — | — | Thin filaments lost + FP noise; V2 = higher res/patches |
| V2 | U-Net | 768 | BCE+Dice | none | **0.668** | **0.510** | PQ **0.312** (U-Net+CC) | — | Small gain vs V1; hard thin cases remain |
| V3 | YOLOv8n-seg | 640 | YOLO | default | — | — | PQ **0.259** | — | Native instances; weak masks + high FN |
| V4 | Hybrid (YOLO n-seg boxes + V2 U-Net + orphans) | 640/768 | — | — | — | — | PQ **0.330** (exclusive later **0.346**) | LB ~0.25 | Best packaging; exclusive masks for Kaggle CSV |
| V5 | Hybrid (YOLOv8s **detect** @1024 + V2 U-Net + orphans) | 1024/768 | — | — | — | — | PQ **0.338** | — | ACCEPT vs 0.330; not SUBMIT_READY (>0.35); s-seg@1024 OOM → detect |
| V6-A | Hybrid (V4 YOLO n + **local U-Net@512** + V2 orphans) | 512 crops / 2048 | BCE+Dice | flips | — | — | PQ **0.292** | — | REJECT vs V4 0.346 (ep5 ckpt); crop Dice 0.787 ≠ PQ gain |

## Per-version rule

When a version completes, update:

1. `experiments/<version>/notes.md`
2. This table
3. Any sample predictions / configs for that version

Final technical report / polished results docs: later.
