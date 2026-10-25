# Methodology

## Problem

Solar filament segmentation on H-alpha images (pixel-level masks). Exact semantic vs instance formulation: confirm from competition files during V0.

## Local-first workflow

Develop and validate locally. Use Kaggle for submission scoring only. Keep the official test set unseen during development.

## Split policy

Create train/validation from training data. Investigate temporal/grouping leakage before assuming a random split.

## Hardware constraints

- GPU: NVIDIA GeForce RTX 3050 Laptop
- VRAM: 4096 MiB
- Design consequence: prefer AMP, small batches, careful resolution / patch strategies. Do not pick models that cannot train on this GPU.

## Versioning

Each milestone (V0, V1, …) gets documented when complete. Final consolidated report later.
