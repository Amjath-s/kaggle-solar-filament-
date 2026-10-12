# Acceptance run (2026-10-12)

- Environment: a fresh venv built from `requirements.txt` only (plus nbconvert 7.17.1 and ipykernel 7.3.0 to execute
  the notebook). Python 3.10.12, torch 2.4.1+cu121, 3x Tesla T4 (driver 610.43.02).
- Command: `SOLAR_GPUS="0 1 2" jupyter nbconvert --to notebook --execute --inplace notebooks/reproduce_submission.ipynb`
- Input: the 180 test images only.

| | V65 | V84 |
|---|---|---|
| regenerated sha256 | 530e89ea0f137757c6efc714c1a5878c2ff7e73bbb9a53e14815604b3b473c66 | 2925e2a28cd789d500efc828be2858ea0edd7324ff98d09bd51e6c5295c4871f |
| submitted sha256 | same (**byte-identical**) | same (**byte-identical**) |
| rows | 1224 | 1112 |
| shards / GPUs | 4 shards on GPUs 0,1,2,0 | 3 shards on GPUs 0,1,2 |
| time per image per process | 1.83 s | 2.48 s (the demote features run on CPU) |
| wall time for 180 images | 136 s | 198 s |

The time per image counts the 4-flip detector, the refiner TTA, post-processing and RLE. Model loading is excluded.

- Row order follows the shard layout (image *i* goes to shard *i* mod *n*). With 3 shards, V65 has the same rows
  (every mask has IoU 1.0) in a different order. With its original 4 shards, it is byte-identical.
- Tests: `pytest` gives 49 passed. `ruff check .` is clean.
