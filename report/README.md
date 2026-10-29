# Report

`report.tex` is the technical report (two-column draft; port it into the host Overleaf template linked on its
first line). Every number in it comes from the 5-fold out-of-fold (OOF) predictions on the 707 training files or
from the pre-registered experiment logs; no test labels are used anywhere.

## Figures

All figures are regenerated on CPU from OOF artefacts that the training steps in the main `README.md` produce
(`outputs/` and `work/` are gitignored). Run from the repo root.

| figure | script | reads | produced by |
|---|---|---|---|
| `figures/host_plots.pdf` (IoU, Dice, 1:n / n:1, host self-evaluation logic) | `report/plots.py` | `outputs/v84/extract_soft.pkl` | `python solar/v84_extract.py soft` |
| `figures/agreement.pdf` (recall vs annotator agreement, FP classes) | `report/agreement.py` | `outputs/v85_analysis/rows.pkl` | `python solar/v85_extract.py`, then `python solar/v85_analyze.py` |
| `figures/examples.pdf` (three OOF examples, fixed selection rule) | `report/examples.py` | `outputs/v87/fold*_v84.csv`, train images | `python solar/v87_export_parity.py <fold>` for folds 0–4 (needs `python solar/v84_demote_fit.py` first) |

`plots.py` and `solar/v85_analyze.py` assert that the V84 bundle reproduces OOF PQ 0.45898; `agreement.py` checks
the row counts (8,199 annotated filaments, 1,881 record-level FP). Each figure script writes a small
`*_stats.json` next to its figure with the numbers quoted in the text.
`solar/v87_export_parity.py` runs the exact test-export code on each fold's held-out training images and checks
the result against the OOF counts; this is the "export reproduces OOF bit-for-bit" check cited in the report.

The scripts were run in the development repository and copied here with only import and data paths changed
(`scripts/` → `solar/`, fold detectors under `work/v61/` as written by the training CLI).
