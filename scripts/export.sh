#!/usr/bin/env bash
# Test export: detector -> sharded export (one process per GPU) -> merge -> validate -> sha256.
# Reads the 180 test IMAGES only. Never submits.
#   V65: MIN_AREA=20  MASK_T=0.45 REFINER=weights/r3all_refiner_state.pt NAME=submission_v65 bash scripts/export.sh
#   V84: MIN_AREA=200 MASK_T=0.50 REFINER=weights/soft_all_refiner_state.pt DEMOTE=weights/demote_all.json \
#        NAME=submission_v84 bash scripts/export.sh
# Env: DET (weights/p2first_all707.pt), IMGSZ (1024), MIN_AREA (200), REFINER (r3all), MASK_T (0.45),
#   DEMOTE (empty = off), GPUS ("0"; one shard per entry, e.g. "0 1 2"), DEVICE (cuda:0; cpu works, slowly),
#   TEST_DIR (data/raw/.../test/test_images), OUT (outputs/reproduce), NAME (submission), PY (python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DET=${DET:-weights/p2first_all707.pt}
OUT=${OUT:-outputs/reproduce}
IMGSZ=${IMGSZ:-1024}
MIN_AREA=${MIN_AREA:-200}
REFINER=${REFINER:-weights/r3all_refiner_state.pt}
MASK_T=${MASK_T:-0.45}
NAME=${NAME:-submission}
TEST_DIR=${TEST_DIR:-data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images}
DEMOTE_ARGS=(); [ -n "${DEMOTE:-}" ] && DEMOTE_ARGS=(--demote-model "$DEMOTE")
read -ra GPUS <<< "${GPUS:-0}"
N=${#GPUS[@]}
SHARDS="$OUT/$NAME"
mkdir -p "$SHARDS"
DSHA=$(sha256sum "$DET" | cut -d' ' -f1); RSHA=$(sha256sum "$REFINER" | cut -d' ' -f1)
echo "$(date -Is) $NAME detector $DET $DSHA | refiner $REFINER $RSHA | imgsz $IMGSZ min_area $MIN_AREA mask_t $MASK_T demote ${DEMOTE:-off}"

pids=()
for s in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$s]} $PY solar/export_v74.py --shard-index "$s" --num-shards "$N" \
    --detector "$DET" --detector-sha "$DSHA" --refiner "$REFINER" --refiner-sha "$RSHA" \
    --imgsz "$IMGSZ" --min-area "$MIN_AREA" --mask-threshold "$MASK_T" --confidence 0.30 "${DEMOTE_ARGS[@]}" \
    --device "${DEVICE:-cuda:0}" --test-dir "$TEST_DIR" --output-dir "$SHARDS" > "$SHARDS/shard$s.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "FAIL shard (see $SHARDS/shard*.log)"; exit 1; }; done

$PY - "$OUT" "$SHARDS" "$N" "$TEST_DIR" "$MIN_AREA" "$NAME" <<'EOF'
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, ".")
from tools.validate_submission import validate_submission
out, shards, n, test, min_area, name = (Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]),
                                        Path(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
for s in range(n):
    m = json.loads((shards / f"shard_{s}.json").read_text())
    assert m["status"] == "complete" and m["num_shards"] == n, f"shard {s} incomplete"
df = pd.concat([pd.read_csv(shards / f"shard_{s}.csv", dtype=str) for s in range(n)], ignore_index=True)
assert df.filament_id.is_unique
csv = out / f"{name}.csv"
df.to_csv(csv, index=False)
stems = {p.stem for p in test.glob("*.jpeg")}
assert len(stems) == 180, len(stems)
report = validate_submission(csv, stems, min_area=min_area)
(out / f"{name}.validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
EOF
(cd "$OUT" && sha256sum "$NAME.csv" > "$NAME.csv.sha256")
echo "$(date -Is) done: $OUT/$NAME.csv $(cut -d' ' -f1 "$OUT/$NAME.csv.sha256") (validated, NOT submitted)"
