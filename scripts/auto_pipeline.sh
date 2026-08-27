#!/usr/bin/env bash
# Autonomous re-run of the patient-level pipeline (steps 2-6). Reuses the
# already-trained patient-level GAN. Guards against low synthetic retention so a
# bad tau can never silently contaminate the classifier. Driven by Claude.
set -uo pipefail
cd /home/rafatokairin/TCC
PY=.venv/bin/python3
CFG="${1:-configs/finetune.yaml}"
LOG=results/auto_pipeline.log
: > "$LOG"
echo "START $(date +%H:%M:%S) config=$CFG" >> "$LOG"

run() { echo ">>> $* ($(date +%H:%M:%S))" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

# --- Step 2: generate + filter (clears stale samples internally) ---
run $PY scripts/02_generate_and_filter.py --config "$CFG" \
    --ckpt results/gan/ckpt_dev.pth --out results/synthetic || { echo "STEP2_FAIL" >> "$LOG"; exit 2; }

B=$(ls results/synthetic/BENIGN_*.png 2>/dev/null | wc -l)
M=$(ls results/synthetic/MALIGNANT_*.png 2>/dev/null | wc -l)
echo "RETAINED benign=$B malignant=$M" >> "$LOG"
if [ "$B" -lt 400 ] || [ "$M" -lt 400 ]; then
  echo "ABORT_LOW_RETENTION benign=$B malignant=$M (need >=400/class)" >> "$LOG"; exit 3
fi

# --- Step 3: fidelity ---
run $PY scripts/03_compute_fidelity.py --config "$CFG" \
    --synthetic results/synthetic --out results/synthetic/fidelity.json || { echo "STEP3_FAIL" >> "$LOG"; exit 4; }

# --- Step 4: classification (fine-tuned, patient-level TEST) ---
run $PY scripts/04_run_classification.py --config "$CFG" \
    --synthetic results/synthetic --out results/classification_ft || { echo "STEP4_FAIL" >> "$LOG"; exit 5; }

# --- Step 6: external validation on MIAS ---
run $PY scripts/06_external_validation.py --config "$CFG" --dataset mias \
    --root /home/rafatokairin/.cache/kagglehub/datasets/kmader/mias-mammography/versions/3 \
    --synthetic results/synthetic --out results/external_mias || { echo "STEP6_FAIL" >> "$LOG"; exit 6; }

# --- Step 5: render LaTeX tables ---
run $PY scripts/05_make_tables.py --summary results/classification_ft/summary.json \
    --fidelity results/synthetic/fidelity.json \
    --external results/external_mias/external_summary.json --external-name MIAS || { echo "STEP5_FAIL" >> "$LOG"; exit 7; }

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
