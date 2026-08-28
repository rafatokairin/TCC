#!/usr/bin/env bash
# Phase 2: LoRA-fine-tune SD1.5 on DEV, generate, filter, and evaluate the
# diffusion synthetic set under the SAME leakage-free classifier pipeline.
# Runs on the local RTX 4060. Driven by Claude.
set -uo pipefail
cd /home/rafatokairin/TCC
export PYTHONPATH=src
PY=.venv/bin/python3
CFG=configs/full_cbis.yaml
LOG=results/auto_diffusion.log
: > "$LOG"
echo "START $(date +%H:%M:%S)" >> "$LOG"
run() { echo ">>> $* ($(date +%H:%M:%S))" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

# 1) Train LoRA on DEV + generate a pool (1200/class) at 512px.
run $PY scripts/09_diffusion_lora.py --config "$CFG" --out results_full/diffusion \
    --steps 1500 --n-per-class 1200 || { echo "DIFF_TRAIN_FAIL" >> "$LOG"; exit 2; }

# 2) LPIPS-filter the pool + memorisation audit (lax tau; diffusion is domain-shifted).
run $PY scripts/10_filter_diffusion.py --config "$CFG" \
    --raw results_full/diffusion/raw --out results_full/diffusion/filtered \
    --tau 0.35 --n-target 780 || { echo "DIFF_FILTER_FAIL" >> "$LOG"; exit 3; }
B=$(ls results_full/diffusion/filtered/BENIGN_*.png 2>/dev/null | wc -l)
M=$(ls results_full/diffusion/filtered/MALIGNANT_*.png 2>/dev/null | wc -l)
echo "DIFF_RETAINED benign=$B malignant=$M" >> "$LOG"

# 3) Fidelity of diffusion set.
run $PY scripts/03_compute_fidelity.py --config "$CFG" \
    --synthetic results_full/diffusion/filtered --out results_full/diffusion/fidelity.json || echo "FID_WARN" >> "$LOG"

# 4) Classification with diffusion synthetic (same protocol, powered TEST).
run $PY scripts/04_run_classification.py --config "$CFG" \
    --synthetic results_full/diffusion/filtered --out results_full/classification_diffusion \
    || { echo "DIFF_CLS_FAIL" >> "$LOG"; exit 5; }

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
