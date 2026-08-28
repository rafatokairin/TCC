#!/usr/bin/env bash
# Phase 1: statistically-powered study on the FULL CBIS-DDSM set (~2857 imgs).
# Runs the whole leakage-free pipeline on the local GPU. Driven by Claude.
set -uo pipefail
cd /home/rafatokairin/TCC
PY=.venv/bin/python3
CFG=configs/full_cbis.yaml
LOG=results/auto_full_cbis.log
: > "$LOG"
echo "START $(date +%H:%M:%S)" >> "$LOG"
run() { echo ">>> $* ($(date +%H:%M:%S))" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

# 1) GAN on DEV (2241 imgs, patient-level) — checkpoints/resumes.
run $PY scripts/01_train_gan.py --config "$CFG" --out results_full/gan/ckpt_dev.pth \
    || { echo "GAN_FAIL" >> "$LOG"; exit 2; }

# 2) generate + auto-threshold + filter
run $PY scripts/02_generate_and_filter.py --config "$CFG" \
    --ckpt results_full/gan/ckpt_dev.pth --out results_full/synthetic --auto-threshold --min-retention 0.05 \
    || { echo "GEN_FAIL" >> "$LOG"; exit 3; }
B=$(ls results_full/synthetic/BENIGN_*.png 2>/dev/null | wc -l)
M=$(ls results_full/synthetic/MALIGNANT_*.png 2>/dev/null | wc -l)
echo "RETAINED benign=$B malignant=$M" >> "$LOG"
[ "$B" -lt 400 ] || [ "$M" -lt 400 ] && { echo "ABORT_LOW_RETENTION b=$B m=$M" >> "$LOG"; exit 4; }

# 3) fidelity
run $PY scripts/03_compute_fidelity.py --config "$CFG" \
    --synthetic results_full/synthetic --out results_full/synthetic/fidelity.json || echo "FID_WARN" >> "$LOG"

# 4) classification (powered TEST, 10 seeds) + significance
run $PY scripts/04_run_classification.py --config "$CFG" \
    --synthetic results_full/synthetic --out results_full/classification \
    || { echo "CLS_FAIL" >> "$LOG"; exit 5; }

# 5) augmentation comparison (GAN vs classic)
run $PY scripts/07_augmentation_comparison.py --config "$CFG" \
    --synthetic results_full/synthetic --gan-ratio 2 --out results_full/augcompare || echo "AUG_WARN" >> "$LOG"

# 6) learning curve (does synthetic help when real is scarce, with a good GAN?)
run $PY scripts/08_learning_curve.py --config "$CFG" \
    --synthetic results_full/synthetic --gan-ratio 2 --levels 100 300 600 --out results_full/learningcurve \
    || echo "LC_WARN" >> "$LOG"

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
