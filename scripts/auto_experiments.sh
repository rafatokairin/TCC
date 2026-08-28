#!/usr/bin/env bash
# Runs the two new experiments (augmentation comparison + learning curve) on the
# local GPU, reusing the patient-level synthetic set. Driven by Claude.
set -uo pipefail
cd /home/rafatokairin/TCC
PY=.venv/bin/python3
LOG=results/auto_experiments.log
: > "$LOG"
echo "START $(date +%H:%M:%S)" >> "$LOG"

echo ">>> 07 augmentation comparison ($(date +%H:%M:%S))" >> "$LOG"
$PY scripts/07_augmentation_comparison.py --config configs/finetune.yaml \
    --synthetic results/synthetic --gan-ratio 2 --out results/augcompare >> "$LOG" 2>&1 \
    || { echo "AUGCOMPARE_FAIL" >> "$LOG"; exit 2; }

echo ">>> 08 learning curve ($(date +%H:%M:%S))" >> "$LOG"
$PY scripts/08_learning_curve.py --config configs/finetune.yaml \
    --synthetic results/synthetic --gan-ratio 2 --levels 40 80 120 160 --out results/learningcurve >> "$LOG" 2>&1 \
    || { echo "LEARNINGCURVE_FAIL" >> "$LOG"; exit 3; }

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
