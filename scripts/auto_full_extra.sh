#!/usr/bin/env bash
# Consistency runs on the FULL CBIS-DDSM: external validation (MIAS) with the
# powered model, and the image-level (leaky) ablation. Driven by Claude.
set -uo pipefail
cd /home/rafatokairin/TCC
export PYTHONPATH=src
PY=.venv/bin/python3
LOG=results/auto_full_extra.log
: > "$LOG"
echo "START $(date +%H:%M:%S)" >> "$LOG"
run() { echo ">>> $* ($(date +%H:%M:%S))" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

# 1) External validation on MIAS with the full-CBIS-trained model.
run $PY scripts/06_external_validation.py --config configs/full_cbis.yaml --dataset mias \
    --root /home/rafatokairin/.cache/kagglehub/datasets/kmader/mias-mammography/versions/3 \
    --synthetic results_full/synthetic --out results_full/external_mias || echo "EXT_WARN" >> "$LOG"

# 2) Image-level (leaky) ablation: same synthetic, non-patient split -> shows if
#    same-patient leakage inflates results even on the powered dataset.
run $PY scripts/04_run_classification.py --config configs/full_cbis_imagelevel.yaml \
    --synthetic results_full/synthetic --out results_full/classification_imagelevel || echo "ABL_WARN" >> "$LOG"

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
