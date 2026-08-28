#!/usr/bin/env bash
# Finalize the full-CBIS study: usable diffusion set (best-780 regardless of tau)
# + its FID + classification; then external validation + image-level ablation.
set -uo pipefail
cd /home/rafatokairin/TCC
export PYTHONPATH=src
PY=.venv/bin/python3
CFG=configs/full_cbis.yaml
LOG=results/auto_finalize.log
: > "$LOG"
echo "START $(date +%H:%M:%S)" >> "$LOG"
run() { echo ">>> $* ($(date +%H:%M:%S))" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

# 1) Keep the 780 most-real diffusion images (no tau cutoff; characterise honestly).
run $PY scripts/10_filter_diffusion.py --config "$CFG" \
    --raw results_full/diffusion/raw --out results_full/diffusion/best780 \
    --tau 0.99 --n-target 780 || echo "DIFF_FILTER_WARN" >> "$LOG"

# 2) FID/KID of the diffusion set vs real DEV.
run $PY scripts/03_compute_fidelity.py --config "$CFG" \
    --synthetic results_full/diffusion/best780 --out results_full/diffusion/fidelity_best.json || echo "FID_WARN" >> "$LOG"

# 3) Classification with diffusion augmentation (powered TEST).
run $PY scripts/04_run_classification.py --config "$CFG" \
    --synthetic results_full/diffusion/best780 --out results_full/classification_diffusion || echo "DIFFCLS_WARN" >> "$LOG"

# 4) External validation (MIAS) with the full-CBIS model.
run $PY scripts/06_external_validation.py --config "$CFG" --dataset mias \
    --root /home/rafatokairin/.cache/kagglehub/datasets/kmader/mias-mammography/versions/3 \
    --synthetic results_full/synthetic --out results_full/external_mias || echo "EXT_WARN" >> "$LOG"

# 5) Image-level (leaky) ablation on the powered dataset.
run $PY scripts/04_run_classification.py --config configs/full_cbis_imagelevel.yaml \
    --synthetic results_full/synthetic --out results_full/classification_imagelevel || echo "ABL_WARN" >> "$LOG"

echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
