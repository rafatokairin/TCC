#!/usr/bin/env bash
# End-to-end leakage-free pipeline (hold-out protocol). Run from repo root.
# Prerequisites: pip install -e . ; a CUDA GPU ; data/manifest.csv built.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
GAN_CKPT="results/gan/ckpt_dev.pth"
SYN_DIR="results/synthetic"
CLS_DIR="results/classification"

echo "[0/5] Build manifest (skip if data/manifest.csv already exists)"
[ -f data/manifest.csv ] || python scripts/00_build_manifest.py \
    --image-root data/dataset128 --legacy-csv data/dataset.csv

echo "[1/5] Train StyleGAN2-ADA on DEV only"
python scripts/01_train_gan.py --config "$CONFIG" --out "$GAN_CKPT"

echo "[2/5] Generate + LPIPS threshold sweep + filter (DEV reference)"
python scripts/02_generate_and_filter.py --config "$CONFIG" --ckpt "$GAN_CKPT" --out "$SYN_DIR"

echo "[3/5] Fidelity (FID/KID)"
python scripts/03_compute_fidelity.py --config "$CONFIG" --synthetic "$SYN_DIR" \
    --out "$SYN_DIR/fidelity.json"

echo "[4/5] Classification on locked TEST + significance tests"
python scripts/04_run_classification.py --config "$CONFIG" --synthetic "$SYN_DIR" --out "$CLS_DIR"

echo "[5/5] Render LaTeX tables for the papers"
python scripts/05_make_tables.py --summary "$CLS_DIR/summary.json" \
    --fidelity "$SYN_DIR/fidelity.json" --out paper/generated

echo "Done. See $CLS_DIR/ and paper/generated/."
