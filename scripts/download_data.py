#!/usr/bin/env python
"""Download a dataset from Kaggle via kagglehub and print its local path.

    python scripts/download_data.py --dataset mias
    python scripts/download_data.py --dataset cbis-ddsm
    python scripts/download_data.py --dataset owner/some-dataset   # raw slug

Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
Known names: cbis-ddsm, mias, inbreast (see breastsynth/data/download.py).
"""
from __future__ import annotations

import argparse

from breastsynth.data.download import KAGGLE_DATASETS, download_kaggle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help=f"name or slug. known: {list(KAGGLE_DATASETS)}")
    args = ap.parse_args()
    path = download_kaggle(args.dataset)
    print(f"\nLocal path: {path}")
    print("Pass this path to scripts/06_external_validation.py (--root) or "
          "scripts/prepare_images.py / 00_build_manifest.py as needed.")


if __name__ == "__main__":
    main()
