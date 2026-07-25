"""Automatic dataset download from Kaggle via kagglehub.

Mirrors the usual snippet:
    import kagglehub
    path = kagglehub.dataset_download("<slug>")

A small registry maps friendly names to Kaggle slugs so scripts can say
`--dataset mias` instead of the full slug. Requires Kaggle credentials to be
configured (kagglehub uses ~/.kaggle/kaggle.json or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables).
"""
from __future__ import annotations

# Friendly name -> Kaggle dataset slug. Add your own here.
KAGGLE_DATASETS = {
    "cbis-ddsm": "awsaf49/cbis-ddsm-breast-cancer-image-dataset",
    "mias": "kmader/mias-mammography",
    "inbreast": "tommyngx/inbreast2012",
}


def resolve_slug(name_or_slug: str) -> str:
    return KAGGLE_DATASETS.get(name_or_slug.lower(), name_or_slug)


def download_kaggle(name_or_slug: str) -> str:
    """Download a Kaggle dataset and return the local path.

    Accepts a friendly name from KAGGLE_DATASETS or a raw 'owner/dataset' slug.
    """
    import kagglehub

    slug = resolve_slug(name_or_slug)
    path = kagglehub.dataset_download(slug)
    print(f"Downloaded '{slug}' to: {path}")
    return path
