"""Data layer: manifest construction, de-duplication, patient-level splits.

Note: `MammogramDataset`/`build_transforms` live in `breastsynth.data.dataset`
and are imported on demand (they require torch), so the torch-free split logic
here stays importable without a GPU stack.
"""
from breastsynth.data.splits import (
    balance_classes,
    foldwise_splits,
    holdout_split,
    inner_validation_split,
)

__all__ = [
    "balance_classes",
    "holdout_split",
    "foldwise_splits",
    "inner_validation_split",
]
