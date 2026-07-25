"""Torch dataset + transforms for the classifier (Reviewer 1: augmentation settings)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 128, train: bool = False) -> T.Compose:
    """Classifier transforms. Train-time augmentation is flip + small rotation only.

    Note: mammogram L/R laterality is normalised at manifest time, so we keep
    augmentation conservative (no aggressive geometric distortion of lesions).
    """
    ops = [T.Resize((image_size, image_size))]
    if train:
        ops += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(5)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class MammogramDataset(Dataset):
    """Serves (image_tensor, label) pairs from a list of paths+labels.

    Accepts explicit path/label lists so callers keep full control over which
    rows (real vs synthetic, which split) go into each dataset — this is what
    keeps the leakage-free protocol auditable.
    """

    def __init__(self, paths: list[str], labels: list[int], transform: T.Compose | None = None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform or build_transforms(train=False)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        transform: T.Compose | None = None,
        path_col: str = "path",
        label_col: str = "label",
    ) -> "MammogramDataset":
        return cls(df[path_col].tolist(), df[label_col].astype(int).tolist(), transform)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)
