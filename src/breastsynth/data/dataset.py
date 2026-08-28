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


def build_transforms(image_size: int = 128, train: bool = False, augment: str = "basic") -> T.Compose:
    """Classifier transforms with selectable train-time augmentation strength.

    augment (train only):
      * "none"   — no augmentation (resize + normalise only);
      * "basic"  — horizontal flip + small rotation (the default used throughout);
      * "strong" — a conventional strong-augmentation recipe (affine translate/
        scale, wider rotation, photometric jitter, and random erasing / cutout),
        used as the classic-augmentation baseline against GAN augmentation.

    Laterality is normalised at manifest time, so geometry stays mild-to-moderate.
    """
    ops = [T.Resize((image_size, image_size))]
    if train and augment != "none":
        if augment == "strong":
            ops += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        else:  # basic
            ops += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(5)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train and augment == "strong":
        ops += [T.RandomErasing(p=0.25, scale=(0.02, 0.12))]  # cutout
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
