"""Generative fidelity/diversity metrics: FID and KID (Reviewer 1 & 2).

LPIPS alone cannot detect low diversity or mode collapse. FID and KID compare the
whole real vs synthetic distributions and complement the LPIPS analysis. We use
`torchmetrics` implementations (Inceptionv3 features). Grayscale mammograms are
replicated to 3 channels.

KID is preferred for small samples (unbiased estimator with a variance), which
matters here given the modest dataset size.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class _ImageFolderList(Dataset):
    def __init__(self, paths: list[str], size: int = 128):
        self.paths = paths
        self.t = transforms.Compose(
            [transforms.Resize((size, size)), transforms.Grayscale(3), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # torchmetrics FID/KID expect uint8 [0,255] tensors.
        return (self.t(Image.open(self.paths[i]).convert("L")) * 255).to(torch.uint8)


def _feed(metric, paths, device, batch_size, real: bool):
    loader = DataLoader(_ImageFolderList(paths), batch_size=batch_size)
    for batch in loader:
        metric.update(batch.to(device), real=real)


@torch.no_grad()
def compute_fid_kid(
    real_paths: list[str],
    fake_paths: list[str],
    device: str = "cuda",
    batch_size: int = 32,
    kid_subset_size: int = 50,
) -> dict:
    """Compute FID and KID between two image sets.

    Returns {"fid": float, "kid_mean": float, "kid_std": float, "n_real":..,
    "n_fake":..}. `kid_subset_size` is clamped to the smaller set size.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    fid = FrechetInceptionDistance(normalize=False).to(device)
    subset = min(kid_subset_size, len(real_paths), len(fake_paths))
    kid = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)

    for metric in (fid, kid):
        _feed(metric, real_paths, device, batch_size, real=True)
        _feed(metric, fake_paths, device, batch_size, real=False)

    kid_mean, kid_std = kid.compute()
    return {
        "fid": float(fid.compute()),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "n_real": len(real_paths),
        "n_fake": len(fake_paths),
    }


def list_images(folder: str | Path, exts=(".png", ".jpg", ".jpeg")) -> list[str]:
    return [str(p) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in exts]
