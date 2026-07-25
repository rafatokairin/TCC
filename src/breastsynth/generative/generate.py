"""Sample synthetic images from a trained conditional generator."""
from __future__ import annotations

import torch

from breastsynth.models.stylegan2ada import combine_vectors, one_hot_labels


@torch.no_grad()
def sample_images(gen, class_label: int, n: int, z_dim: int, n_classes: int, device: str = "cuda"):
    """Yield `n` synthetic images (1x1xHxW tensors in [-1,1]) for one class.

    Generates one at a time so the LPIPS filter can accept/reject per sample
    (rejection sampling) without holding a large batch in memory.
    """
    gen.eval()
    for _ in range(n):
        z = torch.randn(1, z_dim, device=device)
        oh = one_hot_labels(torch.tensor([class_label], device=device), n_classes)
        yield gen(combine_vectors(z, oh)).cpu()
