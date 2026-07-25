"""LPIPS distances + nearest-neighbour / memorisation analysis (Reviewer 1).

`nearest_real` returns, for each candidate (synthetic) image, its minimum LPIPS
distance to a reference real set AND the index of that nearest real image. This
serves both the selection filter and the memorisation check (a synthetic image
whose nearest real neighbour is a near-copy is flagged).
"""
from __future__ import annotations

import numpy as np
import torch


def _to_lpips_input(img: torch.Tensor) -> torch.Tensor:
    """Ensure a 1x3xHxW tensor in [-1, 1] for the LPIPS network."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.size(1) == 1:
        img = img.repeat(1, 3, 1, 1)
    return img


def to_ref_batch(reference_reals: list[torch.Tensor], device: str = "cuda") -> torch.Tensor:
    """Stack reference reals into one [N,3,H,W] tensor on device (build once)."""
    return torch.cat([_to_lpips_input(r) for r in reference_reals], dim=0).to(device).contiguous()


@torch.no_grad()
def nearest_real(
    loss_fn,
    candidate: torch.Tensor,
    reference_reals,
    device: str = "cuda",
) -> tuple[float, int]:
    """Return (min LPIPS distance, argmin index) of `candidate` vs references.

    `reference_reals` may be a list of tensors or a pre-stacked [N,3,H,W] tensor
    (from `to_ref_batch`). All N references are compared in a single batched
    forward pass — dramatically faster than looping one reference at a time.
    """
    ref_batch = (
        reference_reals
        if isinstance(reference_reals, torch.Tensor)
        else to_ref_batch(reference_reals, device)
    )
    cand = _to_lpips_input(candidate.to(device))
    cand_rep = cand.expand(ref_batch.size(0), -1, -1, -1).contiguous()
    dists = loss_fn(ref_batch, cand_rep).view(-1)
    j = int(torch.argmin(dists).item())
    return float(dists[j].item()), j


@torch.no_grad()
def lpips_distribution(
    loss_fn,
    candidates: list[torch.Tensor],
    reference_reals: list[torch.Tensor],
    device: str = "cuda",
) -> np.ndarray:
    """Min-LPIPS distance for every candidate (used for distribution plots)."""
    ref_batch = to_ref_batch(reference_reals, device)
    return np.array([nearest_real(loss_fn, c, ref_batch, device)[0] for c in candidates])


def memorization_rate(min_distances: np.ndarray, eps: float = 0.05) -> dict:
    """Fraction of samples that are suspected near-copies (dist < eps)."""
    min_distances = np.asarray(min_distances, dtype=float)
    flagged = int((min_distances < eps).sum())
    return {
        "eps": eps,
        "n": int(len(min_distances)),
        "n_flagged": flagged,
        "rate": float(flagged / max(1, len(min_distances))),
        "min": float(min_distances.min()) if len(min_distances) else float("nan"),
        "mean": float(min_distances.mean()) if len(min_distances) else float("nan"),
    }
