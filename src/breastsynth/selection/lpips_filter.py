"""LPIPS-based selection of synthetic images (Reviewer 1: threshold + memorisation).

Reference reals come from ONE split only (DEV in the hold-out protocol, or the
fold's training set in the fold-wise protocol). This is the selection-leakage
fix: the filter never references evaluation images.

Two entry points:
  * `generate_and_filter` — rejection sampling to retain `n_target` images per
    class with min-LPIPS < threshold; logs generated/retained/discarded counts
    and the memorisation rate.
  * `threshold_sweep` — generate a fixed pool ONCE, record each sample's
    min-LPIPS, then report retention at several thresholds (justifies τ=0.2
    without regenerating).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from breastsynth.config import SelectionConfig
from breastsynth.generative.generate import sample_images
from breastsynth.metrics.lpips_metrics import memorization_rate, nearest_real, to_ref_batch

_CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def load_reference_reals(
    paths: list[str], img_size: int = 128, max_images: int = 100, seed: int = 42
) -> list[torch.Tensor]:
    """Load up to `max_images` reference reals as 1x1xHxW tensors in [-1,1]."""
    t = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    rng = np.random.RandomState(seed)
    chosen = rng.choice(paths, size=min(max_images, len(paths)), replace=False)
    return [t(Image.open(p).convert("L")).unsqueeze(0) for p in chosen]


def generate_and_filter(
    gen,
    reference_by_class: dict[int, list[torch.Tensor]],
    n_target_per_class: int,
    cfg: SelectionConfig,
    z_dim: int,
    n_classes: int,
    out_dir: str | Path,
    loss_fn=None,
    device: str = "cuda",
) -> dict:
    """Rejection-sample synthetic images until `n_target_per_class` pass the filter.

    Returns a report dict with per-class generated/retained counts, acceptance
    rate, min-LPIPS distribution stats and memorisation rate.
    """
    import lpips

    if loss_fn is None:
        loss_fn = lpips.LPIPS(net=cfg.lpips_net).to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previously-retained samples so a low-yield run can never silently
    # reuse stale images from an earlier run/config.
    for old in list(out_dir.glob("BENIGN_*.png")) + list(out_dir.glob("MALIGNANT_*.png")):
        old.unlink()

    report: dict = {"threshold": cfg.threshold, "per_class": {}}
    for class_label, refs in reference_by_class.items():
        name = _CLASS_NAMES.get(class_label, str(class_label))
        ref_batch = to_ref_batch(refs, device)  # stack once; batched LPIPS per candidate
        retained, generated, retained_dists = 0, 0, []
        max_generate = n_target_per_class * cfg.max_attempts_factor
        # Abort early if the generator can't clear the threshold: after a probe
        # window, if acceptance is ~0 we stop instead of grinding for hours (this
        # happens when the GAN is undertrained or the threshold is too strict).
        probe_after = max(500, n_target_per_class)
        min_acceptance = 0.002
        pbar = tqdm(total=n_target_per_class, desc=f"filter {name}")
        aborted = False
        for img in sample_images(gen, class_label, max_generate, z_dim, n_classes, device):
            generated += 1
            dist, _ = nearest_real(loss_fn, img, ref_batch, device)
            if dist < cfg.threshold:
                retained += 1
                retained_dists.append(dist)
                save_image((img + 1) / 2, out_dir / f"{name}_{retained:04d}.png")
                pbar.update(1)
                if retained >= n_target_per_class:
                    break
            if generated >= probe_after and (retained / generated) < min_acceptance:
                warnings.warn(
                    f"[{name}] acceptance {retained}/{generated} < {min_acceptance:.1%} at "
                    f"tau={cfg.threshold}: stopping early. Train the GAN longer or raise the "
                    f"threshold (check threshold_sweep.json for a feasible tau).",
                    stacklevel=2,
                )
                aborted = True
                break
        pbar.close()
        mem = memorization_rate(np.array(retained_dists), eps=cfg.memorization_eps)
        report["per_class"][name] = {
            "generated": generated,
            "retained": retained,
            "discarded": generated - retained,
            "aborted_low_acceptance": aborted,
            "acceptance_rate": retained / max(1, generated),
            "min_lpips_mean": float(np.mean(retained_dists)) if retained_dists else float("nan"),
            "min_lpips_std": float(np.std(retained_dists)) if retained_dists else float("nan"),
            "memorization": mem,
        }
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    return report


def threshold_sweep(
    gen,
    reference_by_class: dict[int, list[torch.Tensor]],
    cfg: SelectionConfig,
    z_dim: int,
    n_classes: int,
    pool_per_class: int = 1000,
    loss_fn=None,
    device: str = "cuda",
) -> dict:
    """Generate a fixed pool once; report retention/memorisation at each threshold.

    This is what justifies the operating threshold (Reviewer 1): it shows how the
    retained fraction and memorisation risk change with τ, without regenerating.
    """
    import lpips

    if loss_fn is None:
        loss_fn = lpips.LPIPS(net=cfg.lpips_net).to(device)

    dists_by_class: dict[str, list[float]] = {}
    for class_label, refs in reference_by_class.items():
        name = _CLASS_NAMES.get(class_label, str(class_label))
        ref_batch = to_ref_batch(refs, device)  # stack once; batched LPIPS per candidate
        dists = []
        for img in tqdm(
            sample_images(gen, class_label, pool_per_class, z_dim, n_classes, device),
            total=pool_per_class,
            desc=f"sweep {name}",
        ):
            dists.append(nearest_real(loss_fn, img, ref_batch, device)[0])
        dists_by_class[name] = dists

    all_dists = [d for v in dists_by_class.values() for d in v]
    sweep = {}
    for tau in cfg.threshold_sweep:
        per_class = {
            name: {
                "retained_fraction": float(np.mean(np.array(d) < tau)),
                "memorization": memorization_rate(
                    np.array([x for x in d if x < tau]), eps=cfg.memorization_eps
                ),
            }
            for name, d in dists_by_class.items()
        }
        sweep[str(tau)] = {
            "overall_retained_fraction": float(np.mean(np.array(all_dists) < tau)),
            "per_class": per_class,
        }
    return {
        "pool_per_class": pool_per_class,
        "distances": dists_by_class,
        "sweep": sweep,
    }
