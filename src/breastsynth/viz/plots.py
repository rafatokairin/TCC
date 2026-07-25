"""Publication figures. All functions save to disk and return the output path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_lpips_distribution(
    dists_by_class: dict[str, list[float]], out: str | Path, threshold: float | None = None
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in dists_by_class.items():
        ax.hist(d, bins=40, alpha=0.55, label=f"{name} (n={len(d)})", edgecolor="black", linewidth=0.3)
    if threshold is not None:
        ax.axvline(threshold, color="red", ls="--", label=f"τ = {threshold}")
    ax.set_xlabel("Minimum LPIPS distance to real set")
    ax.set_ylabel("Frequency")
    ax.set_title("Synthetic-to-real perceptual distance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return Path(out)


def plot_threshold_sweep(sweep: dict, out: str | Path) -> Path:
    """sweep: {tau_str: {overall_retained_fraction: float, ...}}."""
    taus = sorted(float(t) for t in sweep)
    retained = [sweep[str(t)]["overall_retained_fraction"] for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, retained, "o-")
    ax.set_xlabel("LPIPS threshold τ")
    ax.set_ylabel("Retained fraction")
    ax.set_title("Retention vs LPIPS threshold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return Path(out)


def plot_ratio_results(summary: dict, out: str | Path, metric: str = "auc") -> Path:
    """Bar plot of a metric across ratios with 95% CI error bars."""
    ratios = sorted(summary["per_ratio"], key=lambda r: float(r))
    means = [summary["per_ratio"][r][metric]["mean"] for r in ratios]
    cis = np.array([summary["per_ratio"][r][metric]["ci95"] for r in ratios])
    err = np.abs(cis.T - np.array(means))
    labels = ["real only" if float(r) == 0 else f"{int(float(r))}:1" for r in ratios]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=err, capsize=4, color="#4c78a8")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Synthetic : real ratio")
    ax.set_title(f"TEST {metric.upper()} by augmentation ratio (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return Path(out)
