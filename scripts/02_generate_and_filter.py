#!/usr/bin/env python
"""Generate synthetic images, run the LPIPS threshold sweep, and filter.

Reference reals are drawn from DEV ONLY (selection-leakage fix). Produces:
  * results/synthetic/<class>/*.png  (retained, at the operating threshold)
  * results/synthetic/selection_report.json
  * results/synthetic/threshold_sweep.json + figure  (justifies tau=0.2)

    python scripts/02_generate_and_filter.py --config configs/default.yaml \
        --ckpt results/gan/ckpt_dev.pth --out results/synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips

from breastsynth.config import load_config
from breastsynth.models.stylegan2ada import load_generator
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything
from breastsynth.selection.lpips_filter import (
    generate_and_filter,
    load_reference_reals,
    threshold_sweep,
)
from breastsynth.viz.plots import plot_lpips_distribution, plot_threshold_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="results/gan/ckpt_dev.pth")
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-per-class", type=int, default=1000)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_holdout(cfg)
    gen = load_generator(cfg.gan, args.ckpt, device=args.device)
    loss_fn = lpips.LPIPS(net=cfg.selection.lpips_net).to(args.device)

    # DEV-only reference reals, per class.
    reference_by_class = {}
    for cls in sorted(data.dev_full[cfg.data.label_column].unique()):
        paths = data.dev_full[data.dev_full[cfg.data.label_column] == cls]["path"].tolist()
        reference_by_class[int(cls)] = load_reference_reals(
            paths, cfg.data.image_size, cfg.selection.n_reference_reals, seed=cfg.split.seed
        )

    if not args.skip_sweep:
        sweep = threshold_sweep(
            gen, reference_by_class, cfg.selection, cfg.gan.z_dim, cfg.gan.n_classes,
            pool_per_class=args.pool_per_class, loss_fn=loss_fn, device=args.device,
        )
        (out / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
        plot_lpips_distribution(sweep["distances"], out / "lpips_distribution.png", cfg.selection.threshold)
        plot_threshold_sweep(sweep["sweep"], out / "threshold_sweep.png")
        print("Threshold sweep:")
        for tau, v in sweep["sweep"].items():
            print(f"  tau={tau}: retained fraction={v['overall_retained_fraction']:.3f}")

    # How many synthetic per class do we need? Enough for the largest ratio.
    max_ratio = max(cfg.classifier.ratios)
    per_class_real = int(data.dev_balanced[cfg.data.label_column].value_counts().min())
    n_target = int(max_ratio * per_class_real) if max_ratio > 0 else per_class_real
    print(f"Generating {n_target} synthetic images/class at tau={cfg.selection.threshold}")

    report = generate_and_filter(
        gen, reference_by_class, n_target, cfg.selection, cfg.gan.z_dim, cfg.gan.n_classes,
        out_dir=out, loss_fn=loss_fn, device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
