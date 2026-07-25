#!/usr/bin/env python
"""Train the StyleGAN2-ADA generator on the DEV partition ONLY (leakage-safe).

    python scripts/01_train_gan.py --config configs/default.yaml \
        --out results/gan/ckpt_dev.pth

The DEV/TEST split is derived deterministically from the config, so the TEST
patients are guaranteed absent from GAN training.
"""
from __future__ import annotations

import argparse
import json

from breastsynth.config import load_config
from breastsynth.generative.train_gan import train_stylegan2ada
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/gan/ckpt_dev.pth")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.gan.seed)
    data = prepare_holdout(cfg)

    # GAN sees ALL DEV images (imbalance is fine for a generator); never TEST.
    paths = data.dev_full["path"].tolist()
    labels = data.dev_full[cfg.data.label_column].astype(int).tolist()
    print(f"Training GAN on {len(paths)} DEV images "
          f"({data.balance_report['n_dev_patients']} patients). TEST is untouched.")

    ckpt = train_stylegan2ada(paths, labels, cfg.gan, args.out, device=args.device)
    (ckpt.parent / "gan_split_report.json").write_text(json.dumps(data.balance_report, indent=2, default=str))
    print(f"Saved generator checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
