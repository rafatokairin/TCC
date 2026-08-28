#!/usr/bin/env python
"""Low-data learning curve: does GAN augmentation help when real data is scarce?

Synthetic augmentation is expected to help most in the low-data regime. Under the
SAME patient-level, leakage-free protocol, this subsamples the real DEV pool to
increasing sizes and, at each size, compares real-only vs GAN-augmented (best
ratio) on the locked TEST set. If the GAN helps only below some N real images,
that is a nuanced, publishable positive.

    python scripts/08_learning_curve.py --config configs/finetune.yaml \
        --synthetic results/synthetic --gan-ratio 2 --out results/learningcurve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from breastsynth.config import load_config
from breastsynth.data.dataset import MammogramDataset, build_transforms
from breastsynth.data.splits import inner_validation_split
from breastsynth.evaluation.protocols import compose_training_set
from breastsynth.evaluation.stats import cliffs_delta, wilcoxon_vs_baseline
from breastsynth.metrics.classification import bootstrap_ci, classification_metrics
from breastsynth.metrics.fidelity import list_images
from breastsynth.models.classifier import predict_probs
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything
from breastsynth.training.train_classifier import train_classifier

_CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def _synthetic_by_class(folder):
    out = {c: [] for c in _CLASS_NAMES}
    for p in list_images(folder):
        for cls, name in _CLASS_NAMES.items():
            if Path(p).stem.upper().startswith(name):
                out[cls].append(p)
    return out


def _subsample(dev_df, n_per_class, label_col, seed):
    parts = []
    for cls, g in dev_df.groupby(label_col):
        parts.append(g.sample(n=min(n_per_class, len(g)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _run(dev_df, test_loader, synthetic_by_class, cfg, ratio, device):
    auc = []
    for s in range(cfg.split.n_seeds):
        seed = cfg.split.seed + s
        inner_train, inner_val = inner_validation_split(
            dev_df, cfg.classifier.inner_val_size, cfg.data.label_column,
            cfg.data.patient_column, seed=seed,
        )
        tr_paths, tr_labels = compose_training_set(
            inner_train, synthetic_by_class, ratio, seed, cfg.data.label_column
        )
        model, _ = train_classifier(
            tr_paths, tr_labels, inner_val["path"].tolist(),
            inner_val[cfg.data.label_column].astype(int).tolist(),
            cfg.classifier, device=device, seed=seed, augment="basic",
        )
        yt, yp = predict_probs(model, test_loader, device)
        auc.append(classification_metrics(yt, yp)["auc"])
    return auc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--gan-ratio", type=float, default=2.0)
    ap.add_argument("--levels", type=int, nargs="+", default=[40, 80, 120, 160])
    ap.add_argument("--out", default="results/learningcurve")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = prepare_holdout(cfg)
    synthetic_by_class = _synthetic_by_class(args.synthetic)
    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        MammogramDataset.from_frame(data.test_balanced, build_transforms(cfg.classifier.image_size)),
        batch_size=cfg.classifier.batch_size,
    )
    full_per_class = int(data.dev_balanced[cfg.data.label_column].value_counts().min())
    levels = sorted({min(n, full_per_class) for n in args.levels} | {full_per_class})

    curve = {}
    for n in levels:
        sub = _subsample(data.dev_balanced, n, cfg.data.label_column, seed=cfg.split.seed)
        real = _run(sub, test_loader, {0: [], 1: []}, cfg, 0.0, args.device)
        gan = _run(sub, test_loader, synthetic_by_class, cfg, args.gan_ratio, args.device)
        w = wilcoxon_vs_baseline(real, gan)
        rmean, rlo, rhi = bootstrap_ci(real, seed=cfg.split.seed)
        gmean, glo, ghi = bootstrap_ci(gan, seed=cfg.split.seed)
        curve[str(n)] = {
            "n_per_class": n,
            "real_only_auc": {"mean": rmean, "ci95": [rlo, rhi], "per_seed": real},
            "gan_auc": {"mean": gmean, "ci95": [glo, ghi], "per_seed": gan},
            "delta_auc": gmean - rmean,
            "wilcoxon_p": w["p_value"],
            "cliffs_delta": cliffs_delta(real, gan),
        }
        print(f"  n={n:>3}/class: real={rmean:.3f}  gan={gmean:.3f}  "
              f"Δ={gmean - rmean:+.3f}  p={w['p_value']:.3f}")

    (out / "summary.json").write_text(json.dumps({"gan_ratio": args.gan_ratio, "curve": curve}, indent=2))

    # Figure: real vs GAN AUC across data sizes.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [curve[k]["n_per_class"] for k in curve]
        rm = [curve[k]["real_only_auc"]["mean"] for k in curve]
        gm = [curve[k]["gan_auc"]["mean"] for k in curve]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, rm, "o-", label="real only")
        ax.plot(ns, gm, "s-", label=f"real + GAN {int(args.gan_ratio)}:1")
        ax.set_xlabel("Real images per class (DEV subsample)")
        ax.set_ylabel("TEST AUC")
        ax.set_title("Learning curve: GAN augmentation vs real-only")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=300)
        print(f"Saved {out}/learning_curve.png")
    except Exception as e:
        print("plot skipped:", e)

    print(f"Saved {out}/summary.json")


if __name__ == "__main__":
    main()
