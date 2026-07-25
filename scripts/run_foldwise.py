#!/usr/bin/env python
"""Gold-standard fold-wise protocol: RETRAIN the GAN on each fold's train split.

Expensive (K GAN trainings). Removes leakage by construction: the fold's
validation patients are never seen by that fold's generator or LPIPS filter.

    python scripts/run_foldwise.py --config configs/foldwise.yaml --out results/foldwise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import numpy as np

from breastsynth.config import load_config
from breastsynth.data.splits import balance_classes, foldwise_splits, load_manifest
from breastsynth.evaluation.protocols import compose_training_set
from breastsynth.generative.train_gan import train_stylegan2ada
from breastsynth.metrics.classification import bootstrap_ci, classification_metrics
from breastsynth.models.classifier import predict_probs
from breastsynth.models.stylegan2ada import load_generator
from breastsynth.seed import seed_everything
from breastsynth.selection.lpips_filter import generate_and_filter, load_reference_reals
from breastsynth.training.train_classifier import train_classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/foldwise.yaml")
    ap.add_argument("--out", default="results/foldwise")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg, drop_duplicates=True)

    per_ratio = {r: {"accuracy": [], "f1": [], "auc": []} for r in cfg.classifier.ratios}

    for k, train_df, val_df in foldwise_splits(
        manifest, cfg.split.n_folds, cfg.data.label_column, cfg.data.patient_column, cfg.split.seed
    ):
        fold_dir = out / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {k}: GAN on {len(train_df)} train images (val never seen) ===")

        # 1) Retrain GAN on THIS fold's training images only.
        ckpt = train_stylegan2ada(
            train_df["path"].tolist(), train_df[cfg.data.label_column].astype(int).tolist(),
            cfg.gan, fold_dir / "ckpt.pth", device=args.device,
        )
        gen = load_generator(cfg.gan, str(ckpt), device=args.device)
        loss_fn = lpips.LPIPS(net=cfg.selection.lpips_net).to(args.device)

        # 2) LPIPS reference = fold train only; generate + filter.
        reference_by_class = {
            int(c): load_reference_reals(
                train_df[train_df[cfg.data.label_column] == c]["path"].tolist(),
                cfg.data.image_size, cfg.selection.n_reference_reals, seed=cfg.split.seed,
            )
            for c in sorted(train_df[cfg.data.label_column].unique())
        }
        train_bal, _ = balance_classes(
            train_df, cfg.data.balance_target_per_class, cfg.data.label_column,
            cfg.data.patient_column, seed=cfg.split.seed,
        )
        per_class_real = int(train_bal[cfg.data.label_column].value_counts().min())
        n_target = int(max(cfg.classifier.ratios) * per_class_real)
        generate_and_filter(
            gen, reference_by_class, n_target, cfg.selection, cfg.gan.z_dim, cfg.gan.n_classes,
            out_dir=fold_dir / "synthetic", loss_fn=loss_fn, device=args.device,
        )
        syn = {
            0: sorted(str(p) for p in (fold_dir / "synthetic").glob("BENIGN_*.png")),
            1: sorted(str(p) for p in (fold_dir / "synthetic").glob("MALIGNANT_*.png")),
        }

        # 3) Train classifier per ratio/seed; evaluate on val_k.
        val_bal, _ = balance_classes(
            val_df, int(val_df[cfg.data.label_column].value_counts().min()),
            cfg.data.label_column, cfg.data.patient_column, seed=cfg.split.seed,
        )
        from torch.utils.data import DataLoader

        from breastsynth.data.dataset import MammogramDataset, build_transforms
        val_loader = DataLoader(
            MammogramDataset.from_frame(val_bal, build_transforms(cfg.classifier.image_size)),
            batch_size=cfg.classifier.batch_size,
        )
        for ratio in cfg.classifier.ratios:
            for s in range(cfg.split.n_seeds):
                seed = cfg.split.seed + s
                tr_paths, tr_labels = compose_training_set(train_bal, syn, ratio, seed, cfg.data.label_column)
                model, _ = train_classifier(
                    tr_paths, tr_labels, val_bal["path"].tolist(),
                    val_bal[cfg.data.label_column].astype(int).tolist(),
                    cfg.classifier, device=args.device, seed=seed,
                )
                yt, yp = predict_probs(model, val_loader, args.device)
                m = classification_metrics(yt, yp)
                per_ratio[ratio]["accuracy"].append(m["accuracy"])
                per_ratio[ratio]["f1"].append(m["f1"])
                per_ratio[ratio]["auc"].append(m["auc"])

    # Aggregate across folds x seeds.
    summary = {"per_ratio": {}}
    for ratio, metrics in per_ratio.items():
        entry = {}
        for name, vals in metrics.items():
            mean, lo, hi = bootstrap_ci(vals, seed=cfg.split.seed)
            entry[name] = {"mean": mean, "std": float(np.std(vals)), "ci95": [lo, hi], "per_run": vals}
        summary["per_ratio"][str(ratio)] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nFold-wise summary written to", out / "summary.json")


if __name__ == "__main__":
    main()
