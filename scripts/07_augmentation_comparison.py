#!/usr/bin/env python
"""Compare augmentation strategies (Reviewer 2: "other data augmentation strategies").

Under the SAME patient-level, leakage-free protocol (locked TEST, N seeds), trains
EfficientNet-B0 under four conditions and tests each on the locked TEST set:
  1. Real only, no augmentation
  2. Real + classic strong augmentation (affine, jitter, cutout)
  3. Real + GAN-synthetic (best ratio)
  4. Real + classic augmentation + GAN-synthetic

This contextualises the GAN result: does GAN augmentation beat even simple classic
augmentation? Significance vs the no-augmentation baseline (paired Wilcoxon, Holm).

    python scripts/07_augmentation_comparison.py --config configs/finetune.yaml \
        --synthetic results/synthetic --gan-ratio 2 --out results/augcompare
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
from breastsynth.evaluation.stats import cliffs_delta, holm_correction, wilcoxon_vs_baseline
from breastsynth.metrics.classification import bootstrap_ci, classification_metrics
from breastsynth.metrics.fidelity import list_images
from breastsynth.models.classifier import predict_probs
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything
from breastsynth.training.train_classifier import train_classifier

_CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def _load_synthetic_by_class(folder: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {c: [] for c in _CLASS_NAMES}
    for p in list_images(folder):
        for cls, name in _CLASS_NAMES.items():
            if Path(p).stem.upper().startswith(name):
                out[cls].append(p)
    return out


def _test_loader(test_df, cfg):
    from torch.utils.data import DataLoader

    ds = MammogramDataset.from_frame(test_df, build_transforms(cfg.classifier.image_size, train=False))
    return DataLoader(ds, batch_size=cfg.classifier.batch_size, shuffle=False)


def run_condition(dev_df, test_loader, test_labels, synthetic_by_class, cfg, ratio, augment, device):
    acc, f1, auc = [], [], []
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
            cfg.classifier, device=device, seed=seed, augment=augment,
        )
        yt, yp = predict_probs(model, test_loader, device)
        m = classification_metrics(yt, yp)
        acc.append(m["accuracy"]); f1.append(m["f1"]); auc.append(m["auc"])
    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--gan-ratio", type=float, default=2.0)
    ap.add_argument("--out", default="results/augcompare")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = prepare_holdout(cfg)
    synthetic_by_class = _load_synthetic_by_class(args.synthetic)
    test_loader = _test_loader(data.test_balanced, cfg)
    test_labels = np.asarray(data.test_balanced[cfg.data.label_column].astype(int).tolist())

    r = args.gan_ratio
    conditions = [
        ("real_no_aug", 0.0, "none"),
        ("real_classic_aug", 0.0, "strong"),
        (f"real_gan_{int(r)}to1", r, "basic"),
        (f"real_classic_gan_{int(r)}to1", r, "strong"),
    ]

    results = {}
    for name, ratio, augment in conditions:
        print(f"[cond] {name}: ratio={ratio} augment={augment}")
        results[name] = run_condition(
            data.dev_balanced, test_loader, test_labels, synthetic_by_class, cfg, ratio, augment, args.device
        )

    # Summarise + significance vs the no-augmentation baseline.
    baseline = "real_no_aug"
    summary = {"per_condition": {}, "stats": {}}
    for name, m in results.items():
        entry = {}
        for metric, vals in m.items():
            mean, lo, hi = bootstrap_ci(vals, seed=cfg.split.seed)
            entry[metric] = {"mean": mean, "std": float(np.std(vals)), "ci95": [lo, hi], "per_seed": vals}
        summary["per_condition"][name] = entry

    raw_p = {}
    details = {}
    for name, m in results.items():
        if name == baseline:
            continue
        w = wilcoxon_vs_baseline(results[baseline]["auc"], m["auc"])
        d = cliffs_delta(results[baseline]["auc"], m["auc"])
        raw_p[name] = w["p_value"]
        details[name] = {**w, "cliffs_delta": d}
    for name, c in holm_correction(raw_p).items():
        details[name].update(c)
    summary["stats"]["wilcoxon_auc_vs_no_aug"] = details

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== TEST AUC by condition (mean [95% CI]) ===")
    for name, e in summary["per_condition"].items():
        a = e["auc"]
        print(f"  {name:>26}: acc={e['accuracy']['mean']:.3f} f1={e['f1']['mean']:.3f} "
              f"auc={a['mean']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]")
    print("\nWilcoxon AUC vs no-aug:",
          {k: (round(v['p_adjusted'], 3), v['reject']) for k, v in details.items()})
    print(f"Saved to {out}/summary.json")


if __name__ == "__main__":
    main()
