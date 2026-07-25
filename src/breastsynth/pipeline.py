"""Shared split preparation so every CLI stage derives the SAME partitions.

Determinism here is what guarantees the GAN (trained on DEV_full) and the
classifier (DEV_bal train / TEST_bal eval) agree on which patients are in TEST,
so TEST is never seen by the generator or the LPIPS filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from breastsynth.config import Config
from breastsynth.data.splits import balance_classes, holdout_split, load_manifest


@dataclass
class HoldoutData:
    dev_full: pd.DataFrame     # ALL dev images (feeds GAN + LPIPS reference)
    dev_balanced: pd.DataFrame  # balanced real pool for classifier training
    test_balanced: pd.DataFrame  # locked, balanced real test set
    balance_report: dict


def prepare_holdout(cfg: Config) -> HoldoutData:
    manifest = load_manifest(cfg, drop_duplicates=True)
    dev_full, test_full = holdout_split(
        manifest,
        test_size=cfg.split.test_size,
        label_col=cfg.data.label_column,
        patient_col=cfg.data.patient_column,
        seed=cfg.split.seed,
    )
    # Effective target = min(configured cap, smallest per-class count in DEV).
    # A hard cap alone would leave DEV imbalanced once the hold-out removes
    # patients, so we clamp to the minority-class count within the split.
    dev_min = int(dev_full[cfg.data.label_column].value_counts().min())
    dev_target = min(cfg.data.balance_target_per_class, dev_min)
    dev_bal, dev_rep = balance_classes(
        dev_full, dev_target, cfg.data.label_column,
        cfg.data.patient_column, seed=cfg.split.seed,
    )
    # Balance TEST to its own per-class minimum for an interpretable, balanced eval.
    test_min = int(test_full[cfg.data.label_column].value_counts().min())
    test_bal, test_rep = balance_classes(
        test_full, test_min, cfg.data.label_column, cfg.data.patient_column, seed=cfg.split.seed,
    )
    return HoldoutData(
        dev_full=dev_full,
        dev_balanced=dev_bal,
        test_balanced=test_bal,
        balance_report={
            "dev_full_counts": dev_full[cfg.data.label_column].value_counts().to_dict(),
            "test_full_counts": test_full[cfg.data.label_column].value_counts().to_dict(),
            "dev_balance": dev_rep,
            "test_balance": test_rep,
            "n_dev_patients": int(dev_full[cfg.data.patient_column].nunique()),
            "n_test_patients": int(test_full[cfg.data.patient_column].nunique()),
        },
    )
