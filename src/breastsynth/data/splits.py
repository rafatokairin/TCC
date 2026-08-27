"""Patient-level, leakage-free data splitting (Reviewer 1: data leakage + patient separation).

All splits group by `patient_id` so no patient contributes images to more than
one partition. This is the mechanism that prevents both generator-leakage and
selection-leakage: the TEST (or per-fold validation) patients are never seen by
the GAN or the LPIPS filter.

`load_manifest` drops flagged near-duplicates by default so they never inflate
one side of a split.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from breastsynth.config import Config


def load_manifest(cfg: Config, drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(cfg.data.manifest)
    if drop_duplicates and "is_duplicate" in df.columns:
        df = df[~df["is_duplicate"].astype(bool)].copy()
    required = {"image_id", "patient_id", cfg.data.label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    # `image_root` is authoritative: always resolve `path` from the configured
    # image directory (by image_id stem), overriding any stored path column. This
    # lets the SAME manifest drive different resolutions (e.g. dataset128 vs
    # dataset256) purely via config, and prevents silently reading the wrong set.
    from breastsynth.data.manifest import list_images

    idx = list_images(cfg.data.image_root)
    df["path"] = df["image_id"].map(idx)
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        warnings.warn(
            f"{n_missing}/{len(df)} manifest image_ids have no file under "
            f"'{cfg.data.image_root}'; dropping them.",
            stacklevel=2,
        )
        df = df.dropna(subset=["path"])
    return df.reset_index(drop=True)


def _assert_no_patient_overlap(a: pd.DataFrame, b: pd.DataFrame, patient_col: str) -> None:
    overlap = set(a[patient_col]) & set(b[patient_col])
    if overlap:
        raise AssertionError(f"Patient overlap between splits: {sorted(overlap)[:5]}...")


def balance_classes(
    df: pd.DataFrame,
    target_per_class: int,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Downsample the majority class to `target_per_class`, patient-aware.

    We remove whole patients from the majority class first (to avoid splitting a
    patient across the balance cut), then trim individual images if needed. The
    report records exactly which/how many images were removed (Reviewer 1: "what
    happened to the 63 removed images").
    """
    rng = np.random.RandomState(seed)
    counts = df[label_col].value_counts().to_dict()
    kept_parts = []
    removed_ids: list[str] = []
    for label, group in df.groupby(label_col):
        n = len(group)
        if n <= target_per_class:
            kept_parts.append(group)
            continue
        # Shuffle patients, greedily keep whole patients until target reached.
        patients = list(group[patient_col].unique())
        rng.shuffle(patients)
        kept_rows, kept_n = [], 0
        for pid in patients:
            pat_rows = group[group[patient_col] == pid]
            if kept_n + len(pat_rows) <= target_per_class:
                kept_rows.append(pat_rows)
                kept_n += len(pat_rows)
        kept = pd.concat(kept_rows) if kept_rows else group.iloc[:0]
        # Hit the exact target: trim if over, top up with individual images if
        # under (topping up within one split does not affect DEV/TEST leakage).
        if len(kept) > target_per_class:
            kept = kept.sample(n=target_per_class, random_state=seed)
        elif len(kept) < target_per_class:
            remaining = group[~group["image_id"].isin(set(kept["image_id"]))]
            need = min(target_per_class - len(kept), len(remaining))
            if need > 0:
                kept = pd.concat([kept, remaining.sample(n=need, random_state=seed)])
        removed_ids.extend(set(group["image_id"]) - set(kept["image_id"]))
        kept_parts.append(kept)

    balanced = pd.concat(kept_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    report = {
        "original_counts": {int(k): int(v) for k, v in counts.items()},
        "target_per_class": target_per_class,
        "balanced_counts": {int(k): int(v) for k, v in balanced[label_col].value_counts().items()},
        "n_removed": len(removed_ids),
        "removed_image_ids": sorted(removed_ids),
    }
    return balanced, report


def holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level, stratified DEV/TEST split.

    Implemented via StratifiedGroupKFold (n_splits = round(1/test_size)); the
    first fold is the TEST set. This keeps class balance while never splitting a
    patient across DEV/TEST.
    """
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, df[label_col], groups=df[patient_col]))
    dev = df.iloc[dev_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    _assert_no_patient_overlap(dev, test, patient_col)
    return dev, test


def foldwise_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
):
    """Yield (fold_index, train_df, val_df) for patient-level Stratified GroupKFold.

    Used by the fold-wise (gold-standard) protocol: the GAN is retrained on
    `train_df` of each fold, so `val_df` is never seen by the generator.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(df, df[label_col], groups=df[patient_col])):
        train = df.iloc[tr_idx].reset_index(drop=True)
        val = df.iloc[va_idx].reset_index(drop=True)
        _assert_no_patient_overlap(train, val, patient_col)
        yield k, train, val


def inner_validation_split(
    dev: pd.DataFrame,
    val_size: float = 0.2,
    label_col: str = "label",
    patient_col: str = "patient_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a patient-level inner-validation split out of DEV (for early stopping).

    Never touches TEST. Uses a different seed stream per call via `seed`.
    """
    n_splits = max(2, round(1.0 / val_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(dev, dev[label_col], groups=dev[patient_col]))
    train = dev.iloc[tr_idx].reset_index(drop=True)
    val = dev.iloc[va_idx].reset_index(drop=True)
    _assert_no_patient_overlap(train, val, patient_col)
    return train, val
