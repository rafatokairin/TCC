"""Typed configuration objects loaded from YAML (Reviewer 1: hyperparameters).

Configs are plain dataclasses so they are self-documenting and hashable into
run reports. `load_config` reads a YAML file and overlays it on the defaults so
every field always has a value that ends up in the run log.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    image_root: str = "data/dataset128"
    image_size: int = 128
    benign_token: str = "BENIGN"          # matched case-insensitively (prefix)
    dedup_threshold: int = 4               # pHash Hamming distance
    balance_target_per_class: int = 245    # downsample benign to malignant count
    patient_column: str = "patient_id"
    label_column: str = "label"


@dataclass
class SplitConfig:
    protocol: str = "holdout"              # "holdout" | "foldwise"
    test_size: float = 0.20               # hold-out fraction (patient-level)
    n_folds: int = 5                      # inner CV (holdout) / outer folds (foldwise)
    n_seeds: int = 10                     # repeated classifier trainings
    seed: int = 42


@dataclass
class GanConfig:
    arch: str = "stylegan2ada"            # "stylegan2ada" | "wgan_gp"
    image_size: int = 128                 # generator output resolution (128/256/512)
    z_dim: int = 256
    w_dim: int = 256
    n_classes: int = 2
    base_ch: int = 32
    batch_size: int = 8
    epochs: int = 400
    lr_g: float = 0.0025
    lr_d: float = 0.0025
    beta1: float = 0.0
    beta2: float = 0.99
    r1_gamma: float = 10.0
    r1_interval: int = 32
    ada_target: float = 0.6
    ada_speed: float = 0.01
    mixed_precision: bool = True
    seed: int = 0


@dataclass
class SelectionConfig:
    lpips_net: str = "alex"
    threshold: float = 0.25               # operating point (justified by sweep; 0.20 infeasible)
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30]
    )
    similarity: str = "min"               # nearest-neighbour distance
    n_reference_reals: int = 100          # reference subset per class (DEV only)
    memorization_eps: float = 0.05        # flag near-copies below this LPIPS
    max_attempts_factor: int = 50         # safety cap for rejection sampling


@dataclass
class ClassifierConfig:
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_features: bool = True
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    early_stopping_patience: int = 7
    inner_val_size: float = 0.2           # carved from DEV only
    ratios: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0, 4.0])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    gan: GanConfig = field(default_factory=GanConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    results_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Stable short hash of the full config (goes into run ids/reports)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTION_TYPES = {
    "data": DataConfig,
    "split": SplitConfig,
    "gan": GanConfig,
    "selection": SelectionConfig,
    "classifier": ClassifierConfig,
}


def _build_section(section_cls, overrides: dict[str, Any]):
    valid = {f.name for f in fields(section_cls)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**overrides)


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config from YAML, falling back to dataclass defaults."""
    cfg = Config()
    if path is None:
        return cfg
    raw = yaml.safe_load(Path(path).read_text()) or {}
    kwargs: dict[str, Any] = {}
    for section, section_cls in _SECTION_TYPES.items():
        if section in raw:
            kwargs[section] = _build_section(section_cls, raw.pop(section))
    if "results_dir" in raw:
        kwargs["results_dir"] = raw.pop("results_dir")
    if raw:
        raise ValueError(f"Unknown top-level config keys: {sorted(raw)}")
    return Config(
        data=kwargs.get("data", cfg.data),
        split=kwargs.get("split", cfg.split),
        gan=kwargs.get("gan", cfg.gan),
        selection=kwargs.get("selection", cfg.selection),
        classifier=kwargs.get("classifier", cfg.classifier),
        results_dir=kwargs.get("results_dir", cfg.results_dir),
    )
