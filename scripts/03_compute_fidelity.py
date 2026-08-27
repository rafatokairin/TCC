#!/usr/bin/env python
"""Compute FID and KID between DEV reals and retained synthetic images (per class).

    python scripts/03_compute_fidelity.py --config configs/default.yaml \
        --synthetic results/synthetic --out results/synthetic/fidelity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from breastsynth.config import load_config
from breastsynth.metrics.fidelity import compute_fid_kid, list_images
from breastsynth.pipeline import prepare_holdout

_CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--out", default="results/synthetic/fidelity.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = prepare_holdout(cfg)

    results = {}
    for cls, name in _CLASS_NAMES.items():
        real_paths = data.dev_full[data.dev_full[cfg.data.label_column] == cls]["path"].tolist()
        fake_paths = [p for p in list_images(args.synthetic) if Path(p).stem.upper().startswith(name)]
        if not real_paths or not fake_paths:
            print(f"[skip] {name}: real={len(real_paths)} fake={len(fake_paths)}")
            continue
        results[name] = compute_fid_kid(
            real_paths, fake_paths, device=args.device, image_size=cfg.data.image_size
        )
        print(f"{name}: FID={results[name]['fid']:.2f}  "
              f"KID={results[name]['kid_mean']:.4f}±{results[name]['kid_std']:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
