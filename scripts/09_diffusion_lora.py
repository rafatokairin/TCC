#!/usr/bin/env python
"""Phase 2: LoRA-fine-tune Stable Diffusion 1.5 on DEV and generate synthetic set.

Leakage-safe: LoRA is trained on the DEV partition only; TEST is never seen.
Produces a diffusion synthetic set that can be compared, under the SAME
classifier/evaluation pipeline, against the GAN synthetic set.

    python scripts/09_diffusion_lora.py --config configs/full_cbis.yaml \
        --out results_full/diffusion --steps 1500 --n-per-class 780

Runs on an RTX 4060 (fp16 + gradient checkpointing + LoRA). No A100 needed.
Downstream: reuse scripts 03/04/07 with --synthetic results_full/diffusion/filtered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from breastsynth.config import load_config
from breastsynth.generative.diffusion_lora import generate_lora, train_lora
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_cbis.yaml")
    ap.add_argument("--out", default="results_full/diffusion")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-per-class", type=int, default=780)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing LoRA adapter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.gan.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lora_dir = out / "lora"

    data = prepare_holdout(cfg)
    paths = data.dev_full["path"].tolist()
    labels = data.dev_full[cfg.data.label_column].astype(int).tolist()
    print(f"LoRA training on {len(paths)} DEV images ({data.balance_report['n_dev_patients']} patients); "
          "TEST untouched.")

    if not args.skip_train:
        train_lora(paths, labels, lora_dir, size=args.size, rank=args.rank,
                   steps=args.steps, device=args.device, seed=cfg.gan.seed)

    raw = out / "raw"
    counts = {}
    for cls in (0, 1):
        files = generate_lora(lora_dir, cls, args.n_per_class, raw, size=args.size,
                              device=args.device, seed=cfg.gan.seed)
        counts[cls] = len(files)
    (out / "generation_report.json").write_text(
        json.dumps({"n_per_class": args.n_per_class, "counts": counts, "size": args.size}, indent=2)
    )
    print(f"Generated diffusion synthetic -> {raw}  ({counts})")
    print("Next: LPIPS-filter + memorisation audit, then feed into scripts 03/04/07 for comparison.")


if __name__ == "__main__":
    main()
