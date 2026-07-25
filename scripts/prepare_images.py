#!/usr/bin/env python
"""Resize full mammograms to NxN and orient every breast to the LEFT.

Reproduces the preprocessing used to build data/dataset128/. Run this on the raw
full-mammogram JPEGs (e.g. the CBIS-DDSM 'full mammogram images' extracted from
the awsaf49 Kaggle layout) BEFORE building the manifest.

    python scripts/prepare_images.py --src /path/to/raw_full_mammograms \
        --dst data/dataset128 --size 128

Optionally pass a metadata CSV to take laterality from a 'left or right breast'
or 'laterality' column instead of inferring it from pixels:

    python scripts/prepare_images.py --src RAW --dst data/dataset128 \
        --laterality-csv /path/to/csv/dicom_info.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from breastsynth.data.preprocess import preprocess_folder


def _laterality_map(csv_path: str) -> dict[str, str]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    norm = {c: c.strip().lower().replace("_", " ") for c in df.columns}
    df.columns = [norm[c] for c in df.columns]
    lat_col = next((c for c in df.columns if "laterality" in c or "left or right" in c), None)
    id_col = next(
        (c for c in df.columns if "seriesinstanceuid" in c.replace(" ", "") or "image" in c), None
    )
    if lat_col is None or id_col is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        stem = Path(str(r[id_col])).stem
        out[stem] = str(r[lat_col]).strip().upper()[:1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="folder of raw full-mammogram images")
    ap.add_argument("--dst", default="data/dataset128")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--laterality-csv", default=None)
    ap.add_argument("--keep-rgb", action="store_true", help="do not convert to grayscale")
    args = ap.parse_args()

    lat_map = _laterality_map(args.laterality_csv) if args.laterality_csv else None
    report = preprocess_folder(
        args.src, args.dst, size=args.size, laterality_map=lat_map, to_grayscale=not args.keep_rgb
    )
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    (Path(args.dst).parent / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
