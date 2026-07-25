#!/usr/bin/env python
"""Build data/manifest.csv with patient_id + near-duplicate flags.

Preferred (leakage-safe, patient-level):
    python scripts/00_build_manifest.py \
        --image-root data/dataset128 \
        --cbis-csv mass_case_description_train_set.csv \
        --cbis-csv mass_case_description_test_set.csv \
        --cbis-csv calc_case_description_train_set.csv \
        --cbis-csv calc_case_description_test_set.csv

Recommended when you have the awsaf49 Kaggle layout (recovers patient_id from
dicom_info.csv, keeping labels from the legacy CSV):
    python scripts/00_build_manifest.py --image-root data/dataset128 \
        --legacy-csv data/dataset.csv \
        --dicom-info /path/to/cbis-ddsm/csv/dicom_info.csv

Fallback (NO patient ids -> patient-level separation NOT guaranteed; warns):
    python scripts/00_build_manifest.py --image-root data/dataset128 \
        --legacy-csv data/dataset.csv
"""
from __future__ import annotations

import argparse

from breastsynth.data.manifest import build_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-root", default="data/dataset128")
    ap.add_argument("--cbis-csv", action="append", default=[], help="CBIS-DDSM case CSV (repeatable)")
    ap.add_argument("--legacy-csv", default=None, help="fallback dataset.csv (no patient id)")
    ap.add_argument("--dicom-info", default=None, help="CBIS-DDSM dicom_info.csv (recovers patient_id)")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--dedup-threshold", type=int, default=4)
    args = ap.parse_args()

    df = build_manifest(
        image_root=args.image_root,
        cbis_csvs=args.cbis_csv or None,
        legacy_csv=args.legacy_csv,
        dicom_info_csv=args.dicom_info,
        out=args.out,
        dedup_threshold=args.dedup_threshold,
    )
    n_dup = int(df["is_duplicate"].sum())
    print(f"Manifest written to {args.out}")
    print(f"  images: {len(df)}  (near-duplicates flagged: {n_dup})")
    print(f"  patients: {df['patient_id'].nunique()}")
    print(f"  class counts: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
