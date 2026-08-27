"""Build the canonical dataset manifest (Reviewer 1: objective dataset description).

The manifest is the single table the whole pipeline consumes. It records, for
every full-mammogram image actually present on disk:

    image_id, patient_id, view, pathology, label, is_duplicate

Crucially it carries `patient_id`, recovered from the original CBIS-DDSM case
description CSVs, so that all splits can be **patient-level** (no patient in both
train and test). The original `data/dataset.csv` only had `pathology` +
`image file path`; that is insufficient for patient separation, which is exactly
one of the reviewer's concerns.

Usage (see scripts/00_build_manifest.py):
    build_manifest(
        image_root="data/dataset128",
        cbis_csvs=["mass_case_description_train_set.csv", ...],
        out="data/manifest.csv",
    )
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from breastsynth.data.dedup import flag_near_duplicates

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_PATIENT_RE = re.compile(r"(P_\d{5})")
_LABEL_MAP = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}


def _norm(col: str) -> str:
    return col.strip().lower().replace("_", " ")


def list_images(image_root: str | Path) -> dict[str, str]:
    """Return {image_id (stem) -> path} for every image under image_root."""
    root = Path(image_root)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = str(p)
    if not out:
        raise FileNotFoundError(f"No images found under {image_root}")
    return out


# Extensions stripped when matching a path token to an image id. Includes DICOM
# (.dcm) for CBIS paths; note UIDs themselves contain dots, so we only strip
# these known trailing extensions, never "everything after the last dot".
_MATCH_EXTS = IMAGE_EXTS + (".dcm", ".dicom")


def _strip_image_ext(token: str) -> str:
    low = token.lower()
    for ext in _MATCH_EXTS:
        if low.endswith(ext):
            return token[: -len(ext)]
    return token


def _extract_image_id(path_value: str, known_ids: set[str]) -> str | None:
    """Find which known image_id a path/id string refers to.

    Image ids are DICOM SOP-instance UIDs, which CONTAIN DOTS (e.g.
    '1.3.6.1.4.1.9590...486'), so we must NOT split on dots or use Path.stem
    (both would shred the UID). Two cases are handled:
      * legacy CSV: `image file path` is already the bare UID -> matches directly;
      * CBIS-DDSM: 'Mass-Training_P_00001/<studyUID>/<seriesUID>/<sopUID>.dcm'
        -> the UID appears as a path component, recovered by splitting on path
        separators only (never dots) and stripping any image extension.
    """
    s = str(path_value).strip()
    if s in known_ids:  # bare UID (legacy CSV)
        return s
    if _strip_image_ext(s) in known_ids:
        return _strip_image_ext(s)
    for token in re.split(r"[/\\]", s):  # path separators only, NOT dots
        token = token.strip()
        if token in known_ids:
            return token
        stripped = _strip_image_ext(token)
        if stripped in known_ids:
            return stripped
    return None


def _read_cbis_csv(csv_path: str | Path, known_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [_norm(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        path_val = r.get("image file path", "")
        img_id = _extract_image_id(path_val, known_ids)
        if img_id is None:
            continue
        patient = r.get("patient id")
        if patient is None or (isinstance(patient, float) and pd.isna(patient)):
            m = _PATIENT_RE.search(str(path_val))
            patient = m.group(1) if m else None
        pathology = str(r.get("pathology", "")).strip().upper()
        rows.append(
            {
                "image_id": img_id,
                "patient_id": str(patient) if patient is not None else None,
                "view": r.get("image view"),
                "pathology": pathology,
            }
        )
    return pd.DataFrame(rows)


def _map_patient_from_dicom_info(csv_path: str | Path, known_ids: set[str]) -> dict[str, str]:
    """Build {image_id -> patient_id} from a CBIS-DDSM `dicom_info.csv`.

    This is the reliable bridge in the awsaf49 Kaggle layout: it maps each
    SeriesInstanceUID (our image ids) to its PatientID. Column names are
    auto-detected so the join is robust to minor naming differences.
    """
    df = pd.read_csv(csv_path)

    def _find(cands):
        norm = {c: c.strip().lower().replace("_", "").replace(" ", "") for c in df.columns}
        for c, n in norm.items():
            if n in cands:
                return c
        return None

    patient_col = _find({"patientid", "patient", "subjectid", "subject"})
    if patient_col is None:
        raise ValueError(f"No PatientID-like column found in {csv_path}. Columns: {list(df.columns)}")

    # Prefer named UID/path columns; if none match, scan every column for the
    # one whose values best match our known image ids.
    def _best_uid_col(columns):
        best_col, best = None, 0
        for c in columns:
            if c is None:
                continue
            hits = sum(
                _extract_image_id(v, known_ids) is not None
                for v in df[c].astype(str).head(3000)
            )
            if hits > best:
                best, best_col = hits, c
        return best_col, best

    preferred = _find({"seriesinstanceuid", "imagepath", "imagefilepath", "seriesuid"})
    uid_col, best_hits = _best_uid_col([preferred]) if preferred else (None, 0)
    if best_hits == 0:  # named column absent or didn't match -> scan all columns
        uid_col, best_hits = _best_uid_col(list(df.columns))
    if uid_col is None or best_hits == 0:
        raise ValueError(
            f"Could not find a column in {csv_path} matching the image ids under the "
            "image root. Is this the right dicom_info.csv?"
        )

    mapping: dict[str, str] = {}
    for _, r in df.iterrows():
        img_id = _extract_image_id(r[uid_col], known_ids)
        pid = r.get(patient_col)
        if img_id is None or pid is None or (isinstance(pid, float) and pd.isna(pid)):
            continue
        # In CBIS-DDSM `dicom_info.csv` the PatientID field is the per-image case
        # name (e.g. 'Mass-Training_P_01265_RIGHT_MLO_1'); the real subject is the
        # embedded 'P_#####'. Extract it so images of the same patient share an id
        # (essential for patient-level splitting); fall back to the raw value.
        m = _PATIENT_RE.search(str(pid))
        mapping.setdefault(img_id, m.group(1) if m else str(pid))
    return mapping


def build_manifest(
    image_root: str | Path,
    cbis_csvs: list[str | Path] | None = None,
    legacy_csv: str | Path | None = None,
    dicom_info_csv: str | Path | None = None,
    out: str | Path = "data/manifest.csv",
    dedup_threshold: int = 4,
) -> pd.DataFrame:
    """Assemble the manifest from images on disk + metadata.

    Args:
        image_root: directory of full-mammogram images (each stem is an image_id).
        cbis_csvs: original CBIS-DDSM case description CSVs (provide patient_id).
        legacy_csv: fallback `data/dataset.csv` (pathology + image file path only,
            NO patient id). Used if cbis_csvs is not given. Combine with
            dicom_info_csv to recover patient_id for a leakage-safe split.
        dicom_info_csv: CBIS-DDSM `dicom_info.csv` (awsaf49 Kaggle layout). Maps
            SeriesInstanceUID -> PatientID; used to attach real patient_id on top
            of a legacy/cbis manifest so splits become patient-level.
        out: output manifest path.
        dedup_threshold: pHash Hamming distance for near-duplicate flagging.

    Returns:
        The manifest DataFrame (also written to `out`).
    """
    image_paths = list_images(image_root)
    known_ids = set(image_paths)

    if cbis_csvs:
        missing = [c for c in cbis_csvs if not Path(c).is_file()]
        if missing:
            raise FileNotFoundError(
                "CBIS-DDSM case CSV(s) not found: "
                + ", ".join(str(m) for m in missing)
                + ". These are the metadata files that ship with the CBIS-DDSM "
                "download (e.g. mass_case_description_train_set.csv) and provide "
                "patient_id. Download them from TCIA/Kaggle, or omit --cbis-csv "
                "and pass --legacy-csv data/dataset.csv (no patient-level split)."
            )
        meta = pd.concat([_read_cbis_csv(c, known_ids) for c in cbis_csvs], ignore_index=True)
        meta = meta.drop_duplicates(subset="image_id", keep="first")
    elif legacy_csv is not None:
        if dicom_info_csv is None:
            warnings.warn(
                "Building manifest from legacy CSV WITHOUT patient_id. Patient-level "
                "splitting cannot be guaranteed; each image is treated as its own "
                "patient. Pass dicom_info_csv=... (CBIS-DDSM dicom_info.csv) or "
                "cbis_csvs=... for a leakage-safe, patient-level manifest.",
                stacklevel=2,
            )
        df = pd.read_csv(legacy_csv)
        df.columns = [_norm(c) for c in df.columns]
        rows = []
        for _, r in df.iterrows():
            img_id = _extract_image_id(r.get("image file path", ""), known_ids)
            if img_id is None:
                continue
            rows.append(
                {
                    "image_id": img_id,
                    "patient_id": f"UNKNOWN_{img_id}",  # 1 image == 1 pseudo-patient
                    "view": None,
                    "pathology": str(r.get("pathology", "")).strip().upper(),
                }
            )
        meta = pd.DataFrame(rows)
    else:
        raise ValueError("Provide either cbis_csvs=[...] or legacy_csv=...")

    if meta.empty or "image_id" not in meta.columns:
        raise ValueError(
            f"No CSV rows could be matched to the {len(known_ids)} images under "
            f"'{image_root}'. Check that the CSV's 'image file path' values are the "
            "image UIDs (or contain them as a path component)."
        )

    # Attach real patient_id from dicom_info.csv (SeriesInstanceUID -> PatientID).
    if dicom_info_csv is not None:
        pmap = _map_patient_from_dicom_info(dicom_info_csv, known_ids)
        matched = meta["image_id"].map(pmap)
        n_recovered = int(matched.notna().sum())
        meta["patient_id"] = matched.where(matched.notna(), meta["patient_id"])
        warnings.warn(
            f"Recovered patient_id for {n_recovered}/{len(meta)} images from "
            f"{dicom_info_csv} ({meta['patient_id'].nunique()} unique patients).",
            stacklevel=2,
        )

    # Keep only images that exist on disk and have a usable pathology label.
    meta = meta[meta["image_id"].isin(known_ids)].copy()
    meta["label"] = meta["pathology"].map(_LABEL_MAP)
    unmapped = meta["label"].isna().sum()
    if unmapped:
        warnings.warn(f"{unmapped} rows had unmappable pathology and were dropped.", stacklevel=2)
    meta = meta.dropna(subset=["label"]).copy()
    meta["label"] = meta["label"].astype(int)
    meta["path"] = meta["image_id"].map(image_paths)

    missing_meta = known_ids - set(meta["image_id"])
    if missing_meta:
        warnings.warn(
            f"{len(missing_meta)} images on disk had no metadata match and were "
            "excluded from the manifest.",
            stacklevel=2,
        )

    # Near-duplicate flagging on the matched set.
    dup = flag_near_duplicates(
        {r.image_id: r.path for r in meta.itertuples()}, threshold=dedup_threshold
    )
    meta = meta.merge(dup[["image_id", "phash", "cluster_id", "is_duplicate"]], on="image_id")

    meta = meta[
        ["image_id", "patient_id", "view", "pathology", "label", "path",
         "phash", "cluster_id", "is_duplicate"]
    ].sort_values("image_id").reset_index(drop=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    return meta
