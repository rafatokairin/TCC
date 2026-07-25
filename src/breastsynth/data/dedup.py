"""Near-duplicate detection via perceptual hashing (Reviewer 1).

Reviewer 1 asked whether duplicate or near-duplicate images were removed. We use
perceptual hashing (pHash) and cluster images whose Hamming distance is within a
threshold, keeping one representative per cluster. Duplicates are *flagged* in
the manifest (not silently dropped) so the exact count is reportable.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image


def compute_phash(path: str | Path, hash_size: int = 16) -> imagehash.ImageHash:
    with Image.open(path).convert("L") as img:
        return imagehash.phash(img, hash_size=hash_size)


def flag_near_duplicates(
    image_paths: dict[str, str],
    threshold: int = 4,
    hash_size: int = 16,
) -> pd.DataFrame:
    """Cluster near-duplicate images by pHash Hamming distance.

    Args:
        image_paths: mapping image_id -> file path.
        threshold: images within this Hamming distance are considered duplicates.
        hash_size: pHash size (16 -> 256-bit hash, robust to resizing artefacts).

    Returns:
        DataFrame with columns [image_id, phash, cluster_id, is_duplicate], where
        `is_duplicate=True` marks the non-representative members of a cluster
        (the first image of each cluster is kept as representative).
    """
    ids = list(image_paths)
    hashes = {i: compute_phash(image_paths[i], hash_size) for i in ids}

    cluster_of: dict[str, int] = {}
    representatives: list[tuple[int, imagehash.ImageHash]] = []
    for img_id in ids:
        h = hashes[img_id]
        assigned = None
        for cid, rep_hash in representatives:
            if (h - rep_hash) <= threshold:
                assigned = cid
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append((assigned, h))
        cluster_of[img_id] = assigned

    seen_clusters: set[int] = set()
    rows = []
    for img_id in ids:
        cid = cluster_of[img_id]
        is_dup = cid in seen_clusters
        seen_clusters.add(cid)
        rows.append(
            {
                "image_id": img_id,
                "phash": str(hashes[img_id]),
                "cluster_id": cid,
                "is_duplicate": is_dup,
            }
        )
    return pd.DataFrame(rows)
