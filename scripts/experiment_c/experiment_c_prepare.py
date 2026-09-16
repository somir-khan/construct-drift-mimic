#!/usr/bin/env python3
"""Prepare only the three arrays needed by the minimal Experiment C.

B     : exact frozen 5,000-note MIMIC-III baseline
T     : exact frozen primary 5,000-note MIMIC-IV target (the framework's MMD target)
C_cal : MIMIC-III calibration pool for DriftLens, patient-disjoint from B

The script does not construct null cohorts and does not load the framework PCA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


COHORT_SIZE = 5_000
EMBEDDING_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-emb", required=True)
    parser.add_argument("--baseline-ids", required=True)
    parser.add_argument("--target-emb", required=True, help="Primary 5,000-note MIMIC-IV embeddings.")
    parser.add_argument("--target-ids", required=True, help="HADM_IDs aligned with --target-emb.")
    parser.add_argument(
        "--target-groups",
        required=True,
        help="anchor_year_group labels aligned with --target-emb (groups_mimic4_5000.npy).",
    )
    parser.add_argument(
        "--m3-pool",
        action="append",
        nargs=2,
        required=True,
        metavar=("EMBEDDINGS", "IDS"),
        help="Repeat for every aligned MIMIC-III calibration-pool shard.",
    )
    parser.add_argument(
        "--mimic3-db",
        default=os.getenv("MIMIC3_DB_PATH"),
        help="SQLite database with ADMISSIONS(HADM_ID, SUBJECT_ID).",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(emb_path: str, ids_path: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(emb_path, allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=False)
    ids = np.asarray(ids).reshape(-1)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{label}: expected (*,{EMBEDDING_DIM}), got {embeddings.shape}.")
    if len(embeddings) != len(ids):
        raise ValueError(f"{label}: {len(embeddings)} rows but {len(ids)} IDs.")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(f"{label}: embeddings must be floating point, got {embeddings.dtype}.")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{label}: embeddings contain NaN or infinity.")
    if np.any(np.all(embeddings == 0, axis=1)):
        raise ValueError(f"{label}: embeddings contain an all-zero row.")
    try:
        ids = ids.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label}: IDs must be integer-like.") from exc
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate HADM_IDs found.")
    return np.asarray(embeddings), ids


def map_subjects(connection: sqlite3.Connection, hadm_ids: Iterable[int]) -> dict[int, int]:
    wanted = sorted({int(value) for value in hadm_ids})
    mapping: dict[int, int] = {}
    for start in range(0, len(wanted), 900):
        chunk = wanted[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            'SELECT HADM_ID, SUBJECT_ID FROM "ADMISSIONS" '
            f"WHERE HADM_ID IN ({placeholders})"
        )
        for hadm_id, subject_id in connection.execute(query, chunk):
            hadm, subject = int(hadm_id), int(subject_id)
            if hadm in mapping and mapping[hadm] != subject:
                raise RuntimeError(f"HADM_ID {hadm} maps to multiple subjects.")
            mapping[hadm] = subject
    missing = sorted(set(wanted) - set(mapping))
    if missing:
        raise RuntimeError(
            f"{len(missing)} MIMIC-III HADM_IDs are absent from ADMISSIONS; "
            f"first missing IDs: {missing[:10]}."
        )
    return mapping


def output_record(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    if not args.mimic3_db:
        raise SystemExit("Provide --mimic3-db or set MIMIC3_DB_PATH.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_ids = load_pair(
        args.baseline_emb, args.baseline_ids, "frozen MIMIC-III baseline"
    )
    if len(baseline) != COHORT_SIZE:
        raise ValueError(f"Baseline must have {COHORT_SIZE} rows; got {len(baseline)}.")

    target, target_ids = load_pair(args.target_emb, args.target_ids, "primary MIMIC-IV target")
    if len(target) != COHORT_SIZE:
        raise ValueError(f"Target must have {COHORT_SIZE} rows; got {len(target)}.")
    try:
        target_groups = np.load(args.target_groups, allow_pickle=False)
    except ValueError as exc:
        raise TypeError(
            "--target-groups must be a plain string array (saved without pickled objects)."
        ) from exc
    target_groups = np.asarray(target_groups).reshape(-1).astype(str)
    if len(target_groups) != COHORT_SIZE:
        raise ValueError(f"--target-groups has {len(target_groups)} labels; expected {COHORT_SIZE}.")
    if np.any(np.char.strip(target_groups) == "") or np.any(np.isin(target_groups, ["None", "nan"])):
        raise ValueError("--target-groups contains empty or missing labels.")

    pool_parts: list[np.ndarray] = []
    pool_id_parts: list[np.ndarray] = []
    for index, (emb_path, ids_path) in enumerate(args.m3_pool, start=1):
        part, part_ids = load_pair(emb_path, ids_path, f"MIMIC-III pool shard {index}")
        pool_parts.append(part)
        pool_id_parts.append(part_ids)
    pool = np.concatenate(pool_parts)
    pool_ids = np.concatenate(pool_id_parts)
    if len(np.unique(pool_ids)) != len(pool_ids):
        raise ValueError("MIMIC-III pool shards overlap in HADM_ID.")

    dtypes = {str(array.dtype) for array in [baseline, target, pool]}
    if len(dtypes) != 1:
        raise TypeError(f"Embedding dtypes differ across B, T, and C_cal candidates: {dtypes}.")

    # Sorting makes calibration construction independent of shard order.
    order = np.argsort(pool_ids, kind="stable")
    pool, pool_ids = pool[order], pool_ids[order]
    database = Path(args.mimic3_db).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(str(database)) as connection:
        subject_map = map_subjects(connection, np.concatenate([baseline_ids, pool_ids]))
    baseline_subjects = np.asarray([subject_map[int(x)] for x in baseline_ids], dtype=np.int64)
    pool_subjects = np.asarray([subject_map[int(x)] for x in pool_ids], dtype=np.int64)

    baseline_subject_set = set(baseline_subjects.tolist())
    keep = np.asarray([int(subject) not in baseline_subject_set for subject in pool_subjects])
    calibration = pool[keep]
    calibration_ids = pool_ids[keep]
    calibration_subjects = pool_subjects[keep]
    if len(calibration) < COHORT_SIZE:
        raise RuntimeError(
            f"DriftLens needs at least {COHORT_SIZE} patient-disjoint calibration rows; "
            f"only {len(calibration)} remain."
        )
    overlap = baseline_subject_set & set(calibration_subjects.tolist())
    if overlap:
        raise AssertionError(f"Baseline/calibration subject overlap remains: {len(overlap)} subjects.")

    arrays = {
        "baseline_raw.npy": baseline,
        "baseline_ids.npy": baseline_ids,
        "baseline_subject_ids.npy": baseline_subjects,
        "target_raw.npy": target,
        "target_ids.npy": target_ids,
        "target_source_groups.npy": target_groups,
        "calibration_raw.npy": calibration,
        "calibration_ids.npy": calibration_ids,
        "calibration_subject_ids.npy": calibration_subjects,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array, allow_pickle=False)

    manifest = {
        "design": "Experiment C minimal one-transition comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "construction": {
            "baseline": "exact frozen MIMIC-III baseline; no resampling",
            "target": "exact frozen primary 5,000-note MIMIC-IV target used for the framework MMD; no resampling",
            "calibration": (
                "all MIMIC-III pool rows after sorting by HADM_ID and excluding every "
                "subject represented in the baseline"
            ),
            "target_used_for_calibration_or_tuning": False,
            "framework_mmd_recomputed": False,
        },
        "counts": {
            "baseline_rows": len(baseline),
            "baseline_unique_subjects": len(np.unique(baseline_subjects)),
            "target_rows": len(target),
            "calibration_rows": len(calibration),
            "calibration_unique_subjects": len(np.unique(calibration_subjects)),
            "baseline_calibration_subject_overlap": 0,
        },
        "outputs": {
            name: output_record(output_dir / name, array) for name, array in arrays.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    manifest_path = output_dir / "experiment_c_preparation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
