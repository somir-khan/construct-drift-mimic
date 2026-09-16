#!/usr/bin/env python3
"""
experiment_b_materialize_labels.py

Experiment B — deterministic label/materialization stage.

NO MODEL PERFORMANCE IS COMPUTED HERE.

PURPOSE
-------
Materialize admission-level N x 50 binary diagnosis-label matrices for:

    MIMIC-III train
    MIMIC-III dev
    MIMIC-III test
    MIMIC-IV target

The exact embedding ids_*.npy arrays are the AUTHORITATIVE row order.

A SQL result may arrive in any order. Labels are scattered into rows using:

    row_index = {hadm_id: embedding_row}

This removes SQL-order dependence entirely.

The script also:
- verifies embedding-ID sets against frozen preflight manifests;
- restricts every diagnosis query to exactly the relevant cohort HADM_IDs
  using SQLite TEMP TABLE joins;
- raises a clear RuntimeError if SQL somehow returns an unexpected HADM_ID;
- verifies all materialized code counts against the frozen support CSV;
- materializes M3-test SUBJECT_IDs for the patient-cluster bootstrap;
- reorders M4 SUBJECT_IDs from the frozen manifest into embedding-ID order;
- independently verifies M4 subject mappings against hosp/admissions;
- emits independent row-ID witness arrays for labels and subject vectors;
- writes SHA-256 provenance.

NO embeddings are loaded here; only the ids_*.npy arrays are needed.

Frozen label construction:
- 50 diagnosis-only ICD-9 codes;
- selected from MIMIC-III TRAIN only;
- normalization:
      uppercase
      strip whitespace
      remove decimal point
      reject NAN/NONE/NULL
      preserve leading zeros
"""

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv(override=True)

DEFAULT_MIMIC3_DB = os.getenv("MIMIC3_DB_PATH")
DEFAULT_MIMIC4_DB = os.getenv("MIMIC4_DB_PATH")


# ============================================================================
# FROZEN EXPECTATIONS
# ============================================================================

EXPECTED_K = 50

EXPECTED_SPLITS = {"train": 47723, "dev": 1631, "test": 3372}

EXPECTED_M4_N = 19667


# ============================================================================
# HELPERS
# ============================================================================


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def normalize_icd9(code):
    """
    Canonical ICD-9 normalization.

    THIS FUNCTION MUST REMAIN IDENTICAL ACROSS:
      - Experiment B preflight
      - label materialization
      - frozen probe
    """
    if code is None:
        return None

    code = str(code).strip().upper().replace(".", "")

    if not code or code in {"NAN", "NONE", "NULL"}:
        return None

    return code


def load_ids(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} ID file not found: {path}")

    ids = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)

    if len(ids) == 0:
        raise RuntimeError(f"{label}: ID array is empty.")

    ids = ids.astype(np.int64)

    n_unique = len(np.unique(ids))

    if n_unique != len(ids):
        raise RuntimeError(
            f"{label}: duplicate HADM_IDs detected. "
            f"N={len(ids):,}, unique={n_unique:,}"
        )

    return ids


def load_codes(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Top-50 code file not found: {path}")

    codes = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            code = normalize_icd9(line)

            if code is not None:
                codes.append(code)

    if len(codes) != EXPECTED_K:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_K} codes, " f"found {len(codes)}."
        )

    if len(set(codes)) != EXPECTED_K:
        raise RuntimeError("Top-50 code list contains duplicates.")

    return codes


def assert_same_id_set(ids, expected_ids, label):
    actual = set(int(x) for x in ids.tolist())

    expected = set(int(x) for x in expected_ids)

    missing = sorted(expected - actual)

    extra = sorted(actual - expected)

    if missing or extra:
        raise RuntimeError(
            f"{label}: embedding-ID set does not equal "
            f"the frozen manifest.\n"
            f"  embedding N : {len(actual):,}\n"
            f"  manifest N  : {len(expected):,}\n"
            f"  missing     : {len(missing):,} "
            f"(examples={missing[:10]})\n"
            f"  extra       : {len(extra):,} "
            f"(examples={extra[:10]})"
        )


def create_temp_ids(conn, ids, table_name):
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    conn.execute(
        f"""
        CREATE TEMP TABLE "{table_name}" (
            hadm_id INTEGER PRIMARY KEY
        )
        """
    )

    conn.executemany(
        f"""
        INSERT INTO "{table_name}" (hadm_id)
        VALUES (?)
        """,
        [(int(x),) for x in ids.tolist()],
    )

    n = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM "{table_name}"
        """
    ).fetchone()[0]

    if n != len(ids):
        raise RuntimeError(
            f"Temp table {table_name}: expected " f"{len(ids):,} rows, got {n:,}."
        )


def matrix_from_rows(rows, ids, codes, label):
    row_index = {int(hadm_id): i for i, hadm_id in enumerate(ids.tolist())}

    code_index = {code: j for j, code in enumerate(codes)}

    Y = np.zeros((len(ids), len(codes)), dtype=np.uint8)

    unexpected = set()

    for row in rows.itertuples(index=False):
        hadm_id = int(row.hadm_id)

        if hadm_id not in row_index:
            unexpected.add(hadm_id)
            continue

        code = normalize_icd9(row.code)

        if code is None:
            continue

        j = code_index.get(code)

        if j is not None:
            Y[row_index[hadm_id], j] = 1

    if unexpected:
        examples = sorted(unexpected)[:20]

        raise RuntimeError(
            f"{label}: diagnosis SQL returned HADM_IDs "
            f"outside the exact embedding-ID cohort. "
            f"Offending IDs include: {examples}"
        )

    return Y


# ============================================================================
# MIMIC-III LABELS
# ============================================================================


def materialize_m3_labels(conn, ids, codes, split):
    table = f"expb_{split}_ids"

    create_temp_ids(conn, ids, table)

    df = pd.read_sql_query(
        f"""
        SELECT
            d.HADM_ID AS hadm_id,
            d.ICD9_CODE AS code

        FROM DIAGNOSES_ICD d

        JOIN "{table}" t
          ON t.hadm_id = d.HADM_ID

        WHERE d.ICD9_CODE IS NOT NULL
        """,
        conn,
    )

    return matrix_from_rows(df, ids, codes, f"M3 {split}")


# ============================================================================
# MIMIC-IV LABELS
# ============================================================================


def materialize_m4_labels(conn, ids, codes):
    table = "expb_m4_target_ids"

    create_temp_ids(conn, ids, table)

    df = pd.read_sql_query(
        f"""
        SELECT
            d.hadm_id AS hadm_id,
            d.icd_code AS code

        FROM "hosp/diagnoses_icd" d

        JOIN "{table}" t
          ON t.hadm_id = d.hadm_id

        WHERE d.icd_version = 9
          AND d.icd_code IS NOT NULL
        """,
        conn,
    )

    return matrix_from_rows(df, ids, codes, "M4 target")


# ============================================================================
# SUBJECT-ID MATERIALIZATION
# ============================================================================


def query_m3_test_subjects(conn, ids):
    table = "expb_m3_test_subject_ids"

    create_temp_ids(conn, ids, table)

    df = pd.read_sql_query(
        f"""
        SELECT
            t.hadm_id,
            a.SUBJECT_ID AS subject_id

        FROM "{table}" t

        LEFT JOIN ADMISSIONS a
          ON a.HADM_ID = t.hadm_id
        """,
        conn,
    )

    if df["subject_id"].isna().any():
        bad = df.loc[df["subject_id"].isna(), "hadm_id"].astype(int).tolist()

        raise RuntimeError(
            "M3 test subject mapping is missing "
            "ADMISSIONS rows for HADM_IDs: "
            f"{bad[:20]}"
        )

    if df["hadm_id"].duplicated().any():
        raise RuntimeError("M3 test subject query returned " "duplicate HADM_IDs.")

    expected_set = set(int(x) for x in ids.tolist())

    returned_set = set(int(x) for x in df["hadm_id"].tolist())

    unexpected = sorted(returned_set - expected_set)

    missing = sorted(expected_set - returned_set)

    if unexpected:
        raise RuntimeError(
            "M3 subject query returned unexpected " f"HADM_IDs: {unexpected[:20]}"
        )

    if missing:
        raise RuntimeError(
            "M3 subject query failed to return " f"HADM_IDs: {missing[:20]}"
        )

    mapping = {
        int(row.hadm_id): int(row.subject_id) for row in df.itertuples(index=False)
    }

    subjects = np.array([mapping[int(h)] for h in ids.tolist()], dtype=np.int64)

    aligned = pd.DataFrame({"hadm_id": ids.astype(np.int64), "subject_id": subjects})

    return subjects, aligned


def build_and_verify_m4_subjects(conn, ids, m4_manifest):
    manifest = m4_manifest.copy()

    manifest["hadm_id"] = pd.to_numeric(manifest["hadm_id"], errors="raise").astype(
        np.int64
    )

    manifest["subject_id"] = pd.to_numeric(
        manifest["subject_id"], errors="raise"
    ).astype(np.int64)

    if manifest["hadm_id"].duplicated().any():
        raise RuntimeError("Frozen M4 manifest contains " "duplicate HADM_IDs.")

    manifest_map = dict(
        zip(manifest["hadm_id"].tolist(), manifest["subject_id"].tolist())
    )

    expected_set = set(int(x) for x in ids.tolist())

    manifest_set = set(int(x) for x in manifest_map)

    missing = sorted(expected_set - manifest_set)

    extra = sorted(manifest_set - expected_set)

    if missing or extra:
        raise RuntimeError(
            "M4 subject manifest does not equal "
            "the embedding-ID cohort.\n"
            f"missing={missing[:20]}\n"
            f"extra={extra[:20]}"
        )

    manifest_subjects = np.array(
        [manifest_map[int(h)] for h in ids.tolist()], dtype=np.int64
    )

    # Independent DB verification.
    table = "expb_m4_subject_ids"

    create_temp_ids(conn, ids, table)

    df = pd.read_sql_query(
        f"""
        SELECT
            t.hadm_id,
            a.subject_id

        FROM "{table}" t

        LEFT JOIN "hosp/admissions" a
          ON a.hadm_id = t.hadm_id
        """,
        conn,
    )

    if df["subject_id"].isna().any():
        bad = df.loc[df["subject_id"].isna(), "hadm_id"].astype(int).tolist()

        raise RuntimeError(
            "M4 admissions lookup is missing " f"subject_id for: {bad[:20]}"
        )

    if df["hadm_id"].duplicated().any():
        raise RuntimeError("M4 admissions lookup returned " "duplicate HADM_IDs.")

    returned_set = set(int(x) for x in df["hadm_id"].tolist())

    if returned_set != expected_set:
        raise RuntimeError(
            "M4 admissions subject query did not "
            "return exactly the target HADM_ID set."
        )

    db_map = {
        int(row.hadm_id): int(row.subject_id) for row in df.itertuples(index=False)
    }

    db_subjects = np.array([db_map[int(h)] for h in ids.tolist()], dtype=np.int64)

    mismatch = np.flatnonzero(manifest_subjects != db_subjects)

    if len(mismatch):
        examples = []

        for i in mismatch[:20]:
            examples.append(
                {
                    "row": int(i),
                    "hadm_id": int(ids[i]),
                    "manifest_subject_id": int(manifest_subjects[i]),
                    "database_subject_id": int(db_subjects[i]),
                }
            )

        raise RuntimeError(
            "M4 frozen-manifest subject_id disagrees "
            "with hosp/admissions.\n"
            f"Examples: {examples}"
        )

    aligned = pd.DataFrame(
        {"hadm_id": ids.astype(np.int64), "subject_id": manifest_subjects}
    )

    return manifest_subjects, aligned


# ============================================================================
# SUPPORT VERIFICATION
# ============================================================================


def verify_support(Ys, codes, support_csv):
    support = pd.read_csv(support_csv, dtype={"code": str})

    required = {"code", "m3_train_n", "m3_dev_n", "m3_test_n", "m4_target_n"}

    missing = required - set(support.columns)

    if missing:
        raise RuntimeError("Support CSV is missing columns: " f"{sorted(missing)}")

    support["code"] = support["code"].map(normalize_icd9)

    if support["code"].duplicated().any():
        raise RuntimeError("Support CSV contains duplicate " "normalized ICD-9 codes.")

    support = support.set_index("code")

    expected_columns = {
        "train": "m3_train_n",
        "dev": "m3_dev_n",
        "test": "m3_test_n",
        "m4": "m4_target_n",
    }

    discrepancies = []

    for dataset, expected_col in expected_columns.items():
        observed = Ys[dataset].sum(axis=0, dtype=np.int64)

        for j, code in enumerate(codes):
            if code not in support.index:
                raise RuntimeError(f"Code {code} is absent " "from support CSV.")

            expected = int(support.loc[code, expected_col])

            actual = int(observed[j])

            if actual != expected:
                discrepancies.append(
                    {
                        "dataset": dataset,
                        "code": code,
                        "expected": expected,
                        "materialized": actual,
                    }
                )

    if discrepancies:
        raise RuntimeError(
            "Materialized label support does not "
            "reproduce the frozen preflight counts.\n"
            f"First discrepancies: "
            f"{discrepancies[:20]}"
        )


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Materialize Experiment B top-50 labels " "in exact embedding-ID row order."
        ),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )

    p.add_argument("--m3-train-ids", required=True)

    p.add_argument("--m3-dev-ids", required=True)

    p.add_argument("--m3-test-ids", required=True)

    p.add_argument("--m4-target-ids", required=True)

    p.add_argument("--m3-split-manifest", required=True)

    p.add_argument("--m4-manifest", required=True)

    p.add_argument("--top50-codes", required=True)

    p.add_argument("--support-csv", required=True)

    p.add_argument("--mimic3-db", default=DEFAULT_MIMIC3_DB)

    p.add_argument("--mimic4-db", default=DEFAULT_MIMIC4_DB)

    p.add_argument("--out-dir", default="outputs/experiment_b_labels")

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = parse_args()

    if not args.mimic3_db:
        raise RuntimeError("MIMIC3_DB_PATH unavailable; " "supply --mimic3-db.")

    if not args.mimic4_db:
        raise RuntimeError("MIMIC4_DB_PATH unavailable; " "supply --mimic4-db.")

    required_paths = [
        (args.mimic3_db, "MIMIC-III DB"),
        (args.mimic4_db, "MIMIC-IV DB"),
        (args.m3_split_manifest, "M3 split manifest"),
        (args.m4_manifest, "M4 manifest"),
        (args.top50_codes, "top-50 code list"),
        (args.support_csv, "support CSV"),
    ]

    for path, label in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Exact embedding ID arrays = authoritative order
    # ------------------------------------------------------------------------

    ids = {
        "train": load_ids(args.m3_train_ids, "M3 train"),
        "dev": load_ids(args.m3_dev_ids, "M3 dev"),
        "test": load_ids(args.m3_test_ids, "M3 test"),
        "m4": load_ids(args.m4_target_ids, "M4 target"),
    }

    print("=" * 80)
    print("EXPERIMENT B — LABEL MATERIALIZATION")
    print("=" * 80)

    for key in ["train", "dev", "test", "m4"]:
        print(f"{key:<8}: " f"{len(ids[key]):,} IDs")

    # ------------------------------------------------------------------------
    # Frozen M3 manifest
    # ------------------------------------------------------------------------

    m3_manifest = pd.read_csv(args.m3_split_manifest)

    m3_manifest.columns = [str(c).strip().lower() for c in m3_manifest.columns]

    if not {"split", "hadm_id"}.issubset(m3_manifest.columns):
        raise RuntimeError("M3 manifest requires " "split,hadm_id columns.")

    m3_manifest["hadm_id"] = pd.to_numeric(
        m3_manifest["hadm_id"], errors="raise"
    ).astype(np.int64)

    for split in ["train", "dev", "test"]:
        expected = (
            m3_manifest.loc[m3_manifest["split"] == split, "hadm_id"]
            .astype(np.int64)
            .tolist()
        )

        assert_same_id_set(ids[split], expected, f"M3 {split}")

        expected_n = EXPECTED_SPLITS[split]

        if len(ids[split]) != expected_n:
            raise RuntimeError(
                f"M3 {split}: expected "
                f"N={expected_n:,}, "
                f"got {len(ids[split]):,}."
            )

    # ------------------------------------------------------------------------
    # Frozen M4 manifest
    # ------------------------------------------------------------------------

    m4_manifest = pd.read_csv(args.m4_manifest)

    m4_manifest.columns = [str(c).strip().lower() for c in m4_manifest.columns]

    if not {"hadm_id", "subject_id"}.issubset(m4_manifest.columns):
        raise RuntimeError("M4 manifest requires " "hadm_id,subject_id columns.")

    m4_manifest["hadm_id"] = pd.to_numeric(
        m4_manifest["hadm_id"], errors="raise"
    ).astype(np.int64)

    assert_same_id_set(ids["m4"], m4_manifest["hadm_id"].tolist(), "M4 target")

    if len(ids["m4"]) != EXPECTED_M4_N:
        raise RuntimeError(
            f"M4 target: expected " f"N={EXPECTED_M4_N:,}, " f"got {len(ids['m4']):,}."
        )

    # ------------------------------------------------------------------------
    # Frozen codes
    # ------------------------------------------------------------------------

    codes = load_codes(args.top50_codes)

    print()
    print("Frozen diagnosis label space: " f"{len(codes)} codes")

    # ------------------------------------------------------------------------
    # Materialize labels and subjects
    # ------------------------------------------------------------------------

    conn3 = sqlite3.connect(args.mimic3_db)

    conn4 = sqlite3.connect(args.mimic4_db)

    try:
        Y_train = materialize_m3_labels(conn3, ids["train"], codes, "train")

        Y_dev = materialize_m3_labels(conn3, ids["dev"], codes, "dev")

        Y_test = materialize_m3_labels(conn3, ids["test"], codes, "test")

        Y_m4 = materialize_m4_labels(conn4, ids["m4"], codes)

        (m3_test_subjects, m3_test_subject_df) = query_m3_test_subjects(
            conn3, ids["test"]
        )

        (m4_subjects, m4_subject_df) = build_and_verify_m4_subjects(
            conn4, ids["m4"], m4_manifest
        )

    finally:
        conn3.close()
        conn4.close()

    Ys = {"train": Y_train, "dev": Y_dev, "test": Y_test, "m4": Y_m4}

    # ------------------------------------------------------------------------
    # Reproduce frozen preflight support exactly
    # ------------------------------------------------------------------------

    verify_support(Ys, codes, args.support_csv)

    print()
    print("Frozen support CSV reproduced " "exactly by all four label matrices.")

    # ------------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------------

    output_paths = {
        "labels_train": out_dir / "labels_m3_train_top50.npy",
        "labels_dev": out_dir / "labels_m3_dev_top50.npy",
        "labels_test": out_dir / "labels_m3_test_top50.npy",
        "labels_m4": out_dir / "labels_m4_target_top50.npy",
        "subjects_m3_test": out_dir / "subjects_m3_test.npy",
        "subjects_m4": out_dir / "subjects_m4_target.npy",
        "row_ids_labels_train": out_dir / "row_ids_labels_m3_train.npy",
        "row_ids_labels_dev": out_dir / "row_ids_labels_m3_dev.npy",
        "row_ids_labels_test": out_dir / "row_ids_labels_m3_test.npy",
        "row_ids_labels_m4": out_dir / "row_ids_labels_m4_target.npy",
        "row_ids_subjects_m3_test": out_dir / "row_ids_subjects_m3_test.npy",
        "row_ids_subjects_m4": out_dir / "row_ids_subjects_m4_target.npy",
        "m3_test_subject_csv": out_dir / "experiment_b_m3_test_subjects.csv",
        "m4_subject_csv": out_dir / "experiment_b_m4_target_subjects_aligned.csv",
    }

    # ------------------------------------------------------------------------
    # Save labels
    # ------------------------------------------------------------------------

    np.save(output_paths["labels_train"], Y_train)

    np.save(output_paths["labels_dev"], Y_dev)

    np.save(output_paths["labels_test"], Y_test)

    np.save(output_paths["labels_m4"], Y_m4)

    # ------------------------------------------------------------------------
    # Save subjects
    # ------------------------------------------------------------------------

    np.save(output_paths["subjects_m3_test"], m3_test_subjects)

    np.save(output_paths["subjects_m4"], m4_subjects)

    # ------------------------------------------------------------------------
    # Independently persisted row-order witnesses
    # ------------------------------------------------------------------------

    np.save(output_paths["row_ids_labels_train"], ids["train"])

    np.save(output_paths["row_ids_labels_dev"], ids["dev"])

    np.save(output_paths["row_ids_labels_test"], ids["test"])

    np.save(output_paths["row_ids_labels_m4"], ids["m4"])

    np.save(output_paths["row_ids_subjects_m3_test"], ids["test"])

    np.save(output_paths["row_ids_subjects_m4"], ids["m4"])

    m3_test_subject_df.to_csv(output_paths["m3_test_subject_csv"], index=False)

    m4_subject_df.to_csv(output_paths["m4_subject_csv"], index=False)

    # ------------------------------------------------------------------------
    # Final sanity checks
    # ------------------------------------------------------------------------

    if Y_train.shape != (EXPECTED_SPLITS["train"], EXPECTED_K):
        raise RuntimeError(f"Unexpected train label shape: " f"{Y_train.shape}")

    if Y_dev.shape != (EXPECTED_SPLITS["dev"], EXPECTED_K):
        raise RuntimeError(f"Unexpected dev label shape: " f"{Y_dev.shape}")

    if Y_test.shape != (EXPECTED_SPLITS["test"], EXPECTED_K):
        raise RuntimeError(f"Unexpected test label shape: " f"{Y_test.shape}")

    if Y_m4.shape != (EXPECTED_M4_N, EXPECTED_K):
        raise RuntimeError(f"Unexpected M4 label shape: " f"{Y_m4.shape}")

    # ------------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------------

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "pre-outcome",
        "performance_examined": False,
        "row_order_authority": "embedding ids_*.npy arrays",
        "icd9_normalization": (
            "uppercase; strip whitespace; "
            "remove decimal point; reject "
            "NAN/NONE/NULL; preserve leading zeros"
        ),
        "label_count": len(codes),
        "codes": codes,
        "matrix_shapes": {
            "m3_train": list(Y_train.shape),
            "m3_dev": list(Y_dev.shape),
            "m3_test": list(Y_test.shape),
            "m4_target": list(Y_m4.shape),
        },
        "subject_counts": {
            "m3_test": int(len(np.unique(m3_test_subjects))),
            "m4_target": int(len(np.unique(m4_subjects))),
        },
        "input_hashes_sha256": {
            "m3_train_ids": sha256_file(args.m3_train_ids),
            "m3_dev_ids": sha256_file(args.m3_dev_ids),
            "m3_test_ids": sha256_file(args.m3_test_ids),
            "m4_target_ids": sha256_file(args.m4_target_ids),
            "m3_split_manifest": sha256_file(args.m3_split_manifest),
            "m4_manifest": sha256_file(args.m4_manifest),
            "top50_codes": sha256_file(args.top50_codes),
            "support_csv": sha256_file(args.support_csv),
        },
        "output_hashes_sha256": {
            key: sha256_file(path) for key, path in output_paths.items()
        },
    }

    provenance_path = out_dir / "experiment_b_label_materialization_provenance.json"

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MATERIALIZATION PASSED")
    print("=" * 80)

    for key, path in output_paths.items():
        print(f"{key:<30}: {path}")

    print(f"{'provenance':<30}: " f"{provenance_path}")

    print()
    print("No performance metric was computed.")


if __name__ == "__main__":
    main()
