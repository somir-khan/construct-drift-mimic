#!/usr/bin/env python3
"""
scripts/experiment_b/experiment_b_preflight.py

Experiment B — final PRE-OUTCOME cohort / label-space freeze.

PURPOSE
-------
1. Load published Mullenbach MIMIC-III train/dev/test HADM_ID manifests
   from a single directory.

2. Verify:
   - manifest sizes
   - unique HADM_IDs
   - HADM_ID disjointness
   - SUBJECT_ID disjointness
   - usable discharge-summary coverage under the FROZEN submitted
     latest-row construction:
         PARTITION BY HADM_ID
         ORDER BY CHARTDATE DESC, ROW_ID DESC

3. Derive top-K diagnosis-only ICD-9 codes from MIMIC-III TRAIN ONLY.

4. Freeze labels deterministically:
       admission frequency DESC,
       normalized ICD-9 code ASC

5. Measure support in:
   - MIMIC-III train
   - MIMIC-III dev
   - MIMIC-III test
   - MIMIC-IV 2014-2016 anchor-year-group ICD-9-only target

6. Verify AUROC definability before any classifier is trained.

MIMIC-IV TARGET POLICY
----------------------
An admission enters the final target only if:
    - it has a usable discharge note;
    - its patient's anchor_year_group is the requested group;
    - it has >=1 diagnosis row with icd_version = 9;
    - it has NO diagnosis row with icd_version != 9.

Admissions containing both ICD-9 and another ICD version are counted and
reported, but excluded from the final target.

IMPORTANT
---------
- anchor_year_group is a PATIENT-LEVEL grouping variable, not a note date.
- MIMIC-III uses the submitted frozen latest eligible discharge-summary row.
- Report-preferred preprocessing is NOT substituted into Experiment B.
- No embeddings are loaded.
- No classifier is trained.
- No AUROC or other performance outcome is computed.

EXPECTED MULLENBACH DIRECTORY CONTENTS
--------------------------------------
The directory supplied with --mullenbach-dir should contain:

    train_full_hadm_ids.csv
    dev_full_hadm_ids.csv
    test_full_hadm_ids.csv

Example:
    ../tmis_datasets/datasets/caml-mimic-master/mimicdata/mimic3/

USAGE
-----
python scripts/experiment_b/experiment_b_preflight.py \
    --mullenbach-dir ../tmis_datasets/datasets/caml-mimic-master/mimicdata/mimic3 \
    --mimic3-db ../tmis_datasets/datasets/mimic3.db \
    --mimic4-db ../tmis_datasets/datasets/mimic4.db

If MIMIC3_DB_PATH and MIMIC4_DB_PATH are correctly configured in .env,
the explicit database arguments may be omitted.
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()

DEFAULT_MIMIC3_DB = os.getenv("MIMIC3_DB_PATH")
DEFAULT_MIMIC4_DB = os.getenv("MIMIC4_DB_PATH")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TRAIN_FILENAME = "train_full_hadm_ids.csv"
DEFAULT_DEV_FILENAME = "dev_full_hadm_ids.csv"
DEFAULT_TEST_FILENAME = "test_full_hadm_ids.csv"

EXPECTED_M4_ANY_NATIVE_ICD9_2014_2016 = 19669


# =============================================================================
# HELPERS
# =============================================================================

def print_rule(char="-", width=80):
    print(char * width)


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def overlap_count(a, b):
    return len(set(a) & set(b))


def normalize_icd9(code):
    """
    Canonical ICD-9 normalization applied identically to MIMIC-III and
    MIMIC-IV.

    Rules
    -----
    - convert to string
    - strip surrounding whitespace
    - uppercase
    - remove decimal points

    Leading zeros are preserved exactly as present.
    """
    if code is None:
        return None

    code = str(code).strip().upper().replace(".", "")

    if not code:
        return None

    if code in {"NAN", "NONE", "NULL"}:
        return None

    return code


def load_hadm_manifest(path):
    """
    Load a one-HADM_ID-per-line manifest safely.

    Supported
    ---------
    - .npy
    - single-column CSV/text files

    No delimiter sniffing is used.

    This safely handles either a headerless one-column list of integer IDs or
    the same list preceded by a textual ``HADM_ID`` header. The header is
    converted to NaN under numeric coercion and then removed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if not path.is_file():
        raise ValueError(f"Manifest path is not a file: {path}")

    if path.suffix.lower() == ".npy":
        vals = np.asarray(
            np.load(path, allow_pickle=False)
        ).reshape(-1)

        s = pd.Series(vals)

    else:
        try:
            df = pd.read_csv(
                path,
                header=None,
                usecols=[0],
                names=["hadm_id"],
                dtype=str,
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to read manifest as one-column file: {path}"
            ) from exc

        s = df["hadm_id"]

    vals = pd.to_numeric(
        s,
        errors="coerce",
    ).dropna()

    if len(vals) == 0:
        raise ValueError(
            f"No numeric HADM_ID values could be parsed from {path}"
        )

    # HADM_IDs are integral.
    frac = vals - np.floor(vals)

    if not np.allclose(frac.to_numpy(), 0):
        raise ValueError(
            f"Non-integer values were found in HADM_ID manifest: {path}"
        )

    return vals.astype(np.int64).tolist()


def validate_path(path, label):
    """
    Resolve and validate a filesystem path.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    return path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment B final pre-outcome cohort and label-space freeze."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mullenbach-dir",
        required=True,
        help=(
            "Directory containing train_full_hadm_ids.csv, "
            "dev_full_hadm_ids.csv, and test_full_hadm_ids.csv."
        ),
    )

    parser.add_argument(
        "--train-filename",
        default=DEFAULT_TRAIN_FILENAME,
        help="Training manifest filename inside --mullenbach-dir.",
    )

    parser.add_argument(
        "--dev-filename",
        default=DEFAULT_DEV_FILENAME,
        help="Development manifest filename inside --mullenbach-dir.",
    )

    parser.add_argument(
        "--test-filename",
        default=DEFAULT_TEST_FILENAME,
        help="Test manifest filename inside --mullenbach-dir.",
    )

    parser.add_argument(
        "--mimic3-db",
        default=DEFAULT_MIMIC3_DB,
        help="MIMIC-III SQLite database path.",
    )

    parser.add_argument(
        "--mimic4-db",
        default=DEFAULT_MIMIC4_DB,
        help="MIMIC-IV SQLite database path.",
    )

    parser.add_argument(
        "--m4-window",
        default="2014 - 2016",
        help=(
            "Patient-level MIMIC-IV anchor_year_group used for the "
            "native ICD-9 target. This is NOT interpreted as note-authorship "
            "calendar time."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of TRAIN-only diagnosis codes to freeze.",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/experiment_b_preflight",
        help="Output directory for frozen preflight artifacts.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    # =========================================================================
    # 0. RESOLVE INPUT PATHS
    # =========================================================================

    if args.k <= 0:
        raise ValueError("--k must be greater than zero.")

    mullenbach_dir = validate_path(
        args.mullenbach_dir,
        "Mullenbach manifest directory",
    )

    if not mullenbach_dir.is_dir():
        raise NotADirectoryError(
            f"--mullenbach-dir is not a directory: {mullenbach_dir}"
        )

    train_path = (
        mullenbach_dir / args.train_filename
    ).resolve()

    dev_path = (
        mullenbach_dir / args.dev_filename
    ).resolve()

    test_path = (
        mullenbach_dir / args.test_filename
    ).resolve()

    manifest_paths = {
        "train": train_path,
        "dev": dev_path,
        "test": test_path,
    }

    for split_name, path in manifest_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required {split_name} manifest not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{split_name} manifest path is not a file: {path}"
            )

    if not args.mimic3_db:
        raise RuntimeError(
            "MIMIC3_DB_PATH is unset and --mimic3-db was not supplied."
        )

    if not args.mimic4_db:
        raise RuntimeError(
            "MIMIC4_DB_PATH is unset and --mimic4-db was not supplied."
        )

    mimic3_db = validate_path(
        args.mimic3_db,
        "MIMIC-III database",
    )

    mimic4_db = validate_path(
        args.mimic4_db,
        "MIMIC-IV database",
    )

    if not mimic3_db.is_file():
        raise ValueError(
            f"MIMIC-III DB path is not a file: {mimic3_db}"
        )

    if not mimic4_db.is_file():
        raise ValueError(
            f"MIMIC-IV DB path is not a file: {mimic4_db}"
        )

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("EXPERIMENT B — FINAL PRE-OUTCOME PREFLIGHT")
    print("=" * 80)

    print(f"MIMIC-III DB       : {mimic3_db}")
    print(f"MIMIC-IV DB        : {mimic4_db}")
    print(f"Mullenbach dir     : {mullenbach_dir}")
    print(f"M4 anchor group    : {args.m4_window}")
    print(f"Top-K              : {args.k}")
    print(f"Output directory   : {out_dir}")

    print()
    print("MULLENBACH MANIFEST FILES")
    print_rule()

    for split_name, path in manifest_paths.items():
        print(f"{split_name:<6}: {path}")

    # =========================================================================
    # 1. LOAD MULLENBACH MANIFESTS
    # =========================================================================

    train_ids = load_hadm_manifest(
        train_path
    )

    dev_ids = load_hadm_manifest(
        dev_path
    )

    test_ids = load_hadm_manifest(
        test_path
    )

    splits = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }

    print()
    print("MULLENBACH HADM_ID MANIFESTS")
    print_rule()

    for split_name, ids in splits.items():

        n_rows = len(ids)
        n_unique = len(set(ids))
        n_duplicates = n_rows - n_unique

        print(
            f"{split_name:<6} "
            f"rows={n_rows:>7,}  "
            f"unique={n_unique:>7,}  "
            f"duplicates={n_duplicates:>5,}"
        )

    # Internal duplicate check.
    for split_name, ids in splits.items():

        if len(ids) != len(set(ids)):
            raise RuntimeError(
                f"STOP: {split_name} manifest contains duplicate HADM_IDs."
            )

    # =========================================================================
    # 2. HADM_ID DISJOINTNESS
    # =========================================================================

    hadm_overlap = {
        "train_dev": overlap_count(
            train_ids,
            dev_ids,
        ),

        "train_test": overlap_count(
            train_ids,
            test_ids,
        ),

        "dev_test": overlap_count(
            dev_ids,
            test_ids,
        ),
    }

    print()
    print("HADM_ID OVERLAP")
    print_rule()

    for key, value in hadm_overlap.items():
        print(
            f"{key:<20}: {value:,}"
        )

    if any(
        value != 0
        for value in hadm_overlap.values()
    ):
        raise RuntimeError(
            "STOP: Mullenbach manifests are not HADM_ID-disjoint."
        )

    # =========================================================================
    # 3. MIMIC-III PREFLIGHT
    # =========================================================================

    conn3 = sqlite3.connect(
        str(mimic3_db)
    )

    try:

        # ---------------------------------------------------------------------
        # 3A. Load split IDs into SQLite TEMP table
        # ---------------------------------------------------------------------

        split_rows = []

        for split_name, ids in splits.items():

            split_rows.extend(
                (
                    split_name,
                    int(hadm_id),
                )
                for hadm_id in ids
            )

        conn3.execute(
            """
            CREATE TEMP TABLE expb_split_ids (
                split   TEXT NOT NULL,
                HADM_ID INTEGER NOT NULL
            )
            """
        )

        conn3.executemany(
            """
            INSERT INTO expb_split_ids(
                split,
                HADM_ID
            )
            VALUES (?, ?)
            """,
            split_rows,
        )

        conn3.execute(
            """
            CREATE INDEX idx_expb_split_hadm
            ON expb_split_ids(HADM_ID)
            """
        )

        # ---------------------------------------------------------------------
        # 3B. Verify ADMISSIONS lookup
        # ---------------------------------------------------------------------

        subject_df = pd.read_sql_query(
            """
            SELECT
                s.split,
                s.HADM_ID,
                a.SUBJECT_ID

            FROM expb_split_ids s

            LEFT JOIN ADMISSIONS a
              ON a.HADM_ID = s.HADM_ID
            """,
            conn3,
        )

        missing_admission = (
            subject_df[
                subject_df["SUBJECT_ID"].isna()
            ]
            .copy()
        )

        print()
        print("MIMIC-III ADMISSION LOOKUP")
        print_rule()

        print(
            f"Missing ADMISSIONS rows: "
            f"{len(missing_admission):,}"
        )

        if len(missing_admission) > 0:

            print()
            print(
                missing_admission[
                    [
                        "split",
                        "HADM_ID",
                    ]
                ]
                .head(20)
                .to_string(index=False)
            )

            raise RuntimeError(
                "STOP: some manifest HADM_IDs are absent from ADMISSIONS."
            )

        # ---------------------------------------------------------------------
        # 3C. Verify patient disjointness
        # ---------------------------------------------------------------------

        subjects = {}

        for split_name in [
            "train",
            "dev",
            "test",
        ]:

            vals = (
                subject_df.loc[
                    subject_df["split"] == split_name,
                    "SUBJECT_ID",
                ]
                .astype(np.int64)
                .tolist()
            )

            subjects[
                split_name
            ] = vals

        subject_overlap = {
            "train_dev": overlap_count(
                subjects["train"],
                subjects["dev"],
            ),

            "train_test": overlap_count(
                subjects["train"],
                subjects["test"],
            ),

            "dev_test": overlap_count(
                subjects["dev"],
                subjects["test"],
            ),
        }

        print()
        print("SUBJECT_ID OVERLAP")
        print_rule()

        for key, value in subject_overlap.items():
            print(
                f"{key:<20}: {value:,}"
            )

        if any(
            value != 0
            for value in subject_overlap.values()
        ):
            raise RuntimeError(
                "STOP: Mullenbach splits are not patient-disjoint "
                "against the local MIMIC-III database."
            )

        # ---------------------------------------------------------------------
        # 3D. Verify frozen latest-row discharge-note coverage
        # ---------------------------------------------------------------------

        note_coverage = pd.read_sql_query(
            """
            WITH ranked AS (

                SELECT
                    s.split,
                    n.HADM_ID,
                    n.DESCRIPTION,

                    ROW_NUMBER() OVER (
                        PARTITION BY n.HADM_ID
                        ORDER BY
                            n.CHARTDATE DESC,
                            n.ROW_ID DESC
                    ) AS rn

                FROM expb_split_ids s

                JOIN NOTEEVENTS n
                  ON n.HADM_ID = s.HADM_ID

                WHERE n.CATEGORY = 'Discharge summary'
                  AND (
                        n.ISERROR IS NULL
                        OR n.ISERROR != '1'
                      )
                  AND n.HADM_ID IS NOT NULL
                  AND n.TEXT IS NOT NULL
            ),

            selected AS (

                SELECT
                    split,
                    HADM_ID,
                    DESCRIPTION

                FROM ranked

                WHERE rn = 1
            )

            SELECT
                s.split,

                COUNT(
                    DISTINCT s.HADM_ID
                )
                    AS manifest_n,

                COUNT(
                    DISTINCT x.HADM_ID
                )
                    AS usable_note_n,

                COUNT(
                    DISTINCT CASE
                        WHEN x.DESCRIPTION = 'Addendum'
                        THEN x.HADM_ID
                    END
                )
                    AS addendum_first_n

            FROM expb_split_ids s

            LEFT JOIN selected x
              ON x.HADM_ID = s.HADM_ID
             AND x.split = s.split

            GROUP BY s.split

            ORDER BY CASE s.split
                WHEN 'train' THEN 1
                WHEN 'dev' THEN 2
                WHEN 'test' THEN 3
            END
            """,
            conn3,
        )

        print()
        print("FROZEN LATEST-ROW NOTE COVERAGE")
        print_rule()

        print(
            note_coverage.to_string(
                index=False
            )
        )

        if not np.all(
            note_coverage[
                "manifest_n"
            ].to_numpy()
            ==
            note_coverage[
                "usable_note_n"
            ].to_numpy()
        ):
            raise RuntimeError(
                "STOP: some Mullenbach admissions lack a usable "
                "discharge summary under the frozen latest-row rule."
            )

        # ---------------------------------------------------------------------
        # 3E. Load diagnosis-only ICD-9 labels
        # ---------------------------------------------------------------------

        diag3 = pd.read_sql_query(
            """
            SELECT
                s.split,
                d.HADM_ID,
                d.ICD9_CODE

            FROM expb_split_ids s

            JOIN DIAGNOSES_ICD d
              ON d.HADM_ID = s.HADM_ID

            WHERE d.ICD9_CODE IS NOT NULL
            """,
            conn3,
        )

        diag3["code"] = (
            diag3[
                "ICD9_CODE"
            ]
            .map(
                normalize_icd9
            )
        )

        diag3 = (
            diag3
            .dropna(
                subset=[
                    "code",
                ]
            )
            .drop_duplicates(
                [
                    "split",
                    "HADM_ID",
                    "code",
                ]
            )
        )

        # ---------------------------------------------------------------------
        # 3F. Derive top-K from TRAIN only
        # ---------------------------------------------------------------------

        train_diag = (
            diag3[
                diag3["split"] == "train"
            ]
            .copy()
        )

        train_freq = (
            train_diag
            .groupby(
                "code"
            )[
                "HADM_ID"
            ]
            .nunique()
            .reset_index(
                name="m3_train_n"
            )
            .sort_values(
                [
                    "m3_train_n",
                    "code",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        topk = (
            train_freq
            .head(
                args.k
            )
            .copy()
        )

        if len(topk) != args.k:
            raise RuntimeError(
                f"STOP: requested top-{args.k}, but only "
                f"{len(topk)} diagnosis codes were available."
            )

        topk.insert(
            0,
            "rank",
            np.arange(
                1,
                len(topk) + 1,
            ),
        )

        top_codes = (
            topk[
                "code"
            ]
            .tolist()
        )

        print()
        print(
            f"TOP-{args.k} DIAGNOSIS-ONLY ICD-9 LABEL SPACE"
        )
        print_rule()

        print(
            "Derived from : MIMIC-III TRAIN only"
        )

        print(
            "Counting unit: distinct HADM_IDs per normalized diagnosis code"
        )

        print(
            "Tie-break    : admission frequency DESC, normalized code ASC"
        )

        print(
            "Normalization: uppercase + strip whitespace + remove '.'"
        )

        print(
            "Leading zeros: preserved"
        )

        print()
        print(
            topk[
                [
                    "rank",
                    "code",
                    "m3_train_n",
                ]
            ]
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------------------------------
        # 3G. MIMIC-III support table
        # ---------------------------------------------------------------------

        support = (
            topk[
                [
                    "rank",
                    "code",
                    "m3_train_n",
                ]
            ]
            .copy()
        )

        for split_name in [
            "train",
            "dev",
            "test",
        ]:

            split_diag = (
                diag3[
                    diag3["split"]
                    == split_name
                ]
            )

            counts = (
                split_diag[
                    split_diag[
                        "code"
                    ].isin(
                        top_codes
                    )
                ]
                .groupby(
                    "code"
                )[
                    "HADM_ID"
                ]
                .nunique()
            )

            n_split = len(
                splits[
                    split_name
                ]
            )

            col_n = (
                f"m3_{split_name}_n"
            )

            col_prev = (
                f"m3_{split_name}_prev"
            )

            if split_name == "train":

                support[
                    col_n
                ] = support[
                    "m3_train_n"
                ]

            else:

                support[
                    col_n
                ] = (
                    support[
                        "code"
                    ]
                    .map(
                        counts
                    )
                    .fillna(
                        0
                    )
                    .astype(
                        int
                    )
                )

            support[
                col_prev
            ] = (
                support[
                    col_n
                ]
                / n_split
            )

    finally:

        conn3.close()

    # =========================================================================
    # 4. MIMIC-IV PREFLIGHT
    # =========================================================================

    conn4 = sqlite3.connect(
        str(mimic4_db)
    )

    try:

        # ---------------------------------------------------------------------
        # 4A. Count target candidates:
        #     any native ICD-9
        #     mixed ICD versions
        #     final ICD-9-only
        # ---------------------------------------------------------------------

        m4_cohort_accounting = pd.read_sql_query(
            """
            WITH target_notes AS (

                SELECT DISTINCT
                    n.hadm_id,
                    a.subject_id

                FROM "note/discharge" n

                JOIN "hosp/admissions" a
                  ON a.hadm_id = n.hadm_id

                JOIN "hosp/patients" p
                  ON p.subject_id = a.subject_id

                WHERE n.text IS NOT NULL
                  AND n.hadm_id IS NOT NULL
                  AND p.anchor_year_group = ?
            ),

            icd_summary AS (

                SELECT
                    hadm_id,

                    MAX(
                        CASE
                            WHEN icd_version = 9
                            THEN 1
                            ELSE 0
                        END
                    ) AS has_icd9,

                    MAX(
                        CASE
                            WHEN icd_version IS NOT NULL
                             AND icd_version != 9
                            THEN 1
                            ELSE 0
                        END
                    ) AS has_other_icd

                FROM "hosp/diagnoses_icd"

                WHERE hadm_id IS NOT NULL

                GROUP BY hadm_id
            )

            SELECT

                COUNT(
                    DISTINCT CASE
                        WHEN i.has_icd9 = 1
                        THEN t.hadm_id
                    END
                )
                    AS any_native_icd9_n,

                COUNT(
                    DISTINCT CASE
                        WHEN i.has_icd9 = 1
                         AND i.has_other_icd = 1
                        THEN t.hadm_id
                    END
                )
                    AS mixed_icd_version_n,

                COUNT(
                    DISTINCT CASE
                        WHEN i.has_icd9 = 1
                         AND i.has_other_icd = 0
                        THEN t.hadm_id
                    END
                )
                    AS final_icd9_only_n,

                COUNT(
                    DISTINCT CASE
                        WHEN i.has_icd9 = 1
                         AND i.has_other_icd = 0
                        THEN t.subject_id
                    END
                )
                    AS final_icd9_only_patients

            FROM target_notes t

            JOIN icd_summary i
              ON i.hadm_id = t.hadm_id
            """,
            conn4,
            params=[
                args.m4_window,
            ],
        )

        if len(
            m4_cohort_accounting
        ) != 1:
            raise RuntimeError(
                "STOP: unexpected MIMIC-IV cohort accounting result."
            )

        row = (
            m4_cohort_accounting
            .iloc[
                0
            ]
        )

        n_m4_any_icd9 = int(
            row[
                "any_native_icd9_n"
            ]
            or 0
        )

        n_m4_mixed = int(
            row[
                "mixed_icd_version_n"
            ]
            or 0
        )

        n_m4 = int(
            row[
                "final_icd9_only_n"
            ]
            or 0
        )

        n_m4_patients = int(
            row[
                "final_icd9_only_patients"
            ]
            or 0
        )

        print()
        print("MIMIC-IV NATIVE ICD-9 TARGET")
        print_rule()

        print(
            f"anchor_year_group          : "
            f"{args.m4_window}"
        )

        print(
            "Interpretation              : "
            "patient-level anchor-year group"
        )

        print(
            "Note-date interpretation    : NO"
        )

        print()

        print(
            f"Any native ICD-9 admissions: "
            f"{n_m4_any_icd9:,}"
        )

        print(
            f"Mixed ICD-version excluded : "
            f"{n_m4_mixed:,}"
        )

        print(
            f"Final ICD-9-only admissions: "
            f"{n_m4:,}"
        )

        print(
            f"Final unique patients      : "
            f"{n_m4_patients:,}"
        )

        if (
            args.m4_window
            == "2014 - 2016"
            and
            n_m4_any_icd9
            != EXPECTED_M4_ANY_NATIVE_ICD9_2014_2016
        ):
            print()
            print(
                "WARNING: previous feasibility result for "
                "'any native ICD-9' was "
                f"{EXPECTED_M4_ANY_NATIVE_ICD9_2014_2016:,} admissions, "
                f"but this run returned {n_m4_any_icd9:,}."
            )

            print(
                "Investigate cohort definition/database version before embedding."
            )

        if n_m4 <= 0:
            raise RuntimeError(
                "STOP: final MIMIC-IV ICD-9-only target is empty."
            )

        # ---------------------------------------------------------------------
        # 4B. Build final M4 target manifest
        # ---------------------------------------------------------------------

        m4_target = pd.read_sql_query(
            """
            WITH target_notes AS (

                SELECT DISTINCT
                    n.hadm_id,
                    a.subject_id

                FROM "note/discharge" n

                JOIN "hosp/admissions" a
                  ON a.hadm_id = n.hadm_id

                JOIN "hosp/patients" p
                  ON p.subject_id = a.subject_id

                WHERE n.text IS NOT NULL
                  AND n.hadm_id IS NOT NULL
                  AND p.anchor_year_group = ?
            ),

            icd_summary AS (

                SELECT
                    hadm_id,

                    MAX(
                        CASE
                            WHEN icd_version = 9
                            THEN 1
                            ELSE 0
                        END
                    ) AS has_icd9,

                    MAX(
                        CASE
                            WHEN icd_version IS NOT NULL
                             AND icd_version != 9
                            THEN 1
                            ELSE 0
                        END
                    ) AS has_other_icd

                FROM "hosp/diagnoses_icd"

                WHERE hadm_id IS NOT NULL

                GROUP BY hadm_id
            )

            SELECT
                t.hadm_id,
                t.subject_id

            FROM target_notes t

            JOIN icd_summary i
              ON i.hadm_id = t.hadm_id

            WHERE i.has_icd9 = 1
              AND i.has_other_icd = 0

            ORDER BY
                t.hadm_id
            """,
            conn4,
            params=[
                args.m4_window,
            ],
        )

        if len(
            m4_target
        ) != n_m4:
            raise RuntimeError(
                "STOP: M4 target manifest count does not match "
                "the cohort-accounting query."
            )

        if m4_target[
            "hadm_id"
        ].duplicated().any():
            raise RuntimeError(
                "STOP: duplicate HADM_IDs in final M4 target manifest."
            )

        m4_target[
            "hadm_id"
        ] = (
            m4_target[
                "hadm_id"
            ]
            .astype(
                np.int64
            )
        )

        m4_target[
            "subject_id"
        ] = (
            m4_target[
                "subject_id"
            ]
            .astype(
                np.int64
            )
        )

        # ---------------------------------------------------------------------
        # 4C. Create temporary target-ID table
        # ---------------------------------------------------------------------

        conn4.execute(
            """
            CREATE TEMP TABLE expb_m4_target (
                hadm_id INTEGER PRIMARY KEY
            )
            """
        )

        conn4.executemany(
            """
            INSERT INTO expb_m4_target(
                hadm_id
            )
            VALUES (?)
            """,
            [
                (
                    int(hadm_id),
                )
                for hadm_id
                in m4_target[
                    "hadm_id"
                ].tolist()
            ],
        )

        # ---------------------------------------------------------------------
        # 4D. Fetch native ICD-9 diagnosis labels
        # ---------------------------------------------------------------------

        diag4 = pd.read_sql_query(
            """
            SELECT
                d.hadm_id,
                d.icd_code

            FROM "hosp/diagnoses_icd" d

            JOIN expb_m4_target t
              ON t.hadm_id = d.hadm_id

            WHERE d.icd_version = 9
              AND d.icd_code IS NOT NULL
            """,
            conn4,
        )

        diag4["code"] = (
            diag4[
                "icd_code"
            ]
            .map(
                normalize_icd9
            )
        )

        diag4 = (
            diag4
            .dropna(
                subset=[
                    "code",
                ]
            )
            .drop_duplicates(
                [
                    "hadm_id",
                    "code",
                ]
            )
        )

    finally:

        conn4.close()

    # =========================================================================
    # 5. M4 LABEL SUPPORT
    # =========================================================================

    m4_counts = (
        diag4[
            diag4[
                "code"
            ].isin(
                top_codes
            )
        ]
        .groupby(
            "code"
        )[
            "hadm_id"
        ]
        .nunique()
    )

    support[
        "m4_target_n"
    ] = (
        support[
            "code"
        ]
        .map(
            m4_counts
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    support[
        "m4_target_prev"
    ] = (
        support[
            "m4_target_n"
        ]
        / n_m4
    )

    # =========================================================================
    # 6. AUROC DEFINABILITY
    # =========================================================================

    n_train = len(
        train_ids
    )

    n_dev = len(
        dev_ids
    )

    n_test = len(
        test_ids
    )

    support[
        "train_auc_defined"
    ] = (
        (
            support[
                "m3_train_n"
            ] > 0
        )
        &
        (
            support[
                "m3_train_n"
            ] < n_train
        )
    )

    support[
        "dev_auc_defined"
    ] = (
        (
            support[
                "m3_dev_n"
            ] > 0
        )
        &
        (
            support[
                "m3_dev_n"
            ] < n_dev
        )
    )

    support[
        "test_auc_defined"
    ] = (
        (
            support[
                "m3_test_n"
            ] > 0
        )
        &
        (
            support[
                "m3_test_n"
            ] < n_test
        )
    )

    support[
        "m4_auc_defined"
    ] = (
        (
            support[
                "m4_target_n"
            ] > 0
        )
        &
        (
            support[
                "m4_target_n"
            ] < n_m4
        )
    )

    # These are descriptive sparsity flags only.
    # They do NOT automatically remove labels.
    support[
        "dev_lt_10_positive"
    ] = (
        support[
            "m3_dev_n"
        ]
        < 10
    )

    support[
        "test_lt_10_positive"
    ] = (
        support[
            "m3_test_n"
        ]
        < 10
    )

    support[
        "m4_lt_10_positive"
    ] = (
        support[
            "m4_target_n"
        ]
        < 10
    )

    # =========================================================================
    # 7. ALL-ZERO TOP-K LABEL VECTORS
    # =========================================================================

    top_code_set = set(
        top_codes
    )

    all_zero = {}

    for split_name in [
        "train",
        "dev",
        "test",
    ]:

        split_hadm_set = set(
            splits[
                split_name
            ]
        )

        has_any_topk = set(
            diag3.loc[
                (
                    diag3[
                        "split"
                    ]
                    == split_name
                )
                &
                (
                    diag3[
                        "code"
                    ]
                    .isin(
                        top_code_set
                    )
                ),
                "HADM_ID",
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        all_zero[
            split_name
        ] = len(
            split_hadm_set
            - has_any_topk
        )

    m4_hadm_set = set(
        m4_target[
            "hadm_id"
        ]
        .astype(
            np.int64
        )
        .tolist()
    )

    m4_has_any_topk = set(
        diag4.loc[
            diag4[
                "code"
            ].isin(
                top_code_set
            ),
            "hadm_id",
        ]
        .astype(
            np.int64
        )
        .tolist()
    )

    all_zero[
        "m4_target"
    ] = len(
        m4_hadm_set
        - m4_has_any_topk
    )

    # =========================================================================
    # 8. FINAL SUPPORT TABLE
    # =========================================================================

    display_cols = [
        "rank",
        "code",
        "m3_train_n",
        "m3_dev_n",
        "m3_test_n",
        "m4_target_n",
        "m4_target_prev",
        "dev_auc_defined",
        "test_auc_defined",
        "m4_auc_defined",
    ]

    print()
    print("=" * 80)
    print("FINAL TOP-K SUPPORT TABLE")
    print("=" * 80)

    print(
        support[
            display_cols
        ]
        .to_string(
            index=False,
            formatters={
                "m4_target_prev":
                    lambda value: f"{value:.6f}",
            },
        )
    )

    # =========================================================================
    # 9. ALL-ZERO VECTOR COUNTS
    # =========================================================================

    print()
    print("ALL-ZERO TOP-K LABEL VECTORS")
    print_rule()

    denominators = {
        "train": n_train,
        "dev": n_dev,
        "test": n_test,
        "m4_target": n_m4,
    }

    for key, value in all_zero.items():

        denominator = denominators[
            key
        ]

        print(
            f"{key:<12}: "
            f"{value:>6,}/{denominator:,} "
            f"({pct(value, denominator):.2f}%)"
        )

    # =========================================================================
    # 10. PREFLIGHT FLAGS
    # =========================================================================

    undefined_train = support[
        ~support[
            "train_auc_defined"
        ]
    ]

    undefined_dev = support[
        ~support[
            "dev_auc_defined"
        ]
    ]

    undefined_test = support[
        ~support[
            "test_auc_defined"
        ]
    ]

    undefined_m4 = support[
        ~support[
            "m4_auc_defined"
        ]
    ]

    sparse_dev = support[
        support[
            "dev_lt_10_positive"
        ]
    ]

    sparse_test = support[
        support[
            "test_lt_10_positive"
        ]
    ]

    sparse_m4 = support[
        support[
            "m4_lt_10_positive"
        ]
    ]

    print()
    print("=" * 80)
    print("PREFLIGHT FLAGS")
    print("=" * 80)

    print(
        f"Undefined TRAIN AUROC labels : "
        f"{len(undefined_train)}"
    )

    print(
        f"Undefined DEV AUROC labels   : "
        f"{len(undefined_dev)}"
    )

    print(
        f"Undefined TEST AUROC labels  : "
        f"{len(undefined_test)}"
    )

    print(
        f"Undefined M4 AUROC labels    : "
        f"{len(undefined_m4)}"
    )

    print()

    print(
        f"DEV labels with <10 positives : "
        f"{len(sparse_dev)}"
    )

    print(
        f"TEST labels with <10 positives: "
        f"{len(sparse_test)}"
    )

    print(
        f"M4 labels with <10 positives  : "
        f"{len(sparse_m4)}"
    )

    if len(
        undefined_train
    ):

        print()
        print(
            "Undefined TRAIN labels:"
        )

        print(
            undefined_train[
                [
                    "rank",
                    "code",
                    "m3_train_n",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if len(
        undefined_dev
    ):

        print()
        print(
            "Undefined DEV labels:"
        )

        print(
            undefined_dev[
                [
                    "rank",
                    "code",
                    "m3_dev_n",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if len(
        undefined_test
    ):

        print()
        print(
            "Undefined TEST labels:"
        )

        print(
            undefined_test[
                [
                    "rank",
                    "code",
                    "m3_test_n",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if len(
        undefined_m4
    ):

        print()
        print(
            "Undefined M4 labels:"
        )

        print(
            undefined_m4[
                [
                    "rank",
                    "code",
                    "m4_target_n",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if len(
        sparse_dev
    ):

        print()
        print(
            "DEV labels with <10 positives:"
        )

        print(
            sparse_dev[
                [
                    "rank",
                    "code",
                    "m3_dev_n",
                    "m3_dev_prev",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "m3_dev_prev":
                        lambda value: f"{value:.6f}",
                },
            )
        )

    if len(
        sparse_test
    ):

        print()
        print(
            "TEST labels with <10 positives:"
        )

        print(
            sparse_test[
                [
                    "rank",
                    "code",
                    "m3_test_n",
                    "m3_test_prev",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "m3_test_prev":
                        lambda value: f"{value:.6f}",
                },
            )
        )

    if len(
        sparse_m4
    ):

        print()
        print(
            "M4 labels with <10 positives:"
        )

        print(
            sparse_m4[
                [
                    "rank",
                    "code",
                    "m4_target_n",
                    "m4_target_prev",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "m4_target_prev":
                        lambda value: f"{value:.6f}",
                },
            )
        )

    # =========================================================================
    # 11. SAVE FROZEN ARTIFACTS
    # =========================================================================

    support_path = (
        out_dir
        / "experiment_b_top50_support.csv"
    )

    codes_path = (
        out_dir
        / "experiment_b_top50_codes.txt"
    )

    report_path = (
        out_dir
        / "experiment_b_cohort_report.json"
    )

    m4_manifest_path = (
        out_dir
        / "experiment_b_m4_icd9_only_manifest.csv"
    )

    m3_manifest_path = (
        out_dir
        / "experiment_b_m3_split_manifest.csv"
    )

    note_coverage_path = (
        out_dir
        / "experiment_b_m3_note_coverage.csv"
    )

    # -------------------------------------------------------------------------
    # Save support table
    # -------------------------------------------------------------------------

    support.to_csv(
        support_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save frozen top-K label list
    # -------------------------------------------------------------------------

    with open(
        codes_path,
        "w",
        encoding="utf-8",
    ) as f:

        for code in top_codes:
            f.write(
                f"{code}\n"
            )

    # -------------------------------------------------------------------------
    # Save M4 final target manifest
    # -------------------------------------------------------------------------

    m4_target.to_csv(
        m4_manifest_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save M3 split manifest
    # -------------------------------------------------------------------------

    m3_manifest_df = pd.DataFrame(
        split_rows,
        columns=[
            "split",
            "hadm_id",
        ],
    )

    m3_manifest_df.to_csv(
        m3_manifest_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save note-coverage report
    # -------------------------------------------------------------------------

    note_coverage.to_csv(
        note_coverage_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save JSON protocol record
    # -------------------------------------------------------------------------

    report = {
        "protocol_status": (
            "pre-outcome"
        ),

        "performance_examined": (
            False
        ),

        "mullenbach_manifest_directory": (
            str(
                mullenbach_dir
            )
        ),

        "mullenbach_manifest_files": {
            split_name: str(
                path
            )
            for split_name, path
            in manifest_paths.items()
        },

        "mimic3_db": (
            str(
                mimic3_db
            )
        ),

        "mimic4_db": (
            str(
                mimic4_db
            )
        ),

        "mimic3_document_construction": (
            "submitted frozen latest eligible discharge-summary row; "
            "ROW_NUMBER() OVER (PARTITION BY HADM_ID "
            "ORDER BY CHARTDATE DESC, ROW_ID DESC)"
        ),

        "report_preferred_used_in_experiment_b": (
            False
        ),

        "top_k": (
            int(
                args.k
            )
        ),

        "top_k_source": (
            "MIMIC-III training split only"
        ),

        "top_k_label_type": (
            "diagnosis-only ICD-9"
        ),

        "top_k_counting_unit": (
            "distinct HADM_ID per normalized diagnosis code"
        ),

        "icd9_normalization": (
            "uppercase; strip whitespace; remove decimal point; "
            "preserve leading zeros"
        ),

        "top_k_tie_break": (
            "admission frequency descending; normalized code ascending"
        ),

        "m4_cohort_definition": (
            "patient-level anchor_year_group"
        ),

        "m4_anchor_year_group": (
            args.m4_window
        ),

        "m4_calendar_time_interpretation": (
            False
        ),

        "m4_icd_policy": (
            "include admissions with >=1 ICD-9 diagnosis row and "
            "no diagnosis row from another ICD version"
        ),

        "m4_any_native_icd9_n": (
            int(
                n_m4_any_icd9
            )
        ),

        "m4_mixed_icd_version_excluded_n": (
            int(
                n_m4_mixed
            )
        ),

        "m4_final_icd9_only_n": (
            int(
                n_m4
            )
        ),

        "m4_final_icd9_only_patients": (
            int(
                n_m4_patients
            )
        ),

        "splits": {
            split_name: {
                "n_hadm": int(
                    len(
                        splits[
                            split_name
                        ]
                    )
                ),

                "n_subject": int(
                    len(
                        set(
                            subjects[
                                split_name
                            ]
                        )
                    )
                ),
            }
            for split_name
            in [
                "train",
                "dev",
                "test",
            ]
        },

        "hadm_overlap": {
            key: int(
                value
            )
            for key, value
            in hadm_overlap.items()
        },

        "subject_overlap": {
            key: int(
                value
            )
            for key, value
            in subject_overlap.items()
        },

        "mimic3_latest_row_addendum_first": {
            str(row["split"]): int(
                row[
                    "addendum_first_n"
                ]
            )
            for _, row
            in note_coverage.iterrows()
        },

        "all_zero_topk": {
            key: int(
                value
            )
            for key, value
            in all_zero.items()
        },

        "undefined_auc_labels": {
            "train": int(
                len(
                    undefined_train
                )
            ),

            "dev": int(
                len(
                    undefined_dev
                )
            ),

            "test": int(
                len(
                    undefined_test
                )
            ),

            "m4_target": int(
                len(
                    undefined_m4
                )
            ),
        },

        "lt_10_positive_labels": {
            "dev": int(
                len(
                    sparse_dev
                )
            ),

            "test": int(
                len(
                    sparse_test
                )
            ),

            "m4_target": int(
                len(
                    sparse_m4
                )
            ),
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # =========================================================================
    # 12. PRINT SAVED ARTIFACTS
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVED FROZEN PREFLIGHT ARTIFACTS")
    print("=" * 80)

    print(
        support_path
    )

    print(
        codes_path
    )

    print(
        report_path
    )

    print(
        m3_manifest_path
    )

    print(
        m4_manifest_path
    )

    print(
        note_coverage_path
    )

    # =========================================================================
    # 13. FINAL GO / NO-GO
    # =========================================================================

    hard_fail = (
        any(
            value != 0
            for value
            in hadm_overlap.values()
        )
        or
        any(
            value != 0
            for value
            in subject_overlap.values()
        )
        or
        len(
            undefined_train
        ) > 0
        or
        len(
            undefined_dev
        ) > 0
        or
        len(
            undefined_test
        ) > 0
        or
        len(
            undefined_m4
        ) > 0
    )

    print()
    print("=" * 80)

    if hard_fail:

        print(
            "RESULT: REVIEW REQUIRED BEFORE EMBEDDING."
        )

        print(
            "Do not generate Experiment B embeddings until the "
            "flagged issue is resolved."
        )

    else:

        print(
            "RESULT: HARD PREFLIGHT PASSED."
        )

        print(
            "The MIMIC-III splits, MIMIC-IV target cohort, "
            "and top-K diagnosis label space can now be frozen "
            "before any performance analysis."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
