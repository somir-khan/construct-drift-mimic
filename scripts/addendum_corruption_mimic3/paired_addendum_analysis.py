#!/usr/bin/env python3
"""
paired_addendum_analysis.py

Characterize the 331 MIMIC-III baseline admissions for which the frozen
latest-row rule selected DESCRIPTION='Addendum'.

Input:
    data/addendum_sensitivity/mimic3_baseline_addendum_overlap.csv

Expected input column:
    HADM_ID

Outputs:
    data/addendum_sensitivity/mimic3_baseline_addendum_paired_lengths.csv
    data/addendum_sensitivity/mimic3_baseline_addendum_manual_review_ids.csv

This script does NOT modify the database.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")

OVERLAP_CSV = "data/addendum_sensitivity/mimic3_baseline_addendum_overlap.csv"

OUTPUT_CSV = "data/addendum_sensitivity/mimic3_baseline_addendum_paired_lengths.csv"
REVIEW_IDS_CSV = "data/addendum_sensitivity/mimic3_baseline_addendum_manual_review_ids.csv"


if not MIMIC3_DB_PATH:
    raise RuntimeError(
        "MIMIC3_DB_PATH is not set. "
        "Set it in .env or export it in the shell."
    )

if not os.path.exists(MIMIC3_DB_PATH):
    raise FileNotFoundError(f"MIMIC-III database not found: {MIMIC3_DB_PATH}")

if not os.path.exists(OVERLAP_CSV):
    raise FileNotFoundError(f"Overlap CSV not found: {OVERLAP_CSV}")


# ---------------------------------------------------------------------
# Load the confirmed baseline-overlap IDs
# ---------------------------------------------------------------------

ids_df = pd.read_csv(OVERLAP_CSV)

if "HADM_ID" not in ids_df.columns:
    raise ValueError(
        f"{OVERLAP_CSV} must contain a HADM_ID column. "
        f"Found: {ids_df.columns.tolist()}"
    )

ids_df["HADM_ID"] = ids_df["HADM_ID"].astype(int)

overlap_ids = sorted(ids_df["HADM_ID"].unique().tolist())

print("=" * 72)
print("MIMIC-III BASELINE ADDENDUM-FIRST PAIRED ANALYSIS")
print("=" * 72)
print(f"Overlap IDs loaded: {len(overlap_ids):,}")

if len(overlap_ids) != 331:
    print(
        f"WARNING: expected 331 IDs from the previous overlap analysis, "
        f"but found {len(overlap_ids):,}."
    )


# ---------------------------------------------------------------------
# Query
#
# Important:
# 1. eligibility filters exactly match frozen MIMIC-III embedder
# 2. rn_all reproduces the original latest-row selection rule
# 3. reports CTE independently ranks Reports, so we select the most
#    recent Report belonging to the SAME HADM_ID
# ---------------------------------------------------------------------

placeholders = ",".join(["?"] * len(overlap_ids))

query = f"""
WITH eligible AS (
    SELECT
        HADM_ID,
        DESCRIPTION,
        TEXT,
        LENGTH(TEXT) AS text_len,
        CHARTDATE,
        ROW_ID,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS rn_all

    FROM NOTEEVENTS

    WHERE CATEGORY = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND HADM_ID IS NOT NULL
      AND TEXT IS NOT NULL
      AND HADM_ID IN ({placeholders})
),

selected_addenda AS (
    SELECT
        HADM_ID,
        text_len AS addendum_len,
        CHARTDATE AS addendum_chartdate,
        ROW_ID AS addendum_row_id

    FROM eligible

    WHERE rn_all = 1
      AND DESCRIPTION = 'Addendum'
),

reports_ranked AS (
    SELECT
        HADM_ID,
        text_len AS report_len,
        CHARTDATE AS report_chartdate,
        ROW_ID AS report_row_id,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS report_rn

    FROM eligible

    WHERE DESCRIPTION = 'Report'
),

row_counts AS (
    SELECT
        HADM_ID,
        COUNT(*) AS n_discharge_rows,
        SUM(CASE WHEN DESCRIPTION = 'Report' THEN 1 ELSE 0 END) AS n_reports,
        SUM(CASE WHEN DESCRIPTION = 'Addendum' THEN 1 ELSE 0 END) AS n_addenda

    FROM eligible

    GROUP BY HADM_ID
)

SELECT
    a.HADM_ID,
    a.addendum_len,

    r.report_len,

    CASE
        WHEN r.report_len IS NOT NULL
         AND r.report_len > 0
        THEN CAST(a.addendum_len AS REAL) / r.report_len
        ELSE NULL
    END AS length_ratio,

    c.n_discharge_rows,
    c.n_reports,
    c.n_addenda,

    a.addendum_chartdate,
    r.report_chartdate,

    a.addendum_row_id,
    r.report_row_id

FROM selected_addenda a

LEFT JOIN reports_ranked r
    ON a.HADM_ID = r.HADM_ID
   AND r.report_rn = 1

LEFT JOIN row_counts c
    ON a.HADM_ID = c.HADM_ID

ORDER BY a.HADM_ID;
"""


# ---------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------

conn = sqlite3.connect(MIMIC3_DB_PATH)

try:
    paired = pd.read_sql_query(
        query,
        conn,
        params=overlap_ids,
    )
finally:
    conn.close()


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

if paired["HADM_ID"].duplicated().any():
    dupes = paired.loc[
        paired["HADM_ID"].duplicated(keep=False),
        "HADM_ID",
    ].tolist()

    raise RuntimeError(
        "Paired query unexpectedly returned duplicate HADM_IDs: "
        f"{dupes[:20]}"
    )


queried_ids = set(paired["HADM_ID"].astype(int))
expected_ids = set(overlap_ids)

missing = sorted(expected_ids - queried_ids)
unexpected = sorted(queried_ids - expected_ids)

print()
print("Sanity checks")
print("-" * 72)
print(f"Expected Addendum-first IDs : {len(expected_ids):,}")
print(f"Returned Addendum-first IDs : {len(queried_ids):,}")
print(f"Missing IDs                 : {len(missing):,}")
print(f"Unexpected IDs              : {len(unexpected):,}")

if missing:
    print(f"First missing IDs: {missing[:20]}")

if unexpected:
    print(f"First unexpected IDs: {unexpected[:20]}")


# For your current confirmed overlap, this should be exactly 331.
assert len(paired) == len(overlap_ids), (
    f"Expected {len(overlap_ids)} paired rows but query returned {len(paired)}."
)


# ---------------------------------------------------------------------
# Derived indicators
# ---------------------------------------------------------------------

paired["has_report"] = paired["report_len"].notna()

paired_with_report = paired[paired["has_report"]].copy()

paired["ratio_lt_025"] = paired["length_ratio"] < 0.25
paired["ratio_lt_050"] = paired["length_ratio"] < 0.50
paired["ratio_ge_080"] = paired["length_ratio"] >= 0.80


# ---------------------------------------------------------------------
# Requested summary
# ---------------------------------------------------------------------

n_total = len(paired)

n_has_report = int(paired["has_report"].sum())
n_addendum_only = int((~paired["has_report"]).sum())

median_addendum = paired["addendum_len"].median()

median_report = (
    paired_with_report["report_len"].median()
    if n_has_report
    else np.nan
)

median_ratio = (
    paired_with_report["length_ratio"].median()
    if n_has_report
    else np.nan
)

mean_addendum = paired["addendum_len"].mean()

mean_report = (
    paired_with_report["report_len"].mean()
    if n_has_report
    else np.nan
)

mean_ratio = (
    paired_with_report["length_ratio"].mean()
    if n_has_report
    else np.nan
)

ratio_lt_025 = int(
    (paired_with_report["length_ratio"] < 0.25).sum()
)

ratio_lt_050 = int(
    (paired_with_report["length_ratio"] < 0.50).sum()
)

ratio_ge_080 = int(
    (paired_with_report["length_ratio"] >= 0.80).sum()
)


def pct(n, denominator):
    if denominator == 0:
        return float("nan")
    return 100.0 * n / denominator


print()
print("=" * 72)
print("PAIRED RESULTS")
print("=" * 72)

print(f"addendum_first_baseline : {n_total:,}")
print(
    f"also_has_report         : {n_has_report:,} "
    f"({pct(n_has_report, n_total):.2f}%)"
)
print(
    f"addendum_only           : {n_addendum_only:,} "
    f"({pct(n_addendum_only, n_total):.2f}%)"
)

print()
print("Lengths")
print("-" * 72)

print(f"median Addendum length  : {median_addendum:,.1f}")
print(f"median Report length    : {median_report:,.1f}")
print(f"median length ratio     : {median_ratio:.4f}")

# Means included as secondary diagnostics only.
print()
print(f"mean Addendum length    : {mean_addendum:,.1f}")
print(f"mean Report length      : {mean_report:,.1f}")
print(f"mean length ratio       : {mean_ratio:.4f}")

print()
print("Addendum / Report length-ratio distribution")
print("-" * 72)

print(
    f"ratio < 0.25            : {ratio_lt_025:,} "
    f"({pct(ratio_lt_025, n_has_report):.2f}% of paired)"
)

print(
    f"ratio < 0.50            : {ratio_lt_050:,} "
    f"({pct(ratio_lt_050, n_has_report):.2f}% of paired)"
)

print(
    f"ratio >= 0.80           : {ratio_ge_080:,} "
    f"({pct(ratio_ge_080, n_has_report):.2f}% of paired)"
)


# ---------------------------------------------------------------------
# Extra distribution information — useful because ratio is skewed
# ---------------------------------------------------------------------

if n_has_report:
    ratios = paired_with_report["length_ratio"]

    print()
    print("Length-ratio quantiles")
    print("-" * 72)

    for q in [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]:
        print(
            f"{q:>5.0%} percentile          : "
            f"{ratios.quantile(q):.4f}"
        )


# ---------------------------------------------------------------------
# Save detailed paired table
# ---------------------------------------------------------------------

paired = paired.sort_values(
    ["length_ratio", "HADM_ID"],
    na_position="last",
)

paired.to_csv(OUTPUT_CSV, index=False)

print()
print(f"Detailed paired CSV saved:")
print(f"  {OUTPUT_CSV}")


# ---------------------------------------------------------------------
# Generate HADM_IDs for later manual review.
#
# We deliberately save ONLY IDs / lengths here, not clinical text.
#
# Groups:
#   - 5 lowest ratios
#   - 5 closest to median ratio
#   - up to 10 highest ratios / >= 0.80
# ---------------------------------------------------------------------

if n_has_report:

    tmp = paired_with_report.copy()

    # Lowest ratios
    low = (
        tmp
        .sort_values(["length_ratio", "HADM_ID"])
        .head(5)
        .copy()
    )
    low["review_group"] = "lowest_ratio"

    # Closest to median
    tmp["distance_from_median"] = (
        tmp["length_ratio"] - median_ratio
    ).abs()

    middle = (
        tmp
        .sort_values(["distance_from_median", "HADM_ID"])
        .head(5)
        .copy()
    )
    middle["review_group"] = "near_median"

    # Highest ratios; prioritize >=0.80 if available
    high_pool = tmp[tmp["length_ratio"] >= 0.80].copy()

    if len(high_pool) == 0:
        high_pool = tmp.copy()

    high = (
        high_pool
        .sort_values(
            ["length_ratio", "HADM_ID"],
            ascending=[False, True],
        )
        .head(10)
        .copy()
    )
    high["review_group"] = "highest_ratio"

    review = pd.concat(
        [low, middle, high],
        ignore_index=True,
    )

    review = review.drop_duplicates(
        subset=["HADM_ID", "review_group"]
    )

    review = review[
        [
            "review_group",
            "HADM_ID",
            "addendum_len",
            "report_len",
            "length_ratio",
            "n_reports",
            "n_addenda",
        ]
    ]

    review.to_csv(REVIEW_IDS_CSV, index=False)

    print()
    print("Manual-review ID list saved:")
    print(f"  {REVIEW_IDS_CSV}")


print()
print("=" * 72)
print("DONE")
print("=" * 72)