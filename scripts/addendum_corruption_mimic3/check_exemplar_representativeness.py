#!/usr/bin/env python3

"""
check_exemplar_representativeness.py

Compare the three frozen MIMIC-III Judge exemplars against the exact
5,000-note MIMIC-III baseline used for embeddings.

Measures:
  - character length
  - recognized section count
  - whitespace token count
  - Hospital Course-family section presence
  - empirical percentile for each metric

Does NOT modify any files/database.
"""

import os
import re
import sqlite3

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")

BASELINE_IDS_FILE = "data/embeddings/ids_mimic3_5000.npy"

# Use your actual final Judge manifest.
# This can be the no-text copy because we only need exemplar HADM_IDs.
JUDGE_SAMPLES_FILE = "data/judge_samples_300.csv"


if not MIMIC3_DB_PATH:
    raise RuntimeError("MIMIC3_DB_PATH is not set.")

if not os.path.exists(MIMIC3_DB_PATH):
    raise FileNotFoundError(
        f"MIMIC-III DB not found: {MIMIC3_DB_PATH}"
    )

if not os.path.exists(BASELINE_IDS_FILE):
    raise FileNotFoundError(
        f"Baseline IDs file not found: {BASELINE_IDS_FILE}"
    )

if not os.path.exists(JUDGE_SAMPLES_FILE):
    raise FileNotFoundError(
        f"Judge samples file not found: {JUDGE_SAMPLES_FILE}"
    )


# ---------------------------------------------------------------------
# Exact section vocabulary from surface_features.py
# ---------------------------------------------------------------------

SECTION_HEADERS = [
    "assessment",
    "plan",
    "assessment and plan",
    "hospital course",
    "hospital course by",
    "summary of hospital course",
    "history of present illness",
    "hpi",
    "discharge medications",
    "medications on discharge",
    "medications at discharge",
    "discharge instructions",
    "pertinent results",
    "pertinent labs",
    "pertinent studies",
    "past medical history",
    "pmh",
    "chief complaint",
    "cc",
]

SECTION_RE = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(h) for h in SECTION_HEADERS)
    + r")\s*:?",
    re.MULTILINE | re.IGNORECASE,
)

HOSPITAL_COURSE_RE = re.compile(
    r"^\s*(?:"
    r"brief\s+hospital\s+course"
    r"|hospital\s+course"
    r"|hospital\s+course\s+by.*"
    r"|summary\s+of\s+hospital\s+course"
    r")\s*:?",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_header_match(x):
    """Normalize matched section header before unique counting."""
    return re.sub(r"\s+", " ", x.strip().lower().rstrip(":"))


def features(text):
    text = str(text)

    matches = SECTION_RE.findall(text)
    unique_sections = {
        normalize_header_match(x)
        for x in matches
    }

    return {
        "char_length": len(text),
        "whitespace_tokens": len(text.split()),
        "n_sections": len(unique_sections),
        "has_hospital_course": bool(
            HOSPITAL_COURSE_RE.search(text)
        ),
    }


def empirical_percentile(values, value):
    """
    Percentage of baseline observations <= exemplar value.

    Example:
      25 means exemplar is around the 25th percentile.
      90 means exemplar is longer/more structured than ~90% of baseline.
    """
    values = np.asarray(values)
    return 100.0 * np.mean(values <= value)


# ---------------------------------------------------------------------
# Load baseline IDs
# ---------------------------------------------------------------------

baseline_ids = np.load(BASELINE_IDS_FILE).astype(int)

assert len(baseline_ids) == 5000, (
    f"Expected exactly 5,000 baseline IDs, got {len(baseline_ids)}"
)

assert len(np.unique(baseline_ids)) == 5000, (
    "Baseline HADM_IDs are not unique."
)


# ---------------------------------------------------------------------
# Get frozen exemplar IDs from Judge manifest
# ---------------------------------------------------------------------

judge = pd.read_csv(JUDGE_SAMPLES_FILE)

required = {"hadm_id", "selection_group"}
missing = required - set(judge.columns)

if missing:
    raise ValueError(
        f"Judge manifest missing columns: {sorted(missing)}"
    )

exemplar_df = judge[
    judge["selection_group"].astype(str).str.strip().str.lower()
    == "exemplar"
].copy()

exemplar_ids = (
    exemplar_df["hadm_id"]
    .astype(int)
    .drop_duplicates()
    .tolist()
)

print("=" * 72)
print("EXEMPLAR REPRESENTATIVENESS CHECK")
print("=" * 72)

print(f"Baseline IDs : {len(baseline_ids):,}")
print(f"Exemplar IDs : {len(exemplar_ids):,}")
print(f"Exemplars    : {exemplar_ids}")

if len(exemplar_ids) != 3:
    print(
        f"WARNING: expected 3 unique exemplars, "
        f"found {len(exemplar_ids)}."
    )


# All exemplars should originate from the 5,000 baseline.
not_in_baseline = sorted(
    set(exemplar_ids) - set(baseline_ids.tolist())
)

if not_in_baseline:
    raise RuntimeError(
        "Some exemplar IDs are not in the 5,000 baseline: "
        f"{not_in_baseline}"
    )


# ---------------------------------------------------------------------
# Query exact frozen MIMIC-III latest-row texts
#
# This duplicates the rule used by embed_and_save.py:
#
# CATEGORY='Discharge summary'
# ISERROR valid
# TEXT/HADM_ID non-null
# latest row by CHARTDATE DESC, ROW_ID DESC
# ---------------------------------------------------------------------

all_needed_ids = sorted(
    set(baseline_ids.tolist()) | set(exemplar_ids)
)

placeholders = ",".join("?" * len(all_needed_ids))

query = f"""
WITH ranked AS (
    SELECT
        HADM_ID AS hadm_id,
        TEXT AS text,
        DESCRIPTION AS description,
        CHARTDATE AS chartdate,
        ROW_ID AS row_id,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS rn

    FROM NOTEEVENTS

    WHERE CATEGORY = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND HADM_ID IS NOT NULL
      AND TEXT IS NOT NULL
      AND HADM_ID IN ({placeholders})
)

SELECT
    hadm_id,
    text,
    description,
    chartdate,
    row_id

FROM ranked

WHERE rn = 1
"""


conn = sqlite3.connect(MIMIC3_DB_PATH)

try:
    notes = pd.read_sql_query(
        query,
        conn,
        params=all_needed_ids,
    )
finally:
    conn.close()


notes["hadm_id"] = notes["hadm_id"].astype(int)

print()
print("Retrieved frozen texts:", len(notes))


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

returned = set(notes["hadm_id"])
baseline_set = set(baseline_ids.tolist())

missing_baseline = sorted(baseline_set - returned)

if missing_baseline:
    raise RuntimeError(
        f"Missing {len(missing_baseline)} baseline texts. "
        f"First IDs: {missing_baseline[:20]}"
    )

if notes["hadm_id"].duplicated().any():
    raise RuntimeError(
        "Query unexpectedly returned duplicate HADM_IDs."
    )


# ---------------------------------------------------------------------
# Compute features
# ---------------------------------------------------------------------

feature_rows = []

for _, row in notes.iterrows():

    f = features(row["text"])

    feature_rows.append({
        "hadm_id": int(row["hadm_id"]),
        "description": row["description"],
        **f,
    })


feat = pd.DataFrame(feature_rows)

baseline = feat[
    feat["hadm_id"].isin(baseline_set)
].copy()

exemplars = feat[
    feat["hadm_id"].isin(exemplar_ids)
].copy()


assert len(baseline) == 5000
assert len(exemplars) == len(exemplar_ids)


# Preserve original exemplar ordering from manifest
order = {
    hadm_id: i
    for i, hadm_id in enumerate(exemplar_ids)
}

exemplars["exemplar_number"] = (
    exemplars["hadm_id"].map(order) + 1
)

exemplars = exemplars.sort_values(
    "exemplar_number"
)


# ---------------------------------------------------------------------
# Baseline distribution
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("BASELINE DISTRIBUTION")
print("=" * 72)

for metric in [
    "char_length",
    "whitespace_tokens",
    "n_sections",
]:

    vals = baseline[metric]

    print()
    print(metric)
    print("-" * 40)
    print(f"mean   : {vals.mean():,.2f}")
    print(f"median : {vals.median():,.2f}")
    print(f"p10    : {vals.quantile(0.10):,.2f}")
    print(f"p25    : {vals.quantile(0.25):,.2f}")
    print(f"p75    : {vals.quantile(0.75):,.2f}")
    print(f"p90    : {vals.quantile(0.90):,.2f}")


hc_n = int(
    baseline["has_hospital_course"].sum()
)

print()
print("Hospital Course-family section")
print("-" * 40)
print(
    f"Present: {hc_n:,} / {len(baseline):,} "
    f"({100 * hc_n / len(baseline):.2f}%)"
)


# ---------------------------------------------------------------------
# Exemplar percentiles
# ---------------------------------------------------------------------

output_rows = []

print()
print("=" * 72)
print("EXEMPLARS")
print("=" * 72)


for _, row in exemplars.iterrows():

    length_pct = empirical_percentile(
        baseline["char_length"],
        row["char_length"],
    )

    token_pct = empirical_percentile(
        baseline["whitespace_tokens"],
        row["whitespace_tokens"],
    )

    section_pct = empirical_percentile(
        baseline["n_sections"],
        row["n_sections"],
    )

    print()
    print(
        f"Exemplar {int(row['exemplar_number'])} "
        f"| HADM_ID={int(row['hadm_id'])}"
    )
    print("-" * 72)

    print(
        f"DESCRIPTION              : {row['description']}"
    )

    print(
        f"Character length         : "
        f"{int(row['char_length']):,}"
    )

    print(
        f"Length percentile        : "
        f"{length_pct:.1f}%"
    )

    print(
        f"Whitespace tokens        : "
        f"{int(row['whitespace_tokens']):,}"
    )

    print(
        f"Token-count percentile   : "
        f"{token_pct:.1f}%"
    )

    print(
        f"Recognized sections      : "
        f"{int(row['n_sections'])}"
    )

    print(
        f"Section-count percentile : "
        f"{section_pct:.1f}%"
    )

    print(
        f"Hospital Course present  : "
        f"{bool(row['has_hospital_course'])}"
    )

    output_rows.append({
        "exemplar_number": int(
            row["exemplar_number"]
        ),
        "hadm_id": int(row["hadm_id"]),
        "description": row["description"],
        "char_length": int(
            row["char_length"]
        ),
        "length_percentile": length_pct,
        "whitespace_tokens": int(
            row["whitespace_tokens"]
        ),
        "token_percentile": token_pct,
        "n_sections": int(
            row["n_sections"]
        ),
        "section_percentile": section_pct,
        "has_hospital_course": bool(
            row["has_hospital_course"]
        ),
    })


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

out = pd.DataFrame(output_rows)

out_path = (
    "data/addendum_sensitivity/mimic3_exemplar_representativeness.csv"
)

out.to_csv(out_path, index=False)

print()
print("=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

print(
    out.to_string(
        index=False,
        formatters={
            "length_percentile":
                lambda x: f"{x:.1f}",
            "token_percentile":
                lambda x: f"{x:.1f}",
            "section_percentile":
                lambda x: f"{x:.1f}",
        }
    )
)

print()
print("Saved:")
print(f"  {out_path}")