#!/usr/bin/env python3

"""
embed_addendum_report_sensitivity.py

Sensitivity analysis only.

For the 331 MIMIC-III baseline admissions where the frozen latest-row
rule selected an Addendum:

  - find the most recent eligible DESCRIPTION='Report'
  - embed the 328 admissions that have a Report
  - leave the 3 Addendum-only admissions for later unchanged
  - save Report embeddings + aligned HADM_IDs

IMPORTANT:
  - CPU only
  - does NOT overwrite original 5000 embeddings
  - uses the exact frozen embed_single_note implementation
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Import exact frozen embedding implementation
from embed_and_save import normalize_phi, embed_single_note


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")

OVERLAP_CSV = "data/addendum_sensitivity/mimic3_baseline_addendum_overlap.csv"

OUT_EMBEDDINGS = (
    "data/addendum_sensitivity/"
    "mimic3_report_sensitivity_embeddings.npy"
)

OUT_IDS = (
    "data/addendum_sensitivity/"
    "mimic3_report_sensitivity_ids.npy"
)

OUT_MANIFEST = (
    "data/addendum_sensitivity/"
    "mimic3_report_sensitivity_manifest.csv"
)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
STRIDE = 256


if not MIMIC3_DB_PATH:
    raise RuntimeError("MIMIC3_DB_PATH is not set.")

if not os.path.exists(MIMIC3_DB_PATH):
    raise FileNotFoundError(
        f"MIMIC-III DB not found: {MIMIC3_DB_PATH}"
    )

if not os.path.exists(OVERLAP_CSV):
    raise FileNotFoundError(
        f"Overlap CSV not found: {OVERLAP_CSV}"
    )


# ---------------------------------------------------------------------
# Load confirmed 331 overlap IDs
# ---------------------------------------------------------------------

df_ids = pd.read_csv(OVERLAP_CSV)

if "HADM_ID" not in df_ids.columns:
    raise ValueError(
        f"Expected HADM_ID column in {OVERLAP_CSV}"
    )

overlap_ids = (
    df_ids["HADM_ID"]
    .astype(int)
    .drop_duplicates()
    .tolist()
)

print("=" * 72)
print("MIMIC-III REPORT-PREFERRED SENSITIVITY EMBEDDING")
print("=" * 72)

print(f"Confirmed overlap IDs: {len(overlap_ids):,}")

assert len(overlap_ids) == 331, (
    f"Expected 331 overlap IDs, found {len(overlap_ids)}"
)


# ---------------------------------------------------------------------
# Query most recent eligible Report within each HADM_ID
#
# Eligibility matches frozen pipeline:
# CATEGORY='Discharge summary'
# valid ISERROR
# HADM_ID non-null
# TEXT non-null
#
# IMPORTANT:
# We rank REPORTS only here.
# This is the sensitivity construction, not the submitted primary rule.
# ---------------------------------------------------------------------

placeholders = ",".join(["?"] * len(overlap_ids))

query = f"""
WITH reports AS (

    SELECT
        HADM_ID AS hadm_id,
        TEXT AS text,
        DESCRIPTION AS description,
        CHARTDATE AS chartdate,
        ROW_ID AS row_id,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS report_rn

    FROM NOTEEVENTS

    WHERE CATEGORY = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND HADM_ID IS NOT NULL
      AND TEXT IS NOT NULL
      AND DESCRIPTION = 'Report'
      AND HADM_ID IN ({placeholders})
)

SELECT
    hadm_id,
    text,
    description,
    chartdate,
    row_id

FROM reports

WHERE report_rn = 1

ORDER BY hadm_id
"""


conn = sqlite3.connect(MIMIC3_DB_PATH)

try:
    reports = pd.read_sql_query(
        query,
        conn,
        params=overlap_ids,
    )
finally:
    conn.close()


reports["hadm_id"] = reports["hadm_id"].astype(int)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

print()
print("Report retrieval")
print("-" * 72)

print(f"Overlap admissions : {len(overlap_ids):,}")
print(f"Reports retrieved  : {len(reports):,}")
print(
    f"Addendum-only      : "
    f"{len(overlap_ids) - len(reports):,}"
)

assert not reports["hadm_id"].duplicated().any()

assert len(reports) == 328, (
    f"Expected 328 paired Reports, got {len(reports)}"
)

report_ids = set(reports["hadm_id"])

addendum_only = sorted(
    set(overlap_ids) - report_ids
)

print()
print("Addendum-only IDs:")
print(addendum_only)

assert len(addendum_only) == 3


# ---------------------------------------------------------------------
# Normalize text exactly as frozen pipeline
# ---------------------------------------------------------------------

reports["text_normalized"] = (
    reports["text"]
    .astype(str)
    .map(normalize_phi)
)


# ---------------------------------------------------------------------
# CPU setup only
# ---------------------------------------------------------------------

device = torch.device("cpu")

n_threads = os.cpu_count() or 1

torch.set_num_threads(n_threads)

try:
    torch.set_num_interop_threads(
        max(1, n_threads // 2)
    )
except RuntimeError:
    pass

print()
print(f"Device       : {device}")
print(f"CPU threads  : {n_threads}")
print(f"Model        : {MODEL_NAME}")
print(f"Stride       : {STRIDE}")


# ---------------------------------------------------------------------
# Load exact frozen model/tokenizer
# ---------------------------------------------------------------------

hf_token = os.getenv("HF_TOKEN")

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=hf_token,
)

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    token=hf_token,
).to(device)

model.eval()

for param in model.parameters():
    param.requires_grad = False


# ---------------------------------------------------------------------
# Embed 328 Reports
# ---------------------------------------------------------------------

embeddings = np.zeros(
    (len(reports), 768),
    dtype=np.float32,
)

chunk_counts = []

print()
print("Embedding Reports...")

for i, text in enumerate(
    tqdm(
        reports["text_normalized"].tolist(),
        desc="Reports",
        unit="note",
    )
):

    emb, n_chunks = embed_single_note(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        stride=STRIDE,
    )

    embeddings[i] = emb
    chunk_counts.append(n_chunks)


# ---------------------------------------------------------------------
# Validate output
# ---------------------------------------------------------------------

if not np.isfinite(embeddings).all():
    raise RuntimeError(
        "Non-finite values detected in embeddings."
    )

zero_rows = np.where(
    np.all(embeddings == 0, axis=1)
)[0]

print()
print("Embedding checks")
print("-" * 72)

print(f"Shape             : {embeddings.shape}")
print(f"Zero-vector rows  : {len(zero_rows)}")
print(
    f"Mean chunks/note  : "
    f"{np.mean(chunk_counts):.2f}"
)
print(
    f"Median chunks     : "
    f"{np.median(chunk_counts):.1f}"
)

if len(zero_rows):
    raise RuntimeError(
        f"Found {len(zero_rows)} zero-vector embeddings."
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

report_ids_array = (
    reports["hadm_id"]
    .to_numpy(dtype=np.int64)
)

np.save(
    OUT_EMBEDDINGS,
    embeddings,
)

np.save(
    OUT_IDS,
    report_ids_array,
)

manifest = reports[
    [
        "hadm_id",
        "description",
        "chartdate",
        "row_id",
    ]
].copy()

manifest["char_length"] = (
    reports["text_normalized"]
    .str.len()
)

manifest["n_chunks"] = chunk_counts

manifest.to_csv(
    OUT_MANIFEST,
    index=False,
)


# ---------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------

reload_emb = np.load(OUT_EMBEDDINGS)
reload_ids = np.load(OUT_IDS)

assert reload_emb.shape == (328, 768)
assert reload_ids.shape == (328,)
assert len(np.unique(reload_ids)) == 328


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(OUT_EMBEDDINGS)
print(OUT_IDS)
print(OUT_MANIFEST)

print()
print("PASS: 328 Report embeddings generated.")