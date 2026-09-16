#!/usr/bin/env python3
"""Extract the exact discharge-note rows used for DriftLens prototypes."""

import argparse
import re
import sqlite3

import pandas as pd


def normalize_phi(text):
    text = re.sub(r"\[\*\*.*?\*\*\]", "[PHI]", text)
    return re.sub(r"___", "unknown", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prototypes", required=True)
    p.add_argument("--mimic3-db", required=True)
    p.add_argument("--mimic4-db", required=True)
    p.add_argument(
        "--output",
        required=True,
        help="Restricted local output CSV; do not commit or distribute it.",
    )
    a = p.parse_args()

    prototypes = pd.read_csv(a.prototypes)
    prototypes["hadm_id"] = pd.to_numeric(prototypes["hadm_id"]).astype("int64")
    label_column = next(
        (name for name in ("cohort", "corpus", "dataset") if name in prototypes.columns),
        None,
    )
    if label_column is None:
        raise ValueError(f"No cohort column found. CSV columns: {list(prototypes.columns)}")
    corpus = prototypes[label_column].astype(str).str.lower()
    is_m3 = corpus.str.contains("baseline|mimic.?iii|mimic3")
    is_m4 = corpus.str.contains("target|mimic.?iv|mimic4")
    if not (is_m3 | is_m4).all():
        raise ValueError("Every corpus value must identify the baseline/MIMIC-III or target/MIMIC-IV.")
    prototypes["source"] = is_m3.map({True: "MIMIC-III", False: "MIMIC-IV"})
    m3_ids = prototypes.loc[is_m3, "hadm_id"].tolist()
    m4_ids = prototypes.loc[is_m4, "hadm_id"].tolist()

    marks3 = ",".join("?" for _ in m3_ids)
    marks4 = ",".join("?" for _ in m4_ids)
    sql3 = f"""
    WITH ranked AS (
      SELECT HADM_ID AS hadm_id, TEXT AS note_text,
             ROW_NUMBER() OVER (
               PARTITION BY HADM_ID ORDER BY CHARTDATE DESC, ROW_ID DESC
             ) AS rn
      FROM NOTEEVENTS
      WHERE CATEGORY = 'Discharge summary'
        AND (ISERROR IS NULL OR ISERROR != '1')
        AND HADM_ID IN ({marks3})
    )
    SELECT hadm_id, note_text FROM ranked WHERE rn = 1
    """
    sql4 = f"""
    WITH latest AS (
      SELECT hadm_id, MAX(note_seq) AS max_seq
      FROM "note/discharge"
      WHERE hadm_id IN ({marks4})
      GROUP BY hadm_id
    )
    SELECT n.hadm_id, n.text AS note_text
    FROM "note/discharge" n
    JOIN latest l ON n.hadm_id = l.hadm_id AND n.note_seq = l.max_seq
    """

    with sqlite3.connect(a.mimic3_db) as db:
        notes3 = pd.read_sql_query(sql3, db, params=m3_ids)
    with sqlite3.connect(a.mimic4_db) as db:
        notes4 = pd.read_sql_query(sql4, db, params=m4_ids)

    notes3["hadm_id"] = pd.to_numeric(notes3["hadm_id"]).astype("int64")
    notes4["hadm_id"] = pd.to_numeric(notes4["hadm_id"]).astype("int64")
    notes3["source"] = "MIMIC-III"
    notes4["source"] = "MIMIC-IV"
    notes = pd.concat([notes3, notes4], ignore_index=True)
    notes["note_text"] = notes["note_text"].map(normalize_phi)

    out = prototypes.merge(notes, on=["source", "hadm_id"], how="left", validate="one_to_one")
    if out["note_text"].isna().any():
        missing = out.loc[out["note_text"].isna(), "hadm_id"].tolist()
        raise RuntimeError(f"Notes not found for HADM_IDs: {missing}")
    out.to_csv(a.output, index=False)
    print(f"Saved {len(out)} prototype notes to {a.output}")


if __name__ == "__main__":
    main()
