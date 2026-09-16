#!/usr/bin/env python3
"""Export the 200 frozen Experiment A host notes to a local CSV.

This script is intentionally separate from host selection: it never changes the
host manifest, candidate pool, or any Experiment A result.  It retrieves only
the note row IDs frozen in data/experiment_a_hosts.csv and verifies their
admission, subject, and note sequence before writing the source note text.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


HOSTS_PATH = Path("data/experiment_a_hosts.csv")
OUTPUT_PATH = Path("data/experiment_a_selected_notes.csv")
BATCH_SIZE = 900
N_HOSTS = 200
PHI = re.compile(r"\[\*\*.*?\*\*\]")

load_dotenv()
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("export_experiment_a_selected_notes")


def fail(message: str) -> None:
    LOG.error(message)
    raise SystemExit(1)


def normalize(text: str) -> str:
    """Match generate_experiment_a_variants.py exactly."""
    return PHI.sub("unknown", text).replace("___", "unknown")


def batches(values: list[int], size: int = BATCH_SIZE):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def require_columns(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "host_manifest_order", "cohort", "cohort_code", "arm", "hadm_id",
        "subject_id", "selected_note_rowid", "selected_note_seq",
        "section_header", "section_class", "section_body_start", "section_body_end",
    }
    missing = required - set(frame.columns)
    if missing:
        fail(f"Missing required host-manifest columns in {path}: {sorted(missing)}")


def load_frozen_hosts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Missing frozen host manifest: {path}")
    hosts = pd.read_csv(path)
    require_columns(hosts, path)
    if len(hosts) != N_HOSTS:
        fail(f"Frozen host manifest must contain {N_HOSTS} rows; found {len(hosts)}.")
    for column in ("host_manifest_order", "hadm_id", "subject_id", "selected_note_rowid"):
        hosts[column] = pd.to_numeric(hosts[column], errors="raise").astype("int64")
    if hosts["hadm_id"].duplicated().any() or hosts["selected_note_rowid"].duplicated().any():
        fail("Frozen host manifest contains duplicate admission IDs or selected note row IDs.")
    if set(hosts["host_manifest_order"]) != set(range(1, N_HOSTS + 1)):
        fail("host_manifest_order must be exactly 1 through 200.")
    return hosts.sort_values("host_manifest_order", kind="stable").reset_index(drop=True)


def load_notes(rowids: list[int]) -> dict[int, tuple[int, int, object, str]]:
    if not MIMIC4_DB_PATH or not Path(MIMIC4_DB_PATH).is_file():
        fail("MIMIC4_DB_PATH is not set in .env or does not point to a file.")
    notes: dict[int, tuple[int, int, object, str]] = {}
    connection = None
    try:
        connection = sqlite3.connect(Path(MIMIC4_DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
        for group in batches(rowids):
            placeholders = ",".join("?" for _ in group)
            query = f'''\
SELECT n.rowid AS note_rowid, n.hadm_id, n.note_seq, a.subject_id, n.text
FROM "note/discharge" AS n
JOIN "hosp/admissions" AS a ON a.hadm_id = n.hadm_id
WHERE n.rowid IN ({placeholders}) AND n.text IS NOT NULL
'''
            for rowid, hadm_id, note_seq, subject_id, text in connection.execute(query, group):
                key = int(rowid)
                if key in notes:
                    fail(f"Duplicate database result for selected note rowid {key}.")
                notes[key] = (int(hadm_id), int(subject_id), note_seq, str(text))
    except sqlite3.Error as exc:
        fail(f"Could not read note/discharge and hosp/admissions: {exc}")
    finally:
        if connection is not None:
            connection.close()
    missing = set(rowids) - set(notes)
    if missing:
        fail(f"No nonempty discharge note was found for {len(missing)} frozen row IDs.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Export exact frozen Experiment A host notes to CSV.")
    parser.add_argument("--hosts", type=Path, default=HOSTS_PATH, help=f"Frozen host manifest (default: {HOSTS_PATH})")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"CSV to write (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    hosts = load_frozen_hosts(args.hosts)
    notes = load_notes(hosts["selected_note_rowid"].tolist())
    exported: list[dict[str, object]] = []
    for host in hosts.to_dict(orient="records"):
        rowid = int(host["selected_note_rowid"])
        hadm_id, subject_id, note_seq, note_text = notes[rowid]
        if hadm_id != int(host["hadm_id"]) or subject_id != int(host["subject_id"]):
            fail(f"Frozen note identity mismatch at rowid {rowid}.")
        if str(note_seq) != str(host["selected_note_seq"]):
            fail(f"Frozen note_seq mismatch at rowid {rowid}.")
        normalized_note = normalize(note_text)
        start, end = int(host["section_body_start"]), int(host["section_body_end"])
        if not (0 <= start <= end <= len(normalized_note)):
            fail(f"Frozen section offsets are invalid for note rowid {rowid}.")
        exported.append({
            "host_manifest_order": int(host["host_manifest_order"]),
            "cohort": host["cohort"],
            "cohort_code": host["cohort_code"],
            "arm": host["arm"],
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "selected_note_rowid": rowid,
            "selected_note_seq": note_seq,
            "section_header": host["section_header"],
            "section_class": host["section_class"],
            "section_body_start": start,
            "section_body_end": end,
            # Variants are constructed from this normalized note, not raw_note_text.
            "normalized_selected_section_body": normalized_note[start:end],
            "normalized_note_text": normalized_note,
            "raw_note_text": note_text,
        })

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exported).to_csv(output, index=False)
    LOG.info("Wrote %d frozen source notes: %s", len(exported), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
