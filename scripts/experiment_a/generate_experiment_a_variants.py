#!/usr/bin/env python3
"""Build Experiment A radiology donors and deterministic full-section variants.

Run from the project root after ``data/experiment_a_hosts.csv`` is frozen:

    python scripts/experiment_a/generate_experiment_a_variants.py

The donor table is written and validated before variants are constructed. This
script makes no embedding, diagnostic, Judge, or network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv


HOSTS_PATH = Path("data/experiment_a_hosts.csv")
DONORS_PATH = Path("data/experiment_a_donors.csv")
VARIANTS_PATH = Path("data/experiment_a_variants.csv")
DONOR_LIMIT = 6000
N_HOSTS = 200
N_PER_CELL = 50
MIN_SECTION_CHARS = 500
MAX_POSITIVE_CHUNKS = 80
TOP_VOCAB_SIZE = 500
OVERLAP_QUANTILE = 0.90
SEED = 42
COHORTS = ("2014 - 2016", "2017 - 2019")
ARMS = ("top_positive", "random_structural_negative")

load_dotenv()
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger(__name__)

PHI = re.compile(r"\[\*\*.*?\*\*\]")
TOKEN = re.compile(r"[A-Za-z]{3,}")
FINDINGS = re.compile(r"FINDINGS?:", re.IGNORECASE)
IMPRESSION = re.compile(r"IMPRESSIONS?:", re.IGNORECASE)
INTERNAL_LABEL = re.compile(r"^[ \t]*[A-Z][A-Za-z /\-]{2,40}:[ \t]*", re.MULTILINE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
BOUNDARY_NAMES = (
    "Brief Hospital Course", "Hospital Course by", "Summary of Hospital Course",
    "Hospital Course", "History of Present Illness", "HPI", "Assessment and Plan",
    "Assessment", "Plan", "Discharge Medications", "Medications on Discharge",
    "Medications at Discharge", "Discharge Instructions", "Pertinent Results",
    "Pertinent Labs", "Pertinent Studies", "Past Medical History", "PMH",
    "Chief Complaint", "CC",
)
BOUNDARY = re.compile(
    r"^[ \t]*(" + "|".join(map(re.escape, BOUNDARY_NAMES)) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
PRIMARY = {"brief hospital course", "hospital course"}
SECONDARY = {"history of present illness"}


class GenerationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise GenerationError(message)


def normalize(text: str) -> str:
    return PHI.sub("unknown", text).replace("___", "unknown")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preserve_body_boundary_whitespace(original_body: str, content: str) -> str:
    """Keep the section's existing header/next-section separators intact."""
    leading_size = len(original_body) - len(original_body.lstrip())
    trailing_size = len(original_body) - len(original_body.rstrip())
    leading = original_body[:leading_size]
    trailing = original_body[len(original_body) - trailing_size:] if trailing_size else ""
    return leading + content + trailing


def vocabulary(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        fail(f"Expected an integer identifier or a blank value; found {value!r}.")
        raise AssertionError("unreachable") from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        fail(f"{path} is missing required columns: {missing}")


def connect_read_only(path_value: str | None) -> sqlite3.Connection:
    if not path_value or not Path(path_value).is_file():
        fail("MIMIC4_DB_PATH is not set in .env or does not point to a file.")
    return sqlite3.connect(Path(path_value).resolve().as_uri() + "?mode=ro", uri=True)


def batches(values: list[int], size: int = 900):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def parse_host_section(text: str) -> dict[str, Any]:
    matches = list(BOUNDARY.finditer(text))
    choices: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        header = match.group(1)
        key = header.casefold()
        kind = "primary" if key in PRIMARY else "secondary" if key in SECONDARY else None
        if kind is None:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        choices.append({
            "kind": kind, "header": header, "body_start": start, "body_end": end,
            "raw_length": len(body), "stripped_length": len(body.strip()),
        })
    for kind in ("primary", "secondary"):
        eligible = [
            item for item in choices
            if item["kind"] == kind and item["stripped_length"] >= MIN_SECTION_CHARS
        ]
        if eligible:
            return max(eligible, key=lambda item: (item["raw_length"], -item["body_start"]))
    fail("A frozen host no longer has an eligible Hospital Course or HPI body.")


def load_hosts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Missing frozen host manifest: {path}")
    hosts = pd.read_csv(path)
    required = {
        "host_manifest_order", "cohort", "cohort_code", "hadm_id", "subject_id",
        "selected_note_rowid", "selected_note_seq", "section_header", "section_class",
        "section_body_start", "section_body_end", "section_raw_length",
        "section_stripped_length", "eligibility", "arm", "selection_order",
    }
    require_columns(hosts, required, path)
    if len(hosts) != N_HOSTS or hosts["hadm_id"].duplicated().any():
        fail("Host manifest must contain 200 unique hadm_id values.")
    if list(hosts.sort_values("host_manifest_order")["host_manifest_order"].astype(int)) != list(
        range(1, N_HOSTS + 1)
    ):
        fail("host_manifest_order must contain exactly 1 through 200.")
    counts = hosts.groupby(["cohort", "arm"]).size()
    for cohort in COHORTS:
        for arm in ARMS:
            if int(counts.get((cohort, arm), 0)) != N_PER_CELL:
                fail(f"Expected 50 hosts for {cohort!r} / {arm!r}.")
    top_subjects = set(hosts.loc[hosts["arm"] == ARMS[0], "subject_id"].astype(int))
    random_subjects = set(hosts.loc[hosts["arm"] == ARMS[1], "subject_id"].astype(int))
    if top_subjects & random_subjects:
        fail("Frozen host manifest has subject_id overlap between arms.")
    if (hosts["eligibility"] != "eligible").any() or (
        hosts["section_stripped_length"].astype(int) < MIN_SECTION_CHARS
    ).any():
        fail("Every frozen host must meet the 500-character eligibility floor.")
    return hosts.sort_values("host_manifest_order", kind="stable").reset_index(drop=True)


HOST_NOTE_SQL = """
SELECT n.rowid, n.hadm_id, n.note_seq, a.subject_id, n.text
FROM "note/discharge" AS n
JOIN "hosp/admissions" AS a ON a.hadm_id = n.hadm_id
WHERE n.rowid IN ({placeholders})
"""


def load_and_verify_host_notes(
    connection: sqlite3.Connection, hosts: pd.DataFrame
) -> dict[int, str]:
    records: dict[int, tuple[int, int, int, str]] = {}
    rowids = hosts["selected_note_rowid"].astype(int).tolist()
    try:
        for group in batches(sorted(rowids)):
            query = HOST_NOTE_SQL.format(placeholders=",".join("?" for _ in group))
            for rowid, hadm_id, note_seq, subject_id, text in connection.execute(query, group):
                records[int(rowid)] = (
                    int(hadm_id), int(note_seq), int(subject_id), normalize(str(text))
                )
    except sqlite3.Error as exc:
        fail(f"Could not load frozen discharge notes: {exc}")
    if set(records) != set(rowids):
        fail(f"Could not recover all {len(rowids)} frozen discharge-note rowids.")

    notes: dict[int, str] = {}
    for host in hosts.itertuples(index=False):
        rowid = int(host.selected_note_rowid)
        hadm_id, note_seq, subject_id, text = records[rowid]
        if (hadm_id, note_seq, subject_id) != (
            int(host.hadm_id), int(host.selected_note_seq), int(host.subject_id)
        ):
            fail(f"Frozen note identity changed for hadm_id {int(host.hadm_id)}.")
        parsed = parse_host_section(text)
        expected = (
            str(host.section_header), str(host.section_class), int(host.section_body_start),
            int(host.section_body_end), int(host.section_raw_length),
            int(host.section_stripped_length),
        )
        observed = (
            parsed["header"], parsed["kind"], parsed["body_start"], parsed["body_end"],
            parsed["raw_length"], parsed["stripped_length"],
        )
        if observed != expected:
            fail(f"Frozen section offsets or parser result changed for hadm_id {hadm_id}.")
        notes[hadm_id] = text
    return notes


RADIOLOGY_SQL = """
SELECT rowid, subject_id, hadm_id, text
FROM "note/radiology"
WHERE subject_id IS NOT NULL AND trim(subject_id) <> ''
  AND text IS NOT NULL AND trim(text) <> ''
ORDER BY rowid ASC
LIMIT ?
"""


def load_radiology_sources(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = list(connection.execute(RADIOLOGY_SQL, (DONOR_LIMIT,)))
    except sqlite3.Error as exc:
        fail(f"Could not load note/radiology donors: {exc}")
    if len(rows) != DONOR_LIMIT:
        fail(f"Expected the first {DONOR_LIMIT} eligible radiology rows; found {len(rows)}.")
    sources = []
    for donor_id, (rowid, subject_id, hadm_id, text) in enumerate(rows, start=1):
        sources.append({
            "donor_id": donor_id,
            "source_rowid": int(rowid),
            "subject_id": int(subject_id),
            "hadm_id": nullable_int(hadm_id),
            "text": normalize(str(text)),
        })
    if [x["source_rowid"] for x in sources] != sorted(x["source_rowid"] for x in sources):
        fail("Radiology donor rows are not in ascending SQLite rowid order.")
    return sources


def split_sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, cursor = [], start
    for separator in SENTENCE_SPLIT.finditer(text, start, end):
        spans.append((cursor, separator.start()))
        cursor = separator.end()
    spans.append((cursor, end))
    return spans


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def parse_radiology(text: str) -> dict[str, Any]:
    findings = FINDINGS.search(text)
    if findings is None:
        return {
            "has_findings": False, "findings_header_start": None,
            "findings_body_start": None, "findings_body_end": None,
            "impression_header_start": None, "chunks": [],
        }
    impression = IMPRESSION.search(text, findings.end())
    body_start = findings.end()
    body_end = impression.start() if impression else len(text)
    labels = list(INTERNAL_LABEL.finditer(text, body_start, body_end))
    if labels:
        piece_spans, cursor = [], body_start
        for label in labels:
            piece_spans.append((cursor, label.start()))
            cursor = label.end()
        piece_spans.append((cursor, body_end))
    else:
        # Without internal labels, the protocol requires sentence-level chunks
        # regardless of body length. The >600-character rule below applies only
        # as a second split to an already delimited piece.
        piece_spans = split_sentence_spans(text, body_start, body_end)

    final_spans: list[tuple[int, int]] = []
    for start, end in piece_spans:
        final_spans.extend(
            split_sentence_spans(text, start, end) if end - start > 600 else [(start, end)]
        )
    chunks = []
    for start, end in final_spans:
        start, end = trim_span(text, start, end)
        chunk_text = collapse(text[start:end])
        if len(chunk_text) >= 20:
            chunks.append({"source_start": start, "source_end": end, "text": chunk_text})
    return {
        "has_findings": True,
        "findings_header_start": findings.start(),
        "findings_body_start": body_start,
        "findings_body_end": body_end,
        "impression_header_start": impression.start() if impression else None,
        "chunks": chunks,
    }


def host_top_vocabulary(hosts: pd.DataFrame, notes: dict[int, str]) -> list[str]:
    frequencies: Counter[str] = Counter()
    for host in hosts.itertuples(index=False):
        frequencies.update(sorted(vocabulary(notes[int(host.hadm_id)])))
    ordered = sorted(frequencies, key=lambda token: (-frequencies[token], token))
    return ordered[:TOP_VOCAB_SIZE]


def quantile_type7(values: list[float], probability: float) -> float:
    if not values:
        fail("No radiology chunks were available for lexical-overlap calibration.")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (len(ordered) - 1) * probability
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    weight = position - lower
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


def build_donor_table(
    sources: list[dict[str, Any]], host_vocabulary: list[str]
) -> tuple[pd.DataFrame, float]:
    host_tokens = set(host_vocabulary)
    rows: list[dict[str, Any]] = []
    overlaps: list[float] = []
    for source in sources:
        parsed = parse_radiology(source["text"])
        base = {
            "donor_id": source["donor_id"],
            "source_rowid": source["source_rowid"],
            "subject_id": source["subject_id"],
            "hadm_id": source["hadm_id"],
            "normalized_source_length": len(source["text"]),
            "normalized_source_sha256": sha256_text(source["text"]),
            "has_findings": parsed["has_findings"],
            "findings_header_start": parsed["findings_header_start"],
            "findings_body_start": parsed["findings_body_start"],
            "findings_body_end": parsed["findings_body_end"],
            "impression_header_start": parsed["impression_header_start"],
        }
        if not parsed["chunks"]:
            rows.append(base | {
                "chunk_id": None, "chunk_source_order": None, "chunk_source_start": None,
                "chunk_source_end": None, "chunk_text": None, "chunk_length": None,
                "chunk_vocabulary_size": None, "jaccard_overlap": None,
            })
            continue
        for order, chunk in enumerate(parsed["chunks"], start=1):
            chunk_tokens = vocabulary(chunk["text"])
            overlap = (
                len(chunk_tokens & host_tokens) / len(chunk_tokens | host_tokens)
                if chunk_tokens and host_tokens else 0.0
            )
            overlaps.append(float(overlap))
            rows.append(base | {
                "chunk_id": f"D{int(source['donor_id']):04d}-C{order:03d}",
                "chunk_source_order": order,
                "chunk_source_start": chunk["source_start"],
                "chunk_source_end": chunk["source_end"],
                "chunk_text": chunk["text"],
                "chunk_length": len(chunk["text"]),
                "chunk_vocabulary_size": len(chunk_tokens),
                "jaccard_overlap": float(overlap),
            })
    cutoff = quantile_type7(overlaps, OVERLAP_QUANTILE)
    for row in rows:
        row["overlap_quantile"] = OVERLAP_QUANTILE
        row["overlap_quantile_method"] = "Hyndman-Fan_type_7_linear"
        row["overlap_cutoff"] = cutoff
        row["retained"] = bool(
            row["chunk_id"] is not None and float(row["jaccard_overlap"]) >= cutoff
        )
    donors = pd.DataFrame(rows)
    return donors, cutoff


def validate_donor_table(
    donors: pd.DataFrame, sources: list[dict[str, Any]], host_vocabulary: list[str], cutoff: float
) -> None:
    if donors["donor_id"].nunique() != DONOR_LIMIT:
        fail("Donor table does not archive all 6,000 source reports.")
    mapping = donors[["donor_id", "source_rowid"]].drop_duplicates().sort_values("donor_id")
    if mapping["donor_id"].astype(int).tolist() != list(range(1, DONOR_LIMIT + 1)):
        fail("donor_id is not stable source-row order 1 through 6,000.")
    chunks = donors.loc[donors["chunk_id"].notna()].copy()
    if chunks.empty or chunks["chunk_id"].duplicated().any():
        fail("Radiology chunk IDs must be nonempty and unique.")
    host_tokens = set(host_vocabulary)
    recalculated = []
    for row in chunks.itertuples(index=False):
        tokens = vocabulary(str(row.chunk_text))
        recalculated.append(
            len(tokens & host_tokens) / len(tokens | host_tokens) if tokens and host_tokens else 0.0
        )
        source = sources[int(row.donor_id) - 1]
        start, end = int(row.chunk_source_start), int(row.chunk_source_end)
        if collapse(source["text"][start:end]) != str(row.chunk_text):
            fail(f"Chunk offsets do not reproduce {row.chunk_id}.")
        if not (end <= int(row.findings_body_end)):
            fail(f"Chunk {row.chunk_id} extends beyond its FINDINGS body.")
        if not pd.isna(row.impression_header_start) and int(row.findings_body_end) != int(
            row.impression_header_start
        ):
            fail(f"Donor {int(row.donor_id)} does not stop at its later IMPRESSION header.")
    if not np.allclose(chunks["jaccard_overlap"].astype(float), recalculated, rtol=0, atol=1e-15):
        fail("Saved donor Jaccard values do not match the frozen host vocabulary.")
    observed_cutoff = quantile_type7(recalculated, OVERLAP_QUANTILE)
    if not np.isclose(cutoff, observed_cutoff, rtol=0, atol=1e-15):
        fail("Saved overlap cutoff is not the Hyndman-Fan type-7 0.90 quantile.")
    expected_retained = chunks["jaccard_overlap"].astype(float) >= cutoff
    if not np.array_equal(chunks["retained"].astype(bool).to_numpy(), expected_retained.to_numpy()):
        fail("Saved retained donor set does not apply cutoff-inclusive ties.")


def retained_chunks_by_report(donors: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    retained = donors.loc[donors["retained"].astype(bool) & donors["chunk_id"].notna()]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in retained.sort_values(["source_rowid", "chunk_source_order"]).to_dict("records"):
        grouped[int(row["donor_id"])].append(row)
    if not grouped:
        fail("The lexical-overlap filter retained no donor chunks.")
    return dict(grouped)


def positive_replacement(
    host: Any, donors_by_report: dict[int, list[dict[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    eligible_reports = []
    for donor_id in sorted(
        donors_by_report, key=lambda key: int(donors_by_report[key][0]["source_rowid"])
    ):
        first = donors_by_report[donor_id][0]
        same_admission = (
            nullable_int(first["hadm_id"]) is not None
            and nullable_int(first["hadm_id"]) == int(host.hadm_id)
        )
        if int(first["subject_id"]) != int(host.subject_id) and not same_admission:
            eligible_reports.append(donor_id)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(host.hadm_id)]))
    report_order = rng.permutation(np.asarray(eligible_reports, dtype=np.int64)).tolist()
    chosen: list[dict[str, Any]] = []
    replacement_length = 0
    target_length = int(host.section_raw_length)
    for donor_id in report_order:
        for chunk in donors_by_report[int(donor_id)]:
            if len(chosen) == MAX_POSITIVE_CHUNKS:
                break
            chosen.append(chunk)
            replacement_length += len(str(chunk["chunk_text"])) + (1 if len(chosen) > 1 else 0)
            if replacement_length >= target_length:
                break
        if replacement_length >= target_length or len(chosen) == MAX_POSITIVE_CHUNKS:
            break
    if replacement_length < target_length:
        fail(
            f"Positive host {int(host.hadm_id)} cannot reach its {target_length}-character "
            f"body within {MAX_POSITIVE_CHUNKS} distinct retained chunks."
        )
    return " ".join(str(chunk["chunk_text"]) for chunk in chosen), chosen


def negative_replacement(body: str, hadm_id: int) -> tuple[str, list[str], list[int]]:
    # A source sentence can contain internal line breaks. Collapse them before
    # writing one bullet per sentence; this is the protocol-permitted whitespace
    # normalization and keeps the validator's sentence representation aligned
    # with the emitted one-line bullets.
    sentences = [
        normalized for piece in SENTENCE_SPLIT.split(body.strip())
        if (normalized := collapse(piece))
    ]
    if not sentences:
        fail(f"Negative host {hadm_id} has no nonempty sentences.")
    order = np.arange(len(sentences), dtype=np.int64)
    if len(sentences) >= 2:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, hadm_id]))
        order = rng.permutation(len(sentences))
        if np.array_equal(order, np.arange(len(sentences))):
            order = np.roll(order, -1)
    replacement = "\n".join(f"- {sentences[int(index)]}" for index in order)
    return replacement, sentences, [int(index) for index in order]


def build_variants(
    hosts: pd.DataFrame, notes: dict[int, str], donors: pd.DataFrame
) -> pd.DataFrame:
    by_report = retained_chunks_by_report(donors)
    rows = []
    for variant_order, host in enumerate(hosts.itertuples(index=False), start=1):
        hadm_id = int(host.hadm_id)
        original = notes[hadm_id]
        start, end = int(host.section_body_start), int(host.section_body_end)
        original_body = original[start:end]
        if host.arm == "top_positive":
            filler, chosen = positive_replacement(host, by_report)
            donor_ids = list(dict.fromkeys(int(chunk["donor_id"]) for chunk in chosen))
            chunk_ids = [str(chunk["chunk_id"]) for chunk in chosen]
            offsets = [
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "source_rowid": int(chunk["source_rowid"]),
                    "source_start": int(chunk["chunk_source_start"]),
                    "source_end": int(chunk["chunk_source_end"]),
                }
                for chunk in chosen
            ]
            sentence_count, sentence_order = None, []
        elif host.arm == "random_structural_negative":
            filler, sentences, sentence_order = negative_replacement(original_body, hadm_id)
            chosen, donor_ids, chunk_ids, offsets = [], [], [], []
            sentence_count = len(sentences)
        else:
            fail(f"Unsupported Experiment A arm: {host.arm!r}")
        replacement = preserve_body_boundary_whitespace(original_body, filler)
        variant = original[:start] + replacement + original[end:]
        rows.append({
            "variant_manifest_order": variant_order,
            "variant_id": f"experiment_a_{variant_order:03d}",
            "host_manifest_order": int(host.host_manifest_order),
            "cohort": host.cohort,
            "cohort_code": int(host.cohort_code),
            "arm": host.arm,
            "hadm_id": hadm_id,
            "subject_id": int(host.subject_id),
            "selected_note_rowid": int(host.selected_note_rowid),
            "section_header": host.section_header,
            "section_class": host.section_class,
            "host_body_start": start,
            "host_body_end": end,
            "original_body_length": len(original_body),
            "inserted_filler_length": len(filler),
            "replacement_body_length": len(replacement),
            "note_character_change": len(variant) - len(original),
            "host_seed_sequence_json": json_compact([SEED, hadm_id]),
            "donor_report_count": len(donor_ids),
            "donor_chunk_count": len(chosen),
            "ordered_donor_ids_json": json_compact(donor_ids),
            "ordered_chunk_ids_json": json_compact(chunk_ids),
            "ordered_chunk_source_offsets_json": json_compact(offsets),
            "original_sentence_count": sentence_count,
            "sentence_permutation_zero_based_json": json_compact(sentence_order),
            "replacement_body": replacement,
            "variant_text": variant,
        })
    return pd.DataFrame(rows)


def validate_variants(
    variants: pd.DataFrame, hosts: pd.DataFrame, notes: dict[int, str], donors: pd.DataFrame
) -> None:
    if len(variants) != N_HOSTS or variants["hadm_id"].nunique() != N_HOSTS:
        fail("Variant manifest must contain 200 unique host admissions.")
    if variants["variant_text"].isna().any() or variants["variant_text"].eq("").any():
        fail("Every Experiment A variant must contain nonempty text.")
    retained = {
        str(row["chunk_id"]): row
        for row in donors.loc[donors["retained"].astype(bool)].to_dict("records")
        if row["chunk_id"] is not None
    }
    host_map = {int(row.hadm_id): row for row in hosts.itertuples(index=False)}
    for row in variants.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        host = host_map[hadm_id]
        original = notes[hadm_id]
        start, end = int(row.host_body_start), int(row.host_body_end)
        original_body = original[start:end]
        replacement = str(row.replacement_body)
        expected_variant = original[:start] + replacement + original[end:]
        if str(row.variant_text) != expected_variant:
            fail(f"Variant {row.variant_id} does not wholly replace the frozen body.")
        if int(row.note_character_change) != len(str(row.variant_text)) - len(original):
            fail(f"Variant {row.variant_id} has an incorrect note-level character change.")

        chunk_ids = json.loads(str(row.ordered_chunk_ids_json))
        donor_ids = json.loads(str(row.ordered_donor_ids_json))
        offsets = json.loads(str(row.ordered_chunk_source_offsets_json))
        if row.arm == "top_positive":
            if len(chunk_ids) != len(set(chunk_ids)) or not 1 <= len(chunk_ids) <= MAX_POSITIVE_CHUNKS:
                fail(f"Positive variant {row.variant_id} repeats chunks or violates the 80-chunk cap.")
            if any(chunk_id not in retained for chunk_id in chunk_ids):
                fail(f"Positive variant {row.variant_id} uses a non-retained donor chunk.")
            chosen = [retained[chunk_id] for chunk_id in chunk_ids]
            filler = " ".join(str(chunk["chunk_text"]) for chunk in chosen)
            if preserve_body_boundary_whitespace(original_body, filler) != replacement:
                fail(f"Positive variant {row.variant_id} truncates, reorders, or changes donor chunks.")
            expected_donors = list(dict.fromkeys(int(chunk["donor_id"]) for chunk in chosen))
            if donor_ids != expected_donors or int(row.donor_report_count) != len(expected_donors):
                fail(f"Positive variant {row.variant_id} has inconsistent donor-report provenance.")
            if (
                int(row.donor_chunk_count) != len(chosen)
                or int(row.inserted_filler_length) != len(filler)
                or len(filler) < int(row.original_body_length)
            ):
                fail(f"Positive variant {row.variant_id} has invalid donor length provenance.")
            expected_offsets = [
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "source_rowid": int(chunk["source_rowid"]),
                    "source_start": int(chunk["chunk_source_start"]),
                    "source_end": int(chunk["chunk_source_end"]),
                }
                for chunk in chosen
            ]
            if offsets != expected_offsets:
                fail(f"Positive variant {row.variant_id} has incorrect source offsets.")
            for chunk in chosen:
                if int(chunk["subject_id"]) == int(host.subject_id):
                    fail(f"Positive variant {row.variant_id} uses a same-patient donor.")
                donor_hadm = nullable_int(chunk["hadm_id"])
                if donor_hadm is not None and donor_hadm == hadm_id:
                    fail(f"Positive variant {row.variant_id} uses a same-admission donor.")
        else:
            filler, original_sentences, order = negative_replacement(original_body, hadm_id)
            expected = preserve_body_boundary_whitespace(original_body, filler)
            if (
                replacement != expected
                or int(row.inserted_filler_length) != len(filler)
                or chunk_ids or donor_ids or offsets
            ):
                fail(f"Negative variant {row.variant_id} changed content or used donor text.")
            after = [collapse(line[2:]) for line in filler.splitlines() if line.startswith("- ")]
            if Counter(after) != Counter(collapse(sentence) for sentence in original_sentences):
                fail(f"Negative variant {row.variant_id} changes the sentence multiset.")
            if json.loads(str(row.sentence_permutation_zero_based_json)) != order:
                fail(f"Negative variant {row.variant_id} has incorrect permutation provenance.")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate Experiment A donors and full-section variants."
    )
    parser.add_argument("--hosts", type=Path, default=HOSTS_PATH)
    parser.add_argument("--donors", type=Path, default=DONORS_PATH)
    parser.add_argument("--variants", type=Path, default=VARIANTS_PATH)
    parser.add_argument("--database", default=MIMIC4_DB_PATH, help="Defaults to MIMIC4_DB_PATH.")
    args = parser.parse_args()

    hosts = load_hosts(args.hosts)
    connection = connect_read_only(args.database)
    try:
        host_notes = load_and_verify_host_notes(connection, hosts)
        sources = load_radiology_sources(connection)
    finally:
        connection.close()

    top_vocabulary = host_top_vocabulary(hosts, host_notes)
    donors, cutoff = build_donor_table(sources, top_vocabulary)
    validate_donor_table(donors, sources, top_vocabulary, cutoff)
    atomic_csv(donors, args.donors)
    chunk_count = int(donors["chunk_id"].notna().sum())
    retained_count = int(donors["retained"].astype(bool).sum())
    LOG.info(
        "Wrote donor table: %s | reports=%d chunks=%d retained=%d cutoff=%.12f",
        args.donors, DONOR_LIMIT, chunk_count, retained_count, cutoff,
    )

    variants = build_variants(hosts, host_notes, donors)
    validate_variants(variants, hosts, host_notes, donors)
    atomic_csv(variants, args.variants)
    LOG.info(
        "Wrote and validated %d variants: %s | positive=%d structural-negative=%d",
        len(variants), args.variants,
        int((variants["arm"] == "top_positive").sum()),
        int((variants["arm"] == "random_structural_negative").sum()),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        LOG.error("Experiment A generation stopped: %s", error)
        raise SystemExit(1)
