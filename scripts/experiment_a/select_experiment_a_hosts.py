#!/usr/bin/env python3
"""Select Experiment A hosts; this script never creates variants or calls a Judge.

Run from the project root:

    python scripts/experiment_a/select_experiment_a_hosts.py

That writes the full candidate table and a small verification report. Inspect
the report first. Only after it says every cell is feasible, write the 200-row
host manifest with:

    python scripts/experiment_a/select_experiment_a_hosts.py --write-hosts

The database path comes from ``MIMIC4_DB_PATH`` in ``.env``, matching the v6
scripts. All other input paths are the existing project conventions below.
There are deliberately no checksums, geometry JSON, PCA fitting, variants, or
LLM calls in this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv


# Existing project inputs. Change a path here only if the project layout changes.
BASELINE_PCA = Path("data/baseline_pca.npy")
PCA_MODEL = Path("data/pca_model.pkl")
AUDIT_MANIFEST = Path("data/judge_samples_300.csv")
OUTPUT_DIR = Path("data")
COHORTS = (
    ("2014 - 2016", 1,
     Path("data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2014_2016.npy")),
    ("2017 - 2019", 2,
     Path("data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2017_2019.npy")),
)

N_CANDIDATES = 2500
N_PER_ARM = 50
TOP_END = 50
BOTTOM_START = 2451
MIN_SECTION_CHARS = 500
SEED = 42
# v6 setting: calculate each bandwidth from 1,000 baseline and 1,000 target rows.
BANDWIDTH_SUBSAMPLE = 1000
KERNEL_BLOCK_ROWS = 128

# Same environment-file convention as v6 and the rest of the project scripts.
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
BOUNDARY_NAMES = (
    "Brief Hospital Course", "Hospital Course by", "Summary of Hospital Course", "Hospital Course",
    "History of Present Illness", "HPI", "Assessment and Plan", "Assessment", "Plan",
    "Discharge Medications", "Medications on Discharge", "Medications at Discharge",
    "Discharge Instructions", "Pertinent Results", "Pertinent Labs", "Pertinent Studies",
    "Past Medical History", "PMH", "Chief Complaint", "CC",
)
BOUNDARY = re.compile(
    r"^[ \t]*(" + "|".join(map(re.escape, BOUNDARY_NAMES)) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
PRIMARY = {"brief hospital course", "hospital course"}
SECONDARY = {"history of present illness"}


class SelectionError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise SelectionError(message)


def load_matrix(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        fail(f"Missing {name}: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        fail(f"{name} must be a finite two-dimensional array; got {array.shape}.")
    return np.asarray(array, dtype=np.float64)


def load_ids(path: Path, expected_rows: int, label: str) -> np.ndarray:
    if not path.is_file():
        fail(f"Missing {label} IDs: {path}")
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 1 or len(raw) != expected_rows:
        fail(f"{label} IDs must have {expected_rows} rows; got {raw.shape}.")
    ids = pd.to_numeric(pd.Series(raw), errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        fail(f"{label} IDs contain duplicates.")
    return ids


def median_sigma(pooled: np.ndarray, rng: np.random.Generator) -> float:
    """v6 median heuristic: sigma=sqrt(median(pairwise_sq_distance)/2)."""
    take = min(len(pooled), 2 * BANDWIDTH_SUBSAMPLE)
    sample = pooled[rng.choice(len(pooled), size=take, replace=False)]
    norms = np.sum(sample * sample, axis=1, keepdims=True)
    distances = np.maximum(norms + norms.T - 2.0 * (sample @ sample.T), 0.0)
    upper = distances[np.triu_indices(take, k=1)]
    sigma = float(np.sqrt(np.median(upper) / 2.0))
    if not np.isfinite(sigma) or sigma <= 0:
        fail("Median-heuristic bandwidth is not positive and finite.")
    return sigma


def mean_rbf(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    """Row means of RBF(left, right), in blocks to avoid a giant matrix."""
    right_norm = np.sum(right * right, axis=1)
    result = np.empty(len(left), dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma * sigma)
    for start in range(0, len(left), KERNEL_BLOCK_ROWS):
        block = left[start:start + KERNEL_BLOCK_ROWS]
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        squared = np.maximum(block_norm + right_norm - 2.0 * (block @ right.T), 0.0)
        result[start:start + len(block)] = np.exp(-gamma * squared).mean(axis=1)
    return result


def witness_scores(baseline: np.ndarray, target: np.ndarray, sigma: float) -> np.ndarray:
    return mean_rbf(target, target, sigma) - mean_rbf(target, baseline, sigma)


def normalize(text: str) -> str:
    return PHI.sub("unknown", text).replace("___", "unknown")


def parse_section(text: str) -> dict[str, object]:
    """Pick longest eligible Hospital Course, otherwise longest eligible HPI."""
    matches = list(BOUNDARY.finditer(text))
    choices: list[dict[str, object]] = []
    for i, match in enumerate(matches):
        header = match.group(1)
        header_type = header.casefold()
        kind = "primary" if header_type in PRIMARY else "secondary" if header_type in SECONDARY else None
        if kind is None:
            continue
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        choices.append({
            "kind": kind, "header": header, "body_start": body_start, "body_end": body_end,
            "raw_length": len(body), "stripped_length": len(body.strip()),
        })

    for kind in ("primary", "secondary"):
        eligible = [x for x in choices if x["kind"] == kind and x["stripped_length"] >= MIN_SECTION_CHARS]
        if eligible:
            # Longer body wins; earliest header breaks a length tie.
            return max(eligible, key=lambda x: (x["raw_length"], -x["body_start"])) | {
                "eligibility": "eligible", "reason": f"eligible_{kind}"
            }
    return {
        "kind": "", "header": "", "body_start": pd.NA, "body_end": pd.NA,
        "raw_length": pd.NA, "stripped_length": pd.NA, "eligibility": "ineligible",
        "reason": f"no_Hospital_Course_or_HPI_body_at_least_{MIN_SECTION_CHARS}_characters",
    }


NOTE_SQL = """
WITH ranked AS (
    SELECT n.rowid AS note_rowid, n.hadm_id, n.note_seq, a.subject_id, n.text,
           ROW_NUMBER() OVER (
             PARTITION BY n.hadm_id ORDER BY n.note_seq DESC, n.rowid DESC
           ) AS n
    FROM "note/discharge" AS n
    JOIN "hosp/admissions" AS a ON a.hadm_id = n.hadm_id
    WHERE n.hadm_id IN ({placeholders}) AND n.text IS NOT NULL
)
SELECT note_rowid, hadm_id, note_seq, subject_id, text FROM ranked WHERE n = 1
"""


def batches(values: list[int], size: int = 900):
    """Yield admission-ID batches small enough for SQLite's bound-parameter limit."""
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_notes(hadm_ids: list[int]) -> dict[int, tuple[int, int, object, str]]:
    if not MIMIC4_DB_PATH or not Path(MIMIC4_DB_PATH).is_file():
        fail("MIMIC4_DB_PATH is not set in .env or does not point to a file.")
    notes: dict[int, tuple[int, int, object, str]] = {}
    try:
        connection = sqlite3.connect(Path(MIMIC4_DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
        for group in batches(sorted(hadm_ids)):
            query = NOTE_SQL.format(placeholders=",".join("?" for _ in group))
            for rowid, hadm_id, note_seq, subject_id, text in connection.execute(query, group):
                if hadm_id in notes:
                    fail(f"More than one selected note for hadm_id {hadm_id}.")
                notes[int(hadm_id)] = (int(subject_id), int(rowid), note_seq, normalize(str(text)))
    except sqlite3.Error as exc:
        fail(f"Could not read note/discharge and hosp/admissions: {exc}")
    finally:
        if "connection" in locals():
            connection.close()
    missing = set(hadm_ids) - set(notes)
    if missing:
        fail(f"No discharge note was found for {len(missing)} candidate admissions.")
    return notes


def audit_ids(audit: pd.DataFrame, cohort: str, group: str) -> set[int]:
    need = {"anchor_year_group", "selection_group", "hadm_id"}
    if not need.issubset(audit.columns):
        fail(f"{AUDIT_MANIFEST} must contain {sorted(need)}.")
    values = audit.loc[
        (audit["anchor_year_group"].astype(str) == cohort)
        & (audit["selection_group"].astype(str) == group), "hadm_id"
    ]
    ids = set(pd.to_numeric(values, errors="raise").astype(int))
    if len(ids) != N_PER_ARM:
        fail(f"Audit manifest needs {N_PER_ARM} unique {group!r} IDs for {cohort!r}; found {len(ids)}.")
    return ids


def rank_and_select(frame: pd.DataFrame, code: int) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    order = np.lexsort((result["hadm_id"].to_numpy(), -result["witness_score"].to_numpy()))
    result["rank"] = 0
    result.iloc[order, result.columns.get_loc("rank")] = np.arange(1, len(result) + 1)
    result["raw_band"] = np.where(result["rank"] <= TOP_END, "top",
                            np.where(result["rank"] >= BOTTOM_START, "bottom", "middle"))
    result["arm"] = ""
    result["selection_order"] = pd.NA
    result["passed_over_for_top_ineligibility"] = False
    result["passed_over_for_random_subject_reuse"] = False
    result["random_pool_rank_order"] = pd.NA
    result["random_pool_permuted_order"] = pd.NA

    ranked = result.sort_values("rank", kind="stable")
    selected_top_indices: list[int] = []
    for row_index, row in ranked.iterrows():
        if len(selected_top_indices) == N_PER_ARM:
            break
        if row["eligibility"] == "eligible":
            selected_top_indices.append(int(row_index))
        else:
            result.loc[row_index, "passed_over_for_top_ineligibility"] = True
    top = result.loc[selected_top_indices]
    result.loc[top.index, "arm"] = "top_positive"
    result.loc[top.index, "selection_order"] = np.arange(1, len(top) + 1)

    middle = result.loc[
    (result["raw_band"] == "middle")
    & (result["eligibility"] == "eligible")
    & (~result.index.isin(selected_top_indices))
    ]
    middle = middle.sort_values("rank", kind="stable")
    if set(middle.index) & set(selected_top_indices):
        fail("Random-reference pool overlaps final top_positive hosts.")
    result.loc[middle.index, "random_pool_rank_order"] = np.arange(1, len(middle) + 1)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, code, 991]))
    shuffled = middle.iloc[rng.permutation(len(middle))]
    result.loc[shuffled.index, "random_pool_permuted_order"] = np.arange(1, len(shuffled) + 1)

    # Keep the frozen permutation intact, but do not reuse a subject already
    # represented by a selected host. Top-positive hosts always take priority;
    # a conflicting random candidate is passed over for the next candidate in
    # the saved permutation order. Adding each accepted random subject to this
    # set also prevents a subject from being selected twice in the random arm.
    selected_subject_ids = set(pd.to_numeric(top["subject_id"], errors="raise").astype(int))
    selected_random_indices: list[int] = []
    for row_index, row in shuffled.iterrows():
        if len(selected_random_indices) == N_PER_ARM:
            break
        subject_id = int(row["subject_id"])
        if subject_id in selected_subject_ids:
            result.loc[row_index, "passed_over_for_random_subject_reuse"] = True
            continue
        selected_random_indices.append(int(row_index))
        selected_subject_ids.add(subject_id)
    random_hosts = result.loc[selected_random_indices]
    result.loc[random_hosts.index, "arm"] = "random_structural_negative"
    result.loc[random_hosts.index, "selection_order"] = np.arange(1, len(random_hosts) + 1)

    top_subjects = set(pd.to_numeric(top["subject_id"], errors="raise").astype(int))
    random_subjects = set(pd.to_numeric(random_hosts["subject_id"], errors="raise").astype(int))
    subject_disjoint_arms = top_subjects.isdisjoint(random_subjects)

    def boundary(upper: int, lower: int) -> dict[str, object]:
        return {
            "upper_rank": upper, "lower_rank": lower,
            "upper_hadm_id": int(ranked.iloc[upper - 1].hadm_id),
            "lower_hadm_id": int(ranked.iloc[lower - 1].hadm_id),
            "exact_score_tie": bool(ranked.iloc[upper - 1].witness_score == ranked.iloc[lower - 1].witness_score),
        }

    return result, {
        "rank_key": "(-witness_score, numeric_hadm_id)",
        "ties": {"ranks_50_51": boundary(50, 51), "ranks_2450_2451": boundary(2450, 2451)},
        "eligible_candidates": int((result["eligibility"] == "eligible").sum()),
        "top_ineligible_passes": int(result["passed_over_for_top_ineligibility"].sum()),
        "random_subject_reuse_passes": int(result["passed_over_for_random_subject_reuse"].sum()),
        "top_selected": int(len(top)), "random_middle_pool": int(len(middle)),
        "random_selected": int(len(random_hosts)),
        "subject_disjoint_arms": bool(subject_disjoint_arms),
        "feasible": bool(
            len(top) == N_PER_ARM
            and len(random_hosts) == N_PER_ARM
            and subject_disjoint_arms
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select Experiment A hosts (no variants or Judge calls).")
    parser.add_argument("--write-hosts", action="store_true", help="Write data/experiment_a_hosts.csv after a feasible run.")
    args = parser.parse_args()

    if not AUDIT_MANIFEST.is_file():
        fail(f"Missing audit manifest: {AUDIT_MANIFEST}")
    baseline = load_matrix(BASELINE_PCA, "baseline PCA array")
    pca = joblib.load(PCA_MODEL)
    if getattr(pca, "n_components_", None) != 52:
        fail(f"PCA model must have 52 components; found {getattr(pca, 'n_components_', None)}.")
    if baseline.shape[1] != 52:
        fail(f"Baseline PCA array must have 52 columns; found {baseline.shape[1]}.")
    audit = pd.read_csv(AUDIT_MANIFEST)

    # Matches v6: one generator starts before the two windows, then samples
    # 1,000 baseline and 1,000 target rows for each bandwidth.
    sigma_rng = np.random.default_rng(SEED)
    inputs = []
    all_ids: list[int] = []
    for label, code, embedding_path, id_path in COHORTS:
        raw = load_matrix(embedding_path, f"{label} embeddings")
        ids = load_ids(id_path, len(raw), label)
        if len(raw) != N_CANDIDATES:
            fail(f"{label} must contain {N_CANDIDATES} candidates; found {len(raw)}.")
        transformed = np.asarray(pca.transform(raw), dtype=np.float64)
        if transformed.shape != (N_CANDIDATES, 52):
            fail(f"PCA transform for {label} returned {transformed.shape}, not (2500, 52).")
        base_part = baseline[sigma_rng.choice(len(baseline), min(BANDWIDTH_SUBSAMPLE, len(baseline)), replace=False)]
        target_part = transformed[sigma_rng.choice(len(transformed), min(BANDWIDTH_SUBSAMPLE, len(transformed)), replace=False)]
        sigma = median_sigma(np.vstack([base_part, target_part]), sigma_rng)
        inputs.append((label, code, ids, transformed, sigma))
        all_ids.extend(map(int, ids))
        LOG.info("%s: v6-style sigma = %.8f", label, sigma)
    if len(set(all_ids)) != len(all_ids):
        fail("Candidate pools overlap in hadm_id.")

    notes = load_notes(all_ids)
    frames, report = [], {"eligibility_floor_characters": MIN_SECTION_CHARS, "cohorts": {}}
    for label, code, ids, target, sigma in inputs:
        scores = witness_scores(baseline, target, sigma)
        rows = []
        for source_index, (hadm_id, score) in enumerate(zip(ids, scores, strict=True)):
            subject_id, rowid, note_seq, text = notes[int(hadm_id)]
            section = parse_section(text)
            rows.append({
                "cohort": label, "cohort_code": code, "hadm_id": int(hadm_id), "subject_id": subject_id,
                "source_array_index": source_index, "witness_score": float(score),
                "selected_note_rowid": rowid, "selected_note_seq": note_seq,
                "section_header": section["header"], "section_class": section["kind"],
                "section_body_start": section["body_start"], "section_body_end": section["body_end"],
                "section_raw_length": section["raw_length"], "section_stripped_length": section["stripped_length"],
                "eligibility": section["eligibility"], "eligibility_reason": section["reason"],
            })
        frame, summary = rank_and_select(pd.DataFrame(rows), code)
        summary["sigma"] = float(sigma)
        ranked = frame.sort_values("rank", kind="stable")
        new_top = set(ranked.head(N_PER_ARM).hadm_id.astype(int))
        new_bottom = set(ranked.tail(N_PER_ARM).hadm_id.astype(int))
        old_top, old_bottom = audit_ids(audit, label, "top"), audit_ids(audit, label, "bottom")
        summary["audit_rank_check"] = {
            "top_matches": new_top == old_top, "bottom_matches": new_bottom == old_bottom,
            "top_only_new": sorted(new_top - old_top), "top_only_audit": sorted(old_top - new_top),
            "bottom_only_new": sorted(new_bottom - old_bottom), "bottom_only_audit": sorted(old_bottom - new_bottom),
        }
        if not summary["audit_rank_check"]["top_matches"]:
            LOG.warning(
                "%s: new lexicographic top cut does NOT match audit manifest. "
                "Proceeding with the new cut; the audit set remains only the historical definition.",
                label,
            )
        if not summary["audit_rank_check"]["bottom_matches"]:
            LOG.warning(
                "%s: new lexicographic bottom cut does NOT match audit manifest. "
                "Proceeding with the new cut; the audit set remains only the historical definition.",
                label,
            )
        report["cohorts"][label] = summary
        frames.append(frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat(frames, ignore_index=True).sort_values(["cohort_code", "rank"], kind="stable")
    candidates.to_csv(OUTPUT_DIR / "experiment_a_candidate_pool.csv", index=False)
    feasible = all(x["feasible"] for x in report["cohorts"].values())
    report["all_cohorts_feasible"] = feasible
    with (OUTPUT_DIR / "experiment_a_host_selection_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    LOG.info("Wrote candidate pool and report to %s.", OUTPUT_DIR)

    if args.write_hosts:
        if not feasible:
            fail("At least one cohort is infeasible; no host manifest was written.")
        hosts = candidates.loc[candidates["arm"] != ""].copy()
        if len(hosts) != 200 or hosts.hadm_id.duplicated().any():
            fail("Host selection did not produce 200 unique hosts.")
        top_subjects = set(hosts.loc[hosts["arm"] == "top_positive", "subject_id"].astype(int))
        random_subjects = set(
            hosts.loc[hosts["arm"] == "random_structural_negative", "subject_id"].astype(int)
        )
        overlapping_subjects = sorted(top_subjects & random_subjects)
        if overlapping_subjects:
            fail(
                "Host selection is not subject-disjoint across arms; "
                f"found {len(overlapping_subjects)} overlapping subject_id value(s): "
                f"{overlapping_subjects[:10]}"
            )
        arm_order = {"top_positive": 0, "random_structural_negative": 1}
        hosts["_arm_order"] = hosts.arm.map(arm_order)
        hosts = hosts.sort_values(["cohort_code", "_arm_order", "selection_order"], kind="stable").drop(columns="_arm_order")
        hosts.insert(0, "host_manifest_order", range(1, len(hosts) + 1))
        hosts.to_csv(OUTPUT_DIR / "experiment_a_hosts.csv", index=False)
        LOG.info("Wrote 200-host manifest: %s", OUTPUT_DIR / "experiment_a_hosts.csv")
    elif feasible:
        LOG.info("Dry run is feasible. Inspect the report, then rerun with --write-hosts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as error:
        LOG.error("Host selection stopped: %s", error)
        raise SystemExit(1)
