#!/usr/bin/env python3
"""Read-only pre-Judge validator for the frozen Experiment A protocol.

Run this only after hosts, donors, variants, diagnostics, calibration,
exemplars, Judge input, runner, and launcher have been prepared.  It makes no
Judge, model, or write calls.  Any failed assertion stops the preflight.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import generate_experiment_a_variants as generation


N_CANDIDATES = 2500
N_HOSTS = 200
N_PER_CELL = 50
COHORTS = ("2014 - 2016", "2017 - 2019")
ARMS = ("top_positive", "random_structural_negative")
EXPECTED_CELLS = {(cohort, arm): N_PER_CELL for cohort in COHORTS for arm in ARMS}

HOSTS_PATH = Path("data/experiment_a_hosts.csv")
CANDIDATES_PATH = Path("data/experiment_a_candidate_pool.csv")
SELECTION_REPORT_PATH = Path("data/experiment_a_host_selection_report.json")
DONORS_PATH = Path("data/experiment_a_donors.csv")
VARIANTS_PATH = Path("data/experiment_a_variants.csv")
DIAGNOSTICS_PATH = Path("data/experiment_a_diagnostics.csv")
EXEMPLARS_PATH = Path("data/experiment_a_exemplars.csv")
JUDGE_INPUT_PATH = Path("data/experiment_a_judge_input.csv")
CALIBRATION_PATH = Path("data/experiment_a_threshold_calibration.json")
BASELINE_PCA_PATH = Path("data/baseline_pca.npy")
PCA_MODEL_PATH = Path("data/pca_model.pkl")
DIAGNOSTICS_SCRIPT = Path("scripts/experiment_a/embed_experiment_a_diagnostics.py")
JUDGE_RUNNER = Path("scripts/experiment_a/judge_blind_run_experiment_a.py")
LAUNCHER = Path("scripts/experiment_a/run_judge_blind_experiment_a.slurm")
RESUME_LAUNCHER = Path("scripts/experiment_a/resume_judge_blind_experiment_a.slurm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("validate_experiment_a")


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Missing {label}: {path}")
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def require_exact_frame(saved: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if list(saved.columns) != list(expected.columns):
        fail(f"{label} columns do not match the deterministic reconstruction.")
    try:
        # CSV represents absent donor fields as NaN while reconstruction uses
        # None.  They are the same archived missing value.
        saved = saved.replace({None: np.nan})
        expected = expected.replace({None: np.nan})
        pd.testing.assert_frame_equal(
            saved.reset_index(drop=True), expected.reset_index(drop=True),
            # CSV round-tripping shortens IEEE-754 decimal renderings.  Text,
            # IDs, offsets, and booleans remain exact; this tolerance applies
            # only to numeric cells and is tighter than the donor 1e-15
            # overlap/cutoff checks below.
            check_dtype=False, check_exact=False, rtol=0, atol=1e-15,
        )
    except AssertionError as exc:
        fail(f"{label} differs from its deterministic reconstruction: {exc}")


def validate_candidates_and_selection(
    candidates_path: Path, report_path: Path, hosts: pd.DataFrame
) -> dict[str, float]:
    candidates = read_csv(candidates_path, "candidate pool")
    require_columns(
        candidates,
        {
            "cohort", "cohort_code", "hadm_id", "subject_id", "witness_score", "rank",
            "raw_band", "selected_note_rowid", "selected_note_seq", "eligibility", "arm",
            "random_pool_rank_order", "random_pool_permuted_order",
        },
        "candidate pool",
    )
    if len(candidates) != 2 * N_CANDIDATES or candidates["hadm_id"].duplicated().any():
        fail("Candidate pool must contain 5,000 unique admissions.")
    if not np.isfinite(candidates["witness_score"].astype(float)).all():
        fail("Candidate pool contains a non-finite witness score.")

    for cohort in COHORTS:
        pool = candidates.loc[candidates["cohort"].astype(str) == cohort].copy()
        if len(pool) != N_CANDIDATES:
            fail(f"{cohort} does not contain exactly 2,500 candidates.")
        ranks = pool["rank"].astype(int).to_numpy()
        if set(ranks) != set(range(1, N_CANDIDATES + 1)):
            fail(f"{cohort} candidate ranks are not exactly 1 through 2,500.")
        ids = pool["hadm_id"].astype(np.int64).to_numpy()
        scores = pool["witness_score"].astype(float).to_numpy()
        expected_rank = np.empty(N_CANDIDATES, dtype=np.int64)
        expected_rank[np.lexsort((ids, -scores))] = np.arange(1, N_CANDIDATES + 1)
        if not np.array_equal(ranks, expected_rank):
            fail(f"{cohort} ranks do not use (-witness_score, numeric_hadm_id).")
        raw_band = pool.set_index("rank")["raw_band"].astype(str)
        if not (raw_band.loc[1:50] == "top").all() or not (raw_band.loc[2451:2500] == "bottom").all():
            fail(f"{cohort} raw rank cuts are not frozen at 1–50 and 2,451–2,500.")

    if not report_path.is_file():
        fail(f"Missing host-selection report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("all_cohorts_feasible"):
        fail("Host-selection report does not mark both cohorts feasible.")
    sigmas: dict[str, float] = {}
    for cohort in COHORTS:
        entry = report.get("cohorts", {}).get(cohort)
        if not isinstance(entry, dict):
            fail(f"Host-selection report has no {cohort} entry.")
        if entry.get("rank_key") != "(-witness_score, numeric_hadm_id)":
            fail(f"{cohort} report does not record the frozen rank key.")
        ties = entry.get("ties", {})
        if not {"ranks_50_51", "ranks_2450_2451"}.issubset(ties):
            fail(f"{cohort} report does not record both rank-boundary tie checks.")
        audit = entry.get("audit_rank_check", {})
        if not {"top_matches", "bottom_matches", "top_only_new", "top_only_audit", "bottom_only_new", "bottom_only_audit"}.issubset(audit):
            fail(f"{cohort} report does not retain the complete F4 audit comparison.")
        if entry.get("top_selected") != N_PER_CELL or entry.get("random_selected") != N_PER_CELL:
            fail(f"{cohort} report does not record 50 selected hosts per arm.")
        if entry.get("subject_disjoint_arms") is not True:
            fail(f"{cohort} report does not confirm subject-disjoint arms.")
        sigma = float(entry.get("sigma", np.nan))
        if not np.isfinite(sigma) or sigma <= 0:
            fail(f"{cohort} report has an invalid frozen bandwidth.")
        sigmas[cohort] = sigma

    selected = candidates.loc[candidates["arm"].isin(ARMS)].copy()
    if len(selected) != N_HOSTS or selected["hadm_id"].duplicated().any():
        fail("Candidate pool does not mark exactly 200 unique selected hosts.")
    if set(selected["hadm_id"].astype(int)) != set(hosts["hadm_id"].astype(int)):
        fail("Candidate-pool selection does not match the frozen host manifest.")
    selected_top = selected.loc[selected["arm"] == "top_positive"]
    if selected_top["random_pool_rank_order"].notna().any() or selected_top["random_pool_permuted_order"].notna().any():
        fail("A final top-positive host remains in a random-reference pool.")
    return sigmas


def validate_diagnostics(
    path: Path, variants: pd.DataFrame, candidates: pd.DataFrame, sigmas: dict[str, float],
    baseline_path: Path, pca_path: Path, script_path: Path,
) -> None:
    diagnostics = read_csv(path, "diagnostics")
    required = {
        "variant_id", "host_manifest_order", "cohort", "arm", "hadm_id", "subject_id",
        "original_embedding_windows", "variant_embedding_windows", "cohort_sigma",
        "pca_displacement_l2", "original_witness_reembedded", "variant_witness",
        "witness_change", "frozen_original_witness_score", "frozen_original_rank",
        "post_treatment_rank", "post_treatment_rank_change",
    }
    require_columns(diagnostics, required, "diagnostics")
    if len(diagnostics) != N_HOSTS or diagnostics["hadm_id"].duplicated().any():
        fail("Diagnostics must contain exactly 200 unique hosts.")
    if diagnostics.groupby(["cohort", "arm"]).size().to_dict() != EXPECTED_CELLS:
        fail("Diagnostics does not contain 50 rows in every cohort-by-arm cell.")
    merged = diagnostics.merge(
        variants[["variant_id", "hadm_id", "host_manifest_order", "cohort", "arm", "subject_id"]],
        on=["variant_id", "hadm_id", "host_manifest_order", "cohort", "arm", "subject_id"],
        how="inner", validate="one_to_one",
    )
    if len(merged) != N_HOSTS:
        fail("Diagnostics does not match the frozen variant manifest.")
    numeric = [
        "cohort_sigma", "pca_displacement_l2", "original_witness_reembedded", "variant_witness",
        "witness_change", "frozen_original_witness_score", "post_treatment_rank",
        "post_treatment_rank_change",
    ]
    if not np.isfinite(diagnostics[numeric].astype(float)).all().all():
        fail("Diagnostics contains a non-finite numeric value.")
    if (diagnostics[["original_embedding_windows", "variant_embedding_windows"]].astype(int) < 1).any().any():
        fail("A diagnostic note produced no embedding windows.")
    if (diagnostics["pca_displacement_l2"].astype(float) < 0).any():
        fail("Diagnostics contains a negative PCA displacement.")
    for cohort, sigma in sigmas.items():
        observed = diagnostics.loc[diagnostics["cohort"] == cohort, "cohort_sigma"].astype(float)
        if not np.allclose(observed, sigma, rtol=0, atol=1e-12):
            fail(f"{cohort} diagnostic bandwidth differs from the frozen selection report.")
    frozen = candidates[["hadm_id", "witness_score", "rank"]].rename(
        columns={"witness_score": "candidate_score", "rank": "candidate_rank"}
    )
    score_check = diagnostics.merge(frozen, on="hadm_id", how="inner", validate="one_to_one")
    if len(score_check) != N_HOSTS or not np.allclose(
        score_check["frozen_original_witness_score"], score_check["candidate_score"], rtol=0, atol=1e-10
    ):
        fail("Diagnostics frozen witness scores do not match the candidate pool.")
    if not np.array_equal(score_check["frozen_original_rank"].astype(int), score_check["candidate_rank"].astype(int)):
        fail("Diagnostics frozen ranks do not match the candidate pool.")
    if not baseline_path.is_file() or not pca_path.is_file():
        fail("Missing frozen PCA baseline array or PCA model.")
    try:
        import joblib
    except ImportError as exc:
        fail("joblib is required to validate the frozen PCA model.")
        raise AssertionError("unreachable") from exc
    baseline = np.load(baseline_path, allow_pickle=False)
    pca = joblib.load(pca_path)
    if baseline.ndim != 2 or baseline.shape[1] != 52 or not np.isfinite(baseline).all():
        fail("Frozen baseline PCA representation is not finite with 52 columns.")
    if getattr(pca, "n_components_", None) != 52:
        fail("Frozen PCA model does not have exactly 52 output components.")
    if not script_path.is_file():
        fail(f"Missing diagnostics source: {script_path}")
    source = script_path.read_text(encoding="utf-8")
    required_source = (
        'MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"', "truncation=False",
        "INTERIOR_WINDOW = 510", "STRIDE = 256", "not np.any(embedding)",
        "median_sigma", "np.triu_indices(take, k=1)",
    )
    if any(token not in source for token in required_source):
        fail("Diagnostics source does not match the frozen embedding/bandwidth contract.")


def validate_judge_input(exemplars_path: Path, input_path: Path, variants: pd.DataFrame) -> None:
    exemplars = read_csv(exemplars_path, "frozen exemplars")
    judge_input = read_csv(input_path, "blinded Judge input")
    require_columns(exemplars, {"record_id", "hadm_id", "normalized_text"}, "frozen exemplars")
    if len(exemplars) != 3 or exemplars["record_id"].duplicated().any() or exemplars["normalized_text"].isna().any():
        fail("Exemplar file must contain exactly three unique, nonempty normalized exemplars.")
    required = {"record_id", "group", "note_text"}
    require_columns(judge_input, required, "blinded Judge input")
    if set(judge_input.columns) != required:
        fail("Judge input may contain only record_id, group, and note_text.")
    if len(judge_input) != 203 or judge_input["record_id"].duplicated().any() or judge_input["note_text"].isna().any() or judge_input["note_text"].eq("").any():
        fail("Judge input must contain 203 unique records with nonempty note text.")
    expected_groups = {"exemplar": 3, "top_positive": 100, "random_structural_negative": 100}
    if judge_input["group"].value_counts().to_dict() != expected_groups:
        fail("Judge input does not contain three exemplars and 100 notes per arm.")
    exemplar_input = judge_input.loc[judge_input["group"] == "exemplar"]
    exemplar_check = exemplar_input.merge(
        exemplars[["record_id", "normalized_text"]], on="record_id", how="inner", validate="one_to_one"
    )
    if len(exemplar_check) != 3 or not (exemplar_check["note_text"] == exemplar_check["normalized_text"]).all():
        fail("Judge-input exemplars are not byte-identical to the frozen exemplar text.")
    classifications = judge_input.loc[judge_input["group"] != "exemplar"]
    expected = variants[["variant_id", "arm", "variant_text"]].rename(
        columns={"variant_id": "record_id", "arm": "group", "variant_text": "expected_text"}
    )
    check = classifications.merge(expected, on=["record_id", "group"], how="inner", validate="one_to_one")
    if len(check) != N_HOSTS or not (check["note_text"] == check["expected_text"]).all():
        fail("Judge input does not contain each frozen variant exactly once and byte-identically.")


def validate_runner_and_launcher(
    runner_path: Path, launcher_path: Path, resume_launcher_path: Path
) -> None:
    if not runner_path.is_file() or not launcher_path.is_file() or not resume_launcher_path.is_file():
        fail("Missing restricted Judge runner, fresh launcher, or resume launcher.")
    runner = runner_path.read_text(encoding="utf-8")
    required_runner = (
        "JUDGE_NUM_CTX = 65536", "gemma4:26b", "client.chat(",
        '"num_ctx": JUDGE_NUM_CTX', "--resume", "incomplete", "STOCHASTIC_TEMPERATURE = 0.7",
        "N_STOCHASTIC_CALLS = 5", "MAX_RETRIES = 3", "chat_once(client, model, messages, 0.0",
        "To ground your reference frame", "You do not need to identify",
        "If they are equal, prefer ", "Structural Drift. Reserve Unresolved",
        "default=300", "fewer_than_five_valid_stochastic_runs", "TIE_BREAK_ORDER",
        "run_categories", "category_counts", "verdict_resolution", "--ollama-host",
    )
    if any(token not in runner for token in required_runner) or runner.count("client.chat(") < 2 or runner.count('"num_ctx": JUDGE_NUM_CTX') < 2:
        fail("Judge runner does not meet the frozen context-window or call-site contract.")
    if "not args.resume" not in runner or ".exists()" not in runner:
        fail("Judge runner lacks the required fresh-run stale-incomplete-sidecar guard.")
    launcher = launcher_path.read_text(encoding="utf-8")
    required_launcher = ("203", "incomplete", "ollama")
    if any(token not in launcher.casefold() for token in required_launcher) or "[[ ! -e \"$RESULTS\" ]]" not in launcher or "[[ ! -e \"$INCOMPLETE\" ]]" not in launcher:
        fail("Judge launcher does not declare the 203-row and stale-sidecar safeguards.")
    resume_launcher = resume_launcher_path.read_text(encoding="utf-8")
    required_resume = ("--resume", "--ollama-host", "incomplete", "ollama", "203")
    if any(token not in resume_launcher.casefold() for token in required_resume):
        fail("Judge resume launcher does not declare the required resume safeguards.")
    if "[[ ! -s \"$RESULTS\" && ! -s \"$INCOMPLETE\" ]]" not in resume_launcher:
        fail("Judge resume launcher does not require prior results or an incomplete sidecar.")


def validate_calibration(path: Path) -> None:
    if not path.is_file():
        fail(f"Missing threshold calibration: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("selected_alarm_count_difference") != 13 or data.get("selected_alarm_rate_difference") != 0.13:
        fail("Calibration does not select the frozen 13-point alarm rule.")
    probability = float(data.get("worst_case_false_alarm_probability", np.nan))
    if not np.isclose(probability, 0.0384188161, rtol=0, atol=5e-11):
        fail("Calibration does not reproduce 0.0384188161 to eight decimal places.")
    if data.get("judge_outputs_used") is not False or data.get("threshold_selected_using_outcomes") is not False:
        fail("Calibration artifact indicates that Judge outcomes influenced the threshold.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all frozen Experiment A pre-Judge assertions.")
    parser.add_argument("--hosts", type=Path, default=HOSTS_PATH)
    parser.add_argument("--candidate-pool", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--selection-report", type=Path, default=SELECTION_REPORT_PATH)
    parser.add_argument("--donors", type=Path, default=DONORS_PATH)
    parser.add_argument("--variants", type=Path, default=VARIANTS_PATH)
    parser.add_argument("--diagnostics", type=Path, default=DIAGNOSTICS_PATH)
    parser.add_argument("--exemplars", type=Path, default=EXEMPLARS_PATH)
    parser.add_argument("--judge-input", type=Path, default=JUDGE_INPUT_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--baseline-pca", type=Path, default=BASELINE_PCA_PATH)
    parser.add_argument("--pca-model", type=Path, default=PCA_MODEL_PATH)
    parser.add_argument("--database", default=os.getenv("MIMIC4_DB_PATH"))
    parser.add_argument("--diagnostics-script", type=Path, default=DIAGNOSTICS_SCRIPT)
    parser.add_argument("--judge-runner", type=Path, default=JUDGE_RUNNER)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--resume-launcher", type=Path, default=RESUME_LAUNCHER)
    args = parser.parse_args()

    hosts = generation.load_hosts(args.hosts)
    sigmas = validate_candidates_and_selection(args.candidate_pool, args.selection_report, hosts)
    candidates = read_csv(args.candidate_pool, "candidate pool")
    if not args.database or not Path(args.database).is_file():
        fail("MIMIC4_DB_PATH is not set in .env or does not point to a file.")
    connection = generation.connect_read_only(args.database)
    try:
        notes = generation.load_and_verify_host_notes(connection, hosts)
        sources = generation.load_radiology_sources(connection)
    finally:
        connection.close()
    vocabulary = generation.host_top_vocabulary(hosts, notes)
    donors = read_csv(args.donors, "donor table")
    regenerated_donors, cutoff = generation.build_donor_table(sources, vocabulary)
    require_exact_frame(donors, regenerated_donors, "Donor table")
    stored_cutoffs = donors["overlap_cutoff"].dropna().astype(float).unique()
    if len(stored_cutoffs) != 1 or not np.isclose(stored_cutoffs[0], cutoff, rtol=0, atol=1e-15):
        fail("Saved donor cutoff does not reproduce the frozen type-7 quantile.")
    # Cutoff ties are defined by the archived table.  A CSV-rendered binary
    # float can differ from a fresh reconstruction in its final bit, so use
    # the saved value for the inclusive retained-set check after verifying the
    # two cutoff values above are numerically identical at archival precision.
    generation.validate_donor_table(donors, sources, vocabulary, float(stored_cutoffs[0]))
    variants = read_csv(args.variants, "variant manifest")
    regenerated_variants = generation.build_variants(hosts, notes, donors)
    require_exact_frame(variants, regenerated_variants, "Variant manifest")
    generation.validate_variants(variants, hosts, notes, donors)
    validate_diagnostics(
        args.diagnostics, variants, candidates, sigmas, args.baseline_pca, args.pca_model,
        args.diagnostics_script,
    )
    validate_calibration(args.calibration)
    validate_judge_input(args.exemplars, args.judge_input, variants)
    validate_runner_and_launcher(args.judge_runner, args.launcher, args.resume_launcher)
    LOG.info("All Experiment A pre-Judge assertions passed. Freeze these artifacts before launching Judge.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, generation.GenerationError) as error:
        LOG.error("Experiment A preflight stopped: %s", error)
        raise SystemExit(1)
