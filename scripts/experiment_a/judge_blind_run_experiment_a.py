#!/usr/bin/env python3
"""Restricted blind Judge runner for frozen Experiment A.

The Judge receives only a normalized note and the fixed exemplar prompt.  Arm,
cohort, host, donor, and witness metadata are used only for local integrity
checks and are never placed in a chat message.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from ollama import Client


INPUT_PATH = Path("data/experiment_a_judge_input.csv")
EXEMPLARS_PATH = Path("data/experiment_a_exemplars.csv")
VARIANTS_PATH = Path("data/experiment_a_variants.csv")
RESULTS_PATH = Path("data/experiment_a_results.csv")
INCOMPLETE_PATH = Path("data/experiment_a_incomplete.csv")

MODEL = "gemma4:26b"
JUDGE_NUM_CTX = 65536
N_STOCHASTIC_CALLS = 5
STOCHASTIC_TEMPERATURE = 0.7
MAX_RETRIES = 3
VALID_CATEGORIES = ("Structural Drift", "Lexical Drift", "Unresolved")
TIE_BREAK_ORDER = ("Structural Drift", "Lexical Drift", "Unresolved")
CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}
PHI = re.compile(r"\[\*\*.*?\*\*\]")

RESULT_COLUMNS = [
    "record_id", "group", "category", "confidence", "reasoning", "secondary_evidence",
    "consistency_rate", "n_valid_runs", "run_categories", "category_counts", "verdict_resolution",
    "deterministic_category", "deterministic_reasoning", "deterministic_secondary_evidence",
    "deterministic_error", "stable",
]
INCOMPLETE_COLUMNS = [
    "record_id", "group", "n_valid_runs", "run_categories", "category_counts", "error",
    "raw_responses_json",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("judge_blind_run_experiment_a")


class JudgeRunError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise JudgeRunError(message)


def normalize(text: str) -> str:
    return PHI.sub("unknown", text).replace("___", "unknown")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def atomic_csv(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_inputs(input_path: Path, exemplars_path: Path, variants_path: Path) -> tuple[pd.DataFrame, str]:
    if not input_path.is_file() or not exemplars_path.is_file() or not variants_path.is_file():
        fail("Missing Experiment A Judge input, exemplars, or frozen variant manifest.")
    judge_input = pd.read_csv(input_path)
    if set(judge_input.columns) != {"record_id", "group", "note_text"}:
        fail("Judge input must contain only record_id, group, and note_text.")
    if len(judge_input) != 203 or judge_input["record_id"].duplicated().any():
        fail("Judge input must contain exactly 203 unique records.")
    if judge_input["note_text"].isna().any() or judge_input["note_text"].astype(str).eq("").any():
        fail("Judge input contains an empty note text.")
    expected_groups = {"exemplar": 3, "top_positive": 100, "random_structural_negative": 100}
    if judge_input["group"].value_counts().to_dict() != expected_groups:
        fail("Judge input group counts are not 3 exemplars and 100 notes per arm.")
    if not (judge_input["note_text"].astype(str).map(normalize) == judge_input["note_text"].astype(str)).all():
        fail("Judge input is not already normalized with the frozen PHI convention.")

    exemplars = pd.read_csv(exemplars_path)
    require_columns(exemplars, {"record_id", "hadm_id", "normalized_text"}, "frozen exemplars")
    if len(exemplars) != 3 or exemplars["record_id"].duplicated().any():
        fail("Exemplar manifest must contain exactly three unique records.")
    exemplar_rows = judge_input.loc[judge_input["group"] == "exemplar"]
    exemplar_check = exemplar_rows.merge(
        exemplars[["record_id", "hadm_id", "normalized_text"]], on="record_id", how="inner", validate="one_to_one"
    )
    if len(exemplar_check) != 3 or not (exemplar_check["note_text"] == exemplar_check["normalized_text"]).all():
        fail("Judge-input exemplar text is not byte-identical to the frozen exemplars.")

    variants = pd.read_csv(variants_path)
    require_columns(variants, {"variant_id", "cohort", "arm", "variant_text"}, "frozen variant manifest")
    if len(variants) != 200 or variants["variant_id"].duplicated().any():
        fail("Variant manifest must contain exactly 200 unique variants.")
    expected_cells = {
        (cohort, arm): 50
        for cohort in ("2014 - 2016", "2017 - 2019")
        for arm in ("top_positive", "random_structural_negative")
    }
    if variants.groupby(["cohort", "arm"]).size().to_dict() != expected_cells:
        fail("Variant manifest does not contain 50 notes in every cohort-by-arm cell.")
    classifications = judge_input.loc[judge_input["group"] != "exemplar"]
    expected_variants = variants[["variant_id", "arm", "variant_text"]].rename(
        columns={"variant_id": "record_id", "arm": "group", "variant_text": "expected_text"}
    )
    comparison = classifications.merge(expected_variants, on=["record_id", "group"], how="inner", validate="one_to_one")
    if len(comparison) != 200 or not (comparison["note_text"] == comparison["expected_text"]).all():
        fail("Judge input does not contain every frozen variant exactly once and byte-identically.")

    exemplar_parts = [
        f"--- MIMIC-III Baseline Exemplar {number} (hadm_id={int(row.hadm_id)}) ---\n{row.note_text}"
        for number, row in enumerate(exemplar_check.itertuples(index=False), start=1)
    ]
    exemplar_block = (
        "\n\nTo ground your reference frame, the following are three discharge notes "
        "representative of the MIMIC-III (2001-2012) documentation era. These are "
        "provided as baseline context only — do not classify them:\n\n"
        + "\n\n".join(exemplar_parts)
    )
    return classifications.reset_index(drop=True), exemplar_block


def build_system_prompt(exemplar_block: str) -> str:
    """Original frozen prompt, with only Experiment A's prespecified tie order."""
    return (
        "<|think|>\n"
        "You are a clinical NLP auditor evaluating why a hospital discharge "
        "note feels distributionally distant from a reference era (MIMIC-III, "
        "2001-2012). Distributional drift has already been detected statistically. "
        "Your job is attribution — explaining what observable properties of the "
        "note account for that distance.\n\n"
        "Classify the note into exactly one of these three categories:\n\n"
        "1. \"Structural Drift\" — The note's distance from the baseline is "
        "primarily explained by formatting and template changes: standardized "
        "section headers, altered document structure, length expansion consistent "
        "with template adoption.\n\n"
        "2. \"Lexical Drift\" — The note's distance from the baseline is primarily "
        "explained by vocabulary and terminology differences: unfamiliar clinical "
        "terms, changed phrasing conventions, new abbreviations. You do not need to identify "
        "why vocabulary changed — only that it has.\n\n"
        "3. \"Unresolved\" — The note feels semantically distant from the baseline "
        "exemplars, but neither structural nor lexical patterns adequately explain "
        "that distance. Flag specifically what you observe that cannot be attributed "
        "to the known surface patterns. This is an escalation signal.\n\n"
        "TIE-BREAKING RULE: If the note shows both structural and lexical signals, "
        "classify by whichever is more prominent. If they are equal, prefer "
        "Structural Drift. Reserve Unresolved strictly for notes where neither "
        "category provides an adequate explanation.\n\n"
        "LEXICAL CHECK — before finalising any classification, explicitly scan "
        "the note for: unfamiliar clinical terminology, new abbreviations, changed "
        "diagnostic phrasing, or vocabulary that would not appear in a 2001-2012 "
        "MIMIC-III note. Record any such signals in the secondary_evidence field "
        "even if your primary classification is Structural Drift. If no lexical "
        "divergence is detected, set secondary_evidence to 'none'.\n\n"
        "IMPORTANT: Both MIMIC-III and MIMIC-IV PHI placeholders have been "
        "normalized to the token 'unknown'. Treat this as a corpus artifact, "
        "not a drift signal.\n"
        + exemplar_block
        + "\n\nOutput ONLY a raw JSON object with exactly four keys:\n"
        "- \"category\": one of [\"Structural Drift\", \"Lexical Drift\", \"Unresolved\"]\n"
        "- \"confidence\": one of [\"High\", \"Medium\", \"Low\"]\n"
        "- \"reasoning\": 1-2 sentences explaining your primary classification, citing\n"
        "  specific evidence from the note text\n"
        "- \"secondary_evidence\": 1-2 sentences describing any lexical signals observed\n"
        "  (unfamiliar terms, new abbreviations, changed phrasing), or the string 'none'\n"
        "  if no lexical divergence is detected\n\n"
        "Do not output any preamble, markdown code fences, or text outside the JSON\n"
        "object."
    )


def build_messages(note_text: str, system_prompt: str) -> list[dict[str, str]]:
    # The only content submitted to the Judge is this fixed prompt and note text.
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": note_text}]


def parse_response(raw: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    candidates = [cleaned, re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            result = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        required = {"category", "confidence", "reasoning", "secondary_evidence"}
        if not required.issubset(result):
            if {"category", "confidence", "reasoning"}.issubset(result):
                result["secondary_evidence"] = "none"
            else:
                continue
        if result["category"] not in VALID_CATEGORIES:
            continue
        return result
    return None


def chat_once(
    client: Client, model: str, messages: list[dict[str, str]], temperature: float, max_retries: int
) -> tuple[dict[str, str] | None, list[str]]:
    raw_responses: list[str] = []
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_ctx": JUDGE_NUM_CTX},
            )
            raw = str(response.message.content)
            raw_responses.append(raw)
        except Exception as exc:  # noqa: BLE001
            raw_responses.append(f"[client error] {exc}")
            LOG.warning("Judge call failed on attempt %d/%d: %s", attempt, max_retries, exc)
            continue
        parsed = parse_response(raw)
        if parsed is not None:
            return parsed, raw_responses
        LOG.warning("Judge response failed JSON/category validation on attempt %d/%d.", attempt, max_retries)
    return None, raw_responses


def classify_note(client: Client, model: str, messages: list[dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    stochastic, raw = [], []
    for _ in range(N_STOCHASTIC_CALLS):
        result, attempts = chat_once(client, model, messages, STOCHASTIC_TEMPERATURE, MAX_RETRIES)
        raw.extend(attempts)
        if result is not None:
            stochastic.append(result)
    run_categories = [item["category"] for item in stochastic]
    category_counts = Counter(run_categories)
    if len(stochastic) != N_STOCHASTIC_CALLS:
        return None, {
            "n_valid_runs": len(stochastic), "run_categories": json.dumps(run_categories),
            "category_counts": json.dumps(dict(category_counts), sort_keys=True),
            "raw_responses": raw, "error": "fewer_than_five_valid_stochastic_runs",
        }
    max_count = max(category_counts.values())
    modal_categories = sorted(category for category, count in category_counts.items() if count == max_count)
    if len(modal_categories) == 1:
        modal, verdict_resolution = modal_categories[0], "unique_modal"
    else:
        modal, verdict_resolution = min(modal_categories, key=TIE_BREAK_ORDER.index), "stochastic_tie_break"
    modal_runs = [result for result in stochastic if result["category"] == modal]
    best = max(modal_runs, key=lambda result: CONFIDENCE_ORDER.get(result["confidence"], 0))
    deterministic, diagnostic_raw = chat_once(client, model, messages, 0.0, MAX_RETRIES)
    diagnostic_error = "" if deterministic is not None else "temperature_0_diagnostic_failed"
    result = {
        "category": modal, "confidence": best["confidence"], "reasoning": best["reasoning"],
        "secondary_evidence": best["secondary_evidence"], "consistency_rate": round(category_counts[modal] / N_STOCHASTIC_CALLS, 3),
        "n_valid_runs": N_STOCHASTIC_CALLS,
        "run_categories": json.dumps(run_categories),
        "category_counts": json.dumps(dict(category_counts), sort_keys=True),
        "verdict_resolution": verdict_resolution,
        "deterministic_category": "" if deterministic is None else deterministic["category"],
        "deterministic_reasoning": "" if deterministic is None else deterministic["reasoning"],
        "deterministic_secondary_evidence": "" if deterministic is None else deterministic["secondary_evidence"],
        "deterministic_error": diagnostic_error,
        "stable": "" if deterministic is None else deterministic["category"] == modal,
    }
    return result, {"n_valid_runs": N_STOCHASTIC_CALLS, "raw_responses": raw + diagnostic_raw, "error": ""}


def load_existing(path: Path, columns: list[str], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    require_columns(frame, set(columns), label)
    return frame[columns].to_dict("records")


def assert_source_conformance() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    if JUDGE_NUM_CTX != 65536:
        fail("Frozen Judge context window must equal 65536.")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "chat"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "client"
    ]
    def has_frozen_context(call: ast.Call) -> bool:
        options = next((keyword.value for keyword in call.keywords if keyword.arg == "options"), None)
        return isinstance(options, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "num_ctx"
            and isinstance(value, ast.Name) and value.id == "JUDGE_NUM_CTX"
            for key, value in zip(options.keys, options.values, strict=True)
        )
    if len(calls) != 2 or not all(has_frozen_context(call) for call in calls):
        fail("Both Judge client.chat call sites must explicitly pass num_ctx=JUDGE_NUM_CTX.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen, blind Experiment A Judge protocol.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--exemplars", type=Path, default=EXEMPLARS_PATH)
    parser.add_argument("--variants", type=Path, default=VARIANTS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--incomplete", type=Path, default=INCOMPLETE_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ollama-host", required=True, help="Explicit job-local Ollama endpoint from the launcher.")
    parser.add_argument("--call-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Resume only from the selected incomplete sidecar.")
    args = parser.parse_args()

    assert_source_conformance()
    if args.model != MODEL:
        fail(f"Frozen Judge model is {MODEL}; --model may not change it.")
    if not args.resume and (args.results.exists() or args.incomplete.exists()):
        fail("Fresh run refused: results or incomplete sidecar already exists. Use --resume only for the selected sidecar.")
    classifications, exemplar_block = load_inputs(args.input, args.exemplars, args.variants)
    existing_results = load_existing(args.results, RESULT_COLUMNS, "results") if args.resume else []
    existing_incomplete = load_existing(args.incomplete, INCOMPLETE_COLUMNS, "incomplete sidecar") if args.resume else []
    completed = {str(row["record_id"]) for row in existing_results}
    expected_ids = set(classifications["record_id"].astype(str))
    if not completed.issubset(expected_ids) or len(completed) != len(existing_results):
        fail("Existing results contain invalid or duplicate Experiment A record IDs.")
    if completed & {str(row["record_id"]) for row in existing_incomplete}:
        fail("A record may not appear in both results and the incomplete sidecar.")
    system_prompt = build_system_prompt(exemplar_block)
    client = Client(host=args.ollama_host, timeout=args.call_timeout)
    try:
        # Connectivity ping: num_ctx=JUDGE_NUM_CTX is explicitly frozen here.
        client.chat(
            model=args.model,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_ctx": JUDGE_NUM_CTX},
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"Judge connectivity/model ping failed: {exc}")

    results = existing_results[:]
    incomplete = [row for row in existing_incomplete if str(row["record_id"]) not in completed]
    for row in classifications.itertuples(index=False):
        record_id = str(row.record_id)
        if record_id in completed:
            continue
        messages = build_messages(normalize(str(row.note_text)), system_prompt)
        result, trace = classify_note(client, args.model, messages)
        incomplete = [item for item in incomplete if str(item["record_id"]) != record_id]
        if result is None:
            incomplete.append({
                "record_id": record_id, "group": row.group, "n_valid_runs": trace["n_valid_runs"],
                "run_categories": trace["run_categories"], "category_counts": trace["category_counts"], "error": trace["error"],
                "raw_responses_json": json.dumps(trace["raw_responses"]),
            })
            atomic_csv(incomplete, INCOMPLETE_COLUMNS, args.incomplete)
            LOG.error("%s incomplete after %d valid stochastic calls.", record_id, trace["n_valid_runs"])
            fail(f"Run incomplete at {record_id}; details are in {args.incomplete}. Rerun with --resume.")
        results.append({"record_id": record_id, "group": row.group} | result)
        completed.add(record_id)
        atomic_csv(results, RESULT_COLUMNS, args.results)
        if incomplete:
            atomic_csv(incomplete, INCOMPLETE_COLUMNS, args.incomplete)
        LOG.info("%s: %s (%.1f%% consistency)", record_id, result["category"], 100 * result["consistency_rate"])

    if incomplete:
        atomic_csv(incomplete, INCOMPLETE_COLUMNS, args.incomplete)
        fail(f"Run incomplete: {len(incomplete)} note(s) remain in {args.incomplete}; rerun with --resume.")
    if len(results) != 200 or {row["group"] for row in results} != {"top_positive", "random_structural_negative"}:
        fail("Judge results are not exactly 200 complete Experiment A classifications.")
    if Counter(row["group"] for row in results) != Counter({"top_positive": 100, "random_structural_negative": 100}):
        fail("Judge results do not contain 100 complete outcomes per arm.")
    LOG.info("Completed 200 frozen blind Judge classifications: %s", args.results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JudgeRunError as error:
        LOG.error("Experiment A Judge run stopped: %s", error)
        raise SystemExit(1)
