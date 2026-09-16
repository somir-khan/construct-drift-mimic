"""
judge_blind_run_struct_tie_break_frozen.py
Judge LLM — Semantic Drift Classification

For each discharge note selected by select_judge_samples.py, reads the full
note text from the samples CSV (already stored in the 'text' column) and calls
a local Ollama LLM (default: gemma4:26b) to classify the drift type. Results
are written row-by-row in append mode so interrupted runs can be resumed
without reprocessing completed notes.

Drift taxonomy:
    1. Structural Drift  — formatting and template changes
    2. Lexical Drift     — vocabulary and terminology differences
    3. Unresolved        — unexplained semantic distance; escalation signal

Usage:
    python judge_blind_run_struct_tie_break_frozen.py \\
        --samples-csv data/judge_samples_300.csv \\
        --results-csv data/judge_blind_struct_tie_break_results.csv \\
        --ollama-host http://127.0.0.1:<dynamically-selected-port>

This is the frozen blind Judge configuration. It uses the Structural Drift
tie-break only when structural and lexical evidence are equally persuasive
within a single model response. Across the five stochastic calls, a tied
modal vote is resolved in the fixed order Structural Drift, Lexical Drift,
then Unresolved; the temperature-0 call is diagnostic only.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import ollama
from ollama import Client
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_CATEGORIES = frozenset([
    "Structural Drift",
    "Lexical Drift",
    "Unresolved",
])
TIE_BREAK_ORDER = ("Structural Drift", "Lexical Drift", "Unresolved")
# Tokenizer-verified maximum Judge prompt: 23,921 tokens (three exemplars plus
# the longest audit note). This fixed context window leaves ample room for the
# required JSON response on the A100 80GB deployment.
JUDGE_NUM_CTX = 65536

RESULTS_FIELDNAMES = [
    "hadm_id",
    "anchor_year_group",
    "selection_group",
    "witness_score",
    "category",
    "confidence",
    "reasoning",
    "secondary_evidence",
    "consistency_rate",
    "n_valid_runs",
    "deterministic_category",
    "deterministic_reasoning",
    "deterministic_secondary_evidence",
    "stable",
    "run_categories",
    "category_counts",
    "verdict_resolution",
    "note_length",
    "error",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify drift type in selected discharge notes via a local LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-csv",
        default="data/judge_samples_300.csv",
        help="Frozen input CSV produced by select_judge_samples.py.",
    )
    parser.add_argument(
        "--results-csv",
        default="data/judge_blind_struct_tie_break_results.csv",
        help="New output CSV path. Existing files require --resume.",
    )
    parser.add_argument(
        "--incomplete-csv",
        default=None,
        help="Audit log for unresolved technical failures. Defaults to RESULTS_CSV.incomplete.csv.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an interrupted run produced by this script.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:26b",
        help="Ollama model identifier to use for classification.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API call attempts per note before recording an error.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of independent LLM calls per note; modal category is retained.",
    )
    parser.add_argument(
        "--call-timeout",
        type=int,
        default=300,
        help=(
            "Seconds to wait for a single Ollama API call before treating it "
            "as a timeout and retrying. Prevents the script from hanging "
            "indefinitely on a stalled inference. Default: 300s (5 minutes)."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        required=True,
        help="Ollama endpoint for this run; must be supplied explicitly by the launcher.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature. Must be > 0 for run-to-run variation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHI normalisation
# ---------------------------------------------------------------------------
def normalize_phi(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', 'unknown', text)
    text = re.sub(r'___', 'unknown', text)
    return text


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt(exemplar_block: str) -> str:
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
        + exemplar_block +
        "\n\nOutput ONLY a raw JSON object with exactly four keys:\n"
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

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _strip_fences(text):
    """Remove markdown code fences from an LLM response string."""
    stripped = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text.strip())
    return stripped.strip()


def _build_messages(note_text, system_prompt):
    """Construct the system + user message list for an Ollama chat call."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": note_text},
    ]


def _strip_thinking(text: str) -> str:
    # Gemma 4 official format: <|channel>thought\n...<channel|>
    text = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    # Fallback in case Ollama template wraps differently
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_response(raw_text):
    cleaned = _strip_thinking(raw_text)

    candidates = [
        cleaned.strip(),
        _strip_fences(cleaned),
    ]

    # Fallback: extract first { ... } block in case of preamble/trailing text
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        required = {"category", "confidence", "reasoning", "secondary_evidence"}
        if not required.issubset(parsed.keys()):
            if {"category", "confidence", "reasoning"}.issubset(parsed.keys()):
                parsed["secondary_evidence"] = "none"
            else:
                continue
        if parsed["category"] not in VALID_CATEGORIES:
            continue
        return parsed

    return None

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _single_classify_attempt(
    model, messages, max_retries, temperature, call_timeout=300,
    ollama_host=None,
):
    raw_response = ""
    client = Client(host=ollama_host, timeout=call_timeout)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_ctx": JUDGE_NUM_CTX},
            )
            raw_response = response.message.content

        except Exception as exc:
            log.warning("API error on attempt %d/%d: %s", attempt, max_retries, exc)
            continue

        parsed = _parse_response(raw_response)
        if parsed is not None:
            return parsed, ""

        log.warning("JSON parse failed on attempt %d/%d. Raw: %.120s ...",
                    attempt, max_retries, raw_response)

    return {"category": "", "confidence": "", "reasoning": "",
            "secondary_evidence": ""}, raw_response

def _deterministic_classify(
    model, messages, max_retries, call_timeout=300,
    ollama_host=None,
):
    """
    Run a single classification at temperature=0 for a deterministic
    reasoning field. Used after modal category is established from
    n_runs at temperature=0.7.

    Returns:
        tuple of (result_dict, error_str)
    """
    return _single_classify_attempt(
        model=model,
        messages=messages,
        max_retries=max_retries,
        temperature=0.0,
        call_timeout=call_timeout,
        ollama_host=ollama_host,
    )


def classify_note(
    model, messages, max_retries, n_runs=5, temperature=0.7,
    call_timeout=300, ollama_host=None,
):
    """Call the Ollama LLM n_runs times and return the modal classification.

    Args:
        model:        str, Ollama model identifier.
        messages:     list of dicts, from _build_messages().
        max_retries:  int, maximum API call attempts per run.
        n_runs:       int, number of independent LLM calls.
        temperature:  float, sampling temperature (>0 for variation).
        call_timeout: int, seconds before a single Ollama call is aborted.

    Returns:
        tuple of (result_dict, error_str) where result_dict has keys
        'category', 'confidence', 'reasoning', 'consistency_rate', 'n_valid_runs'.
    """
    all_results = []
    for _ in range(n_runs):
        result, error = _single_classify_attempt(
            model, messages, max_retries, temperature, call_timeout, ollama_host,
        )
        if result["category"]:
            all_results.append(result)

    run_categories = [r["category"] for r in all_results]
    if len(all_results) != n_runs:
        return {
            "category": "", "confidence": "", "reasoning": "",
            "secondary_evidence": "",
            "consistency_rate": 0.0, "n_valid_runs": len(all_results),
            "deterministic_category": "",
            "deterministic_reasoning": "",
            "deterministic_secondary_evidence": "",
            "stable": None,
            "run_categories": json.dumps(run_categories),
            "category_counts": json.dumps(dict(Counter(run_categories)), sort_keys=True),
            "verdict_resolution": "",
        }, "fewer_than_five_valid_stochastic_runs"

    # The five stochastic calls determine the verdict. A tied modal vote is
    # resolved by a fixed pre-specified category order; temperature 0 remains
    # a diagnostic for every note.
    category_counts = Counter(r["category"] for r in all_results)
    max_count = max(category_counts.values())
    modal_categories = sorted(
        category for category, count in category_counts.items() if count == max_count
    )

    det_result, _ = _deterministic_classify(
        model, messages, max_retries, call_timeout, ollama_host,
    )
    det_category = det_result.get("category", "")
    det_reasoning = det_result.get("reasoning", "")
    det_secondary = det_result.get("secondary_evidence", "none")

    if len(modal_categories) == 1:
        modal_category = modal_categories[0]
        verdict_resolution = "unique_modal"
    else:
        modal_category = min(modal_categories, key=TIE_BREAK_ORDER.index)
        verdict_resolution = "stochastic_tie_break"
    stable = (det_category == modal_category) if det_category else None

    consistency     = category_counts[modal_category] / n_runs

    # Reasoning from highest-confidence modal run (temp=0.7)
    conf_order = {"High": 3, "Medium": 2, "Low": 1}
    modal_results = [r for r in all_results if r["category"] == modal_category]
    best_result   = max(modal_results, key=lambda r: conf_order.get(r["confidence"], 0))

    return {
        "category":                        best_result["category"],
        "confidence":                      best_result["confidence"],
        "reasoning":                       best_result["reasoning"],
        "secondary_evidence":              best_result.get("secondary_evidence", "none"),
        "consistency_rate":                round(consistency, 3),
        "n_valid_runs":                    len(all_results),
        "deterministic_category":          det_category,
        "deterministic_reasoning":         det_reasoning,
        "deterministic_secondary_evidence": det_secondary,
        "stable":                          stable,
        "run_categories":                  json.dumps(run_categories),
        "category_counts":                 json.dumps(dict(category_counts), sort_keys=True),
        "verdict_resolution":              verdict_resolution,
    }, ""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_samples(samples_csv):
    """Load the judge_samples.csv produced by select_judge_samples.py."""
    if not os.path.exists(samples_csv):
        log.error("Samples file not found: %s", samples_csv)
        sys.exit(1)

    df = pd.read_csv(samples_csv)

    if len(df) == 0:
        log.error("Samples file is empty: %s", samples_csv)
        sys.exit(1)

    required_cols = {"hadm_id", "witness_score", "selection_group", "anchor_year_group", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error("Samples file missing required columns: %s", missing)
        sys.exit(1)

    return df


def validate_frozen_manifest(df_samples):
    """Fail fast unless the input is the frozen 300-note audit manifest."""
    valid_groups = {"exemplar", "top", "bottom", "random"}
    unexpected = set(df_samples["selection_group"].dropna()) - valid_groups
    if unexpected:
        sys.exit("Unexpected selection_group value(s): " + ", ".join(sorted(unexpected)))

    exemplars = df_samples[df_samples["selection_group"] == "exemplar"].copy()
    audit = df_samples[df_samples["selection_group"].isin({"top", "bottom", "random"})].copy()
    nonempty_text = (
        df_samples["text"].notna()
        & df_samples["text"].astype(str).str.strip().ne("")
    )
    if len(exemplars) != 3 or not nonempty_text.loc[exemplars.index].all():
        sys.exit("Frozen Judge run requires exactly three nonempty baseline exemplars.")
    if (
        len(audit) != 300
        or audit["hadm_id"].duplicated().any()
        or not nonempty_text.loc[audit.index].all()
    ):
        sys.exit("Frozen Judge run requires 300 unique top/random/bottom audit notes with nonempty text.")

    expected_cohorts = {"2014 - 2016", "2017 - 2019"}
    observed_cohorts = set(audit["anchor_year_group"].astype(str).str.strip())
    if observed_cohorts != expected_cohorts:
        sys.exit("Manifest does not contain the expected two anchor-year-group cohorts.")

    counts = audit.groupby(["anchor_year_group", "selection_group"]).size()
    for cohort in expected_cohorts:
        for arm in ("top", "bottom", "random"):
            if int(counts.get((cohort, arm), 0)) != 50:
                sys.exit(f"Manifest requires 50 {arm} notes in cohort {cohort}.")
    return exemplars, audit


def write_incomplete_record(path, row, result, error_code):
    """Keep an audit trail without placing incomplete rows in the results file."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "hadm_id": row.hadm_id,
            "anchor_year_group": getattr(row, "anchor_year_group", ""),
            "selection_group": row.selection_group,
            "witness_score": row.witness_score,
            "category": result.get("category", ""),
            "confidence": result.get("confidence", ""),
            "reasoning": result.get("reasoning", ""),
            "secondary_evidence": result.get("secondary_evidence", ""),
            "consistency_rate": result.get("consistency_rate", ""),
            "n_valid_runs": result.get("n_valid_runs", ""),
            "deterministic_category": result.get("deterministic_category", ""),
            "deterministic_reasoning": result.get("deterministic_reasoning", ""),
            "deterministic_secondary_evidence": result.get("deterministic_secondary_evidence", ""),
            "stable": result.get("stable", ""),
            "run_categories": result.get("run_categories", ""),
            "category_counts": result.get("category_counts", ""),
            "verdict_resolution": result.get("verdict_resolution", ""),
            "note_length": result.get("note_length", 0),
            "error": error_code,
        })


def validate_frozen_configuration(args):
    """Prevent accidental changes to the frozen primary Judge settings."""
    if args.model != "gemma4:26b":
        sys.exit("Frozen primary Judge run requires --model gemma4:26b.")
    if args.n_runs != 5:
        sys.exit("Frozen primary Judge run requires exactly five stochastic runs.")
    if args.temperature != 0.7:
        sys.exit("Frozen primary Judge run requires --temperature 0.7.")
    if args.max_retries != 3:
        sys.exit("Frozen primary Judge run requires --max-retries 3.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Orchestrate LLM drift classification for all selected discharge notes."""
    args = parse_args()
    validate_frozen_configuration(args)

    # --- Stage 1: Load Inputs -----------------------------------------------
    log.info("--- Stage 1: Load Judge Samples ---")
    df_samples = load_samples(args.samples_csv)
    log.info("%d samples loaded from %s", len(df_samples), args.samples_csv)

    df_exemplars, df_classify = validate_frozen_manifest(df_samples)
    parts = []
    for i, (_, erow) in enumerate(df_exemplars.iterrows(), start=1):
        text_snippet = str(erow.get("text", ""))
        parts.append(
            f"--- MIMIC-III Baseline Exemplar {i} (hadm_id={erow['hadm_id']}) ---\n"
            f"{text_snippet}"
        )
    exemplar_block = (
        "\n\nTo ground your reference frame, the following are three discharge notes "
        "representative of the MIMIC-III (2001-2012) documentation era. These are "
        "provided as baseline context only — do not classify them:\n\n"
        + "\n\n".join(parts)
    )
    log.info("Validated frozen manifest and loaded %d baseline exemplars.", len(df_exemplars))

    # Generate the single, static system prompt
    system_prompt = build_system_prompt(exemplar_block)
    log.info("System prompt built.")

    # --- Stage 2: Resume Check ----------------------------------------------
    log.info("--- Stage 2: Resume Check ---")
    done_ids = set()
    if os.path.exists(args.results_csv) and os.path.getsize(args.results_csv) > 0:
        if not args.resume:
            sys.exit(
                f"Results file already exists: {args.results_csv}. Use a new path or --resume."
            )
        df_done = pd.read_csv(args.results_csv)
        required_output = {"hadm_id", "category", "n_valid_runs", "verdict_resolution", "error"}
        missing_output = required_output - set(df_done.columns)
        if missing_output:
            sys.exit("Existing results file is not compatible with this frozen script: "
                     + ", ".join(sorted(missing_output)))
        complete = (
            df_done["error"].fillna("").eq("")
            & df_done["category"].isin(VALID_CATEGORIES)
            & pd.to_numeric(df_done["n_valid_runs"], errors="coerce").eq(args.n_runs)
            & df_done["verdict_resolution"].isin(
                {"unique_modal", "stochastic_tie_break"}
            )
        )
        if not complete.all() or df_done["hadm_id"].duplicated().any():
            sys.exit("Existing results contain incomplete or duplicate rows; use a new output path.")
        done_ids = set(df_done["hadm_id"].astype("int64").tolist())
        manifest_ids = set(df_classify["hadm_id"].astype("int64").tolist())
        if not done_ids <= manifest_ids:
            sys.exit("Existing results contain hadm_id values outside the frozen manifest.")
        log.info("%d completed notes found — resuming.", len(done_ids))

    df_todo = df_classify[~df_classify["hadm_id"].isin(done_ids)].reset_index(drop=True)
    log.info("%d notes remaining to classify.", len(df_todo))

    if len(df_todo) == 0:
        log.info("Nothing to do — all notes already classified.")
        sys.exit(0)

    # --- Stage 3: Validate Ollama Connection --------------------------------
    log.info("--- Stage 3: Validate Ollama Connection ---")
    try:
        _ping_client = Client(host=args.ollama_host, timeout=args.call_timeout)
        _ping_client.chat(
            model=args.model,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_ctx": JUDGE_NUM_CTX},
        )
        log.info("Ollama connection OK (model: %s)", args.model)
    except Exception as exc:
        log.error(
            "Cannot reach Ollama with model '%s': %s\n"
            "  Ensure `ollama serve` is running and the model is pulled.",
            args.model, exc,
        )
        sys.exit(1)

    # --- Stage 4: Classify Notes --------------------------------------------
    log.info("--- Stage 4: Classify Notes ---")
    write_header = not os.path.exists(args.results_csv) or os.path.getsize(args.results_csv) == 0
    incomplete_csv = args.incomplete_csv or f"{args.results_csv}.incomplete.csv"

    out_dir = os.path.dirname(args.results_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.results_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(
            tqdm(df_todo.itertuples(), desc="Classifying notes", unit="note", total=len(df_todo))
        ):
            note_text = getattr(row, "text", None)
            if note_text is None or (isinstance(note_text, float) and pd.isna(note_text)):
                note_text = None

            if note_text is None:
                write_incomplete_record(incomplete_csv, row, {"note_length": 0}, "note_not_found")
                sys.exit(f"Audit incomplete: hadm_id={row.hadm_id} has no note text.")

            note_text = str(note_text)
            note_text = normalize_phi(note_text)
            messages  = _build_messages(note_text, system_prompt)

            try:
                result, error_str = classify_note(
                    args.model, messages, args.max_retries,
                    n_runs=args.n_runs, temperature=args.temperature,
                    call_timeout=args.call_timeout,
                    ollama_host=args.ollama_host,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Fatal Ollama connection failure at hadm_id=%d: %s — aborting.",
                    row.hadm_id, exc,
                )
                sys.exit(1)

            if error_str:
                result["note_length"] = len(note_text)
                write_incomplete_record(incomplete_csv, row, result, error_str)
                sys.exit(
                    f"Audit incomplete: hadm_id={row.hadm_id} ({error_str}). "
                    f"Details written to {incomplete_csv}."
                )

            writer.writerow({
                "hadm_id":                            row.hadm_id,
                "anchor_year_group":                  getattr(row, "anchor_year_group", ""),
                "selection_group":                    row.selection_group,
                "witness_score":                      row.witness_score,
                "category":                           result["category"],
                "confidence":                         result["confidence"],
                "reasoning":                          result["reasoning"],
                "secondary_evidence":                 result.get("secondary_evidence", "none"),
                "consistency_rate":                   result.get("consistency_rate", ""),
                "n_valid_runs":                       result.get("n_valid_runs", ""),
                "deterministic_category":             result.get("deterministic_category", ""),
                "deterministic_reasoning":            result.get("deterministic_reasoning", ""),
                "deterministic_secondary_evidence":   result.get("deterministic_secondary_evidence", "none"),
                "stable":                             result.get("stable", ""),
                "run_categories":                    result.get("run_categories", ""),
                "category_counts":                   result.get("category_counts", ""),
                "verdict_resolution":                result.get("verdict_resolution", ""),
                "note_length":                        len(note_text),
                "error":                              error_str,
            })
            fh.flush()

            log.info(
                "[%d/%d] hadm_id=%-10d  group=%-8s  category=%s  consistency=%.3f",
                idx + 1, len(df_todo),
                row.hadm_id, row.selection_group,
                result["category"] or "(parse error)",
                result.get("consistency_rate", 0.0) or 0.0,
            )

    # --- Stage 5: Final Summary Banner --------------------------------------
    log.info("--- Stage 5: Summary ---")
    df_results = pd.read_csv(args.results_csv)
    n_total   = len(df_results)
    n_success = int((df_results["error"].fillna("") == "").sum())
    n_errors  = n_total - n_success
    if n_total != len(df_classify) or n_success != len(df_classify):
        sys.exit("Audit incomplete: results do not contain one completed row per audit note.")

    log.info("=" * 55)
    log.info("  JUDGE LLM CLASSIFICATION COMPLETE")
    log.info("  Total rows in results  : %d", n_total)
    log.info("  Successful             : %d", n_success)
    log.info("  Errors                 : %d", n_errors)
    log.info("  Output                 : %s", args.results_csv)
    log.info("=" * 55)

    if n_success > 0:
        df_ok = df_results[df_results["error"].fillna("") == ""]

        log.info("Category breakdown by selection group:")
        log.info("\n%s", pd.crosstab(df_ok["selection_group"], df_ok["category"]).to_string())

        if "anchor_year_group" in df_ok.columns:
            log.info("Category breakdown by temporal window:")
            log.info(
                "\n%s",
                pd.crosstab(df_ok["anchor_year_group"], df_ok["category"]).to_string(),
            )

        if "consistency_rate" in df_ok.columns:
            cr = pd.to_numeric(df_ok["consistency_rate"], errors="coerce").dropna()
            if len(cr) > 0:
                log.info(
                    "Consistency rate: mean=%.3f  min=%.3f  max=%.3f",
                    cr.mean(), cr.min(), cr.max(),
                )

        if "stable" in df_ok.columns:
            stable_vals = pd.to_numeric(df_ok["stable"], errors="coerce").dropna()
            if len(stable_vals) > 0:
                n_stable   = int(stable_vals.sum())
                n_unstable = int((stable_vals == False).sum())
                log.info(
                    "Stability (temp-0 agrees with modal): %d stable, "
                    "%d unstable, %d indeterminate",
                    n_stable, n_unstable,
                    len(df_ok) - n_stable - n_unstable,
                )


if __name__ == "__main__":
    main()
