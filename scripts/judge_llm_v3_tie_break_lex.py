"""
scripts/judge_llm_v2.py
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
    python scripts/judge_llm_v2.py
    python scripts/judge_llm_v2.py --model gemma4:26b
    python scripts/judge_llm_v2.py \\
        --samples-csv data/judge_samples.csv \\
        --results-csv data/judge_results_v2.csv

v2 changes: adds secondary_evidence field to capture lexical signals
even when primary classification is Structural Drift. Prompt updated
to force explicit lexical checking before finalising classification.

v3 changes: tie-breaking rule reversed — equal structural and lexical
signals now resolve to Lexical Drift (v2 preferred Structural Drift).
Default results file is judge_results_v3.csv. All other logic identical.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
        default="data/judge_samples.csv",
        help="Input CSV produced by select_judge_samples.py.",
    )
    parser.add_argument(
        "--results-csv",
        default="data/judge_results_v3.csv",
        help="Output CSV path (opened in append mode for resumability).",
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
        "--surface-stats-csv",
        default="outputs/surface_features_by_period.csv",
        help=(
            "CSV produced by surface_features.py. Used to inject window-level "
            "surface statistics into the Judge LLM system prompt. If the file "
            "does not exist, the prompt will note that statistics are unavailable."
        ),
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
        "--baseline-label",
        default="MIMIC-III",
        help=(
            "period_label value in surface_features_by_period.csv that identifies "
            "the MIMIC-III baseline row. Used to read baseline section_rate and "
            "mean_length dynamically rather than relying on hardcoded constants."
        ),
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
def build_surface_context(
    anchor_year_group: str,
    surface_stats: dict,
    baseline_stats: dict | None = None,
) -> str:
    """
    Format window-level surface feature statistics for injection into
    the Judge LLM system prompt.

    Args:
        anchor_year_group: str, e.g. '2017 - 2019'
        surface_stats:     dict with keys matching F1-F5 feature names for
                           the target window. If empty or None, returns a
                           fallback string.
        baseline_stats:    dict for the MIMIC-III baseline row, read from
                           surface_features_by_period.csv. Used to populate
                           the (baseline: ...) annotations. Falls back to
                           'N/A' if not supplied.

    Returns:
        Formatted string block for insertion into the system prompt.
    """
    if not surface_stats:
        return (
            f"  Window: {anchor_year_group}\n"
            "  (Surface feature statistics not available for this window.)"
        )

    section_rate    = surface_stats.get("section_rate", "N/A")
    jaccard         = surface_stats.get("jaccard", "N/A")
    mean_length     = surface_stats.get("mean_length", "N/A")
    numeric_density = surface_stats.get("numeric_density", "N/A")

    bl = baseline_stats or {}
    bl_section_rate = bl.get("section_rate", "N/A")
    bl_jaccard      = bl.get("jaccard", "N/A")
    bl_mean_length  = bl.get("mean_length", "N/A")

    def fmt(v, decimals=3):
        return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

    return (
        f"  Window: {anchor_year_group}\n"
        f"  Structured section rate : {fmt(section_rate)} "
        f"(baseline: {fmt(bl_section_rate)})\n"
        f"  Vocabulary Jaccard      : {fmt(jaccard)} "
        f"(baseline: {fmt(bl_jaccard)})\n"
        f"  Mean note length (chars): {fmt(mean_length, 0)} "
        f"(baseline: {fmt(bl_mean_length, 0)})\n"
        f"  Numeric token density   : {fmt(numeric_density)} "
    )


def build_system_prompt(exemplar_block: str, surface_context: str = "") -> str:
    return (
        "<|think|>\n"
        "You are a clinical NLP auditor evaluating why a hospital discharge "
        "note feels distributionally distant from a reference era (MIMIC-III, "
        "2001-2012). Distributional drift has already been detected statistically. "
        "Your job is attribution — explaining what observable properties of the "
        "note account for that distance.\n\n"
        "WINDOW CONTEXT (aggregate statistics for the temporal window this note "
        "was drawn from):\n"
        + surface_context +
        "\n\nUse these statistics as background context when interpreting the "
        "note below. They describe the window, not this specific note. Your "
        "classification must be grounded in what you observe in the note text "
        "itself. Do not simply echo the window statistics back as your reasoning.\n\n"
        "Classify the note into exactly one of these three categories:\n\n"
        "1. \"Structural Drift\" — The note's distance from the baseline is "
        "primarily explained by formatting and template changes: standardized "
        "section headers, altered document structure, length expansion consistent "
        "with template adoption. The window-level section rate and length "
        "statistics support this explanation.\n\n"
        "2. \"Lexical Drift\" — The note's distance from the baseline is primarily "
        "explained by vocabulary and terminology differences: unfamiliar clinical "
        "terms, changed phrasing conventions, new abbreviations. The window-level "
        "Jaccard drop supports this explanation. You do not need to identify why "
        "vocabulary changed — only that it has.\n\n"
        "3. \"Unresolved\" — The note feels semantically distant from the baseline "
        "exemplars, but neither structural nor lexical patterns adequately explain "
        "that distance. Flag specifically what you observe that cannot be attributed "
        "to the known surface patterns. This is an escalation signal.\n\n"
        "TIE-BREAKING RULE: If the note shows both structural and lexical signals, "
        "classify by whichever is more prominent. If they are equal, prefer "
        "Lexical Drift. Reserve Unresolved strictly for notes where neither "
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
        "- \"category\": one of [\"Structural Drift\", \"Lexical Drift\", "
        "\"Unresolved\"]\n"
        "- \"confidence\": one of [\"High\", \"Medium\", \"Low\"]\n"
        "- \"reasoning\": 1-2 sentences explaining your primary classification, "
        "citing specific evidence from the note text\n"
        "- \"secondary_evidence\": 1-2 sentences describing any lexical signals "
        "observed (unfamiliar terms, new abbreviations, changed phrasing), or "
        "the string 'none' if no lexical divergence is detected\n\n"
        "Do not output any preamble, markdown code fences, or text outside "
        "the JSON object."
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
def _single_classify_attempt(model, messages, max_retries, temperature, call_timeout=7200):
    raw_response = ""
    client = Client(host='http://localhost:11434', timeout=call_timeout)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_ctx": 32768},
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


def _deterministic_classify(model, messages, max_retries, call_timeout=300):
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
    )


def classify_note(model, messages, max_retries, n_runs=5, temperature=0.7, call_timeout=300):
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
            model, messages, max_retries, temperature, call_timeout,
        )
        if result["category"]:
            all_results.append(result)

    if not all_results:
        return {
            "category": "", "confidence": "", "reasoning": "",
            "secondary_evidence": "",
            "consistency_rate": 0.0, "n_valid_runs": 0,
            "deterministic_category": "",
            "deterministic_reasoning": "",
            "deterministic_secondary_evidence": "",
            "stable": None,
        }, "all_runs_failed"

    # Modal category
    category_counts = Counter(r["category"] for r in all_results)
    modal_category  = category_counts.most_common(1)[0][0]
    consistency     = category_counts[modal_category] / n_runs

    # Reasoning from highest-confidence modal run (temp=0.7)
    conf_order = {"High": 3, "Medium": 2, "Low": 1}
    modal_results = [r for r in all_results if r["category"] == modal_category]
    best_result   = max(modal_results, key=lambda r: conf_order.get(r["confidence"], 0))

    # Deterministic run at temperature=0 for stable reasoning field
    det_result, det_error = _deterministic_classify(model, messages, max_retries, call_timeout)
    det_category  = det_result.get("category", "")
    det_reasoning = det_result.get("reasoning", "")
    stable        = (det_category == modal_category) if det_category else None

    return {
        "category":                        best_result["category"],
        "confidence":                      best_result["confidence"],
        "reasoning":                       best_result["reasoning"],
        "secondary_evidence":              best_result.get("secondary_evidence", "none"),
        "consistency_rate":                round(consistency, 3),
        "n_valid_runs":                    len(all_results),
        "deterministic_category":          det_category,
        "deterministic_reasoning":         det_reasoning,
        "deterministic_secondary_evidence": det_result.get("secondary_evidence", "none"),
        "stable":                          stable,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Orchestrate LLM drift classification for all selected discharge notes."""
    args = parse_args()

    # --- Stage 1: Load Inputs -----------------------------------------------
    log.info("--- Stage 1: Load Judge Samples ---")
    df_samples = load_samples(args.samples_csv)
    log.info("%d samples loaded from %s", len(df_samples), args.samples_csv)

    # Split exemplars from rows to classify
    df_exemplars = df_samples[df_samples["selection_group"] == "exemplar"].copy()
    df_classify  = df_samples[df_samples["selection_group"] != "exemplar"].copy()

    if len(df_exemplars) == 0:
        log.warning(
            "No exemplar rows found in %s. Judge LLM will run without baseline "
            "exemplars in the system prompt. Re-run select_judge_samples.py to "
            "generate exemplars.",
            args.samples_csv,
        )
        exemplar_block = ""
    else:
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
        log.info("Loaded %d exemplar notes for system prompt.", len(df_exemplars))

    # --- Stage 1b: Load Surface Feature Statistics --------------------------
    log.info("--- Stage 1b: Load Surface Feature Statistics ---")
    surface_stats_by_window = {}
    if os.path.exists(args.surface_stats_csv):
        df_surf = pd.read_csv(args.surface_stats_csv)
        for _, srow in df_surf.iterrows():
            window = str(srow.get("period_label", srow.get("anchor_year_group", ""))).strip()
            if window:
                surface_stats_by_window[window] = {
                    "section_rate":      srow.get("section_rate", None),
                    "jaccard":           srow.get("jaccard", None),
                    "mean_length":       srow.get("mean_length", None),
                    "numeric_density":   srow.get("numeric_density", None),
                    "phi_density":       srow.get("phi_density", None),
                }
        log.info(
            "Surface stats loaded for %d windows: %s",
            len(surface_stats_by_window),
            list(surface_stats_by_window.keys()),
        )
    else:
        log.warning(
            "Surface stats file not found: %s — prompts will lack window "
            "statistics. Run surface_features.py first.",
            args.surface_stats_csv,
        )

    baseline_stats = surface_stats_by_window.get(args.baseline_label, {})
    if baseline_stats:
        log.info(
            "Baseline stats (%s): section_rate=%.3f  mean_length=%.0f",
            args.baseline_label,
            baseline_stats.get("section_rate", float("nan")),
            baseline_stats.get("mean_length", float("nan")),
        )
    else:
        log.warning(
            "Baseline label '%s' not found in surface stats — baseline "
            "annotations in system prompt will show N/A. Check "
            "--baseline-label matches the period_label column in %s.",
            args.baseline_label, args.surface_stats_csv,
        )

    # Build one system prompt per temporal window, each with its own
    # surface feature context block injected.
    unique_windows = df_classify["anchor_year_group"].unique().tolist()
    system_prompts_by_window = {}
    for window in unique_windows:
        stats  = surface_stats_by_window.get(str(window).strip(), {})
        ctx    = build_surface_context(str(window), stats, baseline_stats)
        system_prompts_by_window[window] = build_system_prompt(exemplar_block, ctx)
        log.info("System prompt built for window: %s", window)

    # --- Stage 2: Resume Check ----------------------------------------------
    log.info("--- Stage 2: Resume Check ---")
    done_ids = set()
    if os.path.exists(args.results_csv) and os.path.getsize(args.results_csv) > 0:
        df_done = pd.read_csv(args.results_csv)
        done_ids = set(df_done["hadm_id"].tolist())
        log.info("%d notes already classified — skipping.", len(done_ids))

    df_todo = df_classify[~df_classify["hadm_id"].isin(done_ids)].reset_index(drop=True)
    log.info("%d notes remaining to classify.", len(df_todo))

    if len(df_todo) == 0:
        log.info("Nothing to do — all notes already classified.")
        sys.exit(0)

    # --- Stage 3: Validate Ollama Connection --------------------------------
    log.info("--- Stage 3: Validate Ollama Connection ---")
    try:
        _ping_client = Client(host='http://localhost:11434', timeout=args.call_timeout)
        _ping_client.chat(
            model=args.model,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_ctx": 32768},
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
    write_header = (
        not os.path.exists(args.results_csv)
        or os.path.getsize(args.results_csv) == 0
    )

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
                log.warning(
                    "[%d/%d] hadm_id=%d — note text missing in samples CSV.",
                    idx + 1, len(df_todo), row.hadm_id,
                )
                writer.writerow({
                    "hadm_id":                            row.hadm_id,
                    "anchor_year_group":                  getattr(row, "anchor_year_group", ""),
                    "selection_group":                    row.selection_group,
                    "witness_score":                      row.witness_score,
                    "category":                           "",
                    "confidence":                         "",
                    "reasoning":                          "",
                    "secondary_evidence":                 "",
                    "consistency_rate":                   "",
                    "n_valid_runs":                       "",
                    "deterministic_category":             "",
                    "deterministic_reasoning":            "",
                    "deterministic_secondary_evidence":   "",
                    "stable":                             "",
                    "note_length":                        0,
                    "error":                              "note_not_found",
                })
                fh.flush()
                continue

            note_text = str(note_text)
            note_text = normalize_phi(note_text)
            window        = getattr(row, "anchor_year_group", "")
            system_prompt = system_prompts_by_window.get(
                window,
                build_system_prompt(exemplar_block)   # fallback if window missing
            )
            messages  = _build_messages(note_text, system_prompt)

            try:
                result, error_str = classify_note(
                    args.model, messages, args.max_retries,
                    n_runs=args.n_runs, temperature=args.temperature,
                    call_timeout=args.call_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Fatal Ollama connection failure at hadm_id=%d: %s — aborting.",
                    row.hadm_id, exc,
                )
                sys.exit(1)

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

    log.info("=" * 55)
    log.info("  JUDGE LLM CLASSIFICATION COMPLETE")
    log.info("  Total rows in results  : %d", n_total)
    log.info("  Successful             : %d", n_success)
    log.info("  Errors                 : %d", n_errors)
    log.info("  Output                 : %s", args.results_csv)
    log.info("=" * 55)

    if n_success > 0:
        df_ok = df_results[df_results["error"] == ""]

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