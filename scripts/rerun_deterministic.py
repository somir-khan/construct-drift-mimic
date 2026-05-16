#!/usr/bin/env python3
"""
rerun_deterministic.py — Patch missing T=0 deterministic runs in judge results.

Reads an existing judge_results CSV, identifies rows where the deterministic
(T=0) classification timed out (empty deterministic_category), re-runs ONLY
that single T=0 call with an increased timeout, and writes a patched CSV.

Usage:
    # V2 (structural tie-break) — 1 missing note
    python rerun_deterministic.py \
        --results-csv data/judge_results_v2.csv \
        --samples-csv data/judge_samples.csv \
        --tie-break structural \
        --call-timeout 900

    # V3 (lexical tie-break) — 6 missing notes
    python rerun_deterministic.py \
        --results-csv data/judge_results_v3.csv \
        --samples-csv data/judge_samples.csv \
        --tie-break lexical \
        --call-timeout 900

The original CSV is backed up to <name>_before_patch.csv before overwriting.
"""

import argparse
import json
import logging
import os
import re
import sys
import shutil

import pandas as pd
from ollama import Client

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(["Structural Drift", "Lexical Drift", "Unresolved"])


# ---------------------------------------------------------------------------
# PHI normalisation (must match the main judge script)
# ---------------------------------------------------------------------------
def normalize_phi(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', 'unknown', text)
    text = re.sub(r'___', 'unknown', text)
    return text


# ---------------------------------------------------------------------------
# Response parsing (identical to judge_llm_v2/v3)
# ---------------------------------------------------------------------------
def _strip_thinking(text: str) -> str:
    text = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_fences(text):
    stripped = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text.strip())
    return stripped.strip()


def _parse_response(raw_text):
    cleaned = _strip_thinking(raw_text)
    candidates = [
        cleaned.strip(),
        _strip_fences(cleaned),
    ]
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
# System prompt builder
# ---------------------------------------------------------------------------
def build_surface_context(anchor_year_group, surface_stats, baseline_stats=None):
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


def build_system_prompt(exemplar_block, surface_context, tie_break):
    """Build system prompt with the specified tie-break direction."""
    if tie_break == "structural":
        tie_break_line = "Structural Drift"
    else:
        tie_break_line = "Lexical Drift"

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
        f"{tie_break_line}. Reserve Unresolved strictly for notes where neither "
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
# Single T=0 classification
# ---------------------------------------------------------------------------
def deterministic_classify(model, messages, max_retries, call_timeout):
    """Run one T=0 classification attempt with retries."""
    client = Client(host='http://localhost:11434', timeout=call_timeout)

    for attempt in range(1, max_retries + 1):
        try:
            log.info("  T=0 attempt %d/%d (timeout=%ds)...", attempt, max_retries, call_timeout)
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.0, "num_ctx": 32768},
            )
            raw = response.message.content
            parsed = _parse_response(raw)
            if parsed is not None:
                return parsed, ""
            log.warning("  JSON parse failed on attempt %d. Raw: %.120s ...", attempt, raw)
        except Exception as exc:
            log.warning("  API error on attempt %d/%d: %s", attempt, max_retries, exc)

    return None, "all_retries_failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Re-run missing deterministic (T=0) judge classifications.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-csv", required=True,
        help="Path to judge_results CSV to patch (e.g. data/judge_results_v3.csv).",
    )
    parser.add_argument(
        "--samples-csv", default="data/judge_samples.csv",
        help="Input CSV with note text (produced by select_judge_samples.py).",
    )
    parser.add_argument(
        "--surface-stats-csv", default="outputs/surface_features_by_period.csv",
        help="Surface features CSV for prompt injection.",
    )
    parser.add_argument(
        "--baseline-label", default="MIMIC-III",
        help="Period label for the baseline row in surface stats.",
    )
    parser.add_argument(
        "--tie-break", required=True, choices=["structural", "lexical"],
        help="Tie-break direction: 'structural' for v2, 'lexical' for v3.",
    )
    parser.add_argument(
        "--model", default="gemma4:26b",
        help="Ollama model identifier.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max API call attempts per note.",
    )
    parser.add_argument(
        "--call-timeout", type=int, default=900,
        help="Seconds to wait for a single Ollama call (default: 900 = 15 min).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List missing notes without re-running.",
    )
    args = parser.parse_args()

    # --- Load results CSV ---
    if not os.path.exists(args.results_csv):
        log.error("Results CSV not found: %s", args.results_csv)
        sys.exit(1)

    df_results = pd.read_csv(args.results_csv)
    log.info("Loaded %d rows from %s", len(df_results), args.results_csv)

    # Identify missing deterministic rows
    missing_mask = df_results["deterministic_category"].isna() | (
        df_results["deterministic_category"].astype(str).str.strip() == ""
    )
    missing_ids = df_results.loc[missing_mask, "hadm_id"].tolist()

    if not missing_ids:
        log.info("No missing deterministic runs found. Nothing to do.")
        sys.exit(0)

    log.info("Found %d notes with missing deterministic runs:", len(missing_ids))
    for _, row in df_results[missing_mask].iterrows():
        log.info(
            "  hadm_id=%d  window=%s  stratum=%s  modal=%s  cons=%.1f  len=%s",
            row["hadm_id"], row["anchor_year_group"], row["selection_group"],
            row["category"], row["consistency_rate"], row.get("note_length", "?"),
        )

    if args.dry_run:
        log.info("Dry run — exiting without re-running.")
        sys.exit(0)

    # --- Load samples CSV for note text ---
    if not os.path.exists(args.samples_csv):
        log.error("Samples CSV not found: %s", args.samples_csv)
        sys.exit(1)

    df_samples = pd.read_csv(args.samples_csv)
    samples_by_id = {}
    for _, srow in df_samples.iterrows():
        samples_by_id[srow["hadm_id"]] = srow

    # --- Load exemplars ---
    df_exemplars = df_samples[df_samples["selection_group"] == "exemplar"]
    if len(df_exemplars) == 0:
        log.warning("No exemplar rows found — running without baseline exemplars.")
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

    # --- Load surface stats ---
    surface_stats_by_window = {}
    if os.path.exists(args.surface_stats_csv):
        df_surf = pd.read_csv(args.surface_stats_csv)
        for _, srow in df_surf.iterrows():
            window = str(srow.get("period_label", srow.get("anchor_year_group", ""))).strip()
            if window:
                surface_stats_by_window[window] = {
                    "section_rate":    srow.get("section_rate", None),
                    "jaccard":         srow.get("jaccard", None),
                    "mean_length":     srow.get("mean_length", None),
                    "numeric_density": srow.get("numeric_density", None),
                    "phi_density":     srow.get("phi_density", None),
                }
        log.info("Surface stats loaded for %d windows.", len(surface_stats_by_window))
    else:
        log.warning("Surface stats not found: %s", args.surface_stats_csv)

    baseline_stats = surface_stats_by_window.get(args.baseline_label, {})

    # --- Build system prompts per window ---
    unique_windows = df_results["anchor_year_group"].unique().tolist()
    system_prompts = {}
    for window in unique_windows:
        stats = surface_stats_by_window.get(str(window).strip(), {})
        ctx = build_surface_context(str(window), stats, baseline_stats)
        system_prompts[window] = build_system_prompt(exemplar_block, ctx, args.tie_break)

    # --- Validate Ollama ---
    log.info("Validating Ollama connection (model: %s)...", args.model)
    try:
        test_client = Client(host='http://localhost:11434', timeout=args.call_timeout)
        test_client.chat(
            model=args.model,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_ctx": 32768},
        )
        log.info("Ollama connection OK.")
    except Exception as exc:
        log.error("Cannot reach Ollama: %s", exc)
        sys.exit(1)

    # --- Backup original CSV ---
    backup_path = args.results_csv.replace(".csv", "_before_patch.csv")
    shutil.copy2(args.results_csv, backup_path)
    log.info("Backed up original to %s", backup_path)

    # --- Re-run missing deterministic calls ---
    patched = 0
    failed = 0

    for hadm_id in missing_ids:
        row_idx = df_results.index[df_results["hadm_id"] == hadm_id].tolist()
        if not row_idx:
            log.warning("hadm_id=%d not found in results CSV — skipping.", hadm_id)
            continue
        row_idx = row_idx[0]
        row = df_results.loc[row_idx]

        # Get note text
        if hadm_id not in samples_by_id:
            log.warning("hadm_id=%d not found in samples CSV — skipping.", hadm_id)
            failed += 1
            continue

        sample = samples_by_id[hadm_id]
        note_text = str(sample.get("text", ""))
        if not note_text or note_text == "nan":
            log.warning("hadm_id=%d has no note text — skipping.", hadm_id)
            failed += 1
            continue

        note_text = normalize_phi(note_text)
        window = row["anchor_year_group"]
        system_prompt = system_prompts.get(window, "")

        if not system_prompt:
            log.warning("No system prompt for window '%s' — skipping hadm_id=%d.", window, hadm_id)
            failed += 1
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": note_text},
        ]

        log.info(
            "Re-running T=0 for hadm_id=%d (window=%s, stratum=%s, modal=%s, len=%d)...",
            hadm_id, window, row["selection_group"], row["category"],
            len(note_text),
        )

        result, error = deterministic_classify(
            args.model, messages, args.max_retries, args.call_timeout,
        )

        if result is None:
            log.error("  FAILED again for hadm_id=%d: %s", hadm_id, error)
            failed += 1
            continue

        det_category = result["category"]
        modal_category = row["category"]
        stable = (det_category == modal_category)

        # Patch the DataFrame
        df_results.at[row_idx, "deterministic_category"] = det_category
        df_results.at[row_idx, "deterministic_reasoning"] = result.get("reasoning", "")
        df_results.at[row_idx, "deterministic_secondary_evidence"] = result.get("secondary_evidence", "none")
        df_results.at[row_idx, "stable"] = stable

        log.info(
            "  PATCHED: det=%s  stable=%s  (modal was %s)",
            det_category, stable, modal_category,
        )
        patched += 1

    # --- Write patched CSV ---
    df_results.to_csv(args.results_csv, index=False)
    log.info(
        "Done. Patched %d/%d notes. Failed: %d. Written to %s",
        patched, len(missing_ids), failed, args.results_csv,
    )

    # --- Summary ---
    still_missing = df_results["deterministic_category"].isna() | (
        df_results["deterministic_category"].astype(str).str.strip() == ""
    )
    if still_missing.any():
        log.warning("%d notes still missing deterministic runs.", still_missing.sum())
    else:
        log.info("All deterministic runs complete.")

    stable_count = df_results["stable"].sum()
    total = len(df_results)
    log.info("Final stability: %d/%d (%.1f%%)", stable_count, total, stable_count/total*100)


if __name__ == "__main__":
    main()