"""
scripts/extract_audit_trail_record.py
Extract the best metadata record for the LaTeX audit-trail box.

Reads judge_results_v2_test_5_run.csv (produced by judge_llm_v2.py) and
prints the exact values to paste into the LaTeX template.

Usage:
    python scripts/extract_audit_trail_record.py
    python scripts/extract_audit_trail_record.py --results-csv data/judge_results_v2_test_5_run.csv
"""

import argparse
import ast
import logging
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column names — match judge_llm_v2.py RESULTS_FIELDNAMES exactly
# ---------------------------------------------------------------------------
STRATUM_COL  = 'selection_group'
WINDOW_COL   = 'anchor_year_group'
LABEL_COL    = 'category'
EVIDENCE_COL = 'secondary_evidence'
SCORE_COL    = 'witness_score'
CONSIST_COL  = 'consistency_rate'


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audit-trail record for LaTeX.")
    parser.add_argument(
        "--results-csv",
        default="data/judge_results_v2_test_5_run.csv",
        help="Path to the judge results CSV (default: data/judge_results_v2_test_5_run.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    # ── 1. Inspect columns ───────────────────────────────────────────────────
    log.info("Columns: %s", df.columns.tolist())
    log.info("First row: %s", df.iloc[0].to_dict())

    # ── 2. Find the ideal record ─────────────────────────────────────────────
    # Criteria: stratum=top, window=2017-2019, label=Structural Drift,
    #           secondary_evidence populated, highest witness score.
    candidates = df[
        (df[STRATUM_COL] == 'top') &
        (df[WINDOW_COL].astype(str).str.contains('2017')) &
        (df[LABEL_COL] == 'Structural Drift') &
        (df[EVIDENCE_COL].notna()) &
        (df[EVIDENCE_COL].astype(str).str.strip() != '') &
        (df[EVIDENCE_COL].astype(str).str.strip() != '[]') &
        (df[EVIDENCE_COL].astype(str).str.strip() != 'nan')
    ].sort_values(SCORE_COL, ascending=False)

    if len(candidates) == 0:
        # Fallback: try 2014-2016 window
        log.warning("No 2017-2019 top-stratum candidates; trying 2014-2016 fallback.")
        candidates = df[
            (df[STRATUM_COL] == 'top') &
            (df[WINDOW_COL].astype(str).str.contains('2014')) &
            (df[LABEL_COL] == 'Structural Drift') &
            (df[EVIDENCE_COL].notna()) &
            (df[EVIDENCE_COL].astype(str).str.strip() != '')
        ].sort_values(SCORE_COL, ascending=False)

    if len(candidates) == 0:
        log.warning("No top-stratum candidates found. Showing all rows:")
        print(df[[STRATUM_COL, WINDOW_COL, LABEL_COL, SCORE_COL]].head(20))
        return

    record = candidates.iloc[0]

    # ── 3. Parse secondary_evidence ──────────────────────────────────────────
    raw_evidence = str(record[EVIDENCE_COL])
    try:
        parsed = ast.literal_eval(raw_evidence)
        if isinstance(parsed, list):
            evidence_text = '; '.join(str(x) for x in parsed)
        else:
            evidence_text = str(parsed)
    except Exception:
        evidence_text = raw_evidence

    # ── 4. Check for stability column ────────────────────────────────────────
    stable_val = 'Stable'
    for col in ['stable', 'is_stable', 'stability', 'deterministic_agrees']:
        if col in df.columns:
            stable_val = 'Stable' if record[col] else 'Unstable'
            break

    # ── 5. Check for confidence column ───────────────────────────────────────
    confidence_val = 'High'
    for col in ['confidence', 'modal_confidence', 'judge_confidence']:
        if col in df.columns:
            confidence_val = str(record[col]).capitalize()
            break

    # ── 6. Print everything needed for the LaTeX box ─────────────────────────
    # re.sub strips spaces around the dash so '2017 - 2019' → '2017--2019'
    window_display = re.sub(r'\s*-\s*', '--', str(record[WINDOW_COL]))
    stratum_display = str(record[STRATUM_COL]).capitalize()

    print("=" * 60)
    print("PASTE THESE VALUES INTO THE LaTeX TEMPLATE")
    print("=" * 60)
    print(f"Window (for header):      {window_display}")
    print(f"Stratum (for header):     {stratum_display}")
    print(f"Witness score:            {record[SCORE_COL]:.4f}")
    print(f"Primary classification:   {record[LABEL_COL]}")
    print(f"Consistency rate:         {record[CONSIST_COL]:.3f}  "
          f"({int(round(record[CONSIST_COL] * 5, 0))}/5)")
    print(f"Stability:                {stable_val}")
    print(f"Confidence:               {confidence_val}")
    print()
    print("Secondary evidence (paste verbatim into LaTeX):")
    print("-" * 60)
    print(evidence_text)
    print("-" * 60)


if __name__ == '__main__':
    main()
