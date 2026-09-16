#!/usr/bin/env python3
"""Create one compact Experiment C table from DriftLens and MCD-DD outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_SEEDS = list(range(1111, 1131))
FRAMEWORK_MMD2 = 0.072665
FRAMEWORK_P = "<0.001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftlens-results", required=True)
    parser.add_argument("--mcddd-runs", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_latest_completed(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[int(record["seed"])] = record
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_number} of {path}.") from exc
    missing = [seed for seed in FINAL_SEEDS if latest.get(seed, {}).get("status") != "completed"]
    extra = sorted(set(latest) - set(FINAL_SEEDS))
    if missing or extra:
        raise RuntimeError(
            f"Final report requires exactly completed seeds 1111-1130; missing={missing}, extra={extra}."
        )
    return latest


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    args = parse_args()
    with Path(args.driftlens_results).open(encoding="utf-8") as handle:
        driftlens = json.load(handle)
    records = load_latest_completed(Path(args.mcddd_runs))
    ordered = [records[seed] for seed in FINAL_SEEDS]

    boundary_hits = sum(bool(row["metrics"]["boundary_hit"]) for row in ordered)
    pre_alarm_runs = sum(int(row["metrics"]["pre_boundary_alarm_count"]) > 0 for row in ordered)
    pre_alarm_total = sum(int(row["metrics"]["pre_boundary_alarm_count"]) for row in ordered)
    later_false_alarm_runs = sum(
        int(row["metrics"]["later_post_boundary_alarm_count"]) > 0 for row in ordered
    )
    later_false_alarm_total = sum(
        int(row["metrics"]["later_post_boundary_alarm_count"]) for row in ordered
    )

    fdd = float(driftlens["fdd"])
    threshold = float(driftlens["threshold"]["value"])
    driftlens_alarm = bool(driftlens["drift_detected"])
    rows = [
        {
            "Dimension": "Detection result",
            "Framework": f"MMD²={FRAMEWORK_MMD2:.6f}; p{FRAMEWORK_P} (existing frozen result)",
            "DriftLens": f"FDD={fdd:.6g}; threshold={threshold:.6g}; alarm={yes_no(driftlens_alarm)}",
            "MCD-DD": (
                f"exact-boundary detection in {boundary_hits}/20 prespecified runs; "
                f"pre-boundary false alarms={pre_alarm_total} across {pre_alarm_runs}/20 runs"
            ),
        },
        {
            "Dimension": "Interpretive artifact",
            "Framework": "Witness ranking and blinded cause taxonomy",
            "DriftLens": "K-means prototype examples (qualitative only)",
            "MCD-DD": "Per-sub-window score and threshold trajectory",
        },
        {
            "Dimension": "Native audit/action output",
            "Framework": "Calibrated close/escalate audit record",
            "DriftLens": "Drift warning; audit/action decision is not a native output",
            "MCD-DD": "Drift flag; audit/action decision is not a native output",
        },
        {
            "Dimension": "Supporting validation in this study",
            "Framework": "Experiment A positive control and Experiment B external-task analysis",
            "DriftLens": "Not separately evaluated",
            "MCD-DD": "Not separately evaluated",
        },
    ]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework": {"mmd2": FRAMEWORK_MMD2, "permutation_p": FRAMEWORK_P, "recomputed": False},
        "driftlens": {
            "fdd": fdd,
            "threshold": threshold,
            "drift_detected": driftlens_alarm,
            "empirical_calibration_percentile": driftlens.get("empirical_calibration_percentile"),
        },
        "mcddd": {
            "exact_boundary_detections": boundary_hits,
            "runs": 20,
            "missed_boundary": 20 - boundary_hits,
            "runs_with_pre_boundary_false_alarm": pre_alarm_runs,
            "total_pre_boundary_false_alarms": pre_alarm_total,
            "runs_with_later_post_boundary_false_alarm": later_false_alarm_runs,
            "total_later_post_boundary_false_alarms": later_false_alarm_total,
            "primary_endpoint": "alarm on exact first target sub-window [5000,5100)",
            "all_prespecified_seeds_retained": True,
            "filtered_denominator_analysis_performed": False,
            "seeds_are_stability_replicates_not_independent_transitions": True,
        },
        "interpretation_notes": {
            "scale": "MMD² and FDD are method-specific quantities on incomparable numerical scales.",
            "granularity": (
                "The framework and DriftLens compare the complete 5,000-note corpora; "
                "MCD-DD evaluates the boundary through adjacent 100-note sub-windows "
                "within its online sliding-window procedure."
            ),
            "representation": (
                "Each method used its native representation rule: the framework used its "
                "frozen 52-component PCA, DriftLens used its baseline-fitted 150-component "
                "PCA, and MCD-DD learned an encoder from the raw 768-dimensional embeddings."
            ),
        },
        "comparison_table": rows,
    }

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "experiment_c_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output / "experiment_c_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Experiment C compact result\n",
        "| Dimension | Framework | DriftLens | MCD-DD |",
        "|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            "| " + " | ".join(str(row[key]).replace("|", "\\|") for key in rows[0]) + " |"
        )
    markdown.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- MMD² and FDD are method-specific quantities computed in different representations and are not numerically comparable; each is interpreted only against its own decision rule.",
            "- The methods operate at different native granularities: the framework and DriftLens compare the complete 5,000-note corpora, whereas MCD-DD evaluates the transition through adjacent 100-note sub-windows within its online sliding-window procedure.",
            "- Each method used its native representation rule: the framework used its frozen 52-component PCA, DriftLens used its baseline-fitted 150-component PCA, and MCD-DD learned an encoder from the raw 768-dimensional embeddings.",
            "- DriftLens prototype clusters are qualitative examples, not validated clinical categories.",
            "",
            "Private QA summary (not manuscript text):",
            "",
            f"- MCD-DD exact-boundary detections: {boundary_hits}/20; all 20 prespecified seeds are retained.",
            f"- MCD-DD pre-boundary false alarms: {pre_alarm_total} across {pre_alarm_runs}/20 runs.",
            f"- MCD-DD later post-boundary false alarms: {later_false_alarm_total} across {later_false_alarm_runs}/20 runs.",
            "- Sustained exact-zero score/threshold sequences were observed in some runs; they remain included as observed implementation behavior and no filtered denominator is calculated.",
            "- No confidence interval is calculated: the 20 seeds are stability replicates of one transition.",
            "",
        ]
    )
    report_path = output / "EXPERIMENT_C_RESULTS.md"
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "boundary_hits": boundary_hits, "driftlens_alarm": driftlens_alarm}, indent=2))


if __name__ == "__main__":
    main()
