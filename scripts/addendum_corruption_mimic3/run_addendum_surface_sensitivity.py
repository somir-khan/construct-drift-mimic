#!/usr/bin/env python3

"""
Block C — Report-preferred MIMIC-III surface-feature sensitivity.

Purpose
-------
Determine whether the MIMIC-III Addendum/latest-row issue materially affects
the separately reported raw-text surface-feature analysis.

Compares:

ORIGINAL:
    latest eligible MIMIC-III discharge-summary row per HADM_ID
    ORDER BY CHARTDATE DESC, ROW_ID DESC

REPORT-PREFERRED:
    latest eligible DESCRIPTION='Report' if one exists;
    otherwise latest eligible discharge-summary row.

Important
---------
- Uses the exact feature functions/constants from surface_features.py.
- Does NOT change MIMIC-IV note selection.
- Recomputes MIMIC-IV counters only because vocabulary Jaccard must be
  recalculated against the alternative MIMIC-III top-500 vocabulary.
- Does NOT overwrite outputs/surface_features_by_period.csv.
- Does NOT involve embeddings, PCA, witness scores, or Judge LLM.

Outputs
-------
data/addendum_sensitivity/
    surface_report_sensitivity_by_period.csv
    surface_report_sensitivity_comparison.csv
    surface_report_sensitivity_summary.json
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Make project .env authoritative before importing surface_features.
# This avoids the stale-shell-variable problem encountered previously.
# ---------------------------------------------------------------------

load_dotenv(override=True)

import surface_features as sf  # noqa: E402


# ---------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------

FROZEN_CSV = Path(
    "outputs/surface_features_by_period.csv"
)

OUT_DIR = Path(
    "data/addendum_sensitivity"
)

OUT_SENSITIVITY_CSV = (
    OUT_DIR
    / "surface_report_sensitivity_by_period.csv"
)

OUT_COMPARISON_CSV = (
    OUT_DIR
    / "surface_report_sensitivity_comparison.csv"
)

OUT_SUMMARY_JSON = (
    OUT_DIR
    / "surface_report_sensitivity_summary.json"
)

SEED = 42
CHUNKSIZE = 10000

# Keep this somewhat conservative instead of launching every available
# CPU process. Change if desired.
N_WORKERS = max(
    1,
    min(
        32,
        (os.cpu_count() or 2) - 1,
    ),
)


# ---------------------------------------------------------------------
# Exact MIMIC-III eligibility filters from surface_features.py.
#
# ORIGINAL differs only in ranking.
# REPORT-PREFERRED prioritizes Report, then uses the original temporal key.
# ---------------------------------------------------------------------

SQL_MIMIC3_ORIGINAL = """
WITH ranked AS (
    SELECT
        HADM_ID     AS hadm_id,
        TEXT        AS text,
        DESCRIPTION AS description,
        CHARTDATE   AS chartdate,
        ROW_ID      AS row_id,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY
                CHARTDATE DESC,
                ROW_ID DESC
        ) AS rn

    FROM NOTEEVENTS

    WHERE CATEGORY = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND TEXT IS NOT NULL
      AND HADM_ID IS NOT NULL
)

SELECT
    hadm_id,
    text,
    description,
    chartdate,
    row_id

FROM ranked

WHERE rn = 1
"""


SQL_MIMIC3_REPORT_PREFERRED = """
WITH ranked AS (
    SELECT
        HADM_ID     AS hadm_id,
        TEXT        AS text,
        DESCRIPTION AS description,
        CHARTDATE   AS chartdate,
        ROW_ID      AS row_id,

        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY
                CASE
                    WHEN DESCRIPTION = 'Report' THEN 0
                    ELSE 1
                END,
                CHARTDATE DESC,
                ROW_ID DESC
        ) AS rn

    FROM NOTEEVENTS

    WHERE CATEGORY = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND TEXT IS NOT NULL
      AND HADM_ID IS NOT NULL
)

SELECT
    hadm_id,
    text,
    description,
    chartdate,
    row_id

FROM ranked

WHERE rn = 1
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_selected_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the surface_features.py handling after SQL retrieval:
      - non-null hadm_id/text
      - strip text
      - remove empty strings
    """

    out = df.copy()

    out = out.dropna(
        subset=[
            "hadm_id",
            "text",
        ]
    )

    out["hadm_id"] = (
        out["hadm_id"]
        .astype(np.int64)
    )

    out["text"] = (
        out["text"]
        .astype(str)
        .str.strip()
    )

    out = out[
        out["text"] != ""
    ].copy()

    return out


def compute_features(
    df: pd.DataFrame,
    label: str,
):
    """
    Use exact feature implementation from surface_features.py.
    """

    print()
    print(
        f"Processing {label}: "
        f"{len(df):,} notes "
        f"on {N_WORKERS} workers..."
    )

    rows, counter = sf._parallel_features(
        df["text"].tolist(),
        N_WORKERS,
    )

    agg = sf._aggregate_rows(
        rows=rows,
        label=label,
        dataset="MIMIC-III",
        status="baseline",
    )

    agg["jaccard"] = 1.0

    return rows, counter, agg


def pct_change(new, old):
    if old == 0:
        return float("nan")

    return (
        100.0
        * (new - old)
        / old
    )


def pp_change(new, old):
    return (
        100.0
        * (new - old)
    )


def assert_close(
    name,
    observed,
    expected,
    atol=1e-10,
):
    if not np.isclose(
        observed,
        expected,
        rtol=0,
        atol=atol,
    ):
        raise RuntimeError(
            f"Frozen replication failed for {name}: "
            f"observed={observed!r}, "
            f"expected={expected!r}"
        )


def top_vocab(counter):
    return {
        word
        for word, _
        in counter.most_common(
            sf.TOP_N_VOCAB
        )
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    mimic3_path = os.getenv(
        "MIMIC3_DB_PATH"
    )

    mimic4_path = os.getenv(
        "MIMIC4_DB_PATH"
    )


    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    if (
        not mimic3_path
        or not os.path.exists(
            mimic3_path
        )
    ):
        raise FileNotFoundError(
            "MIMIC3_DB_PATH not found: "
            f"{mimic3_path}"
        )

    if (
        not mimic4_path
        or not os.path.exists(
            mimic4_path
        )
    ):
        raise FileNotFoundError(
            "MIMIC4_DB_PATH not found: "
            f"{mimic4_path}"
        )

    if not FROZEN_CSV.is_file():
        raise FileNotFoundError(
            f"Frozen surface CSV not found: "
            f"{FROZEN_CSV}"
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 80)
    print(
        "BLOCK C — REPORT-PREFERRED "
        "SURFACE-FEATURE SENSITIVITY"
    )
    print("=" * 80)

    print(
        f"MIMIC-III DB : {mimic3_path}"
    )

    print(
        f"MIMIC-IV DB  : {mimic4_path}"
    )

    print(
        f"Workers       : {N_WORKERS}"
    )

    print(
        f"Top vocab N   : "
        f"{sf.TOP_N_VOCAB}"
    )

    print(
        f"Structured >= : "
        f"{sf.STRUCTURED_THRESHOLD} "
        "recognized sections"
    )


    # -------------------------------------------------------------
    # Frozen output
    # -------------------------------------------------------------

    frozen = pd.read_csv(
        FROZEN_CSV
    )

    if "period_label" not in frozen.columns:
        raise RuntimeError(
            "Frozen CSV does not contain "
            "period_label."
        )

    frozen_by_period = (
        frozen
        .set_index(
            "period_label"
        )
    )

    if "MIMIC-III" not in (
        frozen_by_period.index
    ):
        raise RuntimeError(
            "Frozen CSV has no MIMIC-III row."
        )

    frozen_m3 = (
        frozen_by_period
        .loc["MIMIC-III"]
    )


    # -------------------------------------------------------------
    # MIMIC-III: retrieve BOTH note constructions
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOAD MIMIC-III CONSTRUCTIONS")
    print("=" * 80)


    with sqlite3.connect(
        mimic3_path
    ) as conn3:

        original_df = (
            pd.read_sql_query(
                SQL_MIMIC3_ORIGINAL,
                conn3,
            )
        )

        sensitivity_df = (
            pd.read_sql_query(
                SQL_MIMIC3_REPORT_PREFERRED,
                conn3,
            )
        )


    original_df = clean_selected_df(
        original_df
    )

    sensitivity_df = clean_selected_df(
        sensitivity_df
    )


    print(
        f"Original selected         : "
        f"{len(original_df):,}"
    )

    print(
        f"Report-preferred selected : "
        f"{len(sensitivity_df):,}"
    )


    # Same admission universe required.
    orig_ids = set(
        original_df[
            "hadm_id"
        ].tolist()
    )

    sens_ids = set(
        sensitivity_df[
            "hadm_id"
        ].tolist()
    )

    if orig_ids != sens_ids:
        raise RuntimeError(
            "Original and Report-preferred "
            "constructions do not contain "
            "the same HADM_ID universe."
        )


    if (
        original_df["hadm_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Original construction contains "
            "duplicate HADM_IDs."
        )

    if (
        sensitivity_df["hadm_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Sensitivity construction contains "
            "duplicate HADM_IDs."
        )


    # -------------------------------------------------------------
    # Quantify which selected rows changed
    # -------------------------------------------------------------

    selection_compare = (
        original_df[
            [
                "hadm_id",
                "row_id",
                "description",
            ]
        ]
        .rename(
            columns={
                "row_id":
                    "row_id_original",

                "description":
                    "description_original",
            }
        )
        .merge(
            sensitivity_df[
                [
                    "hadm_id",
                    "row_id",
                    "description",
                ]
            ].rename(
                columns={
                    "row_id":
                        "row_id_sensitivity",

                    "description":
                        "description_sensitivity",
                }
            ),
            on="hadm_id",
            how="inner",
            validate="one_to_one",
        )
    )


    changed_mask = (
        selection_compare[
            "row_id_original"
        ]
        !=
        selection_compare[
            "row_id_sensitivity"
        ]
    )

    n_changed = int(
        changed_mask.sum()
    )

    changed = selection_compare[
        changed_mask
    ].copy()


    print()
    print("Selection exposure")
    print("-" * 72)

    print(
        f"Admissions whose selected row changes : "
        f"{n_changed:,}"
    )

    print(
        f"% of full MIMIC-III baseline          : "
        f"{100*n_changed/len(original_df):.2f}%"
    )


    print()
    print(
        "Original descriptions among changed rows:"
    )

    print(
        changed[
            "description_original"
        ]
        .fillna("<NULL>")
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print()
    print(
        "Sensitivity descriptions among changed rows:"
    )

    print(
        changed[
            "description_sensitivity"
        ]
        .fillna("<NULL>")
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    # Under Report-preferred selection, any selected Addendum means
    # no eligible Report existed for that admission.
    addendum_only_retained = int(
        (
            sensitivity_df[
                "description"
            ]
            == "Addendum"
        ).sum()
    )


    print()
    print(
        "Addendum-only admissions retained : "
        f"{addendum_only_retained:,}"
    )


    # -------------------------------------------------------------
    # Compute exact MIMIC-III surface features
    # -------------------------------------------------------------

    (
        original_rows,
        original_counter,
        original_agg,
    ) = compute_features(
        original_df,
        "MIMIC-III",
    )


    (
        sensitivity_rows,
        sensitivity_counter,
        sensitivity_agg,
    ) = compute_features(
        sensitivity_df,
        "MIMIC-III",
    )


    # -------------------------------------------------------------
    # Replication gate against frozen CSV
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FROZEN MIMIC-III "
        "SURFACE REPLICATION GATE"
    )
    print("=" * 80)


    metrics_to_validate = [
        "mean_length",
        "std_length",
        "section_rate",
        "mean_sections",
        "numeric_density",
        "phi_density",
    ]


    if (
        int(original_agg["n_notes"])
        !=
        int(frozen_m3["n_notes"])
    ):
        raise RuntimeError(
            "MIMIC-III note count did not "
            "reproduce frozen CSV: "
            f"{original_agg['n_notes']} vs "
            f"{frozen_m3['n_notes']}"
        )


    print(
        f"n_notes : "
        f"{original_agg['n_notes']:,} "
        "PASS"
    )


    for metric in metrics_to_validate:

        observed = float(
            original_agg[metric]
        )

        expected = float(
            frozen_m3[metric]
        )

        print(
            f"{metric:<20}: "
            f"observed={observed:.12f} "
            f"frozen={expected:.12f}"
        )

        assert_close(
            metric,
            observed,
            expected,
            atol=1e-10,
        )


    print()
    print(
        "PASS: original MIMIC-III "
        "surface baseline reproduced."
    )


    # -------------------------------------------------------------
    # Vocabulary sensitivity
    # -------------------------------------------------------------

    original_vocab = top_vocab(
        original_counter
    )

    sensitivity_vocab = top_vocab(
        sensitivity_counter
    )


    vocab_overlap_n = len(
        original_vocab
        & sensitivity_vocab
    )

    vocab_union_n = len(
        original_vocab
        | sensitivity_vocab
    )

    baseline_vocab_jaccard = (
        vocab_overlap_n
        / vocab_union_n
    )


    print()
    print("=" * 80)
    print(
        "MIMIC-III TOP-500 "
        "VOCABULARY PROPAGATION"
    )
    print("=" * 80)

    print(
        f"Original top-500 size         : "
        f"{len(original_vocab)}"
    )

    print(
        f"Report-preferred top-500 size : "
        f"{len(sensitivity_vocab)}"
    )

    print(
        f"Shared vocabulary terms       : "
        f"{vocab_overlap_n}/500"
    )

    print(
        f"Top-500 set Jaccard           : "
        f"{baseline_vocab_jaccard:.6f}"
    )


    # -------------------------------------------------------------
    # MIMIC-IV:
    #
    # We rerun the exact existing feature processor because its word
    # counters are not stored in surface_features_by_period.csv.
    # MIMIC-IV note selection itself is unchanged.
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "REBUILD MIMIC-IV VOCABULARY COUNTERS"
    )
    print("=" * 80)


    rng = np.random.default_rng(
        SEED
    )


    with sqlite3.connect(
        mimic4_path
    ) as conn4:

        (
            group_rows,
            group_counter,
        ) = sf._process_mimic4(
            conn=conn4,
            chunksize=CHUNKSIZE,
            sample_size=None,
            rng=rng,
            n_workers=N_WORKERS,
        )


    # -------------------------------------------------------------
    # Verify MIMIC-IV rows still reproduce the frozen output.
    #
    # This also verifies the counters from which original Jaccards
    # are reconstructed.
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FROZEN MIMIC-IV "
        "SURFACE REPLICATION GATE"
    )
    print("=" * 80)


    reconstructed_m4 = {}


    for grp in sf.MIMIC4_GROUP_ORDER:

        rows = group_rows.get(
            grp,
            []
        )

        if not rows:
            continue

        agg = sf._aggregate_rows(
            rows=rows,
            label=grp,
            dataset="MIMIC-IV",
            status=sf.WINDOW_STATUS[
                grp
            ],
        )

        original_jaccard = (
            sf.jaccard(
                group_counter[grp],
                original_vocab,
            )
        )

        sensitivity_jaccard = (
            sf.jaccard(
                group_counter[grp],
                sensitivity_vocab,
            )
        )


        agg[
            "jaccard_original_baseline"
        ] = original_jaccard

        agg[
            "jaccard_report_preferred_baseline"
        ] = sensitivity_jaccard

        reconstructed_m4[
            grp
        ] = agg


        if grp not in (
            frozen_by_period.index
        ):
            raise RuntimeError(
                f"{grp} missing from frozen CSV."
            )

        frozen_row = (
            frozen_by_period
            .loc[grp]
        )


        # Verify note-level aggregate features.
        for metric in [
            "mean_length",
            "std_length",
            "section_rate",
            "mean_sections",
            "numeric_density",
            "phi_density",
        ]:

            assert_close(
                f"{grp}:{metric}",
                float(agg[metric]),
                float(
                    frozen_row[
                        metric
                    ]
                ),
                atol=1e-10,
            )


        # Verify the reconstructed original vocabulary Jaccard.
        assert_close(
            f"{grp}:jaccard",
            original_jaccard,
            float(
                frozen_row[
                    "jaccard"
                ]
            ),
            atol=1e-12,
        )


        print(
            f"{grp:<13} "
            f"n={len(rows):>7,} "
            f"old-J={original_jaccard:.6f} "
            f"PASS"
        )


    print()
    print(
        "PASS: MIMIC-IV surface features "
        "and original Jaccards reproduced."
    )


    # -------------------------------------------------------------
    # Main MIMIC-III comparison
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FINAL BLOCK C — "
        "MIMIC-III BASELINE COMPARISON"
    )
    print("=" * 80)


    comparison_rows = []


    feature_specs = [
        (
            "mean_length",
            "Mean length",
            "relative",
        ),
        (
            "section_rate",
            "Structured section rate",
            "pp",
        ),
        (
            "mean_sections",
            "Mean sections",
            "relative",
        ),
        (
            "numeric_density",
            "Numeric density",
            "relative",
        ),
        (
            "phi_density",
            "PHI density",
            "relative",
        ),
    ]


    print()
    print(
        f"{'Metric':<28}"
        f"{'Original':>16}"
        f"{'Report-pref.':>16}"
        f"{'Change':>16}"
    )

    print("-" * 76)


    for (
        key,
        label,
        change_type,
    ) in feature_specs:

        old = float(
            original_agg[key]
        )

        new = float(
            sensitivity_agg[key]
        )


        if change_type == "pp":

            change = pp_change(
                new,
                old,
            )

            change_display = (
                f"{change:+.3f} pp"
            )

        else:

            change = pct_change(
                new,
                old,
            )

            change_display = (
                f"{change:+.2f}%"
            )


        print(
            f"{label:<28}"
            f"{old:>16.6f}"
            f"{new:>16.6f}"
            f"{change_display:>16}"
        )


        comparison_rows.append({
            "metric": key,
            "original": old,
            "report_preferred": new,
            "change_type":
                change_type,
            "change":
                change,
        })


    # -------------------------------------------------------------
    # Jaccard propagation
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FINAL BLOCK C — "
        "MIMIC-IV JACCARD PROPAGATION"
    )
    print("=" * 80)


    window_comparisons = []


    print()
    print(
        f"{'Window':<16}"
        f"{'Original J':>14}"
        f"{'Report-pref J':>16}"
        f"{'Abs change':>14}"
    )

    print("-" * 62)


    for grp in sf.MIMIC4_GROUP_ORDER:

        if grp not in reconstructed_m4:
            continue

        old_j = float(
            reconstructed_m4[
                grp
            ][
                "jaccard_original_baseline"
            ]
        )

        new_j = float(
            reconstructed_m4[
                grp
            ][
                "jaccard_report_preferred_baseline"
            ]
        )

        delta_j = (
            new_j - old_j
        )


        print(
            f"{grp:<16}"
            f"{old_j:>14.6f}"
            f"{new_j:>16.6f}"
            f"{delta_j:>+14.6f}"
        )


        window_comparisons.append({
            "period_label": grp,
            "jaccard_original":
                old_j,
            "jaccard_report_preferred":
                new_j,
            "jaccard_absolute_change":
                delta_j,
        })


    # -------------------------------------------------------------
    # Main manuscript contrasts for the two analysis windows
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ANALYSIS-WINDOW CONTRASTS "
        "UNDER REPORT-PREFERRED BASELINE"
    )
    print("=" * 80)


    key_contrasts = {}


    for grp in [
        "2014 - 2016",
        "2017 - 2019",
    ]:

        m4 = reconstructed_m4[
            grp
        ]


        old_length_change = (
            100.0
            * (
                m4["mean_length"]
                - original_agg[
                    "mean_length"
                ]
            )
            / original_agg[
                "mean_length"
            ]
        )


        new_length_change = (
            100.0
            * (
                m4["mean_length"]
                - sensitivity_agg[
                    "mean_length"
                ]
            )
            / sensitivity_agg[
                "mean_length"
            ]
        )


        old_section_gap = (
            100.0
            * (
                m4["section_rate"]
                - original_agg[
                    "section_rate"
                ]
            )
        )


        new_section_gap = (
            100.0
            * (
                m4["section_rate"]
                - sensitivity_agg[
                    "section_rate"
                ]
            )
        )


        old_j = float(
            m4[
                "jaccard_original_baseline"
            ]
        )

        new_j = float(
            m4[
                "jaccard_report_preferred_baseline"
            ]
        )


        key_contrasts[grp] = {
            "mimic4_mean_length":
                float(
                    m4["mean_length"]
                ),

            "length_increase_original_baseline_percent":
                old_length_change,

            "length_increase_report_preferred_baseline_percent":
                new_length_change,

            "section_rate_mimic4":
                float(
                    m4["section_rate"]
                ),

            "section_gap_original_baseline_pp":
                old_section_gap,

            "section_gap_report_preferred_baseline_pp":
                new_section_gap,

            "jaccard_original":
                old_j,

            "jaccard_report_preferred":
                new_j,
        }


        print()
        print(grp)
        print("-" * 72)

        print(
            f"Mean-length increase:"
        )

        print(
            f"  original baseline       : "
            f"{old_length_change:+.2f}%"
        )

        print(
            f"  Report-preferred        : "
            f"{new_length_change:+.2f}%"
        )


        print(
            f"Structured-section gap:"
        )

        print(
            f"  original baseline       : "
            f"{old_section_gap:+.2f} pp"
        )

        print(
            f"  Report-preferred        : "
            f"{new_section_gap:+.2f} pp"
        )


        print(
            f"Vocabulary Jaccard:"
        )

        print(
            f"  original baseline       : "
            f"{old_j:.6f}"
        )

        print(
            f"  Report-preferred        : "
            f"{new_j:.6f}"
        )


    # -------------------------------------------------------------
    # Build sensitivity-by-period CSV
    #
    # MIMIC-IV raw features remain unchanged.
    # Only Jaccard is replaced because the baseline vocabulary changed.
    # -------------------------------------------------------------

    sensitivity_period_rows = []


    m3_sens_row = {
        "period_label":
            "MIMIC-III",

        "dataset":
            "MIMIC-III",

        "window_status":
            "baseline",

        "n_notes":
            int(
                sensitivity_agg[
                    "n_notes"
                ]
            ),

        "mean_length":
            sensitivity_agg[
                "mean_length"
            ],

        "std_length":
            sensitivity_agg[
                "std_length"
            ],

        "section_rate":
            sensitivity_agg[
                "section_rate"
            ],

        "mean_sections":
            sensitivity_agg[
                "mean_sections"
            ],

        "numeric_density":
            sensitivity_agg[
                "numeric_density"
            ],

        "phi_density":
            sensitivity_agg[
                "phi_density"
            ],

        "jaccard":
            1.0,
    }


    sensitivity_period_rows.append(
        m3_sens_row
    )


    for grp in sf.MIMIC4_GROUP_ORDER:

        if grp not in reconstructed_m4:
            continue

        r = reconstructed_m4[
            grp
        ]

        sensitivity_period_rows.append({
            "period_label":
                grp,

            "dataset":
                "MIMIC-IV",

            "window_status":
                sf.WINDOW_STATUS[
                    grp
                ],

            "n_notes":
                int(
                    r["n_notes"]
                ),

            "mean_length":
                r["mean_length"],

            "std_length":
                r["std_length"],

            "section_rate":
                r["section_rate"],

            "mean_sections":
                r["mean_sections"],

            "numeric_density":
                r["numeric_density"],

            "phi_density":
                r["phi_density"],

            "jaccard":
                r[
                    "jaccard_report_preferred_baseline"
                ],
        })


    pd.DataFrame(
        sensitivity_period_rows
    ).to_csv(
        OUT_SENSITIVITY_CSV,
        index=False,
    )


    pd.DataFrame(
        comparison_rows
    ).to_csv(
        OUT_COMPARISON_CSV,
        index=False,
    )


    # -------------------------------------------------------------
    # JSON provenance
    # -------------------------------------------------------------

    payload = {
        "construction": {
            "original_rule":
                "latest eligible discharge-summary row "
                "per HADM_ID by CHARTDATE DESC, ROW_ID DESC",

            "sensitivity_rule":
                "prefer DESCRIPTION='Report' when available, "
                "then CHARTDATE DESC, ROW_ID DESC; otherwise "
                "retain latest eligible discharge-summary row",

            "full_mimic3_n":
                len(original_df),

            "selected_rows_changed":
                n_changed,

            "selected_rows_changed_percent":
                (
                    100.0
                    * n_changed
                    / len(original_df)
                ),

            "addendum_only_retained":
                addendum_only_retained,
        },

        "baseline_original":
            {
                key:
                    (
                        int(value)
                        if isinstance(
                            value,
                            (
                                np.integer,
                                int,
                            ),
                        )
                        else float(value)
                    )
                for key, value
                in original_agg.items()
                if key in {
                    "n_notes",
                    "mean_length",
                    "std_length",
                    "section_rate",
                    "mean_sections",
                    "numeric_density",
                    "phi_density",
                    "jaccard",
                }
            },

        "baseline_report_preferred":
            {
                key:
                    (
                        int(value)
                        if isinstance(
                            value,
                            (
                                np.integer,
                                int,
                            ),
                        )
                        else float(value)
                    )
                for key, value
                in sensitivity_agg.items()
                if key in {
                    "n_notes",
                    "mean_length",
                    "std_length",
                    "section_rate",
                    "mean_sections",
                    "numeric_density",
                    "phi_density",
                    "jaccard",
                }
            },

        "baseline_top500_vocabulary": {
            "shared_terms":
                vocab_overlap_n,

            "union_terms":
                vocab_union_n,

            "set_jaccard":
                baseline_vocab_jaccard,
        },

        "mimic4_jaccard":
            window_comparisons,

        "analysis_window_contrasts":
            key_contrasts,
    }


    with open(
        OUT_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )


    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        OUT_SENSITIVITY_CSV
    )

    print(
        OUT_COMPARISON_CSV
    )

    print(
        OUT_SUMMARY_JSON
    )


    print()
    print(
        "PASS: Block C surface-feature "
        "sensitivity completed."
    )


if __name__ == "__main__":
    main()