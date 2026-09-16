#!/usr/bin/env python3

"""
Block B — Report-preferred baseline sensitivity for witness geometry.

Prerequisite:
    scripts/addendum_corruption_mimic3/replay_judge_sample_selection.py must PASS.

This script does NOT revalidate the historical manifest.
Instead, it imports the exact already-validated legacy witness implementation
from replay_judge_sample_selection.py and applies that identical procedure to:

    1. original frozen MIMIC-III geometry
    2. Report-preferred sensitivity geometry

No database access.
No embedding inference.
No Judge LLM.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import replay_judge_sample_selection as legacy


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ORIGINAL_BASELINE_PCA = Path(
    "data/baseline_pca.npy"
)

ORIGINAL_PCA_MODEL = Path(
    "data/pca_model.pkl"
)

SENSITIVITY_BASELINE_PCA = Path(
    "data/addendum_sensitivity/"
    "baseline_pca_report_sensitivity.npy"
)

SENSITIVITY_PCA_MODEL = Path(
    "data/addendum_sensitivity/"
    "pca_model_report_sensitivity.pkl"
)

BASELINE_IDS = Path(
    "data/embeddings/"
    "ids_mimic3_5000.npy"
)

OUT_DIR = Path(
    "data/addendum_sensitivity"
)


# These were just independently reproduced by the canonical replay.
EXPECTED_ORIGINAL_SIGMA = {
    "2014 - 2016": 2.028281897290,
    "2017 - 2019": 2.078442883808,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def centroid_info(
    baseline,
    baseline_ids,
    n=3,
):
    centroid = baseline.mean(axis=0)

    distances = np.linalg.norm(
        baseline - centroid,
        axis=1,
    )

    order = np.argsort(distances)

    ranks = np.empty(
        len(order),
        dtype=np.int64,
    )

    ranks[order] = np.arange(
        1,
        len(order) + 1,
    )

    exemplar_ids = baseline_ids[
        order[:n]
    ].astype(np.int64)

    return {
        "ids": exemplar_ids,
        "distances": distances,
        "ranks": ranks,
    }


def rank_desc(scores):
    order = np.argsort(scores)[::-1]

    ranks = np.empty(
        len(scores),
        dtype=np.int64,
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1,
    )

    return order, ranks


def overlap(a, b):
    return len(
        set(np.asarray(a).tolist())
        & set(np.asarray(b).tolist())
    )


# ---------------------------------------------------------------------
# Run one geometry
# ---------------------------------------------------------------------

def run_geometry(
    label,
    baseline_pca_path,
    pca_model_path,
    baseline_ids,
):

    baseline = np.load(
        baseline_pca_path,
        allow_pickle=False,
    )

    pca = joblib.load(
        pca_model_path
    )

    if len(baseline) != 5000:
        raise RuntimeError(
            f"{label}: baseline has "
            f"{len(baseline)} rows."
        )

    if len(baseline_ids) != len(baseline):
        raise RuntimeError(
            f"{label}: baseline IDs not aligned."
        )

    if baseline.shape[1] != pca.n_components_:
        raise RuntimeError(
            f"{label}: PCA dimension mismatch."
        )


    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print(
        f"Baseline PCA       : {baseline.shape}"
    )

    print(
        f"PCA components     : {pca.n_components_}"
    )

    print(
        "Explained variance:",
        f"{pca.explained_variance_ratio_.sum():.6f}",
    )


    centroid = centroid_info(
        baseline,
        baseline_ids,
        n=3,
    )

    print(
        "Centroid exemplars :",
        centroid["ids"].tolist(),
    )


    # EXACT historical behavior:
    # one RNG for BOTH windows.
    rng = np.random.default_rng(
        legacy.SEED
    )

    windows = {}


    for (
        window_label,
        embedding_path,
        ids_path,
    ) in legacy.WINDOWS:

        raw = np.load(
            embedding_path,
            allow_pickle=False,
        )

        ids = np.load(
            ids_path,
            allow_pickle=False,
        ).astype(np.int64)


        # Exact historical float32 cast.
        target = pca.transform(
            raw
        ).astype(np.float32)


        # -------------------------------------------------------------
        # Exact replay bandwidth sequence
        # -------------------------------------------------------------

        base_part = baseline[
            rng.choice(
                len(baseline),
                size=min(
                    legacy.SUBSAMPLE,
                    len(baseline),
                ),
                replace=False,
            )
        ]

        target_part = target[
            rng.choice(
                len(target),
                size=min(
                    legacy.SUBSAMPLE,
                    len(target),
                ),
                replace=False,
            )
        ]

        pooled = np.vstack(
            [
                base_part,
                target_part,
            ]
        )


        # IMPORTANT:
        # This is imported from the canonical replay,
        # not reimplemented here.
        sigma = (
            legacy.median_heuristic_legacy(
                pooled,
                rng,
            )
        )


        # Exact imported witness implementation.
        scores = (
            legacy.witness_scores_legacy(
                baseline,
                target,
                sigma,
            )
        )


        # Exact imported historical selector.
        #
        # This also consumes the random-stratum RNG draw
        # before proceeding to window 2.
        selected = legacy.select_legacy(
            scores,
            ids,
            rng,
        )


        descending, ranks = rank_desc(
            scores
        )

        ascending = np.argsort(
            scores
        )


        windows[window_label] = {
            "ids": ids,
            "scores": scores,
            "ranks": ranks,

            "sigma": float(sigma),

            "top50":
                selected["top"],

            "bottom50":
                selected["bottom"],

            "random50":
                selected["random"],

            "top100":
                ids[
                    descending[:100]
                ].astype(np.int64),

            "bottom100":
                ids[
                    ascending[:100]
                ].astype(np.int64),
        }


        print()
        print(window_label)
        print("-" * 72)

        print(
            f"sigma       : {sigma:.12f}"
        )

        print(
            f"score min   : {scores.min():.8f}"
        )

        print(
            f"score max   : {scores.max():.8f}"
        )

        print(
            f"score mean  : {scores.mean():.8f}"
        )

        print(
            f"score std   : {scores.std():.8f}"
        )


    return {
        "baseline": baseline,
        "pca": pca,
        "centroid": centroid,
        "windows": windows,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    required = [
        ORIGINAL_BASELINE_PCA,
        ORIGINAL_PCA_MODEL,
        SENSITIVITY_BASELINE_PCA,
        SENSITIVITY_PCA_MODEL,
        BASELINE_IDS,
    ]

    for _, emb, ids in legacy.WINDOWS:
        required.extend(
            [emb, ids]
        )

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    if missing:
        raise SystemExit(
            "Missing file(s):\n  "
            + "\n  ".join(missing)
        )


    baseline_ids = np.load(
        BASELINE_IDS,
        allow_pickle=False,
    ).astype(np.int64)


    print("=" * 72)
    print(
        "BLOCK B — REPORT-PREFERRED "
        "WITNESS-GEOMETRY SENSITIVITY"
    )
    print("=" * 72)

    print(
        "Canonical replay implementation:",
        "replay_judge_sample_selection.py",
    )

    print(
        f"Seed                : {legacy.SEED}"
    )

    print(
        f"Groups              : {legacy.N_PER_GROUP}"
    )

    print(
        f"Bandwidth subsample : {legacy.SUBSAMPLE}"
    )


    # -------------------------------------------------------------
    # Original
    # -------------------------------------------------------------

    original = run_geometry(
        "ORIGINAL FROZEN GEOMETRY",
        ORIGINAL_BASELINE_PCA,
        ORIGINAL_PCA_MODEL,
        baseline_ids,
    )


    # -------------------------------------------------------------
    # Very small sanity check, NOT a second manifest replay.
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("CANONICAL ORIGINAL-SIGMA SANITY CHECK")
    print("=" * 72)

    for window, expected in (
        EXPECTED_ORIGINAL_SIGMA.items()
    ):

        observed = (
            original["windows"][
                window
            ]["sigma"]
        )

        print(
            f"{window}: "
            f"expected={expected:.12f} "
            f"observed={observed:.12f}"
        )

        if not np.isclose(
            observed,
            expected,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Original canonical sigma "
                f"failed for {window}."
            )

    print(
        "PASS: original canonical bandwidths reproduced."
    )


    # -------------------------------------------------------------
    # Report-preferred
    #
    # NEW independent RNG starts at seed 42 inside run_geometry(),
    # so the exact same procedure is applied to the alternative
    # baseline.
    # -------------------------------------------------------------

    sensitivity = run_geometry(
        "REPORT-PREFERRED SENSITIVITY GEOMETRY",
        SENSITIVITY_BASELINE_PCA,
        SENSITIVITY_PCA_MODEL,
        baseline_ids,
    )


    # -------------------------------------------------------------
    # Exemplars
    # -------------------------------------------------------------

    orig_ex = original[
        "centroid"
    ]["ids"]

    sens_ex = sensitivity[
        "centroid"
    ]["ids"]

    ex_overlap = overlap(
        orig_ex,
        sens_ex,
    )


    print()
    print("=" * 72)
    print("CENTROID EXEMPLAR PROPAGATION")
    print("=" * 72)

    print(
        "Original         :",
        orig_ex.tolist(),
    )

    print(
        "Report-preferred :",
        sens_ex.tolist(),
    )

    print(
        f"Overlap          : {ex_overlap}/3"
    )


    # Where do the ORIGINAL exemplars sit
    # under the sensitivity centroid?
    sens_rank = sensitivity[
        "centroid"
    ]["ranks"]

    id_to_row = {
        int(h):
        i
        for i, h
        in enumerate(baseline_ids)
    }

    print()
    print(
        "Original exemplar centroid ranks "
        "under Report-preferred geometry:"
    )

    old_exemplar_sensitivity_ranks = {}

    for hadm_id in orig_ex:

        row = id_to_row[
            int(hadm_id)
        ]

        rank = int(
            sens_rank[row]
        )

        old_exemplar_sensitivity_ranks[
            str(int(hadm_id))
        ] = rank

        print(
            f"  {int(hadm_id)} -> rank {rank}/5000"
        )


    # -------------------------------------------------------------
    # Compare windows
    # -------------------------------------------------------------

    summaries = []

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 80)
    print("FINAL BLOCK B COMPARISON")
    print("=" * 80)


    for (
        window_label,
        _,
        _,
    ) in legacy.WINDOWS:

        a = original[
            "windows"
        ][window_label]

        b = sensitivity[
            "windows"
        ][window_label]


        if not np.array_equal(
            a["ids"],
            b["ids"],
        ):
            raise RuntimeError(
                f"{window_label}: "
                "target IDs changed."
            )


        rho = float(
            spearmanr(
                a["scores"],
                b["scores"],
            ).statistic
        )


        rank_shift = np.abs(
            a["ranks"]
            - b["ranks"]
        )


        result = {
            "window": window_label,

            "sigma_original":
                a["sigma"],

            "sigma_report_preferred":
                b["sigma"],

            "sigma_change_percent":
                float(
                    100.0
                    * (
                        b["sigma"]
                        - a["sigma"]
                    )
                    / a["sigma"]
                ),

            "spearman_rho": rho,

            "median_absolute_rank_shift":
                float(
                    np.median(
                        rank_shift
                    )
                ),

            "mean_absolute_rank_shift":
                float(
                    np.mean(
                        rank_shift
                    )
                ),

            "p90_absolute_rank_shift":
                float(
                    np.quantile(
                        rank_shift,
                        0.90,
                    )
                ),

            "max_absolute_rank_shift":
                int(
                    rank_shift.max()
                ),

            "top50_overlap":
                overlap(
                    a["top50"],
                    b["top50"],
                ),

            "bottom50_overlap":
                overlap(
                    a["bottom50"],
                    b["bottom50"],
                ),

            "random50_overlap":
                overlap(
                    a["random50"],
                    b["random50"],
                ),

            "top100_overlap":
                overlap(
                    a["top100"],
                    b["top100"],
                ),

            "bottom100_overlap":
                overlap(
                    a["bottom100"],
                    b["bottom100"],
                ),
        }


        summaries.append(result)


        print()
        print(window_label)
        print("-" * 72)

        print(
            f"Original sigma             : "
            f"{result['sigma_original']:.12f}"
        )

        print(
            f"Report-preferred sigma     : "
            f"{result['sigma_report_preferred']:.12f}"
        )

        print(
            f"Sigma change               : "
            f"{result['sigma_change_percent']:+.2f}%"
        )

        print(
            f"Spearman rho               : "
            f"{result['spearman_rho']:.6f}"
        )

        print(
            f"Median abs rank shift      : "
            f"{result['median_absolute_rank_shift']:.1f}"
        )

        print(
            f"Mean abs rank shift        : "
            f"{result['mean_absolute_rank_shift']:.1f}"
        )

        print(
            f"90th pct abs rank shift    : "
            f"{result['p90_absolute_rank_shift']:.1f}"
        )

        print(
            f"Max abs rank shift         : "
            f"{result['max_absolute_rank_shift']}"
        )

        print(
            f"Top-50 overlap             : "
            f"{result['top50_overlap']}/50"
        )

        print(
            f"Bottom-50 overlap          : "
            f"{result['bottom50_overlap']}/50"
        )

        print(
            f"Random-50 overlap          : "
            f"{result['random50_overlap']}/50"
        )

        print(
            f"Top-100 overlap            : "
            f"{result['top100_overlap']}/100"
        )

        print(
            f"Bottom-100 overlap         : "
            f"{result['bottom100_overlap']}/100"
        )


        # Save full 2500-note comparison.
        detail = pd.DataFrame({
            "hadm_id":
                a["ids"],

            "witness_original":
                a["scores"],

            "witness_report_preferred":
                b["scores"],

            "witness_delta":
                b["scores"]
                - a["scores"],

            "rank_original":
                a["ranks"],

            "rank_report_preferred":
                b["ranks"],

            "absolute_rank_shift":
                rank_shift,
        })


        suffix = (
            window_label
            .replace(" ", "")
            .replace("-", "_")
        )

        detail.to_csv(
            OUT_DIR
            / (
                "witness_geometry_"
                f"{suffix}.csv"
            ),
            index=False,
        )


    # -------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------

    output = {
        "prerequisite": (
            "Canonical replay_judge_sample_selection.py "
            "independently passed all six historical groups."
        ),

        "implementation": (
            "Block B imports median_heuristic_legacy, "
            "witness_scores_legacy, select_legacy, "
            "SEED, SUBSAMPLE, N_PER_GROUP, and WINDOWS "
            "directly from replay_judge_sample_selection.py."
        ),

        "original_pca_dimensions":
            int(
                original["pca"].n_components_
            ),

        "report_preferred_pca_dimensions":
            int(
                sensitivity["pca"].n_components_
            ),

        "original_exemplars":
            orig_ex.tolist(),

        "report_preferred_exemplars":
            sens_ex.tolist(),

        "exemplar_overlap":
            ex_overlap,

        "original_exemplar_sensitivity_centroid_ranks":
            old_exemplar_sensitivity_ranks,

        "windows":
            summaries,
    }


    summary_path = (
        OUT_DIR
        / "witness_geometry_sensitivity_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )


    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(summary_path)

    print()
    print(
        "PASS: Block B sensitivity completed."
    )


if __name__ == "__main__":
    main()