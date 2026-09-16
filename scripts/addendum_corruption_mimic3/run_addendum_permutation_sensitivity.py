#!/usr/bin/env python3

"""
Complete Block A Addendum sensitivity.

Runs the exact frozen PCA + MMD permutation procedure for:

1. Original submitted MIMIC-III baseline
2. Report-preferred MIMIC-III baseline

The same unchanged MIMIC-IV 5000-note target is used in both.

Outputs:
  - full summary JSON
  - original null distribution .npy
  - Report-preferred null distribution .npy

Does NOT overwrite production artifacts.
"""

import json
import os

import numpy as np

from detect_drift import compress_embeddings, mmd_test


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ORIGINAL_BASELINE = (
    "data/embeddings/"
    "embeddings_mimic3_5000.npy"
)

SENSITIVITY_BASELINE = (
    "data/addendum_sensitivity/"
    "embeddings_mimic3_5000_report_sensitivity.npy"
)

TARGET = (
    "data/embeddings/"
    "embeddings_mimic4_5000.npy"
)

OUTPUT_DIR = "data/addendum_sensitivity"

N_PERMUTATIONS = 1000
ALPHA = 0.05
RNG_SEED = 42

PCA_VARIANCE = 0.90
PCA_MAX_COMPONENTS = 150


# ---------------------------------------------------------------------
# One complete condition
# ---------------------------------------------------------------------

def run_condition(label, X_base_raw, X_target_raw):

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    # -------------------------------------------------------------
    # PCA: exact detector procedure
    # -------------------------------------------------------------

    X_base_pca, X_target_pca, pca = compress_embeddings(
        baseline_emb=X_base_raw,
        target_emb=X_target_raw,
        variance_target=PCA_VARIANCE,
        max_components=PCA_MAX_COMPONENTS,
    )

    d = int(pca.n_components_)

    explained = float(
        pca.explained_variance_ratio_.sum()
    )

    print()
    print("PCA")
    print("----------------------------")
    print(f"Dimensions          : {d}")
    print(f"Explained variance  : {explained:.6f}")

    # -------------------------------------------------------------
    # Exact frozen MMD test
    #
    # IMPORTANT:
    # mmd_test() itself creates ONE rng and uses it sequentially
    # for bandwidth selection and permutation generation.
    # -------------------------------------------------------------

    result = mmd_test(
        X=X_base_pca,
        Y=X_target_pca,
        n_permutations=N_PERMUTATIONS,
        alpha=ALPHA,
        rng_seed=RNG_SEED,
    )

    print()
    print("MMD")
    print("----------------------------")
    print(
        f"MMD^2               : "
        f"{result['mmd2_observed']:.8f}"
    )
    print(
        f"MMD^2 (4 dp)        : "
        f"{result['mmd2_observed']:.4f}"
    )
    print(
        f"Sigma               : "
        f"{result['sigma']:.6f}"
    )
    print(
        f"Permutation p       : "
        f"{result['p_value']:.6f}"
    )
    print(
        f"95% null threshold  : "
        f"{result['threshold_mmd2']:.8f}"
    )
    print(
        f"Drift detected      : "
        f"{result['drift_detected']}"
    )

    return {
        "pca_dimensions": d,
        "explained_variance": explained,

        "mmd2": float(
            result["mmd2_observed"]
        ),

        "sigma": float(
            result["sigma"]
        ),

        "p_value": float(
            result["p_value"]
        ),

        "threshold_mmd2": float(
            result["threshold_mmd2"]
        ),

        "drift_detected": bool(
            result["drift_detected"]
        ),

        "null_distribution":
            result["null_distribution"],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    for path in [
        ORIGINAL_BASELINE,
        SENSITIVITY_BASELINE,
        TARGET,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)


    # -------------------------------------------------------------
    # Load exact matrices
    # -------------------------------------------------------------

    X_original = np.load(
        ORIGINAL_BASELINE
    )

    X_sensitivity = np.load(
        SENSITIVITY_BASELINE
    )

    X_target = np.load(
        TARGET
    )


    print("=" * 72)
    print("FULL ADDENDUM PERMUTATION SENSITIVITY")
    print("=" * 72)

    print()
    print("Original baseline   :", X_original.shape)
    print("Sensitivity baseline:", X_sensitivity.shape)
    print("Target              :", X_target.shape)
    print()
    print("Permutations        :", N_PERMUTATIONS)
    print("Alpha               :", ALPHA)
    print("RNG seed            :", RNG_SEED)


    # -------------------------------------------------------------
    # Strong shape checks
    # -------------------------------------------------------------

    assert X_original.shape == (5000, 768)
    assert X_sensitivity.shape == (5000, 768)
    assert X_target.shape == (5000, 768)

    assert np.isfinite(X_original).all()
    assert np.isfinite(X_sensitivity).all()
    assert np.isfinite(X_target).all()


    # -------------------------------------------------------------
    # ORIGINAL
    # -------------------------------------------------------------

    original = run_condition(
        "ORIGINAL SUBMITTED BASELINE",
        X_original,
        X_target,
    )


    # -------------------------------------------------------------
    # Replication gate
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("ORIGINAL REPLICATION GATE")
    print("=" * 72)

    print(
        f"PCA d reproduced      : "
        f"{original['pca_dimensions']}"
    )

    print(
        f"MMD^2 reproduced      : "
        f"{original['mmd2']:.8f}"
    )

    print(
        f"MMD^2 rounded         : "
        f"{original['mmd2']:.4f}"
    )

    if original["pca_dimensions"] != 52:
        raise RuntimeError(
            "STOP: original PCA d != 52."
        )

    if f"{original['mmd2']:.4f}" != "0.0727":
        raise RuntimeError(
            "STOP: original MMD^2 != 0.0727."
        )

    print("PASS: original detector reproduced.")


    # -------------------------------------------------------------
    # REPORT-PREFERRED
    # -------------------------------------------------------------

    sensitivity = run_condition(
        "REPORT-PREFERRED SENSITIVITY BASELINE",
        X_sensitivity,
        X_target,
    )


    # -------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------

    absolute_change = (
        sensitivity["mmd2"]
        - original["mmd2"]
    )

    relative_change = (
        100.0
        * absolute_change
        / original["mmd2"]
    )


    print()
    print("=" * 72)
    print("FINAL BLOCK A COMPARISON")
    print("=" * 72)

    print()
    print(
        f"{'Quantity':<24}"
        f"{'Original':>18}"
        f"{'Report-pref.':>18}"
    )

    print("-" * 60)

    print(
        f"{'PCA dimensions':<24}"
        f"{original['pca_dimensions']:>18}"
        f"{sensitivity['pca_dimensions']:>18}"
    )

    print(
        f"{'Explained variance':<24}"
        f"{original['explained_variance']:>18.6f}"
        f"{sensitivity['explained_variance']:>18.6f}"
    )

    print(
        f"{'Sigma':<24}"
        f"{original['sigma']:>18.6f}"
        f"{sensitivity['sigma']:>18.6f}"
    )

    print(
        f"{'MMD^2':<24}"
        f"{original['mmd2']:>18.8f}"
        f"{sensitivity['mmd2']:>18.8f}"
    )

    print(
        f"{'Permutation p':<24}"
        f"{original['p_value']:>18.6f}"
        f"{sensitivity['p_value']:>18.6f}"
    )

    print(
        f"{'Null threshold':<24}"
        f"{original['threshold_mmd2']:>18.8f}"
        f"{sensitivity['threshold_mmd2']:>18.8f}"
    )

    print(
        f"{'Drift detected':<24}"
        f"{str(original['drift_detected']):>18}"
        f"{str(sensitivity['drift_detected']):>18}"
    )

    print()
    print(
        f"Absolute MMD^2 change : "
        f"{absolute_change:+.8f}"
    )

    print(
        f"Relative MMD^2 change : "
        f"{relative_change:+.2f}%"
    )


    # -------------------------------------------------------------
    # Save null distributions separately
    # -------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    original_null_path = os.path.join(
        OUTPUT_DIR,
        "mmd_null_original_1000.npy",
    )

    sensitivity_null_path = os.path.join(
        OUTPUT_DIR,
        "mmd_null_report_sensitivity_1000.npy",
    )

    np.save(
        original_null_path,
        original["null_distribution"],
    )

    np.save(
        sensitivity_null_path,
        sensitivity["null_distribution"],
    )


    # -------------------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "mmd_report_sensitivity_full.json",
    )

    payload = {
        "procedure": {
            "n_permutations": N_PERMUTATIONS,
            "alpha": ALPHA,
            "rng_seed": RNG_SEED,
            "pca_variance_target": PCA_VARIANCE,
            "pca_max_components":
                PCA_MAX_COMPONENTS,
        },

        "construction": {
            "baseline_n": 5000,
            "target_n": 5000,
            "addendum_first_exposed": 331,
            "report_replaced": 328,
            "addendum_only_retained": 3,
            "unchanged": 4672,
        },

        "original": {
            k: v
            for k, v in original.items()
            if k != "null_distribution"
        },

        "report_preferred": {
            k: v
            for k, v in sensitivity.items()
            if k != "null_distribution"
        },

        "difference": {
            "absolute_mmd2_change":
                float(absolute_change),

            "relative_mmd2_change_percent":
                float(relative_change),
        },

        "null_distribution_files": {
            "original":
                original_null_path,

            "report_preferred":
                sensitivity_null_path,
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(summary_path)
    print(original_null_path)
    print(sensitivity_null_path)

    print()
    print(
        "PASS: full Block A permutation "
        "sensitivity completed."
    )


if __name__ == "__main__":
    main()