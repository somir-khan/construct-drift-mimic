#!/usr/bin/env python3

"""
run_addendum_mmd_sensitivity.py

A3 — MIMIC-III Addendum-handling sensitivity.

1. Reproduce the submitted detector result using the untouched
   MIMIC-III baseline.
2. Replace the baseline with the Report-preferred 5000-row sensitivity
   baseline.
3. Refit PCA independently under the exact >=90% baseline-variance rule.
4. Transform the SAME unchanged MIMIC-IV raw target embeddings.
5. Recompute the median-heuristic RBF bandwidth and observed unbiased MMD^2.
6. Save sensitivity artifacts separately.

This script DOES NOT overwrite any frozen production artifact.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

# Reuse the exact detector implementation rather than duplicating PCA/MMD code.
from detect_drift import compress_embeddings, _mmd2


# ---------------------------------------------------------------------
# Frozen/reference expectations
# ---------------------------------------------------------------------

EXPECTED_ORIGINAL_MMD4 = "0.0727"
EXPECTED_ORIGINAL_D = 52
RNG_SEED = 42


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--original-baseline",
        default="data/embeddings/embeddings_mimic3_5000.npy",
    )

    p.add_argument(
        "--sensitivity-baseline",
        default=(
            "data/addendum_sensitivity/"
            "embeddings_mimic3_5000_report_sensitivity.npy"
        ),
    )

    p.add_argument(
        "--target-emb",
        default="data/embeddings/embeddings_mimic4_5000.npy",
        help=(
            "EXACT raw 5000x768 MIMIC-IV target embedding matrix used "
            "for the submitted pooled MMD result."
        ),
    )

    p.add_argument(
        "--output-dir",
        default="data/addendum_sensitivity",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Exact bandwidth calculation from detect_drift.py mmd_test()
# ---------------------------------------------------------------------

def compute_sigma_exact(X, Y, rng_seed=42):
    """
    Match detect_drift.py exactly:

      pooled = vstack([X,Y])
      sample <=500 pooled rows without replacement using default_rng(seed)
      sigma = sqrt(median(pairwise squared distances) / 2)
    """

    m, n = len(X), len(Y)

    if m < 2 or n < 2:
        raise ValueError("Need at least two samples per corpus.")

    rng = np.random.default_rng(rng_seed)

    pooled = np.vstack([X, Y])

    sub = pooled[
        rng.choice(
            len(pooled),
            min(500, len(pooled)),
            replace=False,
        )
    ]

    sq = np.sum(
        sub ** 2,
        axis=1,
        keepdims=True,
    )

    sq_dists = np.maximum(
        sq + sq.T - 2.0 * (sub @ sub.T),
        0.0,
    )

    triu = np.triu_indices(
        len(sub),
        k=1,
    )

    sigma = float(
        np.sqrt(
            np.median(
                sq_dists[triu]
            ) / 2.0
        )
    )

    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(
            f"Invalid sigma: {sigma}"
        )

    return sigma


# ---------------------------------------------------------------------
# One complete detector run
# ---------------------------------------------------------------------

def run_detector(label, X_base_raw, X_target_raw):

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    # Exact baseline-fitted PCA logic from detector.
    X_base, X_target, pca = compress_embeddings(
        baseline_emb=X_base_raw,
        target_emb=X_target_raw,
        variance_target=0.90,
        max_components=150,
    )

    explained = float(
        pca.explained_variance_ratio_.sum()
    )

    d = int(pca.n_components_)

    # Exact median-heuristic procedure.
    sigma = compute_sigma_exact(
        X_base,
        X_target,
        rng_seed=RNG_SEED,
    )

    # Exact unbiased MMD^2 implementation.
    mmd2 = float(
        _mmd2(
            X_base,
            X_target,
            sigma,
        )
    )

    print()
    print("Detector result")
    print("----------------------------")
    print(f"Baseline raw shape     : {X_base_raw.shape}")
    print(f"Target raw shape       : {X_target_raw.shape}")
    print(f"PCA dimensions         : {d}")
    print(f"Explained variance     : {explained:.6f}")
    print(f"RBF sigma              : {sigma:.6f}")
    print(f"Observed MMD^2         : {mmd2:.8f}")
    print(f"Observed MMD^2 (4 dp)  : {mmd2:.4f}")

    return {
        "label": label,
        "mmd2": mmd2,
        "sigma": sigma,
        "pca_dimensions": d,
        "explained_variance": explained,
        "baseline_pca": X_base,
        "target_pca": X_target,
        "pca_model": pca,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    paths = {
        "original_baseline": args.original_baseline,
        "sensitivity_baseline": args.sensitivity_baseline,
        "target": args.target_emb,
    }

    for label, path in paths.items():
        if not os.path.exists(path):
            print()
            print(f"ERROR: {label} file not found:")
            print(f"  {path}")

            # Helpful only — do not silently choose another file.
            if label == "target":
                print()
                print(
                    "Do NOT substitute another target automatically. "
                    "A3 must use the exact raw target matrix that produced "
                    "the submitted MMD^2=0.0727."
                )

            raise FileNotFoundError(path)


    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    X_original = np.load(
        args.original_baseline
    )

    X_sensitivity = np.load(
        args.sensitivity_baseline
    )

    X_target = np.load(
        args.target_emb
    )


    print("=" * 72)
    print("A3 — ADDENDUM MMD SENSITIVITY")
    print("=" * 72)

    print()
    print("Inputs")
    print("----------------------------")
    print(
        f"Original baseline    : "
        f"{args.original_baseline}"
    )
    print(
        f"                       "
        f"{X_original.shape}"
    )

    print(
        f"Sensitivity baseline : "
        f"{args.sensitivity_baseline}"
    )
    print(
        f"                       "
        f"{X_sensitivity.shape}"
    )

    print(
        f"Target               : "
        f"{args.target_emb}"
    )
    print(
        f"                       "
        f"{X_target.shape}"
    )


    # -----------------------------------------------------------------
    # Shape / finite checks
    # -----------------------------------------------------------------

    if X_original.shape != (5000, 768):
        raise RuntimeError(
            f"Unexpected original shape: {X_original.shape}"
        )

    if X_sensitivity.shape != (5000, 768):
        raise RuntimeError(
            f"Unexpected sensitivity shape: {X_sensitivity.shape}"
        )

    if X_target.shape != (5000, 768):
        raise RuntimeError(
            "Target must be the exact raw 5000x768 target used for "
            f"the submitted analysis; got {X_target.shape}."
        )

    for name, X in [
        ("original", X_original),
        ("sensitivity", X_sensitivity),
        ("target", X_target),
    ]:
        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{name} contains non-finite values."
            )


    # -----------------------------------------------------------------
    # 1. Reproduce submitted result
    # -----------------------------------------------------------------

    original = run_detector(
        "ORIGINAL SUBMITTED BASELINE",
        X_original,
        X_target,
    )


    # -----------------------------------------------------------------
    # Critical replication gate
    # -----------------------------------------------------------------

    observed_4dp = f"{original['mmd2']:.4f}"

    print()
    print("=" * 72)
    print("SUBMITTED-RESULT REPLICATION GATE")
    print("=" * 72)

    print(
        f"Expected PCA dimensions : "
        f"{EXPECTED_ORIGINAL_D}"
    )

    print(
        f"Observed PCA dimensions : "
        f"{original['pca_dimensions']}"
    )

    print(
        f"Expected MMD^2 (4 dp)   : "
        f"{EXPECTED_ORIGINAL_MMD4}"
    )

    print(
        f"Observed MMD^2 (4 dp)   : "
        f"{observed_4dp}"
    )


    if original["pca_dimensions"] != EXPECTED_ORIGINAL_D:
        raise RuntimeError(
            "\nSTOP: original PCA dimensionality was not reproduced.\n"
            "Do not interpret the sensitivity result."
        )

    if observed_4dp != EXPECTED_ORIGINAL_MMD4:
        raise RuntimeError(
            "\nSTOP: submitted MMD^2=0.0727 was not reproduced.\n"
            "Most likely the wrong MIMIC-IV target embedding matrix "
            "or a different detector artifact is being used.\n"
            "Do not continue to the sensitivity analysis."
        )

    print()
    print(
        "PASS: submitted PCA dimensionality and "
        "MMD^2 reproduced."
    )


    # -----------------------------------------------------------------
    # 2. Report-preferred sensitivity
    # -----------------------------------------------------------------

    sensitivity = run_detector(
        "REPORT-PREFERRED SENSITIVITY BASELINE",
        X_sensitivity,
        X_target,
    )


    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    absolute_change = (
        sensitivity["mmd2"]
        - original["mmd2"]
    )

    percent_change = (
        100.0
        * absolute_change
        / original["mmd2"]
    )


    print()
    print("=" * 72)
    print("ORIGINAL vs REPORT-PREFERRED")
    print("=" * 72)

    print(
        f"Original MMD^2           : "
        f"{original['mmd2']:.8f}"
    )

    print(
        f"Sensitivity MMD^2        : "
        f"{sensitivity['mmd2']:.8f}"
    )

    print(
        f"Absolute change          : "
        f"{absolute_change:+.8f}"
    )

    print(
        f"Relative change          : "
        f"{percent_change:+.2f}%"
    )

    print()

    print(
        f"Original PCA d           : "
        f"{original['pca_dimensions']}"
    )

    print(
        f"Sensitivity PCA d        : "
        f"{sensitivity['pca_dimensions']}"
    )

    print()

    print(
        f"Original sigma           : "
        f"{original['sigma']:.6f}"
    )

    print(
        f"Sensitivity sigma        : "
        f"{sensitivity['sigma']:.6f}"
    )


    # -----------------------------------------------------------------
    # Save ONLY sensitivity artifacts
    # -----------------------------------------------------------------

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_pca_path = (
        outdir
        / "baseline_pca_report_sensitivity.npy"
    )

    target_pca_path = (
        outdir
        / "target_pca_report_sensitivity.npy"
    )

    pca_model_path = (
        outdir
        / "pca_model_report_sensitivity.pkl"
    )

    results_path = (
        outdir
        / "mmd_report_sensitivity_observed.json"
    )


    np.save(
        baseline_pca_path,
        sensitivity["baseline_pca"].astype(
            np.float32
        ),
    )

    np.save(
        target_pca_path,
        sensitivity["target_pca"].astype(
            np.float32
        ),
    )

    joblib.dump(
        sensitivity["pca_model"],
        pca_model_path,
    )


    results = {
        "target_embeddings": args.target_emb,

        "original": {
            "mmd2": original["mmd2"],
            "pca_dimensions":
                original["pca_dimensions"],
            "explained_variance":
                original["explained_variance"],
            "sigma": original["sigma"],
        },

        "report_preferred_sensitivity": {
            "mmd2": sensitivity["mmd2"],
            "pca_dimensions":
                sensitivity["pca_dimensions"],
            "explained_variance":
                sensitivity["explained_variance"],
            "sigma": sensitivity["sigma"],
        },

        "difference": {
            "absolute_mmd2_change":
                absolute_change,
            "relative_mmd2_change_percent":
                percent_change,
        },

        "construction": {
            "n_total_baseline": 5000,
            "n_addendum_first": 331,
            "n_report_replaced": 328,
            "n_addendum_only_retained": 3,
            "n_unchanged": 4672,
        },

        "note": (
            "Observed MMD sensitivity only. "
            "Permutation p-value/null distribution "
            "not yet recomputed."
        ),
    }


    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )


    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(baseline_pca_path)
    print(target_pca_path)
    print(pca_model_path)
    print(results_path)

    print()
    print(
        "PASS: original result reproduced and "
        "Report-preferred observed-MMD sensitivity completed."
    )


if __name__ == "__main__":
    main()