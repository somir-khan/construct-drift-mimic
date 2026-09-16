#!/usr/bin/env python3
"""
experiment_b_validate_embeddings.py

Experiment B — raw embedding and frozen-PCA integrity gate.

NO CLASSIFIER IS FIT.
NO AUROC IS COMPUTED.

Checks:
- exact expected row counts;
- embedding shape N x 768;
- ID length N;
- unique IDs;
- ID set equality against frozen manifests;
- all embedding coordinates finite;
- zero-vector count == 0;
- SHA-256 of embeddings and ID arrays;
- loads the submitted frozen pca_model.pkl;
- catches scikit-learn pickle/version warnings;
- pca.n_features_in_ == 768;
- pca.n_components_ == 52;
- explained_variance_ratio_.sum() ~= 0.900638;
- transforms all four matrices using that exact PCA;
- NEVER refits PCA;
- stores transformed PCA arrays as FLOAT64;
- saves an independent row-ID witness beside every PCA matrix;
- validates transformed shape N x 52;
- validates transformed coordinates are finite;
- writes complete provenance.

The PCA output row order is exactly the source ids_*.npy order.
"""

import argparse
import hashlib
import json
import os
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


# ============================================================================
# FROZEN EXPECTATIONS
# ============================================================================

EXPECTED_EMBED_DIM = 768
EXPECTED_PCA_DIM = 52

EXPECTED_EXPLAINED_VARIANCE = 0.900638

EXPECTED_COUNTS = {
    "m3_train": 47723,
    "m3_dev": 1631,
    "m3_test": 3372,
    "m4_target": 19667,
}


# ============================================================================
# HELPERS
# ============================================================================


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_info():
    """
    Best-effort Git provenance.
    Failure to query Git does not alter the computation.
    """
    result = {"commit": None, "dirty": None}

    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )

        result = {
            "repository_root": root,
            "commit": commit,
            "dirty": bool(status.strip()),
        }

    except Exception:
        pass

    return result


def load_ids(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} ID file not found: {path}")

    ids = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)

    ids = ids.astype(np.int64)

    if len(ids) == 0:
        raise RuntimeError(f"{label}: ID array is empty.")

    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"{label}: duplicate HADM_IDs.")

    return ids


def assert_same_set(ids, expected, label):
    actual_set = set(int(x) for x in ids.tolist())

    expected_set = set(int(x) for x in expected)

    missing = sorted(expected_set - actual_set)

    extra = sorted(actual_set - expected_set)

    if missing or extra:
        raise RuntimeError(
            f"{label}: ID set differs from "
            f"frozen manifest.\n"
            f"missing={len(missing):,}, "
            f"examples={missing[:10]}\n"
            f"extra={len(extra):,}, "
            f"examples={extra[:10]}"
        )


def inspect_embedding(
    emb_path, ids_path, expected_n, expected_ids, label, chunk_size=4096
):
    emb_path = Path(emb_path)

    ids_path = Path(ids_path)

    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    if not ids_path.exists():
        raise FileNotFoundError(ids_path)

    X = np.load(emb_path, mmap_mode="r", allow_pickle=False)

    ids = load_ids(ids_path, label)

    expected_shape = (expected_n, EXPECTED_EMBED_DIM)

    if X.shape != expected_shape:
        raise RuntimeError(
            f"{label}: expected embedding "
            f"shape {expected_shape}, "
            f"got {X.shape}."
        )

    if len(ids) != expected_n:
        raise RuntimeError(
            f"{label}: expected " f"{expected_n:,} IDs, " f"got {len(ids):,}."
        )

    assert_same_set(ids, expected_ids, label)

    n_nonfinite_rows = 0
    n_zero_rows = 0
    max_abs = 0.0

    for start in range(0, expected_n, chunk_size):
        end = min(start + chunk_size, expected_n)

        block = np.asarray(X[start:end])

        finite_rows = np.all(np.isfinite(block), axis=1)

        n_nonfinite_rows += int(np.sum(~finite_rows))

        zero_rows = np.all(block == 0, axis=1)

        n_zero_rows += int(np.sum(zero_rows))

        if block.size:
            block_max = float(np.nanmax(np.abs(block)))

            max_abs = max(max_abs, block_max)

    if n_nonfinite_rows:
        raise RuntimeError(
            f"{label}: " f"{n_nonfinite_rows:,} rows " "contain NaN/Inf."
        )

    if n_zero_rows:
        raise RuntimeError(
            f"{label}: "
            f"{n_zero_rows:,} zero-vector "
            "embeddings found. "
            "This is a hard failure because "
            "the embedder uses zero vectors "
            "when note embedding fails."
        )

    report = {
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "n_zero_rows": n_zero_rows,
        "n_nonfinite_rows": n_nonfinite_rows,
        "max_abs_coordinate": max_abs,
        "embedding_sha256": sha256_file(emb_path),
        "ids_sha256": sha256_file(ids_path),
    }

    return report, ids


def transform_to_npy(emb_path, pca, out_path, expected_n, chunk_size=4096):
    """
    Transform while preserving float64 output.

    No float32 round-trip is introduced here.
    """
    X = np.load(emb_path, mmap_mode="r", allow_pickle=False)

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64, shape=(expected_n, EXPECTED_PCA_DIM)
    )

    for start in range(0, expected_n, chunk_size):
        end = min(start + chunk_size, expected_n)

        # PCA was fitted through sklearn's normal
        # floating-point path; transform explicitly
        # in float64 for reproducibility.
        block = np.asarray(X[start:end], dtype=np.float64)

        transformed = pca.transform(block)

        expected_shape = (end - start, EXPECTED_PCA_DIM)

        if transformed.shape != expected_shape:
            raise RuntimeError(
                "PCA output shape error for "
                f"rows {start}:{end}. "
                f"Expected {expected_shape}, "
                f"got {transformed.shape}."
            )

        if not np.all(np.isfinite(transformed)):
            raise RuntimeError(
                "Frozen PCA produced NaN/Inf " f"for rows {start}:{end}."
            )

        out[start:end] = transformed

    out.flush()

    del out

    check = np.load(out_path, mmap_mode="r", allow_pickle=False)

    if check.shape != (expected_n, EXPECTED_PCA_DIM):
        raise RuntimeError("Saved PCA array has " f"wrong shape: {check.shape}")

    if check.dtype != np.float64:
        raise RuntimeError("Saved PCA output should be " f"float64, got {check.dtype}.")

    return {
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "sha256": sha256_file(out_path),
    }


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate Experiment B embeddings "
            "and transform them using the "
            "submitted frozen PCA."
        ),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )

    for prefix in ["m3-train", "m3-dev", "m3-test", "m4-target"]:
        p.add_argument(f"--{prefix}-emb", required=True)

        p.add_argument(f"--{prefix}-ids", required=True)

    p.add_argument("--m3-split-manifest", required=True)

    p.add_argument("--m4-manifest", required=True)

    p.add_argument("--pca-model", required=True)

    p.add_argument(
        "--expected-pca-variance", type=float, default=(EXPECTED_EXPLAINED_VARIANCE)
    )

    p.add_argument("--pca-variance-tolerance", type=float, default=1e-4)

    p.add_argument(
        "--allow-sklearn-version-warning",
        action="store_true",
        help=(
            "Allow PCA unpickling to continue "
            "after an sklearn version warning. "
            "Default is to stop."
        ),
    )

    p.add_argument("--chunk-size", type=int, default=4096)

    p.add_argument("--out-dir", default=("outputs/" "experiment_b_representation"))

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Frozen manifests
    # ------------------------------------------------------------------------

    m3_manifest = pd.read_csv(args.m3_split_manifest)

    m3_manifest.columns = [str(c).strip().lower() for c in m3_manifest.columns]

    if not {"split", "hadm_id"}.issubset(m3_manifest.columns):
        raise RuntimeError("M3 split manifest must " "contain split,hadm_id.")

    m3_manifest["hadm_id"] = pd.to_numeric(
        m3_manifest["hadm_id"], errors="raise"
    ).astype(np.int64)

    m4_manifest = pd.read_csv(args.m4_manifest)

    m4_manifest.columns = [str(c).strip().lower() for c in m4_manifest.columns]

    if "hadm_id" not in (m4_manifest.columns):
        raise RuntimeError("M4 manifest must contain hadm_id.")

    m4_manifest["hadm_id"] = pd.to_numeric(
        m4_manifest["hadm_id"], errors="raise"
    ).astype(np.int64)

    expected_ids = {
        "m3_train": m3_manifest.loc[
            m3_manifest["split"] == "train", "hadm_id"
        ].tolist(),
        "m3_dev": m3_manifest.loc[m3_manifest["split"] == "dev", "hadm_id"].tolist(),
        "m3_test": m3_manifest.loc[m3_manifest["split"] == "test", "hadm_id"].tolist(),
        "m4_target": m4_manifest["hadm_id"].tolist(),
    }

    inputs = {
        "m3_train": {"emb": args.m3_train_emb, "ids": args.m3_train_ids},
        "m3_dev": {"emb": args.m3_dev_emb, "ids": args.m3_dev_ids},
        "m3_test": {"emb": args.m3_test_emb, "ids": args.m3_test_ids},
        "m4_target": {"emb": args.m4_target_emb, "ids": args.m4_target_ids},
    }

    print("=" * 80)
    print("EXPERIMENT B — " "EMBEDDING / PCA VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Raw embedding validation
    # ------------------------------------------------------------------------

    raw_report = {}
    loaded_ids = {}

    for key in ["m3_train", "m3_dev", "m3_test", "m4_target"]:
        report, cohort_ids = inspect_embedding(
            emb_path=inputs[key]["emb"],
            ids_path=inputs[key]["ids"],
            expected_n=(EXPECTED_COUNTS[key]),
            expected_ids=(expected_ids[key]),
            label=key,
            chunk_size=args.chunk_size,
        )

        raw_report[key] = report

        loaded_ids[key] = cohort_ids

        print()
        print(f"{key}:")

        print("  shape          : " f"{tuple(report['shape'])}")

        print("  dtype          : " f"{report['dtype']}")

        print("  zero rows      : " f"{report['n_zero_rows']}")

        print("  nonfinite rows : " f"{report['n_nonfinite_rows']}")

    # ------------------------------------------------------------------------
    # Frozen PCA load
    # ------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("FROZEN PCA")
    print("-" * 80)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        pca = joblib.load(args.pca_model)

    warning_records = [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught
    ]

    inconsistent = [
        w
        for w in warning_records
        if (
            w["category"] == "InconsistentVersionWarning"
            or (
                "scikit-learn" in w["message"].lower()
                and "version" in w["message"].lower()
            )
        )
    ]

    print("Current scikit-learn version: " f"{sklearn.__version__}")

    if warning_records:
        print("Warnings while loading PCA:")

        for w in warning_records:
            print(f"  [{w['category']}] " f"{w['message']}")

    if inconsistent and not args.allow_sklearn_version_warning:
        raise RuntimeError(
            "A scikit-learn version warning "
            "occurred while loading "
            "pca_model.pkl. Reproduce the "
            "original sklearn environment, "
            "or review the warning and rerun "
            "with --allow-sklearn-version-warning."
        )

    n_features = int(getattr(pca, "n_features_in_", -1))

    n_components = int(getattr(pca, "n_components_", -1))

    if not hasattr(pca, "explained_variance_ratio_"):
        raise RuntimeError("Loaded PCA object has no " "explained_variance_ratio_.")

    variance_sum = float(np.sum(pca.explained_variance_ratio_))

    print(f"n_features_in_     : " f"{n_features}")

    print(f"n_components_      : " f"{n_components}")

    print("explained variance : " f"{variance_sum:.9f}")

    if n_features != EXPECTED_EMBED_DIM:
        raise RuntimeError(
            "Frozen PCA input dimension "
            f"is {n_features}, expected "
            f"{EXPECTED_EMBED_DIM}."
        )

    if n_components != EXPECTED_PCA_DIM:
        raise RuntimeError(
            "Frozen PCA component count "
            f"is {n_components}, expected "
            f"{EXPECTED_PCA_DIM}."
        )

    variance_delta = abs(variance_sum - args.expected_pca_variance)

    if variance_delta > args.pca_variance_tolerance:
        raise RuntimeError(
            "Frozen PCA explained-variance "
            "mismatch.\n"
            f"expected  : "
            f"{args.expected_pca_variance:.9f}\n"
            f"loaded    : "
            f"{variance_sum:.9f}\n"
            f"delta     : "
            f"{variance_delta:.9g}\n"
            f"tolerance : "
            f"{args.pca_variance_tolerance:.9g}"
        )

    # ------------------------------------------------------------------------
    # Transform — NEVER refit
    # ------------------------------------------------------------------------

    transformed_paths = {
        "m3_train": out_dir / "pca_m3_train.npy",
        "m3_dev": out_dir / "pca_m3_dev.npy",
        "m3_test": out_dir / "pca_m3_test.npy",
        "m4_target": out_dir / "pca_m4_target.npy",
    }

    row_id_paths = {
        "m3_train": out_dir / "row_ids_pca_m3_train.npy",
        "m3_dev": out_dir / "row_ids_pca_m3_dev.npy",
        "m3_test": out_dir / "row_ids_pca_m3_test.npy",
        "m4_target": out_dir / "row_ids_pca_m4_target.npy",
    }

    transformed_report = {}

    print()
    print("-" * 80)
    print("FROZEN PCA TRANSFORMS")
    print("-" * 80)

    for key in ["m3_train", "m3_dev", "m3_test", "m4_target"]:
        transformed_report[key] = transform_to_npy(
            emb_path=(inputs[key]["emb"]),
            pca=pca,
            out_path=(transformed_paths[key]),
            expected_n=(EXPECTED_COUNTS[key]),
            chunk_size=(args.chunk_size),
        )

        # Independent row-order witness emitted
        # by the PCA validation stage.
        np.save(row_id_paths[key], loaded_ids[key])

        saved_ids = (
            np.asarray(np.load(row_id_paths[key], allow_pickle=False))
            .reshape(-1)
            .astype(np.int64)
        )

        if not np.array_equal(saved_ids, loaded_ids[key]):
            raise RuntimeError(
                f"{key}: saved PCA row-ID "
                "witness differs from "
                "source embedding IDs."
            )

        transformed_report[key]["row_ids_sha256"] = sha256_file(row_id_paths[key])

        print(
            f"{key:<12}: "
            f"{tuple(transformed_report[key]['shape'])} "
            f"dtype={transformed_report[key]['dtype']}"
        )

    # ------------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------------

    source_script = Path(__file__).resolve()

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "pre-outcome",
        "performance_examined": False,
        "pca_refit": False,
        "additional_scaling": False,
        "pca_output_dtype": "float64",
        "source_script": {
            "path": str(source_script),
            "sha256": sha256_file(source_script),
        },
        "git": git_info(),
        "current_sklearn_version": sklearn.__version__,
        "pca_load_warnings": warning_records,
        "pca": {
            "path": str(args.pca_model),
            "sha256": sha256_file(args.pca_model),
            "n_features_in": n_features,
            "n_components": n_components,
            "explained_variance_sum": variance_sum,
            "expected_explained_variance": args.expected_pca_variance,
            "variance_tolerance": args.pca_variance_tolerance,
        },
        "raw_embeddings": raw_report,
        "pca_outputs": transformed_report,
        "manifest_hashes_sha256": {
            "m3_split_manifest": sha256_file(args.m3_split_manifest),
            "m4_manifest": sha256_file(args.m4_manifest),
        },
    }

    report_path = out_dir / "experiment_b_embedding_validation.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print()
    print("=" * 80)
    print("EMBEDDING / PCA " "VALIDATION PASSED")
    print("=" * 80)

    for key in ["m3_train", "m3_dev", "m3_test", "m4_target"]:
        print(f"{key:<12} PCA : " f"{transformed_paths[key]}")

        print(f"{'':<12} IDs : " f"{row_id_paths[key]}")

    print(f"provenance   : {report_path}")

    print()
    print("No model was fitted and " "no AUROC was computed.")


if __name__ == "__main__":
    main()
