#!/usr/bin/env python3
"""
experiment_b_probe_frozen.py

Experiment B — frozen downstream ICD-9 probe.

===========================================================================
OUTCOME-ACCESS DISCIPLINE
===========================================================================

This script has TWO distinct subcommands.

1. tune
   - loads M3 TRAIN and M3 DEV only;
   - has NO CLI arguments for M3 TEST;
   - has NO CLI arguments for M4;
   - tests C = {0.01, 0.1, 1, 10};
   - selects one GLOBAL C by M3-dev macro-AUROC;
   - metric uses the fixed 47-label evaluation set;
   - exact equality retains the smaller C;
   - retains the already-fitted TRAIN-only model at winning C;
   - does NOT refit on train+dev.

2. evaluate
   - loads the already-frozen winning model;
   - hard-requires the SAME source-script SHA-256 as tune;
   - hard-requires the SAME scikit-learn version as tune;
   - rejects any sklearn InconsistentVersionWarning while unpickling;
   - never refits the model;
   - never re-selects C;
   - evaluates M3 TEST and M4 target;
   - performs patient-cluster bootstrap.

===========================================================================
FROZEN PROTOCOL
===========================================================================

Representation:
    BioClinicalBERT frozen embedding
    -> submitted frozen 52-dimensional PCA
    -> NO additional scaling

The absence of an additional scaler is deliberate. L2 regularization therefore
acts on the raw frozen PCA scale. Introducing a scaler fitted during Experiment B
would create an additional learned transformation beyond the submitted frozen
representation.

Candidate labels:
    50 diagnosis-only ICD-9 codes selected from M3 TRAIN only.

Primary evaluable labels:
    47 fixed labels.

Excluded BEFORE model performance was examined:
    V290
    7742
    V053

Reason:
    AUROC is undefined for these codes in M3 dev/test.

Probe:
    independent binary logistic regression for every one of the 50 codes
    penalty='l2'
    solver='liblinear'
    fit_intercept=True
    class_weight=None
    max_iter=2000
    random_state=42

Hyperparameter:
    one GLOBAL C
    grid = [0.01, 0.1, 1, 10]
    selection metric = M3-dev macro-AUROC over fixed 47 labels
    exact equality -> smaller C
    post-selection refit = NO

Primary evaluation:
    source = M3 held-out test
    target = M4 2014-2016 anchor-year-group ICD-9-only cohort

Metric:
    macro-AUROC

Effect:
    delta = M4 macro-AUROC - M3-test macro-AUROC

No classification threshold is used.

Bootstrap:
    independent patient-cluster bootstrap
    B = 2000 VALID replicates
    CI = two-sided percentile 95%
    percentiles = 2.5, 97.5
    RNG = one np.random.default_rng(42)
    model refit = NO
    C re-selection = NO
    label set = same fixed 47 every replicate

If any bootstrap draw becomes single-class for any fixed label:
    discard the WHOLE replicate and redraw;
    consume the next values from the SAME sequential Generator;
    never reset or reseed.

===========================================================================
ROW ALIGNMENT
===========================================================================

The probe refuses to run unless independently emitted row-ID witnesses match
element-for-element:

TUNE:
    PCA train IDs   == label train IDs
    PCA dev IDs     == label dev IDs

EVALUATE:
    PCA test IDs    == label test IDs    == subject test row IDs
    PCA M4 IDs      == label M4 IDs      == subject M4 row IDs

===========================================================================
TUNE -> EVALUATE INTEGRITY
===========================================================================

The frozen model bundle records:
    - exact source-script SHA-256;
    - scikit-learn version;
    - full frozen protocol;
    - selected C;
    - fixed label ordering/mask.

EVALUATE hard-stops if:
    - the current script SHA-256 differs from the tune-time script hash;
    - current sklearn version differs from tune-time sklearn version;
    - unpickling emits an sklearn version inconsistency warning;
    - the label order/evaluation mask differs;
    - the model bundle indicates post-selection refitting.

This prevents changes to BOOTSTRAP_B, BOOTSTRAP_SEED, CI percentiles,
label rules, or any other script-level protocol parameter between tune
and evaluate.
"""

import argparse
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ============================================================================
# FROZEN CONSTANTS
# ============================================================================

C_GRID = (0.01, 0.1, 1.0, 10.0)

EXCLUDED_CODES = ("V290", "7742", "V053")

EXPECTED_K = 50
EXPECTED_EVALUABLE_K = 47

PCA_DIM = 52

SOLVER = "liblinear"
PENALTY = "l2"
FIT_INTERCEPT = True
CLASS_WEIGHT = None
MAX_ITER = 2000
RANDOM_STATE = 42

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

CI_LOWER = 2.5
CI_UPPER = 97.5


# ============================================================================
# GENERAL HELPERS
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


def normalize_icd9(code):
    """
    Canonical ICD-9 normalization.

    MUST remain identical to preflight/materialization.
    """
    if code is None:
        return None

    code = str(code).strip().upper().replace(".", "")

    if not code or code in {"NAN", "NONE", "NULL"}:
        return None

    return code


def is_sklearn_version_warning(record):
    """
    Determine whether a captured warning indicates sklearn pickle-version drift.
    """
    category = str(record.get("category", ""))

    message = str(record.get("message", "")).lower()

    if category == "InconsistentVersionWarning":
        return True

    if "scikit-learn" in message and "version" in message:
        return True

    if "sklearn" in message and "version" in message:
        return True

    return False


# ============================================================================
# ROW-ORDER VALIDATION
# ============================================================================


def load_row_ids(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} row-ID file not found: {path}")

    ids = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)

    ids = ids.astype(np.int64)

    if len(ids) == 0:
        raise RuntimeError(f"{label}: row-ID array is empty.")

    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"{label}: duplicate HADM_IDs " "in row-ID witness.")

    return ids


def assert_exact_row_ids(path_a, path_b, label):
    a = load_row_ids(path_a, f"{label} A")

    b = load_row_ids(path_b, f"{label} B")

    if a.shape != b.shape:
        raise RuntimeError(
            f"{label}: row-ID shape mismatch. " f"{a.shape} vs {b.shape}"
        )

    mismatch = np.flatnonzero(a != b)

    if len(mismatch):
        i = int(mismatch[0])

        raise RuntimeError(
            f"{label}: row ordering is NOT identical. "
            f"First mismatch at row {i}: "
            f"{int(a[i])} != {int(b[i])}"
        )

    return a


# ============================================================================
# LABEL SPACE
# ============================================================================


def load_codes(path):
    codes = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            code = normalize_icd9(line)

            if code:
                codes.append(code)

    if len(codes) != EXPECTED_K:
        raise RuntimeError(f"Expected {EXPECTED_K} codes, " f"got {len(codes)}.")

    if len(set(codes)) != EXPECTED_K:
        raise RuntimeError("Top-50 code list contains duplicates.")

    missing_excluded = [code for code in EXCLUDED_CODES if code not in codes]

    if missing_excluded:
        raise RuntimeError(
            "Frozen excluded codes are "
            "absent from top-50 list: "
            f"{missing_excluded}"
        )

    eval_indices = np.array(
        [i for i, code in enumerate(codes) if code not in EXCLUDED_CODES],
        dtype=np.int64,
    )

    if len(eval_indices) != EXPECTED_EVALUABLE_K:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUABLE_K} "
            "evaluable labels, got "
            f"{len(eval_indices)}."
        )

    return (codes, eval_indices)


# ============================================================================
# DATA LOADERS
# ============================================================================


def load_X(path, label):
    X = np.asarray(np.load(path, allow_pickle=False))

    if X.ndim != 2:
        raise RuntimeError(f"{label}: expected 2-D X, " f"got {X.shape}.")

    if X.shape[1] != PCA_DIM:
        raise RuntimeError(
            f"{label}: expected " f"{PCA_DIM} PCA dimensions, " f"got {X.shape[1]}."
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"{label}: NaN/Inf present " "in PCA features.")

    if X.dtype != np.float64:
        raise RuntimeError(f"{label}: PCA matrix should " f"be float64, got {X.dtype}.")

    return X


def load_Y(path, expected_n, label):
    Y = np.asarray(np.load(path, allow_pickle=False))

    expected_shape = (expected_n, EXPECTED_K)

    if Y.shape != expected_shape:
        raise RuntimeError(
            f"{label}: expected label " f"shape {expected_shape}, " f"got {Y.shape}."
        )

    unique = np.unique(Y)

    if not set(unique.tolist()).issubset({0, 1}):
        raise RuntimeError(
            f"{label}: labels are not binary. " f"Values={unique.tolist()}"
        )

    return Y.astype(np.uint8, copy=False)


def load_subjects(path, expected_n, label):
    subjects = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)

    subjects = subjects.astype(np.int64)

    if len(subjects) != expected_n:
        raise RuntimeError(
            f"{label}: expected "
            f"{expected_n:,} subject rows, "
            f"got {len(subjects):,}."
        )

    return subjects


# ============================================================================
# CLASS / AUROC HELPERS
# ============================================================================


def validate_classes(Y, indices, label):
    n = len(Y)

    positive = Y[:, indices].sum(axis=0, dtype=np.int64)

    bad = np.flatnonzero((positive <= 0) | (positive >= n))

    if len(bad):
        raise RuntimeError(
            f"{label}: one or more "
            "fixed evaluable labels "
            "are single-class. "
            f"Relative positions={bad.tolist()}"
        )


def macro_auc(Y, scores, eval_indices):
    validate_classes(Y, eval_indices, "macro-AUROC input")

    return float(
        roc_auc_score(Y[:, eval_indices], scores[:, eval_indices], average="macro")
    )


def prevalence_table(Y, codes, eval_indices, prefix):
    rows = []

    for idx in eval_indices:
        positives = int(Y[:, idx].sum())

        rows.append(
            {
                "label_index": int(idx),
                "code": codes[idx],
                f"{prefix}_n": int(len(Y)),
                f"{prefix}_positive": positives,
                f"{prefix}_prevalence": float(positives / len(Y)),
            }
        )

    return pd.DataFrame(rows)


def per_code_auc(Y, scores, codes, eval_indices, prefix):
    rows = []

    for idx in eval_indices:
        y = Y[:, idx]

        positives = int(y.sum())

        negatives = int(len(y) - positives)

        if positives == 0 or negatives == 0:
            raise RuntimeError(
                "Fixed evaluable code " f"{codes[idx]} became " "single-class."
            )

        auc = float(roc_auc_score(y, scores[:, idx]))

        rows.append({"label_index": int(idx), "code": codes[idx], f"{prefix}_auc": auc})

    return pd.DataFrame(rows)


# ============================================================================
# MODEL HELPERS
# ============================================================================


def new_model(C):
    return LogisticRegression(
        penalty=PENALTY,
        C=float(C),
        solver=SOLVER,
        fit_intercept=FIT_INTERCEPT,
        class_weight=CLASS_WEIGHT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )


def fit_50_models(X_train, Y_train, C):
    models = []

    for j in range(EXPECTED_K):
        y = Y_train[:, j]

        positives = int(y.sum())

        if positives == 0 or positives == len(y):
            raise RuntimeError("Training label column " f"{j} is single-class.")

        model = new_model(C)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            model.fit(X_train, y)

        convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]

        if convergence:
            raise RuntimeError(
                "ConvergenceWarning at "
                f"C={C}, label index={j}. "
                "Do not use an "
                "unconverged frozen probe."
            )

        models.append(model)

    return models


def score_models(models, X):
    scores = np.empty((len(X), EXPECTED_K), dtype=np.float64)

    for j, model in enumerate(models):
        scores[:, j] = model.decision_function(X)

    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Non-finite probe scores.")

    return scores


def save_coefficients(models, codes, path):
    rows = []

    for j, (code, model) in enumerate(zip(codes, models)):
        row = {"label_index": j, "code": code, "intercept": float(model.intercept_[0])}

        coef = model.coef_[0]

        for k, value in enumerate(coef):
            row[f"pc_{k + 1:02d}"] = float(value)

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================================
# FROZEN PROTOCOL RECORD
# ============================================================================


def protocol_dict():
    source_script = Path(__file__).resolve()

    return {
        "protocol_status": "frozen_pre_outcome",
        "source_script": {
            "path": str(source_script),
            "sha256": sha256_file(source_script),
        },
        "git": git_info(),
        "representation": {
            "input": (
                "submitted BioClinicalBERT "
                "embeddings transformed by "
                "submitted frozen PCA"
            ),
            "pca_dimensions": PCA_DIM,
            "pca_refit": False,
            "additional_scaling": False,
            "scaling_rationale": (
                "No secondary learned scaling "
                "transformation is introduced. "
                "L2 regularization therefore "
                "acts on the raw frozen PCA "
                "component scale."
            ),
        },
        "labels": {
            "candidate_k": EXPECTED_K,
            "candidate_source": (
                "top diagnosis-only ICD-9 "
                "codes selected from "
                "MIMIC-III training only"
            ),
            "primary_evaluable_k": EXPECTED_EVALUABLE_K,
            "excluded_before_modeling": list(EXCLUDED_CODES),
            "exclusion_reason": ("AUROC undefined in " "MIMIC-III dev/test"),
            "normalization": (
                "uppercase; strip whitespace; "
                "remove decimal; reject "
                "NAN/NONE/NULL; preserve "
                "leading zeros"
            ),
        },
        "probe": {
            "type": ("50 independent one-vs-rest " "binary logistic regressions"),
            "penalty": PENALTY,
            "solver": SOLVER,
            "fit_intercept": FIT_INTERCEPT,
            "class_weight": CLASS_WEIGHT,
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "score": "decision_function",
        },
        "hyperparameter": {
            "global_C_grid": list(C_GRID),
            "selection_dataset": "MIMIC-III dev only",
            "selection_metric": ("macro-AUROC over " "fixed 47 labels"),
            "tie_break": "smaller C on exact equality",
            "post_selection_refit": False,
            "winning_model_policy": (
                "retain already-fitted "
                "train-only models "
                "corresponding to selected C"
            ),
        },
        "primary_evaluation": {
            "source": "MIMIC-III held-out test",
            "target": ("MIMIC-IV 2014-2016 " "anchor-year-group " "ICD-9-only target"),
            "target_calendar_time_interpretation": False,
            "metric": "macro-AUROC",
            "effect": ("M4 macro-AUROC minus " "M3-test macro-AUROC"),
            "decision_threshold": None,
            "target_tuning": False,
        },
        "bootstrap": {
            "type": ("independent " "patient-cluster bootstrap"),
            "valid_replicates": BOOTSTRAP_B,
            "rng": "numpy.random.default_rng",
            "seed": BOOTSTRAP_SEED,
            "ci": "two-sided percentile 95%",
            "percentiles": [CI_LOWER, CI_UPPER],
            "model_refit": False,
            "C_reselection": False,
            "label_set": "fixed 47 labels",
            "single_class_rule": (
                "discard whole replicate "
                "and redraw using subsequent "
                "values from the same "
                "sequential RNG stream"
            ),
        },
        "row_alignment": {
            "tune": ("PCA row IDs must equal " "label row IDs elementwise"),
            "evaluate": (
                "PCA row IDs, label row IDs, "
                "and subject-vector row IDs "
                "must all match elementwise"
            ),
        },
    }


# ============================================================================
# FAST PATIENT CLUSTER CONSTRUCTION
# ============================================================================


def prepare_clusters(subject_ids):
    subject_ids = np.asarray(subject_ids).reshape(-1).astype(np.int64)

    if len(subject_ids) == 0:
        raise RuntimeError("Subject-ID vector is empty.")

    order = np.argsort(subject_ids, kind="stable")

    sorted_subjects = subject_ids[order]

    (unique_subjects, starts) = np.unique(sorted_subjects, return_index=True)

    ends = np.concatenate([starts[1:], np.array([len(order)], dtype=starts.dtype)])

    rows_by_subject = [order[start:end] for start, end in zip(starts, ends)]

    reconstructed_n = sum(len(rows) for rows in rows_by_subject)

    if reconstructed_n != len(subject_ids):
        raise RuntimeError("Patient-cluster construction " "lost or duplicated rows.")

    return (unique_subjects, rows_by_subject)


def draw_cluster_rows(rng, rows_by_subject):
    n_subjects = len(rows_by_subject)

    drawn = rng.integers(0, n_subjects, size=n_subjects)

    return np.concatenate([rows_by_subject[i] for i in drawn])


def bootstrap_macro_auc(Y, scores, eval_indices, row_indices):
    Yb = Y[row_indices]

    Sb = scores[row_indices]

    positives = Yb[:, eval_indices].sum(axis=0, dtype=np.int64)

    n = len(Yb)

    if np.any(positives <= 0) or np.any(positives >= n):
        return None

    return float(
        roc_auc_score(Yb[:, eval_indices], Sb[:, eval_indices], average="macro")
    )


# ============================================================================
# TUNE — DEV ONLY
# ============================================================================


def run_tune(args):
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Row alignment BEFORE loading/fitting model
    # ------------------------------------------------------------------------

    train_ids = assert_exact_row_ids(
        args.train_pca_ids, args.train_label_ids, "M3 train PCA vs labels"
    )

    dev_ids = assert_exact_row_ids(
        args.dev_pca_ids, args.dev_label_ids, "M3 dev PCA vs labels"
    )

    # ------------------------------------------------------------------------
    # Freeze runtime protocol BEFORE fitting
    # ------------------------------------------------------------------------

    protocol = protocol_dict()

    protocol["software"] = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }

    protocol["input_hashes_sha256"] = {
        "train_pca": sha256_file(args.train_pca),
        "train_pca_ids": sha256_file(args.train_pca_ids),
        "train_labels": sha256_file(args.train_labels),
        "train_label_ids": sha256_file(args.train_label_ids),
        "dev_pca": sha256_file(args.dev_pca),
        "dev_pca_ids": sha256_file(args.dev_pca_ids),
        "dev_labels": sha256_file(args.dev_labels),
        "dev_label_ids": sha256_file(args.dev_label_ids),
        "top50_codes": sha256_file(args.top50_codes),
    }

    protocol_path = out_dir / "experiment_b_frozen_probe_protocol.json"

    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print("=" * 80)
    print("EXPERIMENT B — DEV-ONLY TUNE")
    print("=" * 80)

    print("Frozen protocol written before fitting:")

    print(protocol_path)

    # ------------------------------------------------------------------------
    # Labels / evaluation mask
    # ------------------------------------------------------------------------

    (codes, eval_indices) = load_codes(args.top50_codes)

    # ------------------------------------------------------------------------
    # Features / labels
    # ------------------------------------------------------------------------

    X_train = load_X(args.train_pca, "M3 train")

    X_dev = load_X(args.dev_pca, "M3 dev")

    if len(X_train) != len(train_ids):
        raise RuntimeError("M3 train PCA row count " "does not equal row-ID count.")

    if len(X_dev) != len(dev_ids):
        raise RuntimeError("M3 dev PCA row count " "does not equal row-ID count.")

    Y_train = load_Y(args.train_labels, len(X_train), "M3 train")

    Y_dev = load_Y(args.dev_labels, len(X_dev), "M3 dev")

    validate_classes(Y_train, np.arange(EXPECTED_K), "M3 train all 50")

    validate_classes(Y_dev, eval_indices, "M3 dev fixed 47")

    print()
    print(f"M3 train: {X_train.shape}")

    print(f"M3 dev  : {X_dev.shape}")

    print("Candidate labels: " f"{EXPECTED_K}")

    print("Primary evaluable labels: " f"{len(eval_indices)}")

    print("Excluded before modeling: " f"{list(EXCLUDED_CODES)}")

    # ------------------------------------------------------------------------
    # C grid
    # ------------------------------------------------------------------------

    all_models = {}
    tuning_rows = []
    per_code_rows = []

    best_C = None
    best_macro = None

    # Grid is deliberately ascending.
    # Winner changes ONLY on strict improvement.
    # Therefore exact ties retain the smaller C.
    for C in C_GRID:
        print()
        print(f"Fitting C={C:g} ...")

        models = fit_50_models(X_train, Y_train, C)

        scores_dev = score_models(models, X_dev)

        auc_macro = macro_auc(Y_dev, scores_dev, eval_indices)

        all_models[float(C)] = models

        tuning_rows.append({"C": float(C), "m3_dev_macro_auc_47": auc_macro})

        per_df = per_code_auc(Y_dev, scores_dev, codes, eval_indices, prefix="m3_dev")

        per_df.insert(0, "C", float(C))

        per_code_rows.append(per_df)

        print("  M3-dev macro-AUROC (47): " f"{auc_macro:.9f}")

        if best_macro is None or auc_macro > best_macro:
            best_macro = auc_macro
            best_C = float(C)

    # ------------------------------------------------------------------------
    # Retain already-fitted winning train-only model
    # ------------------------------------------------------------------------

    winning_models = all_models[best_C]

    tune_df = pd.DataFrame(tuning_rows)

    tune_path = out_dir / "experiment_b_dev_c_selection.csv"

    tune_df.to_csv(tune_path, index=False)

    per_code_path = out_dir / "experiment_b_dev_auc_by_code_and_c.csv"

    pd.concat(per_code_rows, ignore_index=True).to_csv(per_code_path, index=False)

    # ------------------------------------------------------------------------
    # Frozen winning model
    # ------------------------------------------------------------------------

    model_path = out_dir / "experiment_b_frozen_probe.joblib"

    bundle = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "selected_C": best_C,
        "selected_dev_macro_auc_47": best_macro,
        "codes": codes,
        "eval_indices": eval_indices,
        "excluded_codes": EXCLUDED_CODES,
        "models": winning_models,
        "sklearn_version": sklearn.__version__,
        "post_selection_refit": False,
        "train_row_ids_sha256": sha256_file(args.train_pca_ids),
        "dev_row_ids_sha256": sha256_file(args.dev_pca_ids),
    }

    joblib.dump(bundle, model_path)

    coef_path = out_dir / "experiment_b_frozen_probe_coefficients.csv"

    save_coefficients(winning_models, codes, coef_path)

    tune_summary = {
        "selected_C": best_C,
        "selected_dev_macro_auc_47": best_macro,
        "tie_break_rule": "smaller C on exact equality",
        "post_selection_refit": False,
        "test_loaded": False,
        "m4_loaded": False,
        "source_script_sha256": protocol["source_script"]["sha256"],
        "sklearn_version": sklearn.__version__,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "protocol_sha256": sha256_file(protocol_path),
    }

    summary_path = out_dir / "experiment_b_tune_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(tune_summary, f, indent=2)

    # ------------------------------------------------------------------------
    # Visible dev result
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("DEV-ONLY C SELECTION COMPLETE")
    print("=" * 80)

    print(
        tune_df.to_string(
            index=False, formatters={"m3_dev_macro_auc_47": lambda x: f"{x:.9f}"}
        )
    )

    print()
    print(f"Selected C : {best_C:g}")

    print(f"Dev AUROC  : {best_macro:.9f}")

    print()
    print(f"Frozen model : {model_path}")

    print(f"Tune table   : {tune_path}")

    print(f"Summary      : {summary_path}")

    print()
    print("STOP HERE.")

    print("This command had no access to " "M3 test or M4 data.")


# ============================================================================
# EVALUATE — TEST + M4 ONLY AFTER C IS FROZEN
# ============================================================================


def run_evaluate(args):
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Row alignment BEFORE evaluation
    # ------------------------------------------------------------------------

    test_ids = assert_exact_row_ids(
        args.test_pca_ids, args.test_label_ids, "M3 test PCA vs labels"
    )

    assert_exact_row_ids(
        args.test_pca_ids, args.test_subject_row_ids, "M3 test PCA vs subjects"
    )

    m4_ids = assert_exact_row_ids(
        args.m4_pca_ids, args.m4_label_ids, "M4 PCA vs labels"
    )

    assert_exact_row_ids(args.m4_pca_ids, args.m4_subject_row_ids, "M4 PCA vs subjects")

    # ------------------------------------------------------------------------
    # Frozen model bundle load
    # ------------------------------------------------------------------------

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        bundle = joblib.load(args.model_bundle)

    bundle_warnings = [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught
    ]

    # ------------------------------------------------------------------------
    # Hard gate: sklearn unpickle-version warning
    # ------------------------------------------------------------------------

    inconsistent_warnings = [
        record for record in bundle_warnings if is_sklearn_version_warning(record)
    ]

    if inconsistent_warnings:
        formatted = "\n".join(
            (f"[{w['category']}] " f"{w['message']}") for w in inconsistent_warnings
        )

        raise RuntimeError(
            "Frozen probe bundle emitted a "
            "scikit-learn version warning "
            "while unpickling.\n\n"
            f"{formatted}\n\n"
            "Evaluation is stopped. Use the "
            "same scikit-learn environment "
            "that produced the tune-time model."
        )

    # ------------------------------------------------------------------------
    # Bundle structure
    # ------------------------------------------------------------------------

    required_bundle = {
        "protocol",
        "selected_C",
        "codes",
        "eval_indices",
        "models",
        "post_selection_refit",
        "sklearn_version",
    }

    missing = required_bundle - set(bundle)

    if missing:
        raise RuntimeError("Frozen model bundle is missing " f"keys: {sorted(missing)}")

    # ------------------------------------------------------------------------
    # Hard gate: exact sklearn version
    # ------------------------------------------------------------------------

    tune_sklearn_version = str(bundle["sklearn_version"])

    current_sklearn_version = str(sklearn.__version__)

    if tune_sklearn_version != current_sklearn_version:
        raise RuntimeError(
            "scikit-learn version differs "
            "between tune and evaluate.\n"
            f"  tune     : "
            f"{tune_sklearn_version}\n"
            f"  evaluate : "
            f"{current_sklearn_version}\n"
            "Evaluation is stopped so the "
            "frozen logistic-regression "
            "objects are not used under a "
            "different sklearn runtime."
        )

    # ------------------------------------------------------------------------
    # Hard gate: exact source-script SHA-256
    # ------------------------------------------------------------------------

    protocol = bundle["protocol"]

    if not isinstance(protocol, dict):
        raise RuntimeError("Frozen model bundle protocol " "is not a dictionary.")

    source_record = protocol.get("source_script")

    if not isinstance(source_record, dict):
        raise RuntimeError("Frozen protocol has no " "source_script provenance.")

    tune_script_sha = source_record.get("sha256")

    if not tune_script_sha:
        raise RuntimeError(
            "Frozen protocol does not contain " "a tune-time source-script SHA-256."
        )

    current_script = Path(__file__).resolve()

    current_script_sha = sha256_file(current_script)

    if current_script_sha != tune_script_sha:
        raise RuntimeError(
            "experiment_b_probe_frozen.py "
            "changed between tune and evaluate.\n"
            f"  tune-time SHA-256    : "
            f"{tune_script_sha}\n"
            f"  evaluate-time SHA-256: "
            f"{current_script_sha}\n\n"
            "Evaluation is stopped because "
            "script-level frozen parameters "
            "such as BOOTSTRAP_B, seed, CI "
            "construction, or other protocol "
            "logic may have changed."
        )

    # ------------------------------------------------------------------------
    # Hard gate: no post-selection refit
    # ------------------------------------------------------------------------

    if bundle["post_selection_refit"]:
        raise RuntimeError(
            "Frozen model bundle indicates "
            "post-selection refitting, which "
            "violates the frozen protocol."
        )

    # ------------------------------------------------------------------------
    # Reconstruct current fixed label rules
    # ------------------------------------------------------------------------

    (codes_file, eval_indices_file) = load_codes(args.top50_codes)

    codes_bundle = [normalize_icd9(c) for c in bundle["codes"]]

    if codes_bundle != codes_file:
        raise RuntimeError(
            "Top-50 code order differs "
            "between frozen model bundle "
            "and supplied code file."
        )

    eval_indices_bundle = np.asarray(bundle["eval_indices"], dtype=np.int64)

    if not np.array_equal(eval_indices_bundle, eval_indices_file):
        raise RuntimeError(
            "Fixed 47-label evaluation " "mask differs from frozen bundle."
        )

    models = bundle["models"]

    if len(models) != EXPECTED_K:
        raise RuntimeError(
            "Frozen bundle contains "
            f"{len(models)} models; "
            f"expected {EXPECTED_K}."
        )

    selected_C = float(bundle["selected_C"])

    if selected_C not in C_GRID:
        raise RuntimeError(f"Selected C={selected_C} " "is outside frozen grid.")

    # ------------------------------------------------------------------------
    # Also verify protocol bootstrap constants explicitly
    # ------------------------------------------------------------------------

    frozen_bootstrap = protocol.get("bootstrap", {})

    frozen_B = frozen_bootstrap.get("valid_replicates")

    frozen_seed = frozen_bootstrap.get("seed")

    frozen_percentiles = frozen_bootstrap.get("percentiles")

    if frozen_B != BOOTSTRAP_B:
        raise RuntimeError(
            "Bootstrap replicate count differs "
            "from tune-time protocol.\n"
            f"  frozen : {frozen_B}\n"
            f"  current: {BOOTSTRAP_B}"
        )

    if frozen_seed != BOOTSTRAP_SEED:
        raise RuntimeError(
            "Bootstrap RNG seed differs "
            "from tune-time protocol.\n"
            f"  frozen : {frozen_seed}\n"
            f"  current: {BOOTSTRAP_SEED}"
        )

    expected_percentiles = [CI_LOWER, CI_UPPER]

    if frozen_percentiles != expected_percentiles:
        raise RuntimeError(
            "Bootstrap CI percentiles differ "
            "from tune-time protocol.\n"
            f"  frozen : "
            f"{frozen_percentiles}\n"
            f"  current: "
            f"{expected_percentiles}"
        )

    print("=" * 80)
    print("EXPERIMENT B — FINAL EVALUATION")
    print("=" * 80)

    print(f"Frozen selected C : " f"{selected_C:g}")

    print("Model refit        : NO")

    print("C re-selection     : NO")

    print("Target adaptation  : NONE")

    print(f"sklearn version    : " f"{current_sklearn_version} " "(matches tune)")

    print("source script hash : MATCH")

    # ------------------------------------------------------------------------
    # Evaluation data
    # ------------------------------------------------------------------------

    X_test = load_X(args.test_pca, "M3 test")

    X_m4 = load_X(args.m4_pca, "M4 target")

    if len(X_test) != len(test_ids):
        raise RuntimeError("M3 test feature matrix " "does not match its row IDs.")

    if len(X_m4) != len(m4_ids):
        raise RuntimeError("M4 feature matrix does not " "match its row IDs.")

    Y_test = load_Y(args.test_labels, len(X_test), "M3 test")

    Y_m4 = load_Y(args.m4_labels, len(X_m4), "M4 target")

    subjects_test = load_subjects(args.test_subjects, len(X_test), "M3 test")

    subjects_m4 = load_subjects(args.m4_subjects, len(X_m4), "M4 target")

    validate_classes(Y_test, eval_indices_bundle, "M3 test fixed 47")

    validate_classes(Y_m4, eval_indices_bundle, "M4 target fixed 47")

    # ------------------------------------------------------------------------
    # Prevalence FIRST
    # ------------------------------------------------------------------------

    prev_test = prevalence_table(
        Y_test, codes_file, eval_indices_bundle, prefix="m3_test"
    )

    prev_m4 = prevalence_table(Y_m4, codes_file, eval_indices_bundle, prefix="m4")

    prevalence = prev_test.merge(
        prev_m4, on=["label_index", "code"], how="inner", validate="one_to_one"
    )

    prevalence["prevalence_delta_m4_minus_m3"] = (
        prevalence["m4_prevalence"] - prevalence["m3_test_prevalence"]
    )

    prevalence_path = out_dir / "experiment_b_prevalence_47.csv"

    prevalence.to_csv(prevalence_path, index=False)

    print()
    print("-" * 80)
    print("PREVALENCE — FIXED 47 LABELS")
    print("-" * 80)

    print(
        prevalence[
            [
                "code",
                "m3_test_positive",
                "m3_test_prevalence",
                "m4_positive",
                "m4_prevalence",
            ]
        ].to_string(
            index=False,
            formatters={
                "m3_test_prevalence": lambda x: f"{x:.6f}",
                "m4_prevalence": lambda x: f"{x:.6f}",
            },
        )
    )

    # ------------------------------------------------------------------------
    # Fixed-model predictions
    # ------------------------------------------------------------------------

    scores_test = score_models(models, X_test)

    scores_m4 = score_models(models, X_m4)

    # ------------------------------------------------------------------------
    # Per-code AUROC
    # ------------------------------------------------------------------------

    test_auc_df = per_code_auc(
        Y_test, scores_test, codes_file, eval_indices_bundle, prefix="m3_test"
    )

    m4_auc_df = per_code_auc(
        Y_m4, scores_m4, codes_file, eval_indices_bundle, prefix="m4"
    )

    comparison = prevalence.merge(
        test_auc_df, on=["label_index", "code"], how="inner", validate="one_to_one"
    ).merge(m4_auc_df, on=["label_index", "code"], how="inner", validate="one_to_one")

    comparison["auc_delta_m4_minus_m3"] = (
        comparison["m4_auc"] - comparison["m3_test_auc"]
    )

    per_code_path = out_dir / "experiment_b_prevalence_and_per_code_auc.csv"

    comparison.to_csv(per_code_path, index=False)

    # ------------------------------------------------------------------------
    # Primary point estimates
    # ------------------------------------------------------------------------

    auc_test = macro_auc(Y_test, scores_test, eval_indices_bundle)

    auc_m4 = macro_auc(Y_m4, scores_m4, eval_indices_bundle)

    delta = auc_m4 - auc_test

    # ------------------------------------------------------------------------
    # Fast patient-cluster structures
    # ------------------------------------------------------------------------

    (unique_test, clusters_test) = prepare_clusters(subjects_test)

    (unique_m4, clusters_m4) = prepare_clusters(subjects_m4)

    print()
    print("-" * 80)
    print("PATIENT-CLUSTER BOOTSTRAP")
    print("-" * 80)

    print(f"M3 test patients : " f"{len(unique_test):,}")

    print(f"M4 patients      : " f"{len(unique_m4):,}")

    print(f"Valid replicates : " f"{BOOTSTRAP_B:,}")

    print("RNG              : " "np.random.default_rng(42)")

    # ------------------------------------------------------------------------
    # ONE sequential Generator
    # ------------------------------------------------------------------------

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot_test = np.empty(BOOTSTRAP_B, dtype=np.float64)

    boot_m4 = np.empty(BOOTSTRAP_B, dtype=np.float64)

    boot_delta = np.empty(BOOTSTRAP_B, dtype=np.float64)

    valid = 0
    attempts = 0
    redraws = 0

    max_attempts = BOOTSTRAP_B * 100

    while valid < BOOTSTRAP_B:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Too many invalid bootstrap "
                "draws.\n"
                f"valid={valid}\n"
                f"attempts={attempts}\n"
                f"redraws={redraws}"
            )

        # Draw BOTH cohorts from the SAME
        # sequential RNG object.
        test_rows = draw_cluster_rows(rng, clusters_test)

        m4_rows = draw_cluster_rows(rng, clusters_m4)

        auc_test_b = bootstrap_macro_auc(
            Y_test, scores_test, eval_indices_bundle, test_rows
        )

        auc_m4_b = bootstrap_macro_auc(Y_m4, scores_m4, eval_indices_bundle, m4_rows)

        # Whole replicate is invalid.
        # Do NOT reset/reseed RNG.
        if auc_test_b is None or auc_m4_b is None:
            redraws += 1

            if redraws <= 10 or redraws % 100 == 0:
                print(
                    "  bootstrap redraw: "
                    f"{redraws:,} "
                    f"(attempts={attempts:,}, "
                    f"valid={valid:,})"
                )

            continue

        boot_test[valid] = auc_test_b

        boot_m4[valid] = auc_m4_b

        boot_delta[valid] = auc_m4_b - auc_test_b

        valid += 1

        if valid % 200 == 0:
            print(
                "  valid bootstrap "
                f"replicates: "
                f"{valid:,}/"
                f"{BOOTSTRAP_B:,} "
                f"| redraws={redraws:,} "
                f"| attempts={attempts:,}"
            )

    # ------------------------------------------------------------------------
    # Percentile CIs
    # ------------------------------------------------------------------------

    ci_test = np.percentile(boot_test, [CI_LOWER, CI_UPPER])

    ci_m4 = np.percentile(boot_m4, [CI_LOWER, CI_UPPER])

    ci_delta = np.percentile(boot_delta, [CI_LOWER, CI_UPPER])

    # ------------------------------------------------------------------------
    # Save bootstrap distribution
    # ------------------------------------------------------------------------

    bootstrap_df = pd.DataFrame(
        {
            "replicate": np.arange(1, BOOTSTRAP_B + 1),
            "m3_test_macro_auc": boot_test,
            "m4_macro_auc": boot_m4,
            "delta_m4_minus_m3": boot_delta,
        }
    )

    bootstrap_csv = out_dir / "experiment_b_patient_cluster_bootstrap.csv"

    bootstrap_df.to_csv(bootstrap_csv, index=False)

    bootstrap_delta_npy = out_dir / "experiment_b_bootstrap_delta.npy"

    np.save(bootstrap_delta_npy, boot_delta)

    # ------------------------------------------------------------------------
    # Final result JSON
    # ------------------------------------------------------------------------

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_bundle": str(args.model_bundle),
        "model_bundle_sha256": sha256_file(args.model_bundle),
        "source_script_sha256": current_script_sha,
        "source_script_matches_tune": True,
        "tune_sklearn_version": tune_sklearn_version,
        "evaluate_sklearn_version": current_sklearn_version,
        "sklearn_version_matches_tune": True,
        "selected_C": selected_C,
        "candidate_labels": EXPECTED_K,
        "primary_evaluable_labels": EXPECTED_EVALUABLE_K,
        "excluded_codes": list(EXCLUDED_CODES),
        "m3_test": {
            "n_admissions": int(len(X_test)),
            "n_patients": int(len(unique_test)),
            "macro_auc": auc_test,
            "bootstrap_percentile_95_ci": [float(ci_test[0]), float(ci_test[1])],
        },
        "m4_target": {
            "cohort": ("2014-2016 " "anchor-year-group " "ICD-9-only"),
            "calendar_time_interpretation": False,
            "n_admissions": int(len(X_m4)),
            "n_patients": int(len(unique_m4)),
            "macro_auc": auc_m4,
            "bootstrap_percentile_95_ci": [float(ci_m4[0]), float(ci_m4[1])],
        },
        "primary_effect": {
            "definition": ("M4 macro-AUROC minus " "M3-test macro-AUROC"),
            "delta_macro_auc": delta,
            "bootstrap_percentile_95_ci": [float(ci_delta[0]), float(ci_delta[1])],
        },
        "bootstrap": {
            "valid_replicates": BOOTSTRAP_B,
            "attempts": attempts,
            "redraws": redraws,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.random.default_rng",
            "ci_method": "percentile",
            "ci_percentiles": [CI_LOWER, CI_UPPER],
            "model_refit": False,
            "C_reselection": False,
        },
        "model_bundle_load_warnings": bundle_warnings,
        "software": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "input_hashes_sha256": {
            "test_pca": sha256_file(args.test_pca),
            "test_pca_ids": sha256_file(args.test_pca_ids),
            "test_labels": sha256_file(args.test_labels),
            "test_label_ids": sha256_file(args.test_label_ids),
            "test_subjects": sha256_file(args.test_subjects),
            "test_subject_row_ids": sha256_file(args.test_subject_row_ids),
            "m4_pca": sha256_file(args.m4_pca),
            "m4_pca_ids": sha256_file(args.m4_pca_ids),
            "m4_labels": sha256_file(args.m4_labels),
            "m4_label_ids": sha256_file(args.m4_label_ids),
            "m4_subjects": sha256_file(args.m4_subjects),
            "m4_subject_row_ids": sha256_file(args.m4_subject_row_ids),
            "top50_codes": sha256_file(args.top50_codes),
        },
    }

    result_path = out_dir / "experiment_b_final_results.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ------------------------------------------------------------------------
    # Final visible outcome
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("EXPERIMENT B — PRIMARY RESULT")
    print("=" * 80)

    print(
        "M3 test macro-AUROC : "
        f"{auc_test:.6f} "
        f"[{ci_test[0]:.6f}, "
        f"{ci_test[1]:.6f}]"
    )

    print(
        "M4 macro-AUROC      : "
        f"{auc_m4:.6f} "
        f"[{ci_m4[0]:.6f}, "
        f"{ci_m4[1]:.6f}]"
    )

    print(
        "Delta (M4 - M3)     : "
        f"{delta:+.6f} "
        f"[{ci_delta[0]:+.6f}, "
        f"{ci_delta[1]:+.6f}]"
    )

    print()
    print(f"Bootstrap redraws    : " f"{redraws:,}")

    print(f"Bootstrap attempts   : " f"{attempts:,}")

    print()
    print(f"Prevalence table      : " f"{prevalence_path}")

    print(f"Per-code table        : " f"{per_code_path}")

    print(f"Bootstrap distribution: " f"{bootstrap_csv}")

    print(f"Final JSON            : " f"{result_path}")


# ============================================================================
# CLI
# ============================================================================


def build_parser():
    p = argparse.ArgumentParser(
        description=("Experiment B frozen " "logistic-regression probe."),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )

    sub = p.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------------
    # TUNE
    # ------------------------------------------------------------------------

    tune = sub.add_parser(
        "tune",
        help=("M3 train/dev only. " "No test or M4 inputs exist."),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )

    tune.add_argument("--train-pca", required=True)

    tune.add_argument("--train-pca-ids", required=True)

    tune.add_argument("--train-labels", required=True)

    tune.add_argument("--train-label-ids", required=True)

    tune.add_argument("--dev-pca", required=True)

    tune.add_argument("--dev-pca-ids", required=True)

    tune.add_argument("--dev-labels", required=True)

    tune.add_argument("--dev-label-ids", required=True)

    tune.add_argument("--top50-codes", required=True)

    tune.add_argument("--out-dir", default=("outputs/" "experiment_b_probe"))

    # ------------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------------

    evaluate = sub.add_parser(
        "evaluate",
        help=("Evaluate the already-frozen " "train-only winning model."),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )

    evaluate.add_argument("--model-bundle", required=True)

    evaluate.add_argument("--test-pca", required=True)

    evaluate.add_argument("--test-pca-ids", required=True)

    evaluate.add_argument("--test-labels", required=True)

    evaluate.add_argument("--test-label-ids", required=True)

    evaluate.add_argument("--test-subjects", required=True)

    evaluate.add_argument("--test-subject-row-ids", required=True)

    evaluate.add_argument("--m4-pca", required=True)

    evaluate.add_argument("--m4-pca-ids", required=True)

    evaluate.add_argument("--m4-labels", required=True)

    evaluate.add_argument("--m4-label-ids", required=True)

    evaluate.add_argument("--m4-subjects", required=True)

    evaluate.add_argument("--m4-subject-row-ids", required=True)

    evaluate.add_argument("--top50-codes", required=True)

    evaluate.add_argument("--out-dir", default=("outputs/" "experiment_b_probe"))

    return p


def main():
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "tune":
        run_tune(args)

    elif args.command == "evaluate":
        run_evaluate(args)

    else:
        raise RuntimeError(f"Unknown command: " f"{args.command}")


if __name__ == "__main__":
    main()
