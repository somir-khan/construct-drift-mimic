#!/usr/bin/env python3
"""Run the single prespecified DriftLens comparison for Experiment C.

Scientific settings are intentionally fixed here: 150 PCs, one 5,000-note
target window, 10,000 calibration windows, T_alpha=0.01, and seed 42.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


COHORT_SIZE = 5_000
EMBEDDING_DIM = 768
N_COMPONENTS = 150
N_THRESHOLD_SAMPLES = 10_000
T_ALPHA = 0.01
SEED = 42
PROTOTYPE_K = range(2, 11)
PROTOTYPES_PER_CLUSTER = 4
SILHOUETTE_SAMPLE = 2_000
THRESHOLD_RULE_SOURCE = (
    "experiments/use_case_1_ag_news_science_drift/"
    "use_case_1_drift_detection_accuracy.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--driftlens-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel calibration workers; -1 uses all cores.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_array(path: Path, label: str, rows: int | None = None) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{label}: expected (*,{EMBEDDING_DIM}), got {array.shape}.")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{label}: expected {rows} rows, got {len(array)}.")
    if not np.isfinite(array).all() or np.any(np.all(array == 0, axis=1)):
        raise ValueError(f"{label}: non-finite or all-zero embedding row found.")
    return np.asarray(array)


def load_official_fdd(repo: Path) -> tuple[Any, Path]:
    source = repo / "driftlens" / "distribution_distances" / "frechet_drift_distance.py"
    if not source.is_file():
        raise FileNotFoundError(f"DriftLens FDD source not found: {source}")
    spec = importlib.util.spec_from_file_location("experiment_c_driftlens_fdd", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "frechet_distance", None)):
        raise RuntimeError(f"{source} does not expose frechet_distance().")
    return module.frechet_distance, source


def real_scalar(value: Any, label: str) -> float:
    scalar = np.asarray(value).reshape(()).item()
    if isinstance(scalar, complex):
        if abs(scalar.imag) > 1e-6 * (1.0 + abs(scalar.real)):
            raise FloatingPointError(f"{label} has a material imaginary component: {scalar}")
        scalar = scalar.real
    result = float(scalar)
    if not math.isfinite(result):
        raise FloatingPointError(f"{label} is not finite: {result}")
    return result


def released_formula(
    baseline_mean: np.ndarray,
    window_mean: np.ndarray,
    baseline_covariance: np.ndarray,
    window_covariance: np.ndarray,
) -> float:
    """Formula used by the reviewed DriftLens release (unsquared mean norm)."""
    mean_term = np.linalg.norm(baseline_mean - window_mean)
    root = scipy.linalg.sqrtm(baseline_covariance @ window_covariance)
    covariance_term = real_scalar(
        np.trace(baseline_covariance + window_covariance - 2.0 * root),
        "DriftLens covariance term",
    )
    return float(mean_term + covariance_term)


def distance_for_indices(
    calibration: np.ndarray,
    indices: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_covariance: np.ndarray,
) -> float:
    window = calibration[indices]
    return released_formula(
        baseline_mean,
        window.mean(axis=0),
        baseline_covariance,
        np.cov(window, rowvar=False),
    )


def save_checkpoint(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def calibration_distances(
    calibration: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_covariance: np.ndarray,
    checkpoint: Path,
    jobs: int,
    checkpoint_every: int,
    *,
    samples: int = N_THRESHOLD_SAMPLES,
    window_size: int = COHORT_SIZE,
    seed: int = SEED,
) -> np.ndarray:
    if len(calibration) < window_size:
        raise ValueError(f"Calibration has {len(calibration)} rows; {window_size} are required.")
    if checkpoint.exists():
        distances = np.load(checkpoint, allow_pickle=False)
        if distances.shape != (samples,):
            raise RuntimeError(f"Unexpected checkpoint shape: {distances.shape}")
    else:
        distances = np.full(samples, np.nan, dtype=np.float64)

    missing = np.flatnonzero(~np.isfinite(distances))
    completed = int(missing[0]) if len(missing) else samples
    if np.isfinite(distances[completed:]).any():
        raise RuntimeError("Calibration checkpoint is not a contiguous prefix; start a new output directory.")

    # Replaying these inexpensive draws restores the exact RNG state on resume.
    rng = np.random.RandomState(seed)
    for _ in range(completed):
        selected = rng.choice(len(calibration), window_size, replace=False)
        rng.permutation(window_size)

    for start in range(completed, samples, checkpoint_every):
        stop = min(samples, start + checkpoint_every)
        index_sets: list[np.ndarray] = []
        for _ in range(start, stop):
            selected = rng.choice(len(calibration), window_size, replace=False)
            selected = selected[rng.permutation(window_size)]
            index_sets.append(selected)
        values = Parallel(n_jobs=jobs, max_nbytes="1M", mmap_mode="r")(
            delayed(distance_for_indices)(
                calibration, selected, baseline_mean, baseline_covariance
            )
            for selected in index_sets
        )
        distances[start:stop] = values
        save_checkpoint(checkpoint, distances)
        print(f"DriftLens calibration: {stop}/{samples} windows", flush=True)
    return distances


def threshold_from_release(distances: np.ndarray) -> dict[str, float | int | str]:
    """Match the threshold rule in the released DriftLens experiment script.

    With its default threshold_sensitivity=1, that script retains distances
    strictly between the 1st and 99th empirical quantiles and takes their max.
    """
    lower = float(np.quantile(distances, T_ALPHA))
    upper = float(np.quantile(distances, 1.0 - T_ALPHA))
    retained = distances[(distances > lower) & (distances < upper)]
    if not len(retained):
        raise RuntimeError("DriftLens tail trimming retained no calibration distances.")
    return {
        "value": float(retained.max()),
        "lower_quantile": lower,
        "upper_quantile": upper,
        "retained_windows": int(len(retained)),
        "total_windows": int(len(distances)),
        "t_alpha": T_ALPHA,
        "rule": "max(distance strictly between q_0.01 and q_0.99)",
    }


def prototype_rows(
    embeddings: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    cohort: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scores: dict[int, float] = {}
    for k in PROTOTYPE_K:
        model = KMeans(n_clusters=k, n_init=10, max_iter=1_000, random_state=SEED)
        labels = model.fit_predict(embeddings)
        scores[k] = float(
            silhouette_score(
                embeddings,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE, len(embeddings)),
                random_state=SEED,
            )
        )
    best_k = max(scores, key=lambda k: (scores[k], -k))
    model = KMeans(n_clusters=best_k, n_init=10, max_iter=1_000, random_state=SEED)
    labels = model.fit_predict(embeddings)
    distances = cdist(embeddings, model.cluster_centers_)
    rows: list[dict[str, object]] = []
    for cluster in range(best_k):
        members = np.flatnonzero(labels == cluster)
        nearest = members[
            np.argsort(distances[members, cluster], kind="stable")[:PROTOTYPES_PER_CLUSTER]
        ]
        for rank, row_index in enumerate(nearest, start=1):
            rows.append(
                {
                    "cohort": cohort,
                    "cluster": cluster,
                    "cluster_size": int(len(members)),
                    "prototype_rank": rank,
                    "row_index": int(row_index),
                    "hadm_id": int(ids[row_index]),
                    "source_group": str(groups[row_index]),
                    "distance_to_centroid": float(distances[row_index, cluster]),
                }
            )
    return {
        "best_k": best_k,
        "silhouette_scores": {str(k): value for k, value in scores.items()},
        "silhouette_sample": min(SILHOUETTE_SAMPLE, len(embeddings)),
        "prototypes_per_cluster": PROTOTYPES_PER_CLUSTER,
    }, rows


def main() -> None:
    args = parse_args()
    if args.jobs == 0:
        raise ValueError("--jobs cannot be zero.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")

    prepared = Path(args.prepared_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_array(prepared / "baseline_raw.npy", "baseline", COHORT_SIZE)
    target = load_array(prepared / "target_raw.npy", "target", COHORT_SIZE)
    calibration = load_array(prepared / "calibration_raw.npy", "calibration")
    baseline_ids = np.load(prepared / "baseline_ids.npy", allow_pickle=False).reshape(-1)
    target_ids = np.load(prepared / "target_ids.npy", allow_pickle=False).reshape(-1)
    target_groups = np.load(prepared / "target_source_groups.npy", allow_pickle=False).reshape(-1)

    official_fdd, fdd_source = load_official_fdd(Path(args.driftlens_repo).expanduser().resolve())
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    baseline_reduced = pca.fit_transform(baseline)
    target_reduced = pca.transform(target)
    calibration_reduced = pca.transform(calibration)
    baseline_mean = baseline_reduced.mean(axis=0)
    baseline_covariance = np.cov(baseline_reduced, rowvar=False)
    target_mean = target_reduced.mean(axis=0)
    target_covariance = np.cov(target_reduced, rowvar=False)

    target_distance = real_scalar(
        official_fdd(baseline_mean, target_mean, baseline_covariance, target_covariance),
        "released DriftLens FDD",
    )
    local_check = released_formula(
        baseline_mean, target_mean, baseline_covariance, target_covariance
    )
    if not np.isclose(target_distance, local_check, rtol=1e-6, atol=1e-8):
        raise AssertionError(
            f"Local calibration formula ({local_check}) disagrees with released FDD ({target_distance})."
        )

    distances = calibration_distances(
        calibration_reduced,
        baseline_mean,
        baseline_covariance,
        output / "driftlens_calibration_distances.npy",
        args.jobs,
        args.checkpoint_every,
        samples=N_THRESHOLD_SAMPLES,
        window_size=COHORT_SIZE,
        seed=SEED,
    )
    threshold = threshold_from_release(distances)

    prototype_summary: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    for name, array, ids, groups in (
        ("baseline", baseline, baseline_ids, np.repeat("MIMIC-III", COHORT_SIZE)),
        ("target", target, target_ids, target_groups),
    ):
        summary, rows = prototype_rows(array, ids, groups, name)
        prototype_summary[name] = summary
        all_rows.extend(rows)
    with (output / "driftlens_prototypes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    result = {
        "method": "DriftLens per-batch path",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "frozen primary 5,000-note MIMIC-III baseline vs frozen primary 5,000-note MIMIC-IV target",
        "fdd": target_distance,
        "threshold": threshold,
        "drift_detected": bool(target_distance > float(threshold["value"])),
        "empirical_calibration_percentile": float(np.mean(distances <= target_distance)),
        "settings": {
            "pca_components": N_COMPONENTS,
            "pca_fit": "baseline only",
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "window_size": COHORT_SIZE,
            "threshold_samples": N_THRESHOLD_SAMPLES,
            "threshold_seed": SEED,
            "t_alpha": T_ALPHA,
            "reported_formula": "released implementation; unsquared mean-norm term",
        },
        "qa": {
            "released_function_local_formula_equal": True,
            "absolute_difference": abs(target_distance - local_check),
            "released_source": str(fdd_source),
            "released_source_sha256": sha256_file(fdd_source),
            "threshold_rule_source": THRESHOLD_RULE_SOURCE,
        },
        "prototypes": prototype_summary,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__ if hasattr(scipy, "__version__") else "unknown",
        },
    }
    result_path = output / "driftlens_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"result": str(result_path), "fdd": target_distance, "threshold": threshold["value"], "drift_detected": result["drift_detected"]}, indent=2))


if __name__ == "__main__":
    main()
