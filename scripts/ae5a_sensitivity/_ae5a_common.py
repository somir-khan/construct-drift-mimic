#!/usr/bin/env python3
"""Shared numerical and file utilities for the reduced AE.5a run package."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA


MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
WINDOW_A = "2014 - 2016"
WINDOW_B = "2017 - 2019"

EXPECTED_N = 5_000
EXPECTED_DIM = 768
EXPECTED_WINDOW_N = 2_500
EXPECTED_COMPONENTS = 52
EXPECTED_MMD2 = 0.072665

PCA_LEVELS = (0.80, 0.90, 0.95)
BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)
SAMPLE_SIZES = (500, 1_000, 2_000)
SAMPLE_SEEDS = tuple(range(1111, 1131))
PCA_MAX_COMPONENTS = 150
PRIMARY_SEED = 42
N_PERMUTATIONS = 1_000
ALPHA = 0.05

class ProtocolError(RuntimeError):
    """Raised when a frozen input or required analysis check fails."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _temporary_path(destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(
    path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_analysis_input_arguments(parser: Any) -> None:
    """Add the independent primary pair and separate target-window pools."""
    parser.add_argument(
        "--baseline-embeddings",
        default="data/embeddings/embeddings_mimic3_5000.npy",
    )
    parser.add_argument(
        "--baseline-ids", default="data/embeddings/ids_mimic3_5000.npy"
    )
    parser.add_argument(
        "--target-embeddings",
        default="data/embeddings/embeddings_mimic4_5000.npy",
        help="Independent 5,000-note primary MIMIC-IV embedding array.",
    )
    parser.add_argument(
        "--target-ids",
        default="data/embeddings/ids_mimic4_5000.npy",
        help="Admission IDs aligned to the independent primary target array.",
    )
    parser.add_argument(
        "--window-a-embeddings",
        default="data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy",
    )
    parser.add_argument(
        "--window-a-ids",
        default="data/embeddings_windows/ids_mimic4_2500_2014_2016.npy",
    )
    parser.add_argument(
        "--window-b-embeddings",
        default="data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy",
    )
    parser.add_argument(
        "--window-b-ids",
        default="data/embeddings_windows/ids_mimic4_2500_2017_2019.npy",
    )


def analysis_input_paths(args: Any) -> dict[str, str]:
    return {
        "baseline_embeddings": str(Path(args.baseline_embeddings).expanduser().resolve()),
        "baseline_ids": str(Path(args.baseline_ids).expanduser().resolve()),
        "target_embeddings": str(Path(args.target_embeddings).expanduser().resolve()),
        "target_ids": str(Path(args.target_ids).expanduser().resolve()),
        "window_a_embeddings": str(
            Path(args.window_a_embeddings).expanduser().resolve()
        ),
        "window_a_ids": str(Path(args.window_a_ids).expanduser().resolve()),
        "window_b_embeddings": str(
            Path(args.window_b_embeddings).expanduser().resolve()
        ),
        "window_b_ids": str(Path(args.window_b_ids).expanduser().resolve()),
    }


def _load_npy(path: str | Path, label: str) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProtocolError(f"{label} not found: {resolved}")
    return np.load(resolved, allow_pickle=False)


def load_primary_embeddings(
    baseline_path: str | Path, target_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the independent 5,000-row primary embedding pair."""
    baseline = _load_npy(baseline_path, "Baseline embeddings")
    target = _load_npy(target_path, "Target embeddings")
    for label, array in (("baseline", baseline), ("target", target)):
        if array.shape != (EXPECTED_N, EXPECTED_DIM):
            raise ProtocolError(
                f"{label} embeddings must have shape {(EXPECTED_N, EXPECTED_DIM)}, "
                f"found {array.shape}"
            )
        if array.dtype != np.float32:
            raise ProtocolError(f"{label} embeddings must be float32, found {array.dtype}")
        if not np.isfinite(array).all():
            raise ProtocolError(f"{label} embeddings contain NaN or infinity")
    return baseline, target


def load_analysis_arrays(
    args: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and validate the independent primary pair and separate window pools."""
    baseline, target = load_primary_embeddings(
        args.baseline_embeddings, args.target_embeddings
    )
    baseline_ids = _load_npy(args.baseline_ids, "Baseline IDs")
    target_ids = _load_npy(args.target_ids, "Target IDs")
    window_a = _load_npy(args.window_a_embeddings, "Window A embeddings")
    window_a_ids = _load_npy(args.window_a_ids, "Window A IDs")
    window_b = _load_npy(args.window_b_embeddings, "Window B embeddings")
    window_b_ids = _load_npy(args.window_b_ids, "Window B IDs")

    arrays = (("window A", window_a), ("window B", window_b))
    for label, array in arrays:
        if array.shape != (EXPECTED_WINDOW_N, EXPECTED_DIM):
            raise ProtocolError(
                f"{label} embeddings must have shape "
                f"{(EXPECTED_WINDOW_N, EXPECTED_DIM)}, "
                f"found {array.shape}"
            )
        if array.dtype != np.float32:
            raise ProtocolError(f"{label} embeddings must be float32, found {array.dtype}")
        if not np.isfinite(array).all():
            raise ProtocolError(f"{label} embeddings contain NaN or infinity")

    id_arrays = (
        ("baseline IDs", baseline_ids, EXPECTED_N),
        ("target IDs", target_ids, EXPECTED_N),
        ("window A IDs", window_a_ids, EXPECTED_WINDOW_N),
        ("window B IDs", window_b_ids, EXPECTED_WINDOW_N),
    )
    for label, values, expected_n in id_arrays:
        if values.shape != (expected_n,):
            raise ProtocolError(
                f"{label} must have shape {(expected_n,)}, found {values.shape}"
            )
        try:
            as_int = values.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{label} are not integer-valued") from exc
        if not np.array_equal(values, as_int):
            raise ProtocolError(f"{label} are not integer-valued")
        if len(np.unique(as_int)) != expected_n:
            raise ProtocolError(f"{label} contain duplicate admissions")

    if np.intersect1d(
        window_a_ids.astype(np.int64), window_b_ids.astype(np.int64)
    ).size:
        raise ProtocolError("Window A and Window B contain overlapping admission IDs")

    return (
        baseline,
        baseline_ids.astype(np.int64),
        target,
        target_ids.astype(np.int64),
        window_a,
        window_a_ids.astype(np.int64),
        window_b,
        window_b_ids.astype(np.int64),
    )


def dynamic_pca(
    baseline: np.ndarray,
    target: np.ndarray,
    variance_target: float,
    max_components: int = PCA_MAX_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, PCA, float]:
    """Exact two-pass baseline-only PCA rule copied from detect_drift.py."""
    n_probe = min(max_components, baseline.shape[0] - 1, baseline.shape[1])
    probe = PCA(n_components=n_probe, random_state=PRIMARY_SEED)
    probe.fit(baseline)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = min(n_components, n_probe)
    retained = float(cumulative[n_components - 1])

    pca = PCA(n_components=n_components, random_state=PRIMARY_SEED)
    pca.fit(baseline)
    return pca.transform(baseline), pca.transform(target), pca, retained


def median_sigma(
    values_a: np.ndarray, values_b: np.ndarray, seed: int = PRIMARY_SEED
) -> float:
    """Frozen median heuristic: one at-most-500-row draw from pooled arrays."""
    rng = np.random.default_rng(seed)
    pooled = np.vstack([values_a, values_b])
    count = min(500, len(pooled))
    sub = pooled[rng.choice(len(pooled), count, replace=False)]
    norms = np.sum(sub**2, axis=1, keepdims=True)
    squared = np.maximum(norms + norms.T - 2.0 * (sub @ sub.T), 0.0)
    upper = np.triu_indices(len(sub), k=1)
    sigma = float(np.sqrt(np.median(squared[upper]) / 2.0))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ProtocolError(f"Median heuristic produced invalid sigma={sigma}")
    return sigma


def _rbf_kernel(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    left_norm = np.sum(left**2, axis=1, keepdims=True)
    right_norm = np.sum(right**2, axis=1, keepdims=True)
    squared = np.maximum(left_norm + right_norm.T - 2.0 * (left @ right.T), 0.0)
    return np.exp(-squared / (2.0 * sigma**2))


def mmd2_unbiased(left: np.ndarray, right: np.ndarray, sigma: float) -> float:
    """Unbiased MMD squared U-statistic used by detect_drift.py."""
    m, n = len(left), len(right)
    if m < 2 or n < 2:
        raise ProtocolError("Unbiased MMD requires at least two rows per sample")
    k_ll = _rbf_kernel(left, left, sigma)
    k_rr = _rbf_kernel(right, right, sigma)
    k_lr = _rbf_kernel(left, right, sigma)
    np.fill_diagonal(k_ll, 0.0)
    np.fill_diagonal(k_rr, 0.0)
    return float(
        k_ll.sum() / (m * (m - 1))
        - 2.0 * k_lr.sum() / (m * n)
        + k_rr.sum() / (n * (n - 1))
    )


def primary_geometry(
    baseline: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PCA, float, float, float]:
    base_pca, target_pca, pca, retained = dynamic_pca(baseline, target, 0.90)
    sigma = median_sigma(base_pca, target_pca, PRIMARY_SEED)
    statistic = mmd2_unbiased(base_pca, target_pca, sigma)
    return base_pca, target_pca, pca, retained, sigma, statistic


def assert_primary_reproduction(components: int, statistic: float) -> None:
    """Check only quantities documented for the primary manuscript result."""
    failures: list[str] = []
    if components != EXPECTED_COMPONENTS:
        failures.append(f"components={components}, expected {EXPECTED_COMPONENTS}")
    if not np.isclose(statistic, EXPECTED_MMD2, rtol=0.0, atol=1e-6):
        failures.append(
            f"MMD^2={statistic:.12g}, expected {EXPECTED_MMD2:.6f} to six decimals"
        )
    if failures:
        raise ProtocolError("Primary reproduction failed: " + "; ".join(failures))


def fixed_nested_permutations(
    baseline_n: int,
    target_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One fixed RNG per seed; prefixes create the prespecified nested samples."""
    rng = np.random.default_rng(seed)
    baseline_order = rng.permutation(baseline_n)
    target_order = rng.permutation(target_n)
    return baseline_order, target_order


def permutation_test_precomputed(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PRIMARY_SEED,
    batch_size: int = 25,
) -> tuple[float, np.ndarray, int, float]:
    """Run algebraically identical permutation assignments on one pooled kernel."""
    m, n = len(left), len(right)
    pooled = np.vstack([left, right])
    total_n = len(pooled)
    rng = np.random.default_rng(seed)

    # Replay detect_drift.py's bandwidth draw so its permutation stream is retained.
    rng.choice(total_n, min(500, total_n), replace=False)

    norms = np.sum(pooled**2, axis=1, keepdims=True)
    squared = np.maximum(norms + norms.T - 2.0 * (pooled @ pooled.T), 0.0)
    kernel = np.exp(-squared / (2.0 * sigma**2))
    del squared
    np.fill_diagonal(kernel, 0.0)

    row_sums = kernel.sum(axis=1, dtype=np.float64)
    total_sum = float(row_sums.sum(dtype=np.float64))
    null = np.empty(n_permutations, dtype=np.float64)

    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        width = stop - start
        membership = np.zeros((total_n, width), dtype=kernel.dtype)
        for column in range(width):
            permutation = rng.permutation(total_n)
            membership[permutation[:m], column] = 1.0

        kernel_times_membership = kernel @ membership
        within_left = np.sum(
            membership * kernel_times_membership, axis=0, dtype=np.float64
        )
        selected_row_sum = membership.T.astype(np.float64, copy=False) @ row_sums
        cross = selected_row_sum - within_left
        within_right = total_sum - within_left - 2.0 * cross
        null[start:stop] = (
            within_left / (m * (m - 1))
            - 2.0 * cross / (m * n)
            + within_right / (n * (n - 1))
        )

    observed = mmd2_unbiased(left, right, sigma)
    exceedances = int(np.count_nonzero(null >= observed))
    threshold = float(np.percentile(null, 100.0 * (1.0 - ALPHA)))
    return observed, null, exceedances, threshold


def p_display(exceedances: int, n_permutations: int) -> str:
    if exceedances == 0:
        return f"p < {1.0 / n_permutations:.3f}"
    return f"p = {exceedances / n_permutations:.3f}"


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions
