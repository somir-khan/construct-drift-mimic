#!/usr/bin/env python3
"""Replay the legacy 300-note Judge-sample selection without loading note text.

This is a diagnostic checker for the already completed observational study.
It deliberately reproduces the RNG consumption in select_judge_samples.py:
one Generator seeded with 42 is shared across the two windows.  It does not
query either MIMIC database, create variants, call an LLM, or write files.

Run from the project root:

    python scripts/addendum_corruption_mimic3/replay_judge_sample_selection.py

The default paths are the same project conventions used by the original
selector.  Use --help only if your local layout differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


SEED = 42
N_PER_GROUP = 50
SUBSAMPLE = 1000
WINDOWS = (
    ("2014 - 2016", Path("data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2014_2016.npy")),
    ("2017 - 2019", Path("data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2017_2019.npy")),
)


def median_heuristic_legacy(values: np.ndarray, rng: np.random.Generator) -> float:
    """Exact numerical/RNG behavior of select_judge_samples._median_heuristic."""
    k = min(len(values), len(values))
    indices = rng.choice(len(values), size=k, replace=False)
    sample = values[indices].astype(np.float64)
    norms = np.sum(sample ** 2, axis=1, keepdims=True)
    squared = np.maximum(norms + norms.T - 2.0 * (sample @ sample.T), 0.0)
    upper = squared[np.triu_indices(k, k=1)]
    return max(float(np.sqrt(np.median(upper) / 2.0)), 1e-8)


def witness_scores_legacy(baseline: np.ndarray, target: np.ndarray, sigma: float) -> np.ndarray:
    gamma = 1.0 / (2.0 * sigma ** 2)
    return rbf_kernel(target, target, gamma=gamma).mean(axis=1) - rbf_kernel(
        target, baseline, gamma=gamma
    ).mean(axis=1)


def select_legacy(scores: np.ndarray, hadm_ids: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Exact group-selection and RNG consumption of the old selector."""
    descending = np.argsort(scores)[::-1]
    ascending = np.argsort(scores)
    selected = {"top": descending[:N_PER_GROUP], "bottom": ascending[:N_PER_GROUP]}
    used = set(selected["top"].tolist()) | set(selected["bottom"].tolist())
    remaining = [i for i in range(len(scores)) if i not in used]
    selected["random"] = rng.choice(remaining, size=N_PER_GROUP, replace=False)
    return {name: hadm_ids[indices].astype(np.int64) for name, indices in selected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the legacy two-window sample selection.")
    parser.add_argument("--baseline-pca", type=Path, default=Path("data/baseline_pca.npy"))
    parser.add_argument("--pca-model", type=Path, default=Path("data/pca_model.pkl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/judge_samples_300.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [args.baseline_pca, args.pca_model, args.manifest, *(p for _, *paths in WINDOWS for p in paths)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required file(s):\n  " + "\n  ".join(missing))

    baseline = np.load(args.baseline_pca, allow_pickle=False)
    pca = joblib.load(args.pca_model)
    manifest = pd.read_csv(args.manifest)
    needed = {"hadm_id", "witness_score", "selection_group", "anchor_year_group"}
    if not needed.issubset(manifest.columns):
        raise SystemExit(f"Manifest is missing required columns: {sorted(needed - set(manifest.columns))}")

    rng = np.random.default_rng(SEED)
    all_match = True
    print(f"Legacy replay: seed={SEED}, groups={N_PER_GROUP}, subsample={SUBSAMPLE}")

    for label, embedding_path, ids_path in WINDOWS:
        raw = np.load(embedding_path, allow_pickle=False)
        ids = np.load(ids_path, allow_pickle=False)
        target = pca.transform(raw).astype(np.float32)  # exact old-selector cast

        base_part = baseline[rng.choice(len(baseline), size=min(SUBSAMPLE, len(baseline)), replace=False)]
        target_part = target[rng.choice(len(target), size=min(SUBSAMPLE, len(target)), replace=False)]
        sigma = median_heuristic_legacy(np.vstack([base_part, target_part]), rng)
        scores = witness_scores_legacy(baseline, target, sigma)
        replay = select_legacy(scores, ids, rng)  # consumes the first-window random draw before window two

        saved = manifest.loc[manifest["anchor_year_group"].astype(str) == label]
        print(f"\n{label}: sigma={sigma:.12f}")
        for group in ("top", "bottom", "random"):
            expected = saved.loc[saved["selection_group"] == group, "hadm_id"].astype(np.int64).to_numpy()
            actual = replay[group]
            same_order = np.array_equal(actual, expected)
            same_set = set(actual) == set(expected)
            score_map = dict(zip(ids.astype(np.int64), scores, strict=True))
            saved_scores = saved.loc[saved["selection_group"] == group, ["hadm_id", "witness_score"]]
            errors = [abs(score_map[int(row.hadm_id)] - float(row.witness_score)) for row in saved_scores.itertuples()]
            print(f"  {group:6s} set_match={same_set!s:5s} order_match={same_order!s:5s} "
                  f"max_saved_score_error={max(errors, default=float('nan')):.3g}")
            if not same_set:
                all_match = False
                print("    replay_only:", sorted(set(actual) - set(expected)))
                print("    manifest_only:", sorted(set(expected) - set(actual)))

    print("\nRESULT:", "PASS — all six legacy groups reproduce." if all_match else "FAIL — see differences above.")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
