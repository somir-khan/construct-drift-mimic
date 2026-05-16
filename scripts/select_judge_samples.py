"""
scripts/select_judge_samples.py
Witness-Function Sample Selection for Judge LLM (Two-Window Design)

Computes per-sample witness scores for two MIMIC-IV temporal windows against the
MIMIC-III baseline using an RBF kernel witness function calibrated via the median
heuristic. Selects stratified samples (top / bottom / random by witness score) from
each window independently for downstream semantic classification by the Judge LLM.

The witness function w(x) = E_P[k(x,x')] - E_Q[k(x,y)] assigns high scores
to target samples that are most "alien" to the baseline distribution.

PCA handling: baseline_pca.npy is already 50-dim. Window embeddings are raw 768-dim
and are compressed using the PCA object saved by detect_drift.py (loaded via joblib).

Usage:
    python scripts/select_judge_samples.py
    python scripts/select_judge_samples.py --n-select 10 --rng-seed 0
    python scripts/select_judge_samples.py \\
        --baseline-pca data/baseline_pca.npy \\
        --baseline-raw data/embeddings/embeddings_mimic3_5000.npy \\
        --pca-model    data/pca_model.pkl \\
        --samples-csv  data/judge_samples.csv
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import rbf_kernel

load_dotenv()
MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Select stratified samples by witness score for Judge LLM (two-window).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-pca",
        default="data/baseline_pca.npy",
        help="Path to MIMIC-III PCA embeddings (.npy, shape N×50, already compressed).",
    )
    parser.add_argument(
        "--baseline-raw",
        default="data/embeddings/embeddings_mimic3_5000.npy",
        help=(
            "Path to raw MIMIC-III embeddings (.npy, shape N×768). Used only to load "
            "hadm_ids for centroid exemplar selection; PCA is loaded from --pca-model."
        ),
    )
    parser.add_argument(
        "--baseline-ids",
        default="data/embeddings/ids_mimic3_5000.npy",
        help="Path to MIMIC-III hadm_id array (.npy, shape N) aligned with --baseline-pca.",
    )
    parser.add_argument(
        "--pca-model",
        default="data/pca_model.pkl",
        help="Path to the fitted PCA object saved by detect_drift.py (joblib format).",
    )
    parser.add_argument(
        "--window-a-emb",
        default="data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy",
        help="Path to raw MIMIC-IV window-A embeddings (.npy, shape N×768).",
    )
    parser.add_argument(
        "--window-a-ids",
        default="data/embeddings_windows/ids_mimic4_2500_2014_2016.npy",
        help="Path to MIMIC-IV window-A hadm_id array (.npy, shape N).",
    )
    parser.add_argument(
        "--window-a-label",
        default="2014 - 2016",
        help="anchor_year_group label for window A.",
    )
    parser.add_argument(
        "--window-b-emb",
        default="data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy",
        help="Path to raw MIMIC-IV window-B embeddings (.npy, shape N×768).",
    )
    parser.add_argument(
        "--window-b-ids",
        default="data/embeddings_windows/ids_mimic4_2500_2017_2019.npy",
        help="Path to MIMIC-IV window-B hadm_id array (.npy, shape N).",
    )
    parser.add_argument(
        "--window-b-label",
        default="2017 - 2019",
        help="anchor_year_group label for window B.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=10,
        help="Number of notes to select per group (top / bottom / random) per window.",
    )
    parser.add_argument(
        "--n-exemplars",
        type=int,
        default=3,
        help="Number of MIMIC-III centroid exemplar notes to embed in the Judge LLM prompt.",
    )
    parser.add_argument(
        "--samples-csv",
        default="data/judge_samples.csv",
        help="Output path for the selected sample manifest.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1000,
        help="Number of rows to draw when computing the median-heuristic sigma.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def normalize_phi(text: str) -> str:
    text = re.sub(r'\[\*\*.*?\*\*\]', 'unknown', text)
    text = re.sub(r'___', 'unknown', text)
    return text


def _median_heuristic(X, subsample, rng):
    """Compute the median-heuristic bandwidth sigma from a pooled embedding matrix.

    Subsamples rows from X, computes pairwise squared Euclidean distances on
    that subsample, and returns sqrt(median of those distances). Guarded
    against sigma = 0.

    Args:
        X:         np.ndarray of shape (N, D), the pooled baseline+target matrix.
        subsample: int, number of rows to draw for the distance computation.
        rng:       np.random.Generator for reproducible subsampling.

    Returns:
        float, sigma = max(sqrt(median_sq_dist), 1e-8).
    """
    k = min(subsample, len(X))
    idx = rng.choice(len(X), size=k, replace=False)
    sub = X[idx].astype(np.float64)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    sq = np.sum(sub ** 2, axis=1, keepdims=True)
    sq_dists = np.maximum(sq + sq.T - 2.0 * (sub @ sub.T), 0.0)

    triu_i, triu_j = np.triu_indices(k, k=1)
    sigma = float(np.sqrt(np.median(sq_dists[triu_i, triu_j]) / 2.0))
    return max(sigma, 1e-8)


def compute_witness_scores(X_base, X_target, sigma):
    """Compute per-sample witness function scores for MIMIC-IV embeddings.

    Higher scores indicate embeddings more "alien" to the baseline distribution.
    Uses: w(x) = E_P[k(x,x')] - E_Q[k(x,y)] where P is target, Q is baseline.

    Args:
        X_base:   np.ndarray of shape (N_base, D), MIMIC-III PCA embeddings.
        X_target: np.ndarray of shape (N_target, D), MIMIC-IV PCA embeddings.
        sigma:    float, RBF kernel bandwidth (must be > 0).

    Returns:
        np.ndarray of shape (N_target,), witness score per target sample.
    """
    gamma = 1 / (2 * sigma ** 2)
    K_tt = rbf_kernel(X_target, X_target, gamma).mean(axis=1)
    K_tb = rbf_kernel(X_target, X_base, gamma).mean(axis=1)
    scores = K_tt - K_tb  # higher = more alien to baseline
    return scores


def select_samples(scores, hadm_ids, n, rng):
    """Select top-n, bottom-n, and random-n indices by witness score, deduplicated.

    Deduplication priority: top > bottom > random. Random candidates are drawn
    from indices not already assigned to top or bottom groups.

    Args:
        scores:   np.ndarray of shape (N,), witness scores for target samples.
        hadm_ids: np.ndarray of shape (N,), MIMIC-IV hadm_ids aligned with scores.
        n:        int, number of samples per group.
        rng:      np.random.Generator for reproducible random selection.

    Returns:
        pd.DataFrame with columns hadm_id, witness_score, selection_group.
        selection_group values: 'top', 'bottom', 'random'.
    """
    sorted_desc = np.argsort(scores)[::-1]
    sorted_asc  = np.argsort(scores)

    top_idx    = sorted_desc[:n]
    bottom_idx = sorted_asc[:n]

    used      = set(top_idx.tolist()) | set(bottom_idx.tolist())
    remaining = [i for i in range(len(scores)) if i not in used]

    k = min(n, len(remaining))
    if k < n:
        log.warning(
            "Only %d candidates remain for random group (requested %d); "
            "top/bottom pools overlap.",
            k, n,
        )
    rand_idx = (
        rng.choice(remaining, size=k, replace=False)
        if k > 0
        else np.array([], dtype=int)
    )

    rows = []
    for i in top_idx:
        rows.append({
            "hadm_id":         int(hadm_ids[i]),
            "witness_score":   float(scores[i]),
            "selection_group": "top",
        })
    for i in bottom_idx:
        rows.append({
            "hadm_id":         int(hadm_ids[i]),
            "witness_score":   float(scores[i]),
            "selection_group": "bottom",
        })
    for i in rand_idx:
        rows.append({
            "hadm_id":         int(hadm_ids[i]),
            "witness_score":   float(scores[i]),
            "selection_group": "random",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Orchestrate two-window witness-score computation and stratified sample selection."""
    args = parse_args()

    # --- Stage 1: Load baseline PCA (already 50-dim) ------------------------
    log.info("--- Stage 1: Load Baseline PCA ---")
    if not os.path.exists(args.baseline_pca):
        log.error("File not found: %s  (baseline PCA)", args.baseline_pca)
        sys.exit(1)

    X_base = np.load(args.baseline_pca)
    log.info("Loaded baseline_pca %s", X_base.shape)

    # --- Stage 1b: Baseline Centroid Exemplar Selection ---------------------
    log.info("--- Stage 1b: Baseline Centroid Exemplar Selection ---")

    if not os.path.exists(args.baseline_ids):
        log.error("File not found: %s  (baseline IDs)", args.baseline_ids)
        sys.exit(1)

    baseline_ids = np.load(args.baseline_ids)
    if len(baseline_ids) != len(X_base):
        log.error(
            "baseline_ids has %d rows but baseline_pca has %d — must be aligned.",
            len(baseline_ids), len(X_base),
        )
        sys.exit(1)

    centroid = X_base.mean(axis=0)
    dists = np.linalg.norm(X_base - centroid, axis=1)
    exemplar_indices = np.argsort(dists)[:args.n_exemplars]
    exemplar_hadm_ids = baseline_ids[exemplar_indices].tolist()
    log.info(
        "Selected %d centroid exemplars: hadm_ids=%s",
        args.n_exemplars, exemplar_hadm_ids,
    )

    # Fetch exemplar texts from MIMIC-III
    if not MIMIC3_DB_PATH or not os.path.exists(MIMIC3_DB_PATH):
        log.error("MIMIC3_DB_PATH not set or not found: %s", MIMIC3_DB_PATH)
        sys.exit(1)

    import sqlite3

    placeholders = ",".join("?" * len(exemplar_hadm_ids))
    query_exemplars = f"""
        SELECT HADM_ID, TEXT
        FROM NOTEEVENTS
        WHERE HADM_ID IN ({placeholders})
          AND CATEGORY = 'Discharge summary'
          AND (ISERROR IS NULL OR ISERROR != '1')
    """
    conn3 = sqlite3.connect(MIMIC3_DB_PATH)
    df_exemplars_raw = pd.read_sql_query(
        query_exemplars, conn3, params=exemplar_hadm_ids
    )
    conn3.close()

    if len(df_exemplars_raw) == 0:
        log.error("No MIMIC-III notes found for exemplar hadm_ids: %s", exemplar_hadm_ids)
        sys.exit(1)

    df_exemplars_raw["TEXT"] = df_exemplars_raw["TEXT"].apply(normalize_phi)

    exemplar_rows = []
    for _, erow in df_exemplars_raw.iterrows():
        exemplar_rows.append({
            "hadm_id":           int(erow["HADM_ID"]),
            "witness_score":     float("nan"),
            "selection_group":   "exemplar",
            "anchor_year_group": "MIMIC-III-baseline",
            "text":              str(erow["TEXT"]),
        })
    log.info("Retrieved text for %d exemplar notes.", len(exemplar_rows))

    # --- Stage 2: Load PCA model from detect_drift.py -----------------------
    log.info("--- Stage 2: Load PCA Model ---")

    if not os.path.exists(args.pca_model):
        log.error(
            "PCA model not found: %s\n"
            "  Run detect_drift.py first and ensure it saves the PCA model with:\n"
            "    import joblib; joblib.dump(pca, 'data/pca_model.pkl')",
            args.pca_model,
        )
        sys.exit(1)

    import joblib
    pca = joblib.load(args.pca_model)
    log.info(
        "Loaded PCA model from %s | n_components=%d | explained_variance=%.4f",
        args.pca_model, pca.n_components_,
        pca.explained_variance_ratio_.sum(),
    )

    # --- Stage 3: Per-window processing -------------------------------------
    rng = np.random.default_rng(args.rng_seed)

    # Both files produced by the same PCA object loaded from
    #  --pca-model. Naming reflects window content, not separate models.
    windows = [
        (args.window_a_emb, args.window_a_ids, args.window_a_label, "pca_window_2014_2016.npy"),
        (args.window_b_emb, args.window_b_ids, args.window_b_label, "pca_window_2017_2019.npy"),
    ]

    all_dfs = []

    # Open MIMIC-IV connection once before the windows loop
    if not MIMIC4_DB_PATH or not os.path.exists(MIMIC4_DB_PATH):
        log.error("MIMIC4_DB_PATH not set or not found: %s", MIMIC4_DB_PATH)
        sys.exit(1)

    import sqlite3 as _sqlite3  # noqa: F811 (already imported above)
    conn4 = _sqlite3.connect(MIMIC4_DB_PATH)

    for emb_path, ids_path, label, pca_filename in windows:
        log.info("--- Window: %s ---", label)

        for path, desc in [(emb_path, "embeddings"), (ids_path, "ids")]:
            if not os.path.exists(path):
                log.error("File not found: %s  (%s, window '%s')", path, desc, label)
                conn4.close()
                sys.exit(1)

        # Load raw 768-dim embeddings and transform with loaded PCA
        raw_window = np.load(emb_path)
        log.info("Loaded raw embeddings %s for window '%s'", raw_window.shape, label)

        X_window = pca.transform(raw_window).astype(np.float32)
        log.info("PCA-compressed to %s", X_window.shape)

        # Save PCA-transformed array alongside other window files
        pca_out_dir = os.path.dirname(emb_path)
        pca_out_path = os.path.join(pca_out_dir, pca_filename)
        np.save(pca_out_path, X_window)
        log.info("Saved PCA embeddings -> %s", pca_out_path)

        hadm_ids = np.load(ids_path)
        log.info("Loaded hadm_ids %s", hadm_ids.shape)

        if len(X_window) != len(hadm_ids):
            log.error(
                "Shape mismatch for window '%s': embeddings has %d rows but ids has %d.",
                label, len(X_window), len(hadm_ids),
            )
            conn4.close()
            sys.exit(1)

        # Compute sigma via median heuristic on pooled subsample of baseline + window
        n_sub_base   = min(args.subsample, len(X_base))
        n_sub_window = min(args.subsample, len(X_window))
        sub_base     = X_base[rng.choice(len(X_base),     size=n_sub_base,   replace=False)]
        sub_window   = X_window[rng.choice(len(X_window), size=n_sub_window, replace=False)]
        pooled       = np.vstack([sub_base, sub_window])

        sigma = _median_heuristic(pooled, len(pooled), rng)
        log.info(
            "Bandwidth sigma = %.6f  (subsample=%d per dataset, pooled=%d rows)",
            sigma, args.subsample, len(pooled),
        )

        # Compute witness scores for this window
        log.info(
            "Computing RBF kernel matrices for window '%s' — "
            "allocates ~%d MB at N=%d ...",
            label,
            int(len(X_window) ** 2 * 8 / 1e6 * 2),
            len(X_window),
        )
        scores = compute_witness_scores(X_base, X_window, sigma)
        log.info(
            "Witness scores [%s]: min=%.6f  max=%.6f  mean=%.6f  std=%.6f",
            label, scores.min(), scores.max(), scores.mean(), scores.std(),
        )

        # Select stratified samples for this window
        df_window = select_samples(scores, hadm_ids, args.n_select, rng)
        df_window["anchor_year_group"] = label

        counts = df_window["selection_group"].value_counts()
        for group in ("top", "bottom", "random"):
            log.info("  %-8s : %d notes", group, counts.get(group, 0))

        # Fetch discharge text for selected notes from MIMIC-IV
        selected_ids = df_window["hadm_id"].tolist()
        placeholders = ",".join("?" * len(selected_ids))
        query_text = f"""
            SELECT hadm_id, text
            FROM "note/discharge"
            WHERE hadm_id IN ({placeholders})
            ORDER BY note_seq DESC
        """
        df_texts = pd.read_sql_query(query_text, conn4, params=selected_ids)

        # Keep only the last note per hadm_id (already ordered by note_seq DESC)
        df_texts = df_texts.drop_duplicates(subset="hadm_id", keep="first")
        df_texts["hadm_id"] = df_texts["hadm_id"].astype(np.int64)
        df_texts["text"] = df_texts["text"].apply(normalize_phi)

        df_window = df_window.merge(df_texts, on="hadm_id", how="left")
        n_missing_text = df_window["text"].isna().sum()
        if n_missing_text > 0:
            log.warning(
                "%d selected notes for window '%s' have no text in MIMIC-IV.",
                n_missing_text, label,
            )

        all_dfs.append(df_window)

    conn4.close()

    # --- Stage 4: Concatenate and save output -------------------------------
    log.info("--- Stage 4: Save Output ---")
    df = pd.concat(all_dfs, ignore_index=True)

    df_exemplar_df = pd.DataFrame(exemplar_rows)
    df = pd.concat([df_exemplar_df, df], ignore_index=True)

    out_dir = os.path.dirname(args.samples_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.samples_csv, index=False)

    counts_total = df[df["selection_group"] != "exemplar"]["selection_group"].value_counts()
    log.info("=" * 55)
    log.info("  WITNESS SCORE SAMPLE SELECTION COMPLETE")
    log.info("  Total samples selected : %d", len(df))
    log.info("  exemplar               : %d", (df["selection_group"] == "exemplar").sum())
    log.info("  top                    : %d", counts_total.get("top", 0))
    log.info("  bottom                 : %d", counts_total.get("bottom", 0))
    log.info("  random                 : %d", counts_total.get("random", 0))
    log.info("  Windows                : %s", [w[2] for w in windows])
    log.info("  Output                 : %s", args.samples_csv)
    log.info("=" * 55)


if __name__ == "__main__":
    main()