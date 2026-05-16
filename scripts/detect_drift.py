"""
scripts/detect_drift.py
Drift-Detection Pipeline

Stage 1: Load discharge summaries from CSV and extract [CLS] token embeddings
         (d=768) using Bio_ClinicalBERT (frozen).
Stage 2: Fit PCA on baseline only (frozen anchor), compress both distributions
         to a latent manifold (components selected dynamically to retain >= 90% explained variance, up to 150 max).
Stage 3: MMD two-sample test with Gaussian RBF kernel (median heuristic
         bandwidth) and bootstrap permutation p-value (default 1000 iters).
         Declare drift if p < alpha (default 0.05).

Usage:
    python scripts/detect_drift.py
    python scripts/detect_drift.py --data-dir data --permutations 2000
    python scripts/detect_drift.py --skip-embed   # reuse saved .npy files

    # Load chunked mean-pooling embeddings produced by embed_and_save.py:
    python scripts/detect_drift.py --load-embeddings data/embeddings
"""

import argparse
import glob
import logging
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

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

# Suppress verbose third-party loggers
for _noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMD-based construct drift detection for clinical NLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_PATH", "data"),
        help="Directory containing CSV inputs and where .npy outputs are saved.",
    )
    # Embedding
    parser.add_argument(
        "--model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="HuggingFace model ID for the frozen anchor embedder.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenizer max sequence length.")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding and load existing *_embeddings.npy files from --data-dir.",
    )
    parser.add_argument(
        "--load-embeddings",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing pre-computed embeddings produced by "
            "embed_and_save.py (embeddings_mimic3_*.npy and "
            "embeddings_mimic4_*.npy). When set, the embed_notes() call is "
            "skipped entirely and embeddings are loaded from these files. "
            "The old --skip-embed path remains as a fallback when this flag "
            "is not provided."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "If set, randomly sample this many notes from each database "
            "before embedding. Recommended for CPU runs: 2000-5000. "
            "If None, embed all notes (requires GPU for full datasets). "
            "Sampling is stratified by note source for reproducibility."
        ),
    )

    # PCA
    parser.add_argument("--pca-variance",       type=float, default=0.90,
                        help="Target cumulative explained variance for dynamic component selection.")
    parser.add_argument("--pca-components-max", type=int,   default=150,
                        help="Upper bound on dynamic PCA component search.")

    # MMD
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Number of bootstrap permutations for the null distribution.")
    parser.add_argument("--alpha",        type=float, default=0.05,
                        help="Significance threshold for drift declaration.")
    parser.add_argument("--rng-seed",     type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Device to run embeddings on. Defaults to cpu. Pass --device cuda "
            "only if a compatible GPU is confirmed available (check with "
            "nvidia-smi and verify torch.cuda.is_available())."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: Embedding
# ---------------------------------------------------------------------------
def normalize_phi(text: str) -> str:
    """
    Normalize PHI placeholders to a single [PHI] token before embedding.
    MIMIC-III uses [**...**] bracket notation.
    MIMIC-IV uses ___ triple underscore.
    Without this, the deidentification format difference dominates the MMD
    signal rather than genuine semantic drift.
    """
    import re
    text = re.sub(r'\[\*\*.*?\*\*\]', 'unknown', text)
    text = re.sub(r'___', 'unknown', text)
    return text


def load_notes_mimic3(
    conn: sqlite3.Connection,
    sample_size: int | None = None,
    rng_seed: int = 42,
) -> list[str]:
    """Load discharge summaries from MIMIC-III NOTEEVENTS table.

    Deduplication: latest note per hadm_id (CHARTDATE DESC, ROW_ID DESC),
    matching the canonical key established in surface_features.py (SF1) and
    embed_and_save.py (E1). Only one note per admission enters the pool.
    """
    query = """
    WITH deduped AS (
        SELECT
            TEXT AS text,
            ROW_NUMBER() OVER (
                PARTITION BY HADM_ID
                ORDER BY CHARTDATE DESC, ROW_ID DESC
            ) AS rn
        FROM NOTEEVENTS
        WHERE CATEGORY = 'Discharge summary'
          AND (ISERROR IS NULL OR ISERROR != '1')
          AND HADM_ID IS NOT NULL
          AND TEXT IS NOT NULL
    )
    SELECT text FROM deduped WHERE rn = 1
    """
    df = pd.read_sql_query(query, conn)
    log.info("Loaded %d MIMIC-III discharge summaries", len(df))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=rng_seed)
        log.info(
            "Sampled %d notes from MIMIC-III (random_state=%d)",
            sample_size, rng_seed,
        )
    texts = [normalize_phi(t) for t in df["text"].tolist()]
    log.info("PHI normalization applied to all MIMIC-III notes")
    return texts


def load_notes_mimic4(
    conn: sqlite3.Connection,
    sample_size: int | None = None,
    rng_seed: int = 42,
) -> list[str]:
    """Load discharge summaries from MIMIC-IV note/discharge table.

    Deduplication: latest note per hadm_id (MAX(note_seq)), matching
    surface_features.py SQL_MIMIC4 and embed_and_save.py (E2).
    Only one note per admission enters the pool.
    """
    query = """
    WITH latest_note AS (
        SELECT hadm_id, MAX(note_seq) AS max_note_seq
        FROM "note/discharge"
        WHERE hadm_id IS NOT NULL
          AND text IS NOT NULL
        GROUP BY hadm_id
    )
    SELECT d.text
    FROM "note/discharge" d
    JOIN latest_note ln
      ON d.hadm_id = ln.hadm_id
     AND d.note_seq = ln.max_note_seq
    """
    df = pd.read_sql_query(query, conn)
    log.info("Loaded %d MIMIC-IV discharge summaries", len(df))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=rng_seed)
        log.info(
            "Sampled %d notes from MIMIC-IV (random_state=%d)",
            sample_size, rng_seed,
        )
    texts = [normalize_phi(t) for t in df["text"].tolist()]
    log.info("PHI normalization applied to all MIMIC-IV notes")
    return texts





def load_notes(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    log.info("Loaded %d notes from %s | columns: %s", len(df), csv_path, list(df.columns))
    return df["text"].tolist()


def embed_notes(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Disabled: raises RuntimeError to enforce use of --load-embeddings.

    This fallback path produces single-pass truncated CLS embeddings that are
    incompatible with the chunked uniform mean-pooling embeddings generated by
    embed_and_save.py. Using it silently produces a different embedding
    distribution than the paper pipeline and yields invalid MMD results.

    Always run embed_and_save.py first and pass --load-embeddings to this
    script. Do not call this function directly.
    """
    raise RuntimeError(
        "embed_notes() is disabled. This path produces truncated single-pass "
        "CLS embeddings that are incompatible with the chunked mean-pooling "
        "embeddings required for paper-consistent results.\n\n"
        "Run embed_and_save.py first to generate chunked embeddings, then "
        "invoke detect_drift.py with --load-embeddings <dir>."
    )


# ---------------------------------------------------------------------------
# Loading pre-computed embeddings from embed_and_save.py
# ---------------------------------------------------------------------------
def load_precomputed_embeddings(embed_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate and load the embeddings produced by embed_and_save.py.

    Searches `embed_dir` for files matching the naming convention:
      embeddings_mimic3_{N}.npy  (baseline, MIMIC-III)
      embeddings_mimic4_{N}.npy  (target,   MIMIC-IV)

    If multiple files match (e.g. from different sample sizes), the most
    recently modified one is used and a warning is logged.

    Args:
        embed_dir: Directory written by embed_and_save.py.

    Returns:
        (baseline_emb, target_emb) — both np.ndarray, shape (N, 768).

    Raises:
        FileNotFoundError: If no matching file is found for either dataset.
    """
    def _find_latest(pattern: str, label: str) -> str:
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No {label} embedding file found matching: {pattern}\n"
                "Run embed_and_save.py first to generate the files."
            )
        if len(matches) > 1:
            matches.sort(key=os.path.getmtime, reverse=True)
            log.warning(
                "Multiple %s embedding files found — using most recent: %s",
                label,
                matches[0],
            )
        return matches[0]

    baseline_path = _find_latest(
        os.path.join(embed_dir, "embeddings_mimic3_*.npy"), "MIMIC-III"
    )
    target_path = _find_latest(
        os.path.join(embed_dir, "embeddings_mimic4_*.npy"), "MIMIC-IV"
    )

    baseline_emb = np.load(baseline_path)
    target_emb   = np.load(target_path)
    log.info(
        "Loaded pre-computed embeddings | baseline %s from %s",
        baseline_emb.shape,
        baseline_path,
    )
    log.info(
        "Loaded pre-computed embeddings | target   %s from %s",
        target_emb.shape,
        target_path,
    )
    return baseline_emb, target_emb


# ---------------------------------------------------------------------------
# Stage 2: PCA Manifold Compression
# ---------------------------------------------------------------------------
def compress_embeddings(
    baseline_emb: np.ndarray,
    target_emb: np.ndarray,
    variance_target: float,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on baseline only (frozen anchor), transform both distributions.
    Number of components is selected dynamically to retain >= variance_target
    explained variance, up to max_components.
    Returns (baseline_compressed, target_compressed, fitted_pca).
    """
    # Probe pass: fit with upper bound to find the variance curve
    n_probe = min(max_components, baseline_emb.shape[0] - 1, baseline_emb.shape[1])
    pca_probe = PCA(n_components=n_probe, random_state=42)
    pca_probe.fit(baseline_emb)

    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_target) + 1)
    n_components = min(n_components, n_probe)  # safety clamp

    actual_variance = cumvar[n_components - 1]
    log.info(
        "Dynamic PCA: %d components retain %.4f variance (target >= %.2f)",
        n_components, actual_variance, variance_target,
    )
    log.info(
        "Per-component variance (first 10): %s",
        pca_probe.explained_variance_ratio_[:10].round(4).tolist(),
    )

    # Final pass: refit with exact n_components for a clean object
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(baseline_emb)

    baseline_c = pca.transform(baseline_emb)
    target_c   = pca.transform(target_emb)
    log.info(
        "Compressed: baseline %s -> %s | target %s -> %s",
        baseline_emb.shape, baseline_c.shape, target_emb.shape, target_c.shape,
    )
    return baseline_c, target_c, pca


# ---------------------------------------------------------------------------
# Stage 3: MMD Two-Sample Test
# ---------------------------------------------------------------------------
def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix K[i,j] = exp(-||X_i - Y_j||^2 / (2*sigma^2))."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = np.maximum(XX + YY.T - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """
    Unbiased MMD^2 estimator with Gaussian RBF kernel.
    MMD^2(P,Q) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    Diagonal excluded from within-set terms (i != j only).
    """
    m, n = len(X), len(Y)
    K_XX = _rbf_kernel(X, X, sigma)
    K_YY = _rbf_kernel(Y, Y, sigma)
    K_XY = _rbf_kernel(X, Y, sigma)
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    return float(
        K_XX.sum() / (m * (m - 1))
        - 2.0 * K_XY.sum() / (m * n)
        + K_YY.sum() / (n * (n - 1))
    )


def mmd_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    alpha: float,
    rng_seed: int,
) -> dict:
    """
    Two-sample MMD test.

    Bandwidth: median heuristic on a pooled subsample (<=500 points).
    P-value:   fraction of permuted MMD^2 values >= observed.
    Threshold: (1-alpha) percentile of the null distribution.
    """
    m, n = len(X), len(Y)
    rng = np.random.default_rng(rng_seed)

    # --- Bandwidth via median heuristic ---
    pooled = np.vstack([X, Y])
    sub = pooled[rng.choice(len(pooled), min(500, len(pooled)), replace=False)]
    sq = np.sum(sub ** 2, axis=1, keepdims=True)
    sq_dists = np.maximum(sq + sq.T - 2.0 * (sub @ sub.T), 0.0)
    triu = np.triu_indices(len(sub), k=1)
    sigma = float(np.sqrt(np.median(sq_dists[triu]) / 2.0))
    log.info("Bandwidth sigma: %.6f  (median heuristic, subsample=%d)", sigma, len(sub))

    # --- Observed MMD^2 ---
    observed = _mmd2(X, Y, sigma)
    log.info("Observed MMD^2: %.6f", observed)

    # --- Bootstrap permutation under H0 ---
    log.info("Running %d permutations ...", n_permutations)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(m + n)
        null[i] = _mmd2(pooled[perm[:m]], pooled[perm[m:]], sigma)

    p_value  = float(np.mean(null >= observed))
    threshold = float(np.percentile(null, 100 * (1.0 - alpha)))

    return {
        "mmd2_observed":     observed,
        "sigma":             sigma,
        "p_value":           p_value,
        "threshold_mmd2":    threshold,
        "drift_detected":    p_value < alpha,
        "null_distribution": null,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not MIMIC3_DB_PATH or not MIMIC4_DB_PATH:
        log.error("MIMIC3_DB_PATH and MIMIC4_DB_PATH must both be set in .env")
        sys.exit(1)
    conn_iii = sqlite3.connect(MIMIC3_DB_PATH)
    conn_iv  = sqlite3.connect(MIMIC4_DB_PATH)
    log.info("Connected to MIMIC-III: %s", MIMIC3_DB_PATH)
    log.info("Connected to MIMIC-IV:  %s", MIMIC4_DB_PATH)

    # Resolve paths
    data_dir     = args.data_dir
    baseline_npy = os.path.join(data_dir, "baseline_embeddings.npy")
    target_npy   = os.path.join(data_dir, "target_embeddings.npy")
    baseline_pca_npy = os.path.join(data_dir, "baseline_pca.npy")
    target_pca_npy   = os.path.join(data_dir, "target_pca.npy")

    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Embedding
    # ------------------------------------------------------------------
    if args.load_embeddings is not None:
        log.info(
            "--- Stage 1: Loading chunked embeddings from %s (--load-embeddings) ---",
            args.load_embeddings,
        )
        baseline_emb, target_emb = load_precomputed_embeddings(args.load_embeddings)
    elif args.skip_embed:
        log.info("--- Stage 1: Loading saved embeddings (--skip-embed) ---")
        baseline_emb = np.load(baseline_npy)
        target_emb   = np.load(target_npy)
        log.info("Loaded baseline %s | target %s", baseline_emb.shape, target_emb.shape)
    else:
        log.info("--- Stage 1: Embedding ---")
        device = torch.device(args.device)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested (--device cuda) but torch.cuda.is_available() "
                    "returned False. Check your PyTorch installation."
                )
            logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logging.info(f"Using CPU with {os.cpu_count()} cores available")
            torch.set_num_threads(os.cpu_count())
            torch.set_num_interop_threads(max(1, os.cpu_count() // 2))

        if args.device == "cpu" and args.sample_size is None:
            log.warning(
                "Running on CPU with no --sample-size set. "
                "Full dataset embedding (390k+ notes with chunking) will "
                "take 50-80 hours. Strongly recommend: "
                "--sample-size 3000 for development runs on CPU. "
                "Use --sample-size None only with GPU."
            )
        elif args.device == "cpu":
            estimated_hours = (args.sample_size * 6) / (10 * 3600)
            log.info(
                "CPU run with sample_size=%d | "
                "Estimated embedding time: %.1f hours",
                args.sample_size, estimated_hours,
            )

        hf_token = os.getenv("HF_TOKEN")
        log.info("Loading model: %s", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        model     = AutoModel.from_pretrained(args.model, token=hf_token).to(device)

        for p in model.parameters():
            p.requires_grad = False
        log.info("Model ready | parameters: %s (all frozen)",
                 f"{sum(p.numel() for p in model.parameters()):,}")

        log.info("Embedding baseline (P_III) ...")
        baseline_texts = load_notes_mimic3(
            conn_iii,
            sample_size=args.sample_size,
            rng_seed=args.rng_seed,
        )
        baseline_emb   = embed_notes(baseline_texts, tokenizer, model, device,
                                     args.batch_size, args.max_length)
        np.save(baseline_npy, baseline_emb)
        log.info("Saved %s -> %s", baseline_npy, baseline_emb.shape)

        log.info("Embedding target (P_IV) ...")
        target_texts = load_notes_mimic4(
            conn_iv,
            sample_size=args.sample_size,
            rng_seed=args.rng_seed,
        )
        target_emb   = embed_notes(target_texts, tokenizer, model, device,
                                   args.batch_size, args.max_length)
        np.save(target_npy, target_emb)
        log.info("Saved %s -> %s", target_npy, target_emb.shape)

    # ------------------------------------------------------------------
    # Stage 2: PCA Manifold Compression
    # ------------------------------------------------------------------
    log.info("--- Stage 2: PCA Manifold Compression ---")
    baseline_c, target_c, pca = compress_embeddings(
        baseline_emb, target_emb, args.pca_variance, args.pca_components_max
    )
    np.save(baseline_pca_npy, baseline_c)
    np.save(target_pca_npy,   target_c)
    log.info("Saved %s -> %s", baseline_pca_npy, baseline_c.shape)
    log.info("Saved %s -> %s", target_pca_npy,   target_c.shape)

    import joblib
    pca_model_path = os.path.join(data_dir, "pca_model.pkl")
    joblib.dump(pca, pca_model_path)
    log.info("Saved PCA model -> %s", pca_model_path)

    # ------------------------------------------------------------------
    # Stage 3: MMD Test
    # ------------------------------------------------------------------
    log.info("--- Stage 3: MMD Two-Sample Test ---")
    result = mmd_test(
        baseline_c, target_c,
        n_permutations=args.permutations,
        alpha=args.alpha,
        rng_seed=args.rng_seed,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    verdict = "DRIFT DETECTED" if result["drift_detected"] else "NO DRIFT"
    log.info("=" * 55)
    log.info("  MMD DRIFT REPORT")
    log.info("  MMD^2 observed  : %.6f", result["mmd2_observed"])
    log.info("  MMD^2 threshold : %.6f  (alpha=%.2f)", result["threshold_mmd2"], args.alpha)
    log.info("  p-value         : %.4f", result["p_value"])
    log.info("  sigma (RBF bw)  : %.6f", result["sigma"])
    log.info("  Permutations    : %d", args.permutations)
    log.info("  Verdict         : %s", verdict)
    log.info("=" * 55)

    conn_iii.close()
    conn_iv.close()
    log.info("Database connections closed.")


if __name__ == "__main__":
    main()