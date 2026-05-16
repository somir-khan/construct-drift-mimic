"""
scripts/embed_and_save.py
Standalone Chunked Embedding Script
Embeds MIMIC-III and/or MIMIC-IV discharge summaries using Bio_ClinicalBERT
with chunked uniform mean pooling over overlapping 512-token windows.
This script is designed to run once. All downstream scripts (detect_drift.py,
evaluate.py, judge_llm.py) load from the saved .npy files rather than
re-running the expensive embedding step.
Pooling strategy (uniform_mean):
  1. Tokenize each note WITHOUT truncation (add_special_tokens=True).
  2. Strip the [CLS] and [SEP] special tokens from the full token sequence.
     Split interior tokens into overlapping windows of up to 510 tokens
     (stride=256 by default). Re-prepend [CLS] and append [SEP] to every
     window so each window is a well-formed BERT input.
       Window 1: [CLS] + interior[0:510]   + [SEP]
       Window 2: [CLS] + interior[256:766] + [SEP]
       ...
       Last window always ends at the final interior token (may be < 510 tokens).
  3. For each window, run Bio_ClinicalBERT and extract the [CLS] embedding
     (position 0 of last_hidden_state), shape (768,).
  4. Final note embedding = arithmetic mean of all [CLS] vectors, shape (768,).
     Equivalently: np.mean(chunk_cls_embeddings, axis=0).
  5. Zero vector (768,) is returned for notes that tokenize to 0 tokens.
Usage:
    python scripts/embed_and_save.py
    python scripts/embed_and_save.py --dataset mimic3 --sample-size 2000
    python scripts/embed_and_save.py --output-dir data/embeddings --stride 256
    python scripts/embed_and_save.py --dataset both --seed 42
"""
import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
load_dotenv()
MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")
_EMBED_DIM    = 768
_WINDOW_SIZE  = 512   # BERT hard limit
_INTERIOR_SIZE = _WINDOW_SIZE - 2  # 510 — slots remaining after [CLS] + [SEP]
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
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunked Bio_ClinicalBERT embedder for MIMIC-III / MIMIC-IV "
            "discharge summaries. Saves embeddings and HADM_IDs to .npy files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--anchor-year-groups", nargs="+", default=None, metavar="GROUP")
    parser.add_argument("--anchor-year-group", default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--num-threads", type=int, default=None,
        help="PyTorch intraop threads for CPU (default: all available cores).",
    )
    return parser.parse_args()
# ---------------------------------------------------------------------------
# PHI normalisation
# ---------------------------------------------------------------------------
def normalize_phi(text: str) -> str:
    """Replace PHI placeholders with 'unknown' (single in-vocabulary token)."""
    text = re.sub(r'\[\*\*.*?\*\*\]', 'unknown', text)
    text = re.sub(r'___', 'unknown', text)
    return text
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_notes_mimic3(
    conn: sqlite3.Connection,
    sample_size: int | None = None,
    rng_seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Load deduplicated MIMIC-III discharge summaries.
    E1 fix: ROW_NUMBER() CTE keeps the latest note per admission
    (PARTITION BY HADM_ID ORDER BY CHARTDATE DESC, ROW_ID DESC).
    Dedup key is identical to surface_features.py SQL_MIMIC3 and detect_drift.py D3.
    """
    query = """
    WITH ranked AS (
        SELECT
            HADM_ID AS hadm_id,
            TEXT    AS text,
            ROW_NUMBER() OVER (
                PARTITION BY HADM_ID ORDER BY CHARTDATE DESC, ROW_ID DESC
            ) AS rn
        FROM NOTEEVENTS
        WHERE CATEGORY   = 'Discharge summary'
          AND (ISERROR IS NULL OR ISERROR != '1')
          AND HADM_ID    IS NOT NULL
          AND TEXT       IS NOT NULL
    )
    SELECT hadm_id, text FROM ranked WHERE rn = 1
    """
    df = pd.read_sql_query(query, conn)
    log.info("Loaded %d MIMIC-III discharge summaries (deduplicated)", len(df))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=rng_seed)
        log.info("Sampled %d notes from MIMIC-III (seed=%d)", sample_size, rng_seed)
    texts    = [normalize_phi(t) for t in df["text"].tolist()]
    hadm_ids = df["hadm_id"].tolist()
    log.info("PHI normalization applied to all %d MIMIC-III notes", len(texts))
    return texts, hadm_ids


def load_notes_mimic4(
    conn: sqlite3.Connection,
    sample_size: int | None = None,
    rng_seed: int = 42,
    anchor_year_groups: list[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """
    Load deduplicated MIMIC-IV discharge summaries.
    E2 fix: MAX(note_seq) CTE keeps the latest note per admission,
    matching surface_features.py SQL_MIMIC4.
    """
    if anchor_year_groups is not None:
        placeholders = ",".join(["?"] * len(anchor_year_groups))
        query = f"""
        WITH latest AS (
            SELECT hadm_id, MAX(note_seq) AS max_seq
            FROM   "note/discharge"
            GROUP  BY hadm_id
        )
        SELECT n.hadm_id, n.text, p.anchor_year_group
        FROM        "note/discharge"  n
        JOIN        latest            l  ON n.hadm_id = l.hadm_id AND n.note_seq = l.max_seq
        JOIN        "hosp/admissions" a  ON n.hadm_id = a.hadm_id
        JOIN        "hosp/patients"   p  ON a.subject_id = p.subject_id
        WHERE n.text IS NOT NULL
          AND n.hadm_id IS NOT NULL
          AND p.anchor_year_group IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=anchor_year_groups)
        log.info(
            "Loaded %d MIMIC-IV discharge summaries (deduplicated, anchor_year_groups=%s)",
            len(df), anchor_year_groups,
        )
    else:
        query = """
        WITH latest AS (
            SELECT hadm_id, MAX(note_seq) AS max_seq
            FROM   "note/discharge"
            GROUP  BY hadm_id
        )
        SELECT n.hadm_id, n.text, p.anchor_year_group
        FROM        "note/discharge"  n
        JOIN        latest            l  ON n.hadm_id = l.hadm_id AND n.note_seq = l.max_seq
        JOIN        "hosp/admissions" a  ON n.hadm_id = a.hadm_id
        JOIN        "hosp/patients"   p  ON a.subject_id = p.subject_id
        WHERE n.text IS NOT NULL
          AND n.hadm_id IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        log.info("Loaded %d MIMIC-IV discharge summaries (deduplicated)", len(df))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=rng_seed)
        log.info("Sampled %d notes from MIMIC-IV (seed=%d)", sample_size, rng_seed)
    texts        = [normalize_phi(t) for t in df["text"].tolist()]
    hadm_ids     = df["hadm_id"].tolist()
    group_labels = df["anchor_year_group"].tolist()
    log.info("PHI normalization applied to all %d MIMIC-IV notes", len(texts))
    return texts, hadm_ids, group_labels
# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def build_windows(
    interior_ids: list[int],
    window_size: int = _INTERIOR_SIZE,
    stride: int = 256,
) -> list[list[int]]:
    """
    Split interior token ids (CLS and SEP already stripped) into overlapping
    windows of up to `window_size` tokens.
    Callers prepend [CLS] and append [SEP] before passing to the model.
    """
    if not interior_ids:
        return []
    n       = len(interior_ids)
    windows = []
    start   = 0
    while start < n:
        end = min(start + window_size, n)
        windows.append(interior_ids[start:end])
        if end == n:
            break
        start += stride
    return windows
# ---------------------------------------------------------------------------
# Per-note embedding
# ---------------------------------------------------------------------------
def embed_single_note(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    stride: int = 256,
) -> tuple[np.ndarray, int]:
    """
    Produce a single 768-dimensional embedding for one discharge note.
    E3 fix: strip original [CLS]/[SEP], window interior tokens at 510,
    then re-prepend [CLS] and append [SEP] to every window so every window
    has a genuine [CLS] at position 0 as Bio_ClinicalBERT expects.
    """
    encoded       = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids_all = encoded["input_ids"][0].tolist()

    if not input_ids_all:
        return np.zeros(_EMBED_DIM, dtype=np.float32), 0

    # E3: strip [CLS] (index 0) and [SEP] (index -1); window only interior tokens
    cls_id   = tokenizer.cls_token_id
    sep_id   = tokenizer.sep_token_id
    interior = input_ids_all[1:-1]

    if not interior:
        return np.zeros(_EMBED_DIM, dtype=np.float32), 0

    interior_windows = build_windows(interior, window_size=_INTERIOR_SIZE, stride=stride)

    if not interior_windows:
        return np.zeros(_EMBED_DIM, dtype=np.float32), 0

    # Re-add [CLS] and [SEP] to every window — well-formed BERT input
    windows = [[cls_id] + w + [sep_id] for w in interior_windows]

    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(w) for w in windows)
    ids_rows, msk_rows = [], []
    for w in windows:
        pad = max_len - len(w)
        ids_rows.append(w + [pad_id] * pad)
        msk_rows.append([1] * len(w) + [0] * pad)

    input_ids_t = torch.tensor(ids_rows, dtype=torch.long).to(device)
    attn_mask_t = torch.tensor(msk_rows, dtype=torch.long).to(device)

    # token_type_ids = all zeros: single-segment input (no sentence-pair task).
    # HuggingFace defaults to zeros when omitted, but explicit is safer and
    # documents intent for reviewers.
    token_type_ids_t = torch.zeros_like(input_ids_t)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            token_type_ids=token_type_ids_t,
        )

    cls_vecs  = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embedding = np.mean(cls_vecs, axis=0).astype(np.float32)
    return embedding, len(windows)
# ---------------------------------------------------------------------------
# Batch embedding loop
# ---------------------------------------------------------------------------
def embed_notes_chunked(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    stride: int = 256,
) -> tuple[np.ndarray, int, float]:
    model.eval()
    N            = len(texts)
    embeddings   = np.zeros((N, _EMBED_DIM), dtype=np.float32)
    n_errors     = 0
    total_chunks = 0
    for i, text in enumerate(tqdm(texts, desc="Embedding notes", unit="note")):
        try:
            emb, n_chunks = embed_single_note(text, tokenizer, model, device, stride=stride)
            embeddings[i] = emb
            total_chunks += max(n_chunks, 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Error embedding note %d: %s — storing zero vector", i, exc)
            embeddings[i] = np.zeros(_EMBED_DIM, dtype=np.float32)
            n_errors     += 1
            total_chunks += 1
    return embeddings, n_errors, total_chunks / max(N, 1)
# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_outputs(
    embeddings: np.ndarray,
    hadm_ids: list[int],
    dataset: str,
    output_dir: str,
    sample_size: int | None,
    stride: int,
    model_name: str,
    device: str,
    anchor_year_groups: list[str] | None = None,
    output_suffix: str = "",
) -> None:
    n = len(embeddings)
    os.makedirs(output_dir, exist_ok=True)
    emb_path  = os.path.join(output_dir, f"embeddings_{dataset}_{n}{output_suffix}.npy")
    ids_path  = os.path.join(output_dir, f"ids_{dataset}_{n}{output_suffix}.npy")
    meta_path = os.path.join(output_dir, f"metadata_{dataset}_{n}{output_suffix}.json")
    np.save(emb_path, embeddings.astype(np.float32))
    np.save(ids_path, np.array(hadm_ids, dtype=np.int64))
    if anchor_year_groups is not None:
        grp_path = os.path.join(output_dir, f"groups_mimic4_{n}{output_suffix}.npy")
        np.save(grp_path, np.array(anchor_year_groups, dtype=str))
        log.info("Saved groups     -> %s  shape=(%d,)", grp_path, n)
    metadata = {
        "n_notes": n, "sample_size": sample_size, "stride": stride,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name, "pooling_strategy": "uniform_mean", "device": device,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Saved embeddings -> %s  shape=%s", emb_path, embeddings.shape)
    log.info("Saved IDs        -> %s  shape=(%d,)", ids_path, n)
    log.info("Saved metadata   -> %s", meta_path)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    datasets_to_run: list[str] = (
        ["mimic3", "mimic4"] if args.dataset == "both" else [args.dataset]
    )
    if args.anchor_year_group is not None and "mimic3" in datasets_to_run:
        log.info(
            "--anchor-year-group is set ('%s'). Skipping MIMIC-III embed "
            "(baseline already saved to data/embeddings/).",
            args.anchor_year_group,
        )
        datasets_to_run = [d for d in datasets_to_run if d != "mimic3"]
    if "mimic3" in datasets_to_run and not MIMIC3_DB_PATH:
        log.error("MIMIC3_DB_PATH is not set in .env — cannot embed MIMIC-III.")
        sys.exit(1)
    if "mimic4" in datasets_to_run and not MIMIC4_DB_PATH:
        log.error("MIMIC4_DB_PATH is not set in .env — cannot embed MIMIC-IV.")
        sys.exit(1)
    effective_output_dir = (
        "data/embeddings_windows" if args.anchor_year_group is not None else args.output_dir
    )
    os.makedirs(effective_output_dir, exist_ok=True)
    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() returned False.")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        n_threads = args.num_threads if args.num_threads is not None else os.cpu_count()
        logging.info(f"Using CPU: {n_threads} threads ({os.cpu_count()} cores available)")
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
    if args.device == "cpu" and args.sample_size is None:
        log.warning(
            "Running on CPU with no --sample-size. Full-dataset embedding "
            "will take 50-80 hours. Recommended: --sample-size 5000."
        )
    hf_token  = os.getenv("HF_TOKEN")
    log.info("Loading tokenizer and model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model     = AutoModel.from_pretrained(args.model, token=hf_token).to(device)
    for param in model.parameters():
        param.requires_grad = False
    log.info("Model loaded and frozen | parameters: %s",
             f"{sum(p.numel() for p in model.parameters()):,}")
    if args.device == "cuda":
        try:
            test = torch.zeros(1, 1, dtype=torch.long).to(device)
            model(input_ids=test, attention_mask=test)
            logging.info("CUDA validation passed.")
        except Exception as e:
            raise RuntimeError(f"CUDA validation failed: {e}") from e
    t_start = time.monotonic()
    for dataset in datasets_to_run:
        log.info("=" * 60)
        log.info("  DATASET: %s", dataset.upper())
        log.info("=" * 60)
        t0            = time.monotonic()
        mimic4_groups = None
        if dataset == "mimic3":
            conn = sqlite3.connect(MIMIC3_DB_PATH)
            texts, hadm_ids = load_notes_mimic3(conn, args.sample_size, args.seed)
            conn.close()
        else:
            conn = sqlite3.connect(MIMIC4_DB_PATH)
            effective_groups = (
                [args.anchor_year_group] if args.anchor_year_group is not None
                else args.anchor_year_groups
            )
            texts, hadm_ids, mimic4_groups = load_notes_mimic4(
                conn, args.sample_size, args.seed, effective_groups
            )
            conn.close()
        log.info("Notes to embed: %d | window_size: %d | stride: %d",
                 len(texts), _WINDOW_SIZE, args.stride)
        embeddings, n_errors, mean_chunks = embed_notes_chunked(
            texts, tokenizer, model, device, stride=args.stride
        )
        elapsed = time.monotonic() - t0
        save_outputs(
            embeddings=embeddings, hadm_ids=hadm_ids, dataset=dataset,
            output_dir=effective_output_dir, sample_size=args.sample_size,
            stride=args.stride, model_name=args.model, device=args.device,
            anchor_year_groups=mimic4_groups if dataset == "mimic4" else None,
            output_suffix=args.output_suffix,
        )
        log.info("-" * 60)
        log.info("  %s COMPLETE", dataset.upper())
        log.info("  Total notes        : %d", len(texts))
        log.info("  Notes with errors  : %d", n_errors)
        log.info("  Mean chunks / note : %.2f", mean_chunks)
        log.info("  Elapsed            : %.1f s", elapsed)
        log.info("-" * 60)
    log.info("All datasets complete. Total runtime: %.1f s", time.monotonic() - t_start)
if __name__ == "__main__":
    main()