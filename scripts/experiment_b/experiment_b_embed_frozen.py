#!/usr/bin/env python3
"""
experiment_b_embed_frozen.py

Manifest-driven raw embedding generation for Experiment B.

IMPORTANT
---------
This script deliberately preserves the submitted frozen anchor exactly:

MIMIC-III document construction:
    Latest eligible discharge-summary row per HADM_ID:
        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID
            ORDER BY CHARTDATE DESC, ROW_ID DESC
        )

MIMIC-IV document construction:
    Latest discharge note by MAX(note_seq), matching embed_and_save.py.

PHI normalization:
    [** ... **] -> "unknown"
    ___         -> "unknown"

BioClinicalBERT document embedding:
    - tokenize complete note WITHOUT truncation
    - remove original CLS/SEP
    - overlapping windows of <=510 interior tokens
    - stride 256
    - prepend CLS and append SEP to every window
    - extract CLS from every window
    - arithmetic mean across window CLS vectors
    - output float32, 768 dimensions

This script does NOT:
    - sample admissions randomly
    - fit PCA
    - train a classifier
    - inspect ICD prediction performance
    - alter the frozen Experiment B cohorts

It consumes:
    experiment_b_m3_split_manifest.csv
    experiment_b_m4_icd9_only_manifest.csv

Outputs one aligned embedding/ID pair for each requested cohort.

The script is resumable.  While a cohort is running it maintains:
    <stem>.partial.npy
    <stem>.progress.json

On successful completion these are finalized as ordinary .npy files.

Example
-------
python experiment_b_embed_frozen.py \
    --m3-manifest outputs/experiment_b_preflight/experiment_b_m3_split_manifest.csv \
    --m4-manifest outputs/experiment_b_preflight/experiment_b_m4_icd9_only_manifest.csv \
    --mimic3-db /path/to/mimic3.db \
    --mimic4-db /path/to/mimic4.db \
    --device cuda \
    --cohorts m3_train m3_dev m3_test m4_target \
    --output-dir data/experiment_b_embeddings
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================================
# Environment
# ============================================================================

load_dotenv()

DEFAULT_MIMIC3_DB = os.getenv("MIMIC3_DB_PATH")
DEFAULT_MIMIC4_DB = os.getenv("MIMIC4_DB_PATH")

EMBED_DIM = 768
WINDOW_SIZE = 512
INTERIOR_SIZE = WINDOW_SIZE - 2  # 510


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

log = logging.getLogger(__name__)

for noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "filelock",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Manifest-driven Experiment B BioClinicalBERT embedding pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--m3-manifest",
        required=True,
        help=(
            "experiment_b_m3_split_manifest.csv "
            "with columns split,hadm_id."
        ),
    )

    parser.add_argument(
        "--m4-manifest",
        required=True,
        help=(
            "experiment_b_m4_icd9_only_manifest.csv "
            "with at least column hadm_id."
        ),
    )

    parser.add_argument(
        "--mimic3-db",
        default=DEFAULT_MIMIC3_DB,
        help="MIMIC-III SQLite DB.",
    )

    parser.add_argument(
        "--mimic4-db",
        default=DEFAULT_MIMIC4_DB,
        help="MIMIC-IV SQLite DB.",
    )

    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=[
            "m3_train",
            "m3_dev",
            "m3_test",
            "m4_target",
        ],
        default=[
            "m3_train",
            "m3_dev",
            "m3_test",
            "m4_target",
        ],
    )

    parser.add_argument(
        "--output-dir",
        default="data/experiment_b_embeddings",
    )

    parser.add_argument(
        "--model",
        default="emilyalsentzer/Bio_ClinicalBERT",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush partial embedding array every N completed notes.",
    )

    parser.add_argument(
        "--max-note-retries",
        type=int,
        default=2,
        help=(
            "Retry a note after an embedding exception. "
            "Persistent failures stop the run; they are never silently "
            "accepted as zero vectors."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Use only the first N manifest rows. "
            "ONLY for CPU/GPU equivalence testing. "
            "Do not use for the final Experiment B run."
        ),
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="CPU PyTorch threads when --device cpu.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already completed cohort outputs.",
    )

    return parser.parse_args()


# ============================================================================
# Small helpers
# ============================================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    os.replace(tmp, path)


def normalize_phi(text):
    """
    EXACT submitted PHI normalization.
    """
    text = re.sub(r"\[\*\*.*?\*\*\]", "unknown", text)
    text = re.sub(r"___", "unknown", text)

    return text


# ============================================================================
# Manifest loading
# ============================================================================

def load_manifests(args):
    m3 = pd.read_csv(args.m3_manifest)
    m4 = pd.read_csv(args.m4_manifest)

    required_m3 = {"split", "hadm_id"}

    if not required_m3.issubset(m3.columns):
        raise ValueError(
            f"M3 manifest requires {required_m3}; "
            f"found {m3.columns.tolist()}"
        )

    if "hadm_id" not in m4.columns:
        raise ValueError(
            f"M4 manifest requires hadm_id; "
            f"found {m4.columns.tolist()}"
        )

    m3["hadm_id"] = m3["hadm_id"].astype(np.int64)
    m4["hadm_id"] = m4["hadm_id"].astype(np.int64)

    if m3.duplicated(["split", "hadm_id"]).any():
        raise RuntimeError(
            "Duplicate split/HADM_ID rows in M3 manifest."
        )

    if m4["hadm_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate HADM_ID rows in M4 manifest."
        )

    valid_splits = {"train", "dev", "test"}

    unexpected = set(m3["split"].unique()) - valid_splits

    if unexpected:
        raise RuntimeError(
            f"Unexpected M3 split labels: {sorted(unexpected)}"
        )

    cohort_frames = {
        "m3_train": (
            m3.loc[
                m3["split"] == "train",
                ["hadm_id"],
            ]
            .reset_index(drop=True)
        ),

        "m3_dev": (
            m3.loc[
                m3["split"] == "dev",
                ["hadm_id"],
            ]
            .reset_index(drop=True)
        ),

        "m3_test": (
            m3.loc[
                m3["split"] == "test",
                ["hadm_id"],
            ]
            .reset_index(drop=True)
        ),

        "m4_target": (
            m4[["hadm_id"]]
            .reset_index(drop=True)
        ),
    }

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be >0.")

        log.warning(
            "TEST MODE: --limit=%d. "
            "These are NOT final Experiment B embeddings.",
            args.limit,
        )

        cohort_frames = {
            k: df.head(args.limit).copy()
            for k, df in cohort_frames.items()
        }

    return cohort_frames


# ============================================================================
# Exact manifest-driven note loading
# ============================================================================

def make_temp_manifest(
    conn,
    ids,
    table_name="expb_embed_manifest",
):
    conn.execute(
        f'DROP TABLE IF EXISTS "{table_name}"'
    )

    conn.execute(
        f"""
        CREATE TEMP TABLE "{table_name}" (
            ord INTEGER PRIMARY KEY,
            hadm_id INTEGER NOT NULL UNIQUE
        )
        """
    )

    conn.executemany(
        f"""
        INSERT INTO "{table_name}"(ord, hadm_id)
        VALUES (?, ?)
        """,
        [
            (i, int(hadm_id))
            for i, hadm_id in enumerate(ids)
        ],
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{table_name}_hadm
        ON "{table_name}"(hadm_id)
        """
    )


def load_mimic3_manifest_notes(
    db_path,
    manifest_df,
):
    """
    Frozen MIMIC-III construction.

    This is the submitted latest eligible discharge-summary rule:
        PARTITION BY HADM_ID
        ORDER BY CHARTDATE DESC, ROW_ID DESC

    Output order is EXACTLY the input manifest order.
    """
    ids = manifest_df["hadm_id"].astype(np.int64).tolist()

    conn = sqlite3.connect(db_path)

    try:
        make_temp_manifest(conn, ids)

        query = """
        WITH ranked AS (
            SELECT
                m.ord,
                n.HADM_ID AS hadm_id,
                n.TEXT AS text,

                ROW_NUMBER() OVER (
                    PARTITION BY n.HADM_ID
                    ORDER BY
                        n.CHARTDATE DESC,
                        n.ROW_ID DESC
                ) AS rn

            FROM expb_embed_manifest m

            JOIN NOTEEVENTS n
              ON n.HADM_ID = m.hadm_id

            WHERE n.CATEGORY = 'Discharge summary'
              AND (n.ISERROR IS NULL OR n.ISERROR != '1')
              AND n.HADM_ID IS NOT NULL
              AND n.TEXT IS NOT NULL
        )

        SELECT
            ord,
            hadm_id,
            text

        FROM ranked

        WHERE rn = 1

        ORDER BY ord
        """

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    if len(df) != len(ids):
        returned = set(
            df["hadm_id"].astype(np.int64).tolist()
        )

        missing = [
            x for x in ids
            if x not in returned
        ]

        raise RuntimeError(
            "M3 note extraction did not return every manifest ID. "
            f"expected={len(ids):,}, returned={len(df):,}, "
            f"missing={len(missing):,}, "
            f"first_missing={missing[:20]}"
        )

    observed_ids = (
        df["hadm_id"]
        .astype(np.int64)
        .tolist()
    )

    if observed_ids != ids:
        raise RuntimeError(
            "M3 note extraction order does not match manifest order."
        )

    texts = [
        normalize_phi(str(x))
        for x in df["text"].tolist()
    ]

    return texts, observed_ids


def load_mimic4_manifest_notes(
    db_path,
    manifest_df,
):
    """
    Frozen MIMIC-IV construction matching embed_and_save.py:
        MAX(note_seq) per HADM_ID.

    The current preflight found one discharge row per target admission,
    but we preserve the submitted loader definition regardless.

    Output order is EXACTLY the input manifest order.
    """
    ids = manifest_df["hadm_id"].astype(np.int64).tolist()

    conn = sqlite3.connect(db_path)

    try:
        make_temp_manifest(conn, ids)

        query = """
        WITH latest AS (
        SELECT
        n.hadm_id,
        MAX(n.note_seq) AS max_seq
        FROM "note/discharge" n
        JOIN expb_embed_manifest m
        ON m.hadm_id = n.hadm_id
        GROUP BY n.hadm_id
        )

    SELECT
        m.ord,
        n.hadm_id,
        n.text

    FROM expb_embed_manifest m
    JOIN latest l
        ON l.hadm_id = m.hadm_id
    JOIN "note/discharge" n
        ON n.hadm_id = l.hadm_id
        AND n.note_seq = l.max_seq
    WHERE n.text IS NOT NULL
        AND n.hadm_id IS NOT NULL

ORDER BY m.ord
"""

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    if len(df) != len(ids):
        returned = set(
            df["hadm_id"].astype(np.int64).tolist()
        )

        missing = [
            x for x in ids
            if x not in returned
        ]

        raise RuntimeError(
            "M4 note extraction did not return every manifest ID. "
            f"expected={len(ids):,}, returned={len(df):,}, "
            f"missing={len(missing):,}, "
            f"first_missing={missing[:20]}"
        )

    observed_ids = (
        df["hadm_id"]
        .astype(np.int64)
        .tolist()
    )

    if observed_ids != ids:
        raise RuntimeError(
            "M4 note extraction order does not match manifest order."
        )

    texts = [
        normalize_phi(str(x))
        for x in df["text"].tolist()
    ]

    return texts, observed_ids


# ============================================================================
# Exact submitted chunking
# ============================================================================

def build_windows(
    interior_ids,
    window_size=INTERIOR_SIZE,
    stride=256,
):
    """
    EXACT mathematical window definition from embed_and_save.py.
    """
    if not interior_ids:
        return []

    n = len(interior_ids)
    windows = []
    start = 0

    while start < n:
        end = min(
            start + window_size,
            n,
        )

        windows.append(
            interior_ids[start:end]
        )

        if end == n:
            break

        start += stride

    return windows


# ============================================================================
# Exact submitted per-note embedding
# ============================================================================

def embed_single_note(
    text,
    tokenizer,
    model,
    device,
    stride=256,
):
    """
    Preserve the submitted embedding operation exactly.

    No autocast.
    No FP16.
    No BF16.
    No cross-note batching.
    No window microbatching.

    All windows belonging to one note are evaluated together, just as in
    embed_and_save.py.
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    )

    input_ids_all = (
        encoded["input_ids"][0]
        .tolist()
    )

    if not input_ids_all:
        return (
            np.zeros(
                EMBED_DIM,
                dtype=np.float32,
            ),
            0,
        )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Strip original CLS / SEP.
    interior = input_ids_all[1:-1]

    if not interior:
        return (
            np.zeros(
                EMBED_DIM,
                dtype=np.float32,
            ),
            0,
        )

    interior_windows = build_windows(
        interior,
        window_size=INTERIOR_SIZE,
        stride=stride,
    )

    if not interior_windows:
        return (
            np.zeros(
                EMBED_DIM,
                dtype=np.float32,
            ),
            0,
        )

    # Re-add CLS / SEP to every window.
    windows = [
        [cls_id] + w + [sep_id]
        for w in interior_windows
    ]

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    max_len = max(
        len(w)
        for w in windows
    )

    ids_rows = []
    mask_rows = []

    for w in windows:
        pad = max_len - len(w)

        ids_rows.append(
            w + [pad_id] * pad
        )

        mask_rows.append(
            [1] * len(w)
            + [0] * pad
        )

    input_ids_t = torch.tensor(
        ids_rows,
        dtype=torch.long,
        device=device,
    )

    attn_mask_t = torch.tensor(
        mask_rows,
        dtype=torch.long,
        device=device,
    )

    token_type_ids_t = torch.zeros_like(
        input_ids_t
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            token_type_ids=token_type_ids_t,
        )

    cls_vecs = (
        outputs
        .last_hidden_state[:, 0, :]
        .detach()
        .cpu()
        .numpy()
    )

    embedding = np.mean(
        cls_vecs,
        axis=0,
    ).astype(np.float32)

    return embedding, len(windows)


# ============================================================================
# Runtime provenance
# ============================================================================

def configure_runtime(
    device_name,
    num_threads,
):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False."
            )

        # Keep inference FP32 and avoid TF32 approximation on Ampere/Hopper.
        torch.backends.cuda.matmul.allow_tf32 = False

        if hasattr(
            torch.backends,
            "cudnn",
        ):
            torch.backends.cudnn.allow_tf32 = False

        try:
            torch.set_float32_matmul_precision(
                "highest"
            )
        except Exception:
            pass

        device = torch.device("cuda")

        log.info(
            "CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

        log.info(
            "CUDA capability: %s",
            torch.cuda.get_device_capability(0),
        )

    else:
        n_threads = (
            num_threads
            if num_threads is not None
            else os.cpu_count()
        )

        torch.set_num_threads(
            max(1, int(n_threads))
        )

        try:
            torch.set_num_interop_threads(
                max(
                    1,
                    int(n_threads) // 2,
                )
            )
        except RuntimeError:
            pass

        device = torch.device("cpu")

        log.info(
            "CPU mode: %d threads",
            torch.get_num_threads(),
        )

    return device


def runtime_metadata(
    args,
    model,
    device,
):
    model_commit = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    payload = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "model_name":
            args.model,

        "model_commit_hash":
            model_commit,

        "device":
            str(device),

        "pooling_strategy":
            "uniform_mean",

        "embedding_dim":
            EMBED_DIM,

        "window_size":
            WINDOW_SIZE,

        "interior_size":
            INTERIOR_SIZE,

        "stride":
            args.stride,

        "phi_normalization": {
            "mimic3_pattern":
                r"\[\*\*.*?\*\*\]",

            "mimic4_pattern":
                "___",

            "replacement":
                "unknown",
        },

        "precision_policy": {
            "model_dtype":
                str(
                    next(
                        model.parameters()
                    ).dtype
                ),

            "autocast":
                False,

            "tf32_cuda_matmul":
                (
                    torch.backends.cuda.matmul.allow_tf32
                    if torch.cuda.is_available()
                    else None
                ),
        },
    }

    if device.type == "cuda":
        payload["cuda"] = {
            "device_name":
                torch.cuda.get_device_name(0),

            "device_capability":
                list(
                    torch.cuda.get_device_capability(0)
                ),

            "cuda_runtime":
                torch.version.cuda,
        }

    return payload


# ============================================================================
# Cohort embedding with checkpoint / resume
# ============================================================================

def cohort_paths(
    output_dir,
    cohort,
    n,
):
    stem = f"{cohort}_{n}"

    return {
        "stem":
            stem,

        "partial":
            output_dir
            / f"embeddings_{stem}.partial.npy",

        "progress":
            output_dir
            / f"embeddings_{stem}.progress.json",

        "embeddings":
            output_dir
            / f"embeddings_{stem}.npy",

        "ids":
            output_dir
            / f"ids_{stem}.npy",

        "metadata":
            output_dir
            / f"metadata_{stem}.json",
    }


def embed_cohort(
    cohort,
    manifest_df,
    texts,
    ids,
    tokenizer,
    model,
    device,
    args,
    common_metadata,
):
    n = len(ids)

    if len(texts) != n:
        raise RuntimeError(
            f"{cohort}: text/ID count mismatch."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = cohort_paths(
        output_dir,
        cohort,
        n,
    )

    if (
        paths["embeddings"].exists()
        and paths["ids"].exists()
        and paths["metadata"].exists()
        and not args.overwrite
    ):
        log.info(
            "%s already complete; skipping: %s",
            cohort,
            paths["embeddings"],
        )

        return

    if args.overwrite:
        for key in (
            "partial",
            "progress",
            "embeddings",
            "ids",
            "metadata",
        ):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass

    cohort_manifest_hash = hashlib.sha256(
        np.asarray(
            ids,
            dtype=np.int64,
        ).tobytes()
    ).hexdigest()

    # ------------------------------------------------------------------------
    # Resume or initialize memmap
    # ------------------------------------------------------------------------

    next_index = 0
    cumulative_chunks = 0

    if (
        paths["partial"].exists()
        and paths["progress"].exists()
    ):
        with open(
            paths["progress"],
            "r",
            encoding="utf-8",
        ) as fh:
            progress = json.load(fh)

        checks = {
            "cohort":
                cohort,

            "n_notes":
                n,

            "manifest_id_sha256":
                cohort_manifest_hash,

            "model_name":
                args.model,

            "stride":
                args.stride,
        }

        for key, expected in checks.items():
            observed = progress.get(key)

            if observed != expected:
                raise RuntimeError(
                    f"{cohort}: resume mismatch for {key}: "
                    f"progress={observed!r}, current={expected!r}"
                )

        arr = np.load(
            paths["partial"],
            mmap_mode="r+",
        )

        if arr.shape != (
            n,
            EMBED_DIM,
        ):
            raise RuntimeError(
                f"{cohort}: partial array shape mismatch: "
                f"{arr.shape}"
            )

        next_index = int(
            progress["next_index"]
        )

        cumulative_chunks = int(
            progress.get(
                "total_chunks",
                0,
            )
        )

        log.info(
            "%s resuming at %d/%d.",
            cohort,
            next_index,
            n,
        )

    else:
        arr = np.lib.format.open_memmap(
            paths["partial"],
            mode="w+",
            dtype=np.float32,
            shape=(n, EMBED_DIM),
        )

        progress = {
            "cohort":
                cohort,

            "n_notes":
                n,

            "manifest_id_sha256":
                cohort_manifest_hash,

            "model_name":
                args.model,

            "stride":
                args.stride,

            "next_index":
                0,

            "total_chunks":
                0,

            "started_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        write_json_atomic(
            paths["progress"],
            progress,
        )

    # ------------------------------------------------------------------------
    # Main sequential embedding loop
    # ------------------------------------------------------------------------

    started = time.monotonic()

    pbar = tqdm(
        range(
            next_index,
            n,
        ),
        total=n,
        initial=next_index,
        desc=cohort,
        unit="note",
    )

    for i in pbar:
        text = texts[i]

        last_exc = None

        for attempt in range(
            1,
            args.max_note_retries + 2,
        ):
            try:
                emb, n_chunks = embed_single_note(
                    text=text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    stride=args.stride,
                )

                if emb.shape != (
                    EMBED_DIM,
                ):
                    raise RuntimeError(
                        f"Unexpected embedding shape {emb.shape}"
                    )

                if not np.isfinite(
                    emb
                ).all():
                    raise RuntimeError(
                        "Embedding contains non-finite values."
                    )

                arr[i] = emb

                cumulative_chunks += int(
                    n_chunks
                )

                last_exc = None

                break

            except Exception as exc:
                last_exc = exc

                log.warning(
                    "%s index=%d hadm_id=%s "
                    "attempt=%d failed: %s",
                    cohort,
                    i,
                    ids[i],
                    attempt,
                    exc,
                )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                time.sleep(1)

        if last_exc is not None:
            arr.flush()

            progress.update(
                {
                    "next_index":
                        i,

                    "total_chunks":
                        cumulative_chunks,

                    "failed_index":
                        i,

                    "failed_hadm_id":
                        int(ids[i]),

                    "failed_error":
                        repr(last_exc),

                    "updated_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }
            )

            write_json_atomic(
                paths["progress"],
                progress,
            )

            raise RuntimeError(
                f"{cohort}: persistent embedding failure "
                f"at index={i}, HADM_ID={ids[i]}. "
                "Partial work is preserved and the run can be resumed."
            ) from last_exc

        # Save NEXT unprocessed index.
        if (
            (i + 1) % args.checkpoint_every == 0
            or i + 1 == n
        ):
            arr.flush()

            progress.update(
                {
                    "next_index":
                        i + 1,

                    "total_chunks":
                        cumulative_chunks,

                    "updated_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }
            )

            # Remove stale failure metadata after a successful retry/resume.
            for key in (
                "failed_index",
                "failed_hadm_id",
                "failed_error",
            ):
                progress.pop(
                    key,
                    None,
                )

            write_json_atomic(
                paths["progress"],
                progress,
            )

    elapsed = (
        time.monotonic()
        - started
    )

    arr.flush()
    del arr

    # ------------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------------

    # Rename a valid NPY file; no numerical rewrite.
    os.replace(
        paths["partial"],
        paths["embeddings"],
    )

    np.save(
        paths["ids"],
        np.asarray(
            ids,
            dtype=np.int64,
        ),
    )

    final_emb = np.load(
        paths["embeddings"],
        mmap_mode="r",
    )

    final_ids = np.load(
        paths["ids"],
    )

    if final_emb.shape != (
        n,
        EMBED_DIM,
    ):
        raise RuntimeError(
            f"{cohort}: final embedding shape invalid: "
            f"{final_emb.shape}"
        )

    if final_ids.shape != (
        n,
    ):
        raise RuntimeError(
            f"{cohort}: final ID shape invalid: "
            f"{final_ids.shape}"
        )

    if not np.array_equal(
        final_ids,
        np.asarray(
            ids,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"{cohort}: final ID file is not aligned "
            "to the frozen manifest."
        )

    # Full scan is okay here and detects zero/NaN corruption.
    nonfinite_rows = int(
        np.sum(
            ~np.isfinite(
                final_emb
            ).all(
                axis=1
            )
        )
    )

    zero_rows = int(
        np.sum(
            np.all(
                final_emb == 0,
                axis=1,
            )
        )
    )

    if nonfinite_rows != 0:
        raise RuntimeError(
            f"{cohort}: {nonfinite_rows} final rows "
            "contain non-finite values."
        )

    if zero_rows != 0:
        raise RuntimeError(
            f"{cohort}: {zero_rows} all-zero embedding rows. "
            "Do not continue to PCA/probe evaluation."
        )

    metadata = dict(
        common_metadata
    )

    metadata.update(
        {
            "cohort":
                cohort,

            "n_notes":
                n,

            "manifest_id_sha256":
                cohort_manifest_hash,

            "embeddings_filename":
                paths["embeddings"].name,

            "ids_filename":
                paths["ids"].name,

            "embeddings_sha256":
                sha256_file(
                    paths["embeddings"]
                ),

            "ids_sha256":
                sha256_file(
                    paths["ids"]
                ),

            "zero_embedding_rows":
                zero_rows,

            "nonfinite_embedding_rows":
                nonfinite_rows,

            "total_chunks":
                cumulative_chunks,

            "mean_chunks_per_note":
                (
                    cumulative_chunks / n
                    if n
                    else 0.0
                ),

            "elapsed_seconds_this_session":
                elapsed,

            "completed_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "document_construction":
                (
                    "MIMIC-III submitted latest-row rule"
                    if cohort.startswith("m3_")
                    else
                    "MIMIC-IV submitted MAX(note_seq) rule"
                ),

            "experiment_b_pca_fitted_here":
                False,
        }
    )

    write_json_atomic(
        paths["metadata"],
        metadata,
    )

    try:
        paths["progress"].unlink()
    except FileNotFoundError:
        pass

    log.info(
        "%s COMPLETE | embeddings=%s | ids=%s",
        cohort,
        paths["embeddings"],
        paths["ids"],
    )

    log.info(
        "%s chunks=%d mean_chunks/note=%.3f",
        cohort,
        cumulative_chunks,
        cumulative_chunks / n,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------------

    for path in (
        args.m3_manifest,
        args.m4_manifest,
    ):
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    if any(
        c.startswith("m3_")
        for c in args.cohorts
    ):
        if (
            not args.mimic3_db
            or not os.path.exists(
                args.mimic3_db
            )
        ):
            raise FileNotFoundError(
                f"MIMIC-III DB not found: {args.mimic3_db}"
            )

    if "m4_target" in args.cohorts:
        if (
            not args.mimic4_db
            or not os.path.exists(
                args.mimic4_db
            )
        ):
            raise FileNotFoundError(
                f"MIMIC-IV DB not found: {args.mimic4_db}"
            )

    if args.stride != 256:
        log.warning(
            "The submitted frozen pipeline used stride=256. "
            "You supplied stride=%d.",
            args.stride,
        )

    cohort_frames = load_manifests(
        args
    )

    expected_full_counts = {
        "m3_train": 47723,
        "m3_dev": 1631,
        "m3_test": 3372,
        "m4_target": 19667,
    }

    if args.limit is None:
        for cohort in args.cohorts:
            observed = len(
                cohort_frames[cohort]
            )

            expected = expected_full_counts[
                cohort
            ]

            if observed != expected:
                raise RuntimeError(
                    f"{cohort}: expected frozen count "
                    f"{expected:,}, found {observed:,}."
                )

    log.info(
        "Requested cohorts: %s",
        args.cohorts,
    )

    for cohort in args.cohorts:
        log.info(
            "%s manifest rows: %d",
            cohort,
            len(
                cohort_frames[
                    cohort
                ]
            ),
        )

    # ------------------------------------------------------------------------
    # Configure device
    # ------------------------------------------------------------------------

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    device = configure_runtime(
        args.device,
        args.num_threads,
    )

    # ------------------------------------------------------------------------
    # Load frozen anchor
    # ------------------------------------------------------------------------

    hf_token = os.getenv(
        "HF_TOKEN"
    )

    log.info(
        "Loading tokenizer: %s",
        args.model,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
    )

    log.info(
        "Loading model: %s",
        args.model,
    )

    model = AutoModel.from_pretrained(
        args.model,
        token=hf_token,
    )

    model = model.to(
        device
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    log.info(
        "Model frozen | parameters=%s | dtype=%s",
        f"{sum(p.numel() for p in model.parameters()):,}",
        next(model.parameters()).dtype,
    )

    common_metadata = runtime_metadata(
        args,
        model,
        device,
    )

    common_metadata.update(
        {
            "m3_manifest_path":
                os.path.abspath(
                    args.m3_manifest
                ),

            "m3_manifest_file_sha256":
                sha256_file(
                    args.m3_manifest
                ),

            "m4_manifest_path":
                os.path.abspath(
                    args.m4_manifest
                ),

            "m4_manifest_file_sha256":
                sha256_file(
                    args.m4_manifest
                ),

            "mimic3_db_path":
                (
                    os.path.abspath(
                        args.mimic3_db
                    )
                    if args.mimic3_db
                    else None
                ),

            "mimic4_db_path":
                (
                    os.path.abspath(
                        args.mimic4_db
                    )
                    if args.mimic4_db
                    else None
                ),

            "report_preferred_used":
                False,

            "pca_policy":
                (
                    "NO PCA is fitted by this script. "
                    "Downstream Experiment B must use the existing "
                    "frozen submitted pca_model.pkl."
                ),

            "test_limit":
                args.limit,
        }
    )

    # ------------------------------------------------------------------------
    # Extract + embed each cohort
    # ------------------------------------------------------------------------

    for cohort in args.cohorts:
        manifest_df = (
            cohort_frames[
                cohort
            ]
        )

        log.info(
            "=" * 72
        )

        log.info(
            "Preparing cohort %s",
            cohort,
        )

        if cohort.startswith(
            "m3_"
        ):
            texts, ids = load_mimic3_manifest_notes(
                args.mimic3_db,
                manifest_df,
            )

        else:
            texts, ids = load_mimic4_manifest_notes(
                args.mimic4_db,
                manifest_df,
            )

        log.info(
            "%s loaded %d exact manifest-aligned notes.",
            cohort,
            len(texts),
        )

        embed_cohort(
            cohort=cohort,
            manifest_df=manifest_df,
            texts=texts,
            ids=ids,
            tokenizer=tokenizer,
            model=model,
            device=device,
            args=args,
            common_metadata=common_metadata,
        )

    log.info(
        "=" * 72
    )

    log.info(
        "ALL REQUESTED EXPERIMENT B EMBEDDING COHORTS COMPLETE."
    )

    log.info(
        "Do NOT refit PCA. Next stage must transform these arrays "
        "with the existing frozen submitted pca_model.pkl."
    )


if __name__ == "__main__":
    main()