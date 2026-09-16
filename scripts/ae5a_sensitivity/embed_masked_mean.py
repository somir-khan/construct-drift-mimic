#!/usr/bin/env python3
"""Embed one frozen primary corpus with masked-mean pooling."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:  # Optional convenience only.
    load_dotenv = None

from _ae5a_common import (
    EXPECTED_DIM,
    EXPECTED_N,
    MODEL_ID,
    ProtocolError,
    assert_primary_reproduction,
    atomic_save_npy,
    atomic_write_json,
    environment_versions,
    load_primary_embeddings,
    primary_geometry,
)


WINDOW_SIZE = 512
INTERIOR_SIZE = 510
STRIDE = 256
SQL_CHUNK = 400
CLS_CHECK_COUNT = 50
CLS_RTOL = 1e-5
CLS_ATOL = 5e-6


def load_ml_dependencies() -> None:
    global torch, tqdm, AutoModel, AutoTokenizer
    try:
        import torch as _torch
        from tqdm import tqdm as _tqdm
        from transformers import AutoModel as _AutoModel
        from transformers import AutoTokenizer as _AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ProtocolError(
            "Masked-mean embedding requires torch, transformers, and tqdm."
        ) from exc
    torch = _torch
    tqdm = _tqdm
    AutoModel = _AutoModel
    AutoTokenizer = _AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create masked-mean embeddings for one frozen primary corpus."
    )
    parser.add_argument("--corpus", required=True, choices=("baseline", "target"))
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "Frozen primary admission-ID array. Defaults to the corresponding "
            "5,000-note file under data/embeddings."
        ),
    )
    parser.add_argument(
        "--baseline-embeddings",
        default="data/embeddings/embeddings_mimic3_5000.npy",
        help="Independent 5,000-note primary MIMIC-III embedding array.",
    )
    parser.add_argument(
        "--target-embeddings",
        default="data/embeddings/embeddings_mimic4_5000.npy",
        help="Independent 5,000-note primary MIMIC-IV embedding array.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mimic3-db", default=os.getenv("MIMIC3_DB_PATH"))
    parser.add_argument("--mimic4-db", default=os.getenv("MIMIC4_DB_PATH"))
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ae5a_sensitivity")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace this corpus's prior masked-mean outputs.",
    )
    return parser.parse_args()


def normalize_phi(text: str) -> str:
    """Map both source placeholder formats to the literal word unknown."""
    text = re.sub(r"\[\*\*.*?\*\*\]", "unknown", text)
    return re.sub(r"___", "unknown", text)


def chunks(values: list[int], size: int = SQL_CHUNK) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_ids(path: str | Path) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProtocolError(f"Frozen ID array not found: {resolved}")
    values = np.load(resolved, allow_pickle=False)
    if values.shape != (EXPECTED_N,):
        raise ProtocolError(
            f"Frozen ID array must have shape {(EXPECTED_N,)}, found {values.shape}"
        )
    try:
        as_int = values.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Frozen IDs are not integer-valued") from exc
    if not np.array_equal(values, as_int):
        raise ProtocolError("Frozen IDs are not integer-valued")
    if len(np.unique(as_int)) != EXPECTED_N:
        raise ProtocolError("Frozen IDs contain duplicate admissions")
    return as_int


def verify_primary_before_model(
    baseline_path: str | Path,
    target_path: str | Path,
    corpus: str,
) -> np.ndarray:
    """Reproduce the manuscript geometry before importing or loading BERT."""
    baseline_resolved = Path(baseline_path).expanduser().resolve()
    target_resolved = Path(target_path).expanduser().resolve()
    print(f"Primary baseline embeddings: {baseline_resolved}")
    print(f"Primary target embeddings: {target_resolved}")
    baseline, target = load_primary_embeddings(baseline_resolved, target_resolved)
    base_pca, target_pca, pca, _retained, sigma, statistic = primary_geometry(
        baseline, target
    )
    assert_primary_reproduction(int(pca.n_components_), statistic)
    print(
        "PASS primary reproduction before model load: "
        f"components={pca.n_components_}, sigma={sigma:.6f}, "
        f"MMD^2={statistic:.6f}"
    )
    reference = baseline if corpus == "baseline" else target
    del base_pca, target_pca, pca
    return reference


def fetch_mimic3_texts(database: str | Path, ordered_ids: np.ndarray) -> list[str]:
    requested = [int(value) for value in ordered_ids]
    found: dict[int, str] = {}
    with sqlite3.connect(str(database)) as connection:
        for part in chunks(requested):
            placeholders = ",".join("?" for _ in part)
            query = f"""
                WITH ranked AS (
                    SELECT HADM_ID AS hadm_id,
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
                      AND HADM_ID IN ({placeholders})
                )
                SELECT hadm_id, text FROM ranked WHERE rn = 1
            """
            for hadm_id, text in connection.execute(query, part):
                key = int(hadm_id)
                if key in found:
                    raise ProtocolError(f"Duplicate latest MIMIC-III row for HADM_ID={key}")
                found[key] = str(text)
    missing = [value for value in requested if value not in found]
    if missing:
        raise ProtocolError(
            f"Missing {len(missing)} frozen MIMIC-III admissions; first IDs: {missing[:10]}"
        )
    return [normalize_phi(found[value]) for value in requested]


def fetch_mimic4_texts(database: str | Path, ordered_ids: np.ndarray) -> list[str]:
    requested = [int(value) for value in ordered_ids]
    found: dict[int, str] = {}
    with sqlite3.connect(str(database)) as connection:
        for part in chunks(requested):
            placeholders = ",".join("?" for _ in part)
            query = f"""
                WITH latest AS (
                    SELECT hadm_id, MAX(note_seq) AS max_seq
                    FROM "note/discharge"
                    WHERE hadm_id IN ({placeholders})
                    GROUP BY hadm_id
                )
                SELECT n.hadm_id, n.text
                FROM "note/discharge" n
                JOIN latest l
                  ON n.hadm_id = l.hadm_id AND n.note_seq = l.max_seq
                WHERE n.text IS NOT NULL AND n.hadm_id IS NOT NULL
            """
            for hadm_id, text in connection.execute(query, part):
                key = int(hadm_id)
                if key in found:
                    raise ProtocolError(f"Duplicate latest MIMIC-IV row for HADM_ID={key}")
                found[key] = str(text)
    missing = [value for value in requested if value not in found]
    if missing:
        raise ProtocolError(
            f"Missing {len(missing)} frozen MIMIC-IV admissions; first IDs: {missing[:10]}"
        )
    return [normalize_phi(found[value]) for value in requested]


def build_windows(interior_ids: list[int]) -> list[list[int]]:
    if not interior_ids:
        return []
    windows: list[list[int]] = []
    start = 0
    while start < len(interior_ids):
        end = min(start + INTERIOR_SIZE, len(interior_ids))
        windows.append(interior_ids[start:end])
        if end == len(interior_ids):
            break
        start += STRIDE
    return windows


def prepare_windows(
    text: str, tokenizer: AutoTokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = encoded["input_ids"][0].tolist()
    if len(all_ids) < 2:
        empty = torch.empty((0, 0), dtype=torch.long)
        return empty, empty, empty

    interior_windows = build_windows(all_ids[1:-1])
    if not interior_windows:
        empty = torch.empty((0, 0), dtype=torch.long)
        return empty, empty, empty

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ProtocolError("Tokenizer does not define CLS and SEP token IDs")

    windows = [[cls_id] + values + [sep_id] for values in interior_windows]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(values) for values in windows)

    id_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    pooling_rows: list[list[int]] = []
    for values in windows:
        padding = max_len - len(values)
        id_rows.append(values + [pad_id] * padding)
        attention_rows.append([1] * len(values) + [0] * padding)
        pooling_rows.append(
            [0] + [1] * (len(values) - 2) + [0] + [0] * padding
        )

    return (
        torch.tensor(id_rows, dtype=torch.long),
        torch.tensor(attention_rows, dtype=torch.long),
        torch.tensor(pooling_rows, dtype=torch.bool),
    )


def embed_note(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    pooling: str = "masked_mean",
) -> tuple[np.ndarray, int]:
    input_ids, attention_mask, pooling_mask = prepare_windows(text, tokenizer)
    if input_ids.numel() == 0:
        raise ProtocolError("Note produced no genuine interior-token windows")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pooling_mask = pooling_mask.to(device)
    token_type_ids = torch.zeros_like(input_ids)
    with torch.no_grad():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

    if pooling == "cls":
        window_vectors = hidden[:, 0, :]
    elif pooling == "masked_mean":
        weights = pooling_mask.unsqueeze(-1).to(hidden.dtype)
        counts = weights.sum(dim=1)
        if torch.any(counts == 0):
            raise ProtocolError("A nonempty window has zero non-special tokens")
        window_vectors = (hidden * weights).sum(dim=1) / counts
    else:
        raise ProtocolError(f"Unknown pooling mode: {pooling}")

    window_array = window_vectors.detach().cpu().numpy()
    document = np.mean(window_array, axis=0).astype(np.float32)
    return document, int(len(input_ids))


def verify_cls_compatibility(
    texts: list[str],
    reference: np.ndarray,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> float:
    """Check the unchanged CLS path on 50 fixed, evenly spaced notes."""
    indices = np.linspace(0, EXPECTED_N - 1, CLS_CHECK_COUNT, dtype=np.int64)
    candidate = np.empty((CLS_CHECK_COUNT, EXPECTED_DIM), dtype=np.float32)
    for position, note_index in enumerate(
        tqdm(indices, desc="CLS compatibility check", unit="note")
    ):
        candidate[position], _ = embed_note(
            texts[int(note_index)], tokenizer, model, device, pooling="cls"
        )

    expected = reference[indices]
    max_absolute_difference = float(
        np.max(
            np.abs(
                candidate.astype(np.float64, copy=False)
                - expected.astype(np.float64, copy=False)
            )
        )
    )
    if not np.allclose(candidate, expected, rtol=CLS_RTOL, atol=CLS_ATOL):
        raise ProtocolError(
            "CLS compatibility check failed on the fixed 50-note subset; "
            f"maximum absolute difference={max_absolute_difference:.9g}"
        )
    print(
        "PASS CLS compatibility check: "
        f"notes={CLS_CHECK_COUNT}, "
        f"maximum absolute difference={max_absolute_difference:.9g}"
    )
    return max_absolute_difference


def embed_corpus(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    result = np.empty((len(texts), EXPECTED_DIM), dtype=np.float32)
    window_counts: list[int] = []
    for index, text in enumerate(
        tqdm(texts, desc="Masked-mean embedding", unit="note")
    ):
        embedding, count = embed_note(
            text, tokenizer, model, device, pooling="masked_mean"
        )
        result[index] = embedding
        window_counts.append(count)
    if not np.isfinite(result).all():
        raise ProtocolError("Masked-mean embeddings contain NaN or infinity")
    return result, window_counts


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()

    default_ids = (
        "data/embeddings/ids_mimic3_5000.npy"
        if args.corpus == "baseline"
        else "data/embeddings/ids_mimic4_5000.npy"
    )
    source_ids_path = Path(args.ids or default_ids).expanduser().resolve()
    ids = load_ids(source_ids_path)

    output_dir = args.output_dir.expanduser().resolve() / "embeddings"
    stem = "mimic3_5000" if args.corpus == "baseline" else "mimic4_5000"
    embedding_path = output_dir / f"embeddings_{stem}_masked_mean.npy"
    ids_path = output_dir / f"ids_{stem}_masked_mean.npy"
    metadata_path = output_dir / f"metadata_{stem}_masked_mean.json"
    existing = [path for path in (embedding_path, ids_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise ProtocolError(
            "Refusing to overwrite prior outputs: " + ", ".join(map(str, existing))
        )

    reference_embeddings = verify_primary_before_model(
        args.baseline_embeddings, args.target_embeddings, args.corpus
    )

    if args.corpus == "baseline":
        if not args.mimic3_db or not Path(args.mimic3_db).is_file():
            raise ProtocolError(f"MIMIC3 database not found: {args.mimic3_db}")
        texts = fetch_mimic3_texts(args.mimic3_db, ids)
    else:
        if not args.mimic4_db or not Path(args.mimic4_db).is_file():
            raise ProtocolError(f"MIMIC4 database not found: {args.mimic4_db}")
        texts = fetch_mimic4_texts(args.mimic4_db, ids)

    load_ml_dependencies()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("--device cuda was requested, but CUDA is unavailable")
    if args.device == "cpu" and args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModel.from_pretrained(MODEL_ID, token=token).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    verify_cls_compatibility(texts, reference_embeddings, tokenizer, model, device)
    embeddings, window_counts = embed_corpus(texts, tokenizer, model, device)
    atomic_save_npy(embedding_path, embeddings)
    atomic_save_npy(ids_path, ids)
    metadata = {
        "schema": "ae5a-masked-mean-metadata-minimal-v3",
        "corpus": args.corpus,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_ids": str(source_ids_path),
        "model_id": MODEL_ID,
        "device": str(device),
        "pooling": {
            "within_window": (
                "mean final-layer non-special token states; CLS, SEP, and padding excluded"
            ),
            "across_windows": "unweighted NumPy arithmetic mean",
            "l2_normalization": False,
            "dtype": "float32",
        },
        "windowing": {
            "window_size_with_special_tokens": WINDOW_SIZE,
            "interior_tokens": INTERIOR_SIZE,
            "stride": STRIDE,
        },
        "window_count_summary": {
            "minimum": min(window_counts),
            "maximum": max(window_counts),
            "mean": float(np.mean(window_counts)),
        },
        "software_environment": environment_versions(),
    }
    atomic_write_json(metadata_path, metadata)
    print(f"Saved masked-mean embeddings: {embedding_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
