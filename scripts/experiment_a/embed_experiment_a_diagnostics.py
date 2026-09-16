#!/usr/bin/env python3
"""Compute frozen Experiment A embedding and witness diagnostics.

Run only after the 200-host variant manifest has passed generation validation:

    python scripts/experiment_a/embed_experiment_a_diagnostics.py --device cuda

This script never changes arm membership, host selection, donor selection, or
variant text, and it makes no Judge calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer


BASELINE_PCA = Path("data/baseline_pca.npy")
PCA_MODEL = Path("data/pca_model.pkl")
HOSTS_PATH = Path("data/experiment_a_hosts.csv")
VARIANTS_PATH = Path("data/experiment_a_variants.csv")
CANDIDATES_PATH = Path("data/experiment_a_candidate_pool.csv")
SELECTION_REPORT_PATH = Path("data/experiment_a_host_selection_report.json")
OUTPUT_PATH = Path("data/experiment_a_diagnostics.csv")
WINDOWS = (
    ("2014 - 2016", 1,
     Path("data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2014_2016.npy")),
    ("2017 - 2019", 2,
     Path("data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy"),
     Path("data/embeddings_windows/ids_mimic4_2500_2017_2019.npy")),
)
N_HOSTS = 200
N_CANDIDATES = 2500
N_PER_CELL = 50
SEED = 42
BANDWIDTH_SUBSAMPLE = 1000
KERNEL_BLOCK_ROWS = 128
MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 768
INTERIOR_WINDOW = 510
STRIDE = 256
SCORE_TOLERANCE = 1e-10

load_dotenv()
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger(__name__)
for noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

PHI = re.compile(r"\[\*\*.*?\*\*\]")


class DiagnosticError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise DiagnosticError(message)


def normalize(text: str) -> str:
    return PHI.sub("unknown", text).replace("___", "unknown")


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        fail(f"{path} is missing required columns: {missing}")


def load_matrix(path: Path, name: str, columns: int | None = None) -> np.ndarray:
    if not path.is_file():
        fail(f"Missing {name}: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        fail(f"{name} must be a finite two-dimensional array; got {array.shape}.")
    if columns is not None and array.shape[1] != columns:
        fail(f"{name} must have {columns} columns; got {array.shape[1]}.")
    return np.asarray(array, dtype=np.float64)


def load_ids(path: Path, expected_rows: int, label: str) -> np.ndarray:
    if not path.is_file():
        fail(f"Missing {label} IDs: {path}")
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 1 or len(raw) != expected_rows:
        fail(f"{label} IDs must have {expected_rows} rows; got {raw.shape}.")
    ids = pd.to_numeric(pd.Series(raw), errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        fail(f"{label} IDs contain duplicates.")
    return ids


def median_sigma(pooled: np.ndarray, rng: np.random.Generator) -> float:
    take = min(len(pooled), 2 * BANDWIDTH_SUBSAMPLE)
    sample = pooled[rng.choice(len(pooled), size=take, replace=False)]
    norms = np.sum(sample * sample, axis=1, keepdims=True)
    squared = np.maximum(norms + norms.T - 2.0 * (sample @ sample.T), 0.0)
    distances = squared[np.triu_indices(take, k=1)]
    sigma = float(np.sqrt(np.median(distances) / 2.0))
    if not np.isfinite(sigma) or sigma <= 0:
        fail("Median-heuristic bandwidth is not positive and finite.")
    return sigma


def mean_rbf(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    right_norm = np.sum(right * right, axis=1)
    result = np.empty(len(left), dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma * sigma)
    for start in range(0, len(left), KERNEL_BLOCK_ROWS):
        block = left[start:start + KERNEL_BLOCK_ROWS]
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        squared = np.maximum(block_norm + right_norm - 2.0 * (block @ right.T), 0.0)
        result[start:start + len(block)] = np.exp(-gamma * squared).mean(axis=1)
    return result


def witness(points: np.ndarray, baseline: np.ndarray, target: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate the fixed witness function against unmodified B and T_c."""
    return mean_rbf(points, target, sigma) - mean_rbf(points, baseline, sigma)


def build_windows(interior_ids: list[int], stride: int = STRIDE) -> list[list[int]]:
    if not interior_ids:
        return []
    windows, start = [], 0
    while start < len(interior_ids):
        end = min(start + INTERIOR_WINDOW, len(interior_ids))
        windows.append(interior_ids[start:end])
        if end == len(interior_ids):
            break
        start += stride
    return windows


def embed_note(
    text: str, tokenizer: AutoTokenizer, model: AutoModel, device: torch.device
) -> tuple[np.ndarray, int]:
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = encoded["input_ids"][0].tolist()
    interior = all_ids[1:-1]
    windows = build_windows(interior)
    if not windows:
        fail("A note produced no interior tokenizer windows.")
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        fail("Tokenizer is missing CLS or SEP token IDs.")
    windows = [[cls_id] + window + [sep_id] for window in windows]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    width = max(len(window) for window in windows)
    ids = [window + [pad_id] * (width - len(window)) for window in windows]
    masks = [[1] * len(window) + [0] * (width - len(window)) for window in windows]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention = torch.tensor(masks, dtype=torch.long, device=device)
    token_types = torch.zeros_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention, token_type_ids=token_types)
    embedding = output.last_hidden_state[:, 0, :].cpu().numpy().mean(axis=0).astype(np.float32)
    if embedding.shape != (EMBED_DIM,) or not np.isfinite(embedding).all() or not np.any(embedding):
        fail("Embedding is invalid or a zero vector.")
    return embedding, len(windows)


def load_hosts_and_variants(hosts_path: Path, variants_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hosts_path.is_file() or not variants_path.is_file():
        fail("Missing frozen host or variant manifest.")
    hosts, variants = pd.read_csv(hosts_path), pd.read_csv(variants_path)
    require_columns(
        hosts,
        {"host_manifest_order", "cohort", "cohort_code", "arm", "hadm_id", "subject_id",
         "selected_note_rowid", "selected_note_seq"},
        hosts_path,
    )
    require_columns(
        variants,
        {"variant_manifest_order", "variant_id", "host_manifest_order", "cohort", "cohort_code",
         "arm", "hadm_id", "subject_id", "selected_note_rowid", "variant_text"},
        variants_path,
    )
    if len(hosts) != N_HOSTS or len(variants) != N_HOSTS:
        fail("Both host and variant manifests must contain exactly 200 rows.")
    if hosts["hadm_id"].duplicated().any() or variants["hadm_id"].duplicated().any():
        fail("Host and variant manifests must have unique hadm_id values.")
    if variants["variant_text"].isna().any() or variants["variant_text"].eq("").any():
        fail("Variant manifest contains missing or empty variant text.")
    expected_cells = {(cohort, arm): N_PER_CELL for cohort in ("2014 - 2016", "2017 - 2019")
                      for arm in ("top_positive", "random_structural_negative")}
    observed = variants.groupby(["cohort", "arm"]).size().to_dict()
    if observed != expected_cells:
        fail("Variant manifest does not contain 50 rows in every cohort-by-arm cell.")
    merged = variants.merge(
        hosts[["host_manifest_order", "hadm_id", "subject_id", "selected_note_rowid", "selected_note_seq"]],
        on=["host_manifest_order", "hadm_id", "subject_id", "selected_note_rowid"],
        how="inner", validate="one_to_one",
    )
    if len(merged) != N_HOSTS:
        fail("Variant manifest does not match the frozen host manifest.")
    return hosts.sort_values("host_manifest_order", kind="stable"), variants.sort_values(
        "variant_manifest_order", kind="stable"
    )


HOST_NOTE_SQL = """
SELECT n.rowid, n.hadm_id, n.note_seq, a.subject_id, n.text
FROM "note/discharge" AS n
JOIN "hosp/admissions" AS a ON a.hadm_id = n.hadm_id
WHERE n.rowid IN ({placeholders})
"""


def batches(values: list[int], size: int = 900):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_original_notes(connection: sqlite3.Connection, hosts: pd.DataFrame) -> dict[int, str]:
    rowids = hosts["selected_note_rowid"].astype(int).tolist()
    records: dict[int, tuple[int, int, int, str]] = {}
    try:
        for group in batches(sorted(rowids)):
            query = HOST_NOTE_SQL.format(placeholders=",".join("?" for _ in group))
            for rowid, hadm_id, note_seq, subject_id, text in connection.execute(query, group):
                records[int(rowid)] = (int(hadm_id), int(note_seq), int(subject_id), normalize(str(text)))
    except sqlite3.Error as exc:
        fail(f"Could not reload frozen host notes: {exc}")
    if set(records) != set(rowids):
        fail("Could not recover every frozen host note rowid.")
    notes = {}
    for host in hosts.itertuples(index=False):
        hadm_id, sequence, subject_id, text = records[int(host.selected_note_rowid)]
        if (hadm_id, sequence, subject_id) != (
            int(host.hadm_id), int(host.selected_note_seq), int(host.subject_id)
        ):
            fail(f"Frozen host identity changed for hadm_id {int(host.hadm_id)}.")
        notes[hadm_id] = text
    return notes


def load_fixed_witness_inputs(
    baseline_path: Path, pca_path: Path, candidates_path: Path, report_path: Path
) -> tuple[np.ndarray, Any, dict[str, dict[str, Any]]]:
    baseline = load_matrix(baseline_path, "baseline PCA array", columns=52)
    if not pca_path.is_file():
        fail(f"Missing PCA model: {pca_path}")
    pca = joblib.load(pca_path)
    if getattr(pca, "n_components_", None) != 52:
        fail("Frozen PCA model must have 52 components.")
    if not candidates_path.is_file() or not report_path.is_file():
        fail("Missing candidate pool or host-selection report.")
    candidates = pd.read_csv(candidates_path)
    require_columns(candidates, {"cohort", "hadm_id", "witness_score", "rank"}, candidates_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("all_cohorts_feasible"):
        fail("Host-selection report is not feasible; diagnostics must not proceed.")

    fixed: dict[str, dict[str, Any]] = {}
    sigma_rng = np.random.default_rng(SEED)
    for label, code, embedding_path, id_path in WINDOWS:
        raw = load_matrix(embedding_path, f"{label} raw embeddings", columns=EMBED_DIM)
        if len(raw) != N_CANDIDATES:
            fail(f"{label} requires {N_CANDIDATES} target embeddings.")
        ids = load_ids(id_path, N_CANDIDATES, label)
        target = np.asarray(pca.transform(raw), dtype=np.float64)
        if target.shape != (N_CANDIDATES, 52) or not np.isfinite(target).all():
            fail(f"Frozen PCA transform for {label} is invalid.")
        base_part = baseline[sigma_rng.choice(len(baseline), min(BANDWIDTH_SUBSAMPLE, len(baseline)), replace=False)]
        target_part = target[sigma_rng.choice(len(target), min(BANDWIDTH_SUBSAMPLE, len(target)), replace=False)]
        sigma = median_sigma(np.vstack([base_part, target_part]), sigma_rng)
        reported_sigma = float(report.get("cohorts", {}).get(label, {}).get("sigma", np.nan))
        if not np.isclose(sigma, reported_sigma, rtol=0, atol=1e-12):
            fail(f"Recomputed {label} sigma does not match the frozen selection report.")
        pool = candidates.loc[candidates["cohort"].astype(str) == label].copy()
        if len(pool) != N_CANDIDATES or pool["hadm_id"].duplicated().any():
            fail(f"Candidate pool for {label} is not a unique 2,500-row frozen set.")
        pool_ids = pool["hadm_id"].astype(np.int64).to_numpy()
        scores = pool["witness_score"].astype(float).to_numpy()
        recomputed = witness(target, baseline, target, sigma)
        aligned = pd.Series(recomputed, index=ids).reindex(pool_ids).to_numpy(dtype=np.float64)
        if not np.allclose(aligned, scores, rtol=0, atol=SCORE_TOLERANCE):
            difference = float(np.max(np.abs(aligned - scores)))
            fail(f"{label} frozen witness scores do not reproduce (max error {difference:.3g}).")
        fixed[label] = {
            "code": code, "target": target, "sigma": sigma, "pool_ids": pool_ids,
            "pool_scores": scores, "pool_ranks": pool["rank"].astype(int).to_numpy(),
        }
    return baseline, pca, fixed


def post_treatment_rank(hadm_id: int, score: float, fixed: dict[str, Any]) -> int:
    scores = np.array(fixed["pool_scores"], dtype=np.float64, copy=True)
    ids = np.asarray(fixed["pool_ids"], dtype=np.int64)
    matches = np.flatnonzero(ids == hadm_id)
    if len(matches) != 1:
        fail(f"hadm_id {hadm_id} is absent or duplicated in its frozen candidate pool.")
    scores[int(matches[0])] = score
    order = np.lexsort((ids, -scores))
    return int(np.flatnonzero(ids[order] == hadm_id)[0]) + 1


def configure_device(name: str, num_threads: int | None) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA was requested but is unavailable.")
        LOG.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    threads = num_threads if num_threads is not None else os.cpu_count() or 1
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, threads // 2))
    LOG.info("Using CPU with %d Torch threads.", threads)
    return torch.device("cpu")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed Experiment A originals and variants for frozen diagnostics.")
    parser.add_argument("--hosts", type=Path, default=HOSTS_PATH)
    parser.add_argument("--variants", type=Path, default=VARIANTS_PATH)
    parser.add_argument("--candidate-pool", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--selection-report", type=Path, default=SELECTION_REPORT_PATH)
    parser.add_argument("--baseline-pca", type=Path, default=BASELINE_PCA)
    parser.add_argument("--pca-model", type=Path, default=PCA_MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--database", default=MIMIC4_DB_PATH, help="Defaults to MIMIC4_DB_PATH.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    hosts, variants = load_hosts_and_variants(args.hosts, args.variants)
    if not args.database or not Path(args.database).is_file():
        fail("MIMIC4_DB_PATH is not set in .env or does not point to a file.")
    connection = sqlite3.connect(Path(args.database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        originals = load_original_notes(connection, hosts)
    finally:
        connection.close()
    baseline, pca, fixed_by_cohort = load_fixed_witness_inputs(
        args.baseline_pca, args.pca_model, args.candidate_pool, args.selection_report
    )
    device = configure_device(args.device, args.num_threads)
    token = os.getenv("HF_TOKEN")
    LOG.info("Loading frozen embedding model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModel.from_pretrained(MODEL_ID, token=token).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    diagnostics = []
    for number, row in enumerate(variants.itertuples(index=False), start=1):
        hadm_id = int(row.hadm_id)
        original_embedding, original_windows = embed_note(originals[hadm_id], tokenizer, model, device)
        variant_embedding, variant_windows = embed_note(str(row.variant_text), tokenizer, model, device)
        original_pca = np.asarray(pca.transform(original_embedding.reshape(1, -1)), dtype=np.float64)
        variant_pca = np.asarray(pca.transform(variant_embedding.reshape(1, -1)), dtype=np.float64)
        if original_pca.shape != (1, 52) or variant_pca.shape != (1, 52):
            fail(f"PCA diagnostic transform failed for hadm_id {hadm_id}.")
        fixed = fixed_by_cohort.get(str(row.cohort))
        if fixed is None:
            fail(f"Variant {row.variant_id} has an unknown cohort.")
        original_witness = float(witness(original_pca, baseline, fixed["target"], fixed["sigma"])[0])
        variant_witness = float(witness(variant_pca, baseline, fixed["target"], fixed["sigma"])[0])
        pool_index = np.flatnonzero(np.asarray(fixed["pool_ids"]) == hadm_id)
        if len(pool_index) != 1:
            fail(f"Variant hadm_id {hadm_id} is missing from its frozen cohort pool.")
        index = int(pool_index[0])
        frozen_score = float(fixed["pool_scores"][index])
        frozen_rank = int(fixed["pool_ranks"][index])
        post_rank = post_treatment_rank(hadm_id, variant_witness, fixed)
        diagnostics.append({
            "variant_manifest_order": int(row.variant_manifest_order),
            "variant_id": row.variant_id,
            "host_manifest_order": int(row.host_manifest_order),
            "cohort": row.cohort,
            "cohort_code": int(row.cohort_code),
            "arm": row.arm,
            "hadm_id": hadm_id,
            "subject_id": int(row.subject_id),
            "original_embedding_windows": original_windows,
            "variant_embedding_windows": variant_windows,
            "cohort_sigma": float(fixed["sigma"]),
            "pca_displacement_l2": float(np.linalg.norm(variant_pca[0] - original_pca[0])),
            "original_witness_reembedded": original_witness,
            "variant_witness": variant_witness,
            "witness_change": variant_witness - original_witness,
            "frozen_original_witness_score": frozen_score,
            "frozen_original_rank": frozen_rank,
            "post_treatment_rank": post_rank,
            "post_treatment_rank_change": post_rank - frozen_rank,
        })
        if number % 10 == 0 or number == N_HOSTS:
            LOG.info("Embedded and diagnosed %d/%d variants.", number, N_HOSTS)
    output = pd.DataFrame(diagnostics)
    if len(output) != N_HOSTS or output["hadm_id"].duplicated().any() or not np.isfinite(
        output[["pca_displacement_l2", "original_witness_reembedded", "variant_witness", "witness_change"]]
    ).all().all():
        fail("Diagnostic output failed completeness, uniqueness, or finiteness checks.")
    atomic_csv(output, args.output)
    LOG.info("Wrote frozen Experiment A diagnostics: %s", args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        LOG.error("Experiment A diagnostics stopped: %s", error)
        raise SystemExit(1)
