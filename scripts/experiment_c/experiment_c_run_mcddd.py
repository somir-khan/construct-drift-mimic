#!/usr/bin/env python3
"""Run released MCD-DD on one shuffled B||T pseudo-stream over 20 seeds.

The primary endpoint is an alarm on the exact first target sub-window
[5000, 5100). Pre-boundary alarms and later alarms are reported separately.
"""

from __future__ import annotations

import os

# Required for deterministic CUDA matrix multiplication when strict mode is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


COHORT_SIZE = 5_000
EMBEDDING_DIM = 768
FINAL_SEEDS = list(range(1111, 1131))

# Paper-matching settings fixed before the runs.
WINDOW_SIZE = 1_000
SUB_WINDOW_NUM = 10
M = 30
K = 10
HIDDEN_SIZE = 200
OUTPUT_SIZE = 150
LEARNING_RATE = 0.005
EPOCHS = 1
EPS_SMALL = 1.0
EPS_BIG = 10.0
TEMPERATURE = 0.1
LAMBDA = 1.0
PERCENTILE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--mcddd-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=FINAL_SEEDS,
        help="Use one seed for the pilot; omit for the fixed 20-seed run.",
    )
    parser.add_argument(
        "--determinism",
        choices=["strict", "warn", "off"],
        default="strict",
        help="Use warn for the one-seed pilot; use strict for the final run if the pilot is clean.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, source: Path) -> Any:
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_embeddings(path: Path, label: str) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != (COHORT_SIZE, EMBEDDING_DIM):
        raise ValueError(f"{label}: expected {(COHORT_SIZE, EMBEDDING_DIM)}, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label}: expected floating point, got {array.dtype}.")
    if not np.isfinite(array).all() or np.any(np.all(array == 0, axis=1)):
        raise ValueError(f"{label}: non-finite or all-zero embedding row found.")
    return np.asarray(array)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def set_seed(seed: int, determinism: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if determinism == "off":
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True, warn_only=determinism == "warn")


def verify_api(encoder_class: type, detector_class: type) -> None:
    expected_encoder = ["input_size", "hidden_size", "output_size"]
    expected_detector = [
        "model", "optimizer", "epochs", "sub_window_num", "n", "k",
        "eps_small", "eps_big", "temperature", "lamb", "percentile", "device",
    ]
    actual_encoder = list(inspect.signature(encoder_class).parameters)
    actual_detector = list(inspect.signature(detector_class).parameters)
    missing = [name for name in ("train", "test") if not callable(getattr(detector_class, name, None))]
    if actual_encoder != expected_encoder or actual_detector != expected_detector or missing:
        raise RuntimeError(
            "MCD-DD API mismatch before computation: "
            f"Encoder{tuple(actual_encoder)}, MCD{tuple(actual_detector)}, missing={missing}."
        )


def finite_scalar(value: Any, label: str) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{label} is not finite: {result}")
    return result


def order_sha256(baseline_order: np.ndarray, target_order: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(baseline_order, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(target_order, dtype=np.int64).tobytes())
    return digest.hexdigest()


def summarize_alarms(
    steps: list[dict[str, Any]], boundary_index: int, slide: int
) -> dict[str, Any]:
    alarm_starts = [int(row["new_subwindow_start"]) for row in steps if row["alarm"]]
    first_post = min((x for x in alarm_starts if x >= boundary_index), default=None)
    boundary_row = next(row for row in steps if row["new_subwindow_start"] == boundary_index)
    return {
        "boundary_hit": bool(boundary_row["alarm"]),
        "primary_endpoint": (
            f"alarm on exact first target sub-window "
            f"[{boundary_index},{boundary_index + slide})"
        ),
        "boundary_score": boundary_row["adjacent_score"],
        "boundary_threshold": boundary_row["threshold"],
        "pre_boundary_alarm_count": sum(x < boundary_index for x in alarm_starts),
        "later_post_boundary_alarm_count": sum(x > boundary_index for x in alarm_starts),
        "first_post_boundary_alarm_start": first_post,
        "first_post_boundary_delay_subwindows": (
            (first_post - boundary_index) // slide if first_post is not None else None
        ),
        "total_alarm_count": len(alarm_starts),
        "evaluated_subwindows": len(steps),
    }


def run_one(
    seed: int,
    baseline: np.ndarray,
    target: np.ndarray,
    encoder_class: type,
    detector_class: type,
    device: torch.device,
    determinism: str,
) -> dict[str, Any]:
    started = time.time()
    set_seed(seed, determinism)
    baseline_order = np.random.default_rng(seed).permutation(COHORT_SIZE)
    target_order = np.random.default_rng(seed + 200_000).permutation(COHORT_SIZE)
    stream = np.concatenate([baseline[baseline_order], target[target_order]])
    stream_tensor = torch.as_tensor(stream, dtype=torch.float32, device=device)

    slide = WINDOW_SIZE // SUB_WINDOW_NUM
    boundary_index = COHORT_SIZE
    if boundary_index % slide:
        raise AssertionError("The constructed boundary must align with a tested sub-window.")

    model = encoder_class(EMBEDDING_DIM, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    detector = detector_class(
        model,
        optimizer,
        EPOCHS,
        SUB_WINDOW_NUM,
        M,
        K,
        EPS_SMALL,
        EPS_BIG,
        TEMPERATURE,
        LAMBDA,
        PERCENTILE,
        device,
    )

    number_windows = ((len(stream) - WINDOW_SIZE) // slide) + 1
    threshold_tensor: torch.Tensor | None = None
    steps: list[dict[str, Any]] = []
    for step in range(number_windows):
        start = step * slide
        window = stream_tensor[start : start + WINDOW_SIZE]
        if step > 0:
            if threshold_tensor is None:
                raise AssertionError("MCD-DD returned no threshold after the first window.")
            distances = [finite_scalar(x, "MCD distance") for x in detector.test(window)]
            if len(distances) != SUB_WINDOW_NUM - 1:
                raise RuntimeError(
                    "MCD-DD test() returned "
                    f"{len(distances)} distances; expected {SUB_WINDOW_NUM - 1}."
                )
            threshold = finite_scalar(threshold_tensor, "MCD threshold")
            new_start = start + WINDOW_SIZE - slide
            # Released mcd.py appends distances from each earlier sub-window to
            # the newest one in order, so the last item is the adjacent pair.
            adjacent_score = distances[-1]
            steps.append(
                {
                    "step": step,
                    "new_subwindow_start": new_start,
                    "new_subwindow_end": new_start + slide,
                    "adjacent_score": adjacent_score,
                    "threshold": threshold,
                    "alarm": bool(adjacent_score > threshold),
                    "distances_to_latest": distances,
                }
            )
        threshold_tensor = detector.train(window)
        finite_scalar(threshold_tensor, "updated MCD threshold")

    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics = summarize_alarms(steps, boundary_index, slide)
    return {
        "status": "completed",
        "seed": seed,
        "method": "released MCD-DD",
        "sampling": "released torch.randint sampling with replacement",
        "device": str(device),
        "determinism": determinism,
        "stream": {
            "length": len(stream),
            "boundary_index": boundary_index,
            "semantics": "shuffled MIMIC-III baseline followed by independently shuffled MIMIC-IV target; not chronology",
            "baseline_shuffle_seed": seed,
            "target_shuffle_seed": seed + 200_000,
            "order_sha256": order_sha256(baseline_order, target_order),
        },
        "parameters": {
            "window_size": WINDOW_SIZE,
            "window_fraction": WINDOW_SIZE / len(stream),
            "sub_window_num": SUB_WINDOW_NUM,
            "slide": slide,
            "m": M,
            "k": K,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "eps_small": EPS_SMALL,
            "eps_big": EPS_BIG,
            "temperature": TEMPERATURE,
            "lambda": LAMBDA,
            "percentile": PERCENTILE,
        },
        "metrics": metrics,
        "steps": steps,
        "elapsed_seconds": time.time() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def read_latest_records(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[int(record["seed"])] = record
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise RuntimeError(f"Invalid JSONL record at line {line_number} of {path}.") from exc
    return latest


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.torch_threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Torch threads must be positive and seeds must be unique.")
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)

    prepared = Path(args.prepared_dir).expanduser().resolve()
    baseline = load_embeddings(prepared / "baseline_raw.npy", "baseline")
    target = load_embeddings(prepared / "target_raw.npy", "target")
    if baseline.dtype != target.dtype:
        raise TypeError(f"B/T dtypes differ: {baseline.dtype} vs {target.dtype}.")

    repo = Path(args.mcddd_repo).expanduser().resolve()
    encoder_source, detector_source = repo / "encoder.py", repo / "mcd.py"
    sys.path.insert(0, str(repo))
    encoder_module = load_module("experiment_c_mcddd_encoder", encoder_source)
    detector_module = load_module("experiment_c_mcddd_detector", detector_source)
    verify_api(encoder_module.Encoder, detector_module.MCD)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "mcddd_runs.jsonl"
    latest = read_latest_records(records_path)
    source_text = detector_source.read_text(encoding="utf-8", errors="replace")
    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seeds": args.seeds,
        "device": str(device),
        "determinism": args.determinism,
        "fixed_parameters": {
            "window_size": WINDOW_SIZE,
            "sub_window_num": SUB_WINDOW_NUM,
            "m": M,
            "k": K,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "eps_small": EPS_SMALL,
            "eps_big": EPS_BIG,
            "temperature": TEMPERATURE,
            "lambda": LAMBDA,
            "percentile": PERCENTILE,
        },
        "endpoint": "boundary_hit only when alarm start equals 5000; no tolerance horizon",
        "source": {
            "encoder_py_sha256": sha256_file(encoder_source),
            "mcd_py_sha256": sha256_file(detector_source),
            "torch_randint_present_in_mcd_py": "torch.randint" in source_text,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    info_path = output / "mcddd_run_info.json"
    if not info_path.exists():
        with info_path.open("w", encoding="utf-8") as handle:
            json.dump(run_info, handle, indent=2)
            handle.write("\n")

    failures = 0
    for seed in args.seeds:
        if latest.get(seed, {}).get("status") == "completed":
            print(f"Seed {seed}: already complete; skipping.", flush=True)
            continue
        print(f"Seed {seed}: starting on {device}.", flush=True)
        try:
            record = run_one(
                seed,
                baseline,
                target,
                encoder_module.Encoder,
                detector_module.MCD,
                device,
                args.determinism,
            )
        except Exception as exc:  # preserve the failure before ending the job
            failures += 1
            record = {
                "status": "failed",
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            }
        append_record(records_path, record)
        print(f"Seed {seed}: {record['status']}.", flush=True)

    if failures:
        raise RuntimeError(f"{failures} seed(s) failed; inspect {records_path}.")
    print(f"MCD-DD records: {records_path}")


if __name__ == "__main__":
    main()
