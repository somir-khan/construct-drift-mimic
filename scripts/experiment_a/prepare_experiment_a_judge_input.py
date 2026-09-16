#!/usr/bin/env python3
"""Create frozen Experiment A exemplars and the blinded Judge-input manifest.

The three exemplars are copied verbatim from the completed legacy audit
manifest.  This script never reselects exemplars and never reads a database,
makes an embedding, or calls a Judge.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd


LEGACY_SAMPLES_PATH = Path("data/judge_samples.csv")
VARIANTS_PATH = Path("data/experiment_a_variants.csv")
EXEMPLARS_PATH = Path("data/experiment_a_exemplars.csv")
JUDGE_INPUT_PATH = Path("data/experiment_a_judge_input.csv")
N_EXEMPLARS = 3
N_VARIANTS = 200
ARMS = {"top_positive", "random_structural_negative"}
PHI = re.compile(r"\[\*\*.*?\*\*\]")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("prepare_experiment_a_judge_input")


class PreparationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise PreparationError(message)


def normalize(text: str) -> str:
    return PHI.sub("unknown", text).replace("___", "unknown")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def legacy_column(frame: pd.DataFrame, choices: tuple[str, ...], label: str) -> str:
    found = [column for column in choices if column in frame.columns]
    if len(found) != 1:
        fail(f"{label} must contain exactly one of {list(choices)}.")
    return found[0]


def load_legacy_exemplars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"Missing completed legacy audit manifest: {path}. Provide its saved "
            "judge-sample CSV with the original three exemplar rows; do not reselect them."
        )
    legacy = pd.read_csv(path)
    require_columns(legacy, {"hadm_id"}, "completed legacy audit manifest")
    group_column = legacy_column(legacy, ("selection_group", "group"), "completed legacy audit manifest")
    text_column = legacy_column(legacy, ("text", "note_text", "normalized_text"), "completed legacy audit manifest")
    exemplars = legacy.loc[legacy[group_column].astype(str) == "exemplar", ["hadm_id", text_column]].copy()
    if len(exemplars) != N_EXEMPLARS or exemplars["hadm_id"].duplicated().any():
        fail("Completed legacy audit manifest must contain exactly three unique exemplar admissions.")
    if exemplars[text_column].isna().any() or exemplars[text_column].astype(str).eq("").any():
        fail("Completed legacy audit manifest has an empty exemplar text.")
    exemplars["normalized_text"] = exemplars[text_column].astype(str)
    if not (exemplars["normalized_text"].map(normalize) == exemplars["normalized_text"]).all():
        fail(
            "Legacy exemplar text is not already PHI-normalized. Refusing to alter it because "
            "Experiment A requires byte-identical normalized exemplar text from the completed audit."
        )
    exemplars = exemplars[["hadm_id", "normalized_text"]].reset_index(drop=True)
    exemplars.insert(0, "record_id", [f"exemplar_{number:03d}" for number in range(1, N_EXEMPLARS + 1)])
    return exemplars


def load_variants(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Missing frozen variant manifest: {path}")
    variants = pd.read_csv(path)
    require_columns(variants, {"variant_manifest_order", "variant_id", "arm", "variant_text"}, "variant manifest")
    if len(variants) != N_VARIANTS or variants["variant_id"].duplicated().any():
        fail("Variant manifest must contain exactly 200 unique variant IDs.")
    if variants["variant_text"].isna().any() or variants["variant_text"].astype(str).eq("").any():
        fail("Variant manifest contains an empty variant text.")
    if set(variants["arm"].astype(str)) != ARMS or variants["arm"].value_counts().to_dict() != {
        "top_positive": 100, "random_structural_negative": 100,
    }:
        fail("Variant manifest must contain exactly 100 variants in each frozen arm.")
    if set(variants["variant_manifest_order"].astype(int)) != set(range(1, N_VARIANTS + 1)):
        fail("variant_manifest_order must be exactly 1 through 200.")
    variants["variant_text"] = variants["variant_text"].astype(str)
    if not (variants["variant_text"].map(normalize) == variants["variant_text"]).all():
        fail("Variant text is not normalized with the frozen PHI convention.")
    return variants.sort_values("variant_manifest_order", kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy frozen legacy exemplars and construct blinded Experiment A Judge input."
    )
    parser.add_argument("--legacy-samples", type=Path, default=LEGACY_SAMPLES_PATH)
    parser.add_argument("--variants", type=Path, default=VARIANTS_PATH)
    parser.add_argument("--exemplars-output", type=Path, default=EXEMPLARS_PATH)
    parser.add_argument("--judge-input-output", type=Path, default=JUDGE_INPUT_PATH)
    args = parser.parse_args()

    exemplars = load_legacy_exemplars(args.legacy_samples)
    variants = load_variants(args.variants)
    exemplar_input = pd.DataFrame({
        "record_id": exemplars["record_id"],
        "group": "exemplar",
        "note_text": exemplars["normalized_text"],
    })
    variant_input = variants.rename(columns={"variant_id": "record_id", "arm": "group", "variant_text": "note_text"})[
        ["record_id", "group", "note_text"]
    ]
    judge_input = pd.concat([exemplar_input, variant_input], ignore_index=True)
    if len(judge_input) != N_EXEMPLARS + N_VARIANTS or judge_input["record_id"].duplicated().any():
        fail("Judge input did not produce 203 unique records.")
    if judge_input["group"].value_counts().to_dict() != {
        "top_positive": 100, "random_structural_negative": 100, "exemplar": 3,
    }:
        fail("Judge input arm counts are invalid.")
    if list(judge_input.columns) != ["record_id", "group", "note_text"]:
        fail("Judge input schema must contain only record_id, group, and note_text.")
    atomic_csv(exemplars, args.exemplars_output)
    atomic_csv(judge_input, args.judge_input_output)
    LOG.info("Wrote three frozen legacy exemplars: %s", args.exemplars_output)
    LOG.info("Wrote blinded 203-record Judge input: %s", args.judge_input_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        LOG.error("Experiment A Judge-input preparation stopped: %s", error)
        raise SystemExit(1)
