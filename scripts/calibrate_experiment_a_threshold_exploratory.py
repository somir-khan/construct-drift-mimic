#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np


N_PER_ARM = 100
NOMINAL_ALPHA = Fraction(5, 100)
NOMINAL_ALPHA_FLOAT = float(NOMINAL_ALPHA)
GRID_RESOLUTION_DEFAULT = 101
GRID_TOLERANCE = 5e-4
OUTPUT_PATH = Path("data/experiment_a_threshold_calibration.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger(__name__)


class CalibrationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CalibrationError(message)


def simple_walk_tail_coefficients(n: int, threshold: int) -> list[Fraction]:
    coefficients = []
    for m in range(n + 1):
        favorable = sum(comb(m, plus) for plus in range(m + 1) if 2 * plus - m >= threshold)
        coefficients.append(Fraction(favorable, 2**m))
    return coefficients


def left_half_bernstein_coefficients(coefficients: list[Fraction]) -> list[Fraction]:
    working = coefficients[:]
    left = [working[0]]
    while len(working) > 1:
        working = [
            (working[index] + working[index + 1]) / 2
            for index in range(len(working) - 1)
        ]
        left.append(working[0])
    return left


def prove_equal_rate_maximum(n: int, threshold: int) -> dict[str, object]:
    tail = simple_walk_tail_coefficients(n, threshold)
    derivative = [n * (tail[index + 1] - tail[index]) for index in range(n)]
    left_half = left_half_bernstein_coefficients(derivative)
    if any(value < 0 for value in left_half):
        fail("Could not verify monotonicity of the equal-rate null tail on [0, 1/2].")
    return {
        "method": "exact_Bernstein_derivative_with_de_Casteljau_subdivision",
        "paired_difference_parameter": "t = 2*p*(1-p)",
        "verified_t_interval": [0.0, 0.5],
        "subdivided_derivative_coefficients": len(left_half),
        "negative_subdivided_derivative_coefficients": 0,
        "minimum_subdivided_derivative_coefficient": str(min(left_half)),
        "conclusion": "equal-rate alarm probability is nondecreasing in t",
    }


def exact_probability_at_half(n: int, threshold: int) -> Fraction:
    numerator = 0
    for negative_count in range(n - threshold + 1):
        positive_tail = sum(
            comb(n, positive_count)
            for positive_count in range(negative_count + threshold, n + 1)
        )
        numerator += comb(n, negative_count) * positive_tail
    return Fraction(numerator, 2 ** (2 * n))


def binomial_pmf_array(n: int, p: float) -> np.ndarray:
    if p <= 0.0:
        pmf = np.zeros(n + 1)
        pmf[0] = 1.0
        return pmf
    if p >= 1.0:
        pmf = np.zeros(n + 1)
        pmf[n] = 1.0
        return pmf
    k = np.arange(n + 1)
    log_binom = np.array(
        [math.lgamma(n + 1) - math.lgamma(kk + 1) - math.lgamma(n - kk + 1) for kk in k]
    )
    log_pmf = log_binom + k * math.log(p) + (n - k) * math.log(1.0 - p)
    return np.exp(log_pmf)


def survival_function(pmf: np.ndarray) -> np.ndarray:
    tail = np.cumsum(pmf[::-1])[::-1]
    return np.concatenate([tail, [0.0]])


def directional_null_tail_probability(
    n: int, pmf_pos: np.ndarray, pmf_neg: np.ndarray, threshold: int
) -> float:
    sf_pos = survival_function(pmf_pos)
    y_max = n - threshold
    if y_max < 0:
        return 0.0
    y_vals = np.arange(0, y_max + 1)
    return float(np.sum(pmf_neg[y_vals] * sf_pos[y_vals + threshold]))


def grid_search_directional_null(
    n: int, threshold: int, resolution: int = GRID_RESOLUTION_DEFAULT
) -> dict[str, object]:
    p_vals = np.linspace(0.0, 1.0, resolution)
    pmf_cache = {float(p): binomial_pmf_array(n, float(p)) for p in p_vals}

    best_prob = -1.0
    best_pair: tuple[float | None, float | None] = (None, None)
    exceed_alpha_count = 0
    grid_points_evaluated = 0
    diagonal_points: list[tuple[float, float]] = []

    for p_pos in p_vals:
        pmf_pos = pmf_cache[float(p_pos)]
        for p_neg in p_vals:
            if p_neg < p_pos:
                continue
            grid_points_evaluated += 1
            pmf_neg = pmf_cache[float(p_neg)]
            probability = directional_null_tail_probability(n, pmf_pos, pmf_neg, threshold)
            if probability > best_prob:
                best_prob = probability
                best_pair = (float(p_pos), float(p_neg))
            if probability >= NOMINAL_ALPHA_FLOAT:
                exceed_alpha_count += 1
            if abs(float(p_pos) - float(p_neg)) < 1e-12:
                diagonal_points.append((float(p_pos), probability))

    diagonal_argmax_p, diagonal_argmax_prob = max(diagonal_points, key=lambda item: item[1])
    return {
        "method": "floating_point_full_grid_scan_over_directional_null_region",
        "region": "p_positive <= p_negative, both in [0, 1]",
        "grid_resolution_per_axis": resolution,
        "grid_points_evaluated": grid_points_evaluated,
        "grid_max_probability": best_prob,
        "grid_max_probability_location_p_positive": best_pair[0],
        "grid_max_probability_location_p_negative": best_pair[1],
        "grid_points_at_or_above_nominal_alpha": exceed_alpha_count,
        "diagonal_argmax_p": diagonal_argmax_p,
        "diagonal_argmax_probability": diagonal_argmax_prob,
        "diagonal_argmax_within_one_grid_step_of_one_half": abs(diagonal_argmax_p - 0.5)
        <= (1.0 / (resolution - 1)) + 1e-12,
    }


def select_threshold() -> tuple[int, Fraction, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for threshold in range(1, N_PER_ARM + 1):
        probability = exact_probability_at_half(N_PER_ARM, threshold)
        controls_alpha = probability < NOMINAL_ALPHA
        candidates.append(
            {
                "alarm_count_difference": threshold,
                "alarm_rate_difference": float(Fraction(threshold, N_PER_ARM)),
                "worst_case_false_alarm_probability": float(probability),
                "worst_case_false_alarm_probability_10dp": f"{float(probability):.10f}",
                "controls_nominal_alpha": controls_alpha,
            }
        )
        if controls_alpha:
            return threshold, probability, candidates
    fail("No positive count-difference threshold controls the nominal alpha level.")


def calibrate(
    run_grid_search: bool = True, grid_resolution: int = GRID_RESOLUTION_DEFAULT
) -> dict[str, object]:
    threshold, probability, candidates = select_threshold()
    last_rejected = candidates[-2] if len(candidates) > 1 else None
    proof = prove_equal_rate_maximum(N_PER_ARM, threshold)
    probability_float = float(probability)
    probability_10dp = f"{probability_float:.10f}"
    if probability >= NOMINAL_ALPHA:
        fail("The selected rule does not control the nominal 5% level.")

    grid_confirmation = None
    grid_confirms_diagonal_is_global_max = None
    if run_grid_search:
        grid_confirmation = grid_search_directional_null(
            N_PER_ARM, threshold, resolution=grid_resolution
        )
        grid_max = grid_confirmation["grid_max_probability"]
        if grid_max - probability_float > GRID_TOLERANCE:
            fail(
                "Numerical grid scan found a directional-null point exceeding the "
                f"selected threshold's exact worst case beyond tolerance: "
                f"grid_max={grid_max:.6f}, exact={probability_float:.6f}."
            )
        if grid_confirmation["grid_points_at_or_above_nominal_alpha"] > 0:
            fail(
                "Numerical grid scan found admissible directional-null points at or "
                "above the nominal 5% level."
            )
        if not grid_confirmation["diagonal_argmax_within_one_grid_step_of_one_half"]:
            fail(
                "Numerical grid scan's own equal-rate maximum did not land at p=0.5 "
                "within one grid step; inconsistent with the exact algebraic proof."
            )
        grid_confirms_diagonal_is_global_max = (
            grid_max - probability_float <= GRID_TOLERANCE
            and grid_confirmation["grid_points_at_or_above_nominal_alpha"] == 0
        )

    return {
        "purpose": "pre_outcome_exact_calibration_and_archive",
        "design_frozen": True,
        "n_per_arm": N_PER_ARM,
        "selection_rule": (
            "Choose the smallest positive integer count difference whose exact "
            "worst-case directional-null false-alarm probability is below 0.05."
        ),
        "candidate_thresholds_evaluated": candidates,
        "selection_result": {
            "last_rejected_count_difference": (
                None if last_rejected is None else last_rejected["alarm_count_difference"]
            ),
            "last_rejected_false_alarm_probability_10dp": (
                None
                if last_rejected is None
                else last_rejected["worst_case_false_alarm_probability_10dp"]
            ),
            "first_accepted_count_difference": threshold,
            "first_accepted_false_alarm_probability_10dp": probability_10dp,
        },
        "selected_alarm_count_difference": threshold,
        "selected_alarm_rate_difference": float(Fraction(threshold, N_PER_ARM)),
        "directional_null": "p_positive <= p_negative",
        "alarm_event": f"X - Y >= {threshold}",
        "independence_assumption": "X and Y are independent binomial counts",
        "directional_null_reduction": (
            "The event is increasing in p_positive and decreasing in p_negative; "
            "under p_positive <= p_negative the supremum lies on equality."
        ),
        "equal_rate_supremum_proof": proof,
        "grid_search_confirmation": grid_confirmation,
        "grid_search_confirms_diagonal_is_global_max_in_null_region": grid_confirms_diagonal_is_global_max,
        "supremum_p_positive": 0.5,
        "supremum_p_negative": 0.5,
        "exact_probability_numerator": probability.numerator,
        "exact_probability_denominator": probability.denominator,
        "worst_case_false_alarm_probability": probability_float,
        "worst_case_false_alarm_probability_10dp": probability_10dp,
        "nominal_alpha": float(NOMINAL_ALPHA),
        "controls_nominal_alpha": True,
        "judge_outputs_used": False,
        "negative_control_outcomes_used": False,
        "threshold_selected_using_outcomes": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the Experiment A directional alarm threshold."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=GRID_RESOLUTION_DEFAULT,
        help=(
            "Points per axis for the numerical directional-null grid scan "
            f"(default {GRID_RESOLUTION_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip the numerical grid scan; keep only the open search and exact proof.",
    )
    args = parser.parse_args()

    result = calibrate(run_grid_search=not args.no_grid_search, grid_resolution=args.grid_resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    LOG.info(
        "Selected %s-point rule: worst-case false-alarm probability=%s < 0.05",
        result["selected_alarm_count_difference"],
        result["worst_case_false_alarm_probability_10dp"],
    )
    selection = result["selection_result"]
    LOG.info(
        "Threshold search: %s points fails (%s); %s points is the first passing value (%s).",
        selection["last_rejected_count_difference"],
        selection["last_rejected_false_alarm_probability_10dp"],
        selection["first_accepted_count_difference"],
        selection["first_accepted_false_alarm_probability_10dp"],
    )
    if result["grid_search_confirmation"] is not None:
        grid = result["grid_search_confirmation"]
        LOG.info(
            "Grid scan (%dx%d, %d directional-null points): max=%.6f at (p+=%.3f, p-=%.3f); "
            "points >= alpha: %d",
            args.grid_resolution,
            args.grid_resolution,
            grid["grid_points_evaluated"],
            grid["grid_max_probability"],
            grid["grid_max_probability_location_p_positive"],
            grid["grid_max_probability_location_p_negative"],
            grid["grid_points_at_or_above_nominal_alpha"],
        )
    LOG.info("Wrote calibration artifact: %s", args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as error:
        LOG.error("Threshold-calibration verification stopped: %s", error)
        raise SystemExit(1)
