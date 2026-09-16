#!/usr/bin/env python3
"""Run, validate, and report the reduced AE.5a sensitivity bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _ae5a_common import (
    BANDWIDTH_MULTIPLIERS,
    EXPECTED_DIM,
    EXPECTED_N,
    EXPECTED_MMD2,
    MODEL_ID,
    N_PERMUTATIONS,
    PCA_LEVELS,
    PRIMARY_SEED,
    SAMPLE_SEEDS,
    SAMPLE_SIZES,
    WINDOW_A,
    WINDOW_B,
    ProtocolError,
    add_analysis_input_arguments,
    analysis_input_paths,
    assert_primary_reproduction,
    atomic_save_npy,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    dynamic_pca,
    environment_versions,
    fixed_nested_permutations,
    load_analysis_arrays,
    median_sigma,
    mmd2_unbiased,
    p_display,
    percent_change,
    permutation_test_precomputed,
    primary_geometry,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all fixed AE.5a sensitivity cells and write the reports."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ae5a_sensitivity")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace prior sensitivity result and report files.",
    )
    add_analysis_input_arguments(parser)
    return parser.parse_args()


def quantiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    try:
        q1, median, q3 = np.quantile(array, [0.25, 0.50, 0.75], method="linear")
    except TypeError:  # NumPy before 1.22.
        q1, median, q3 = np.quantile(
            array, [0.25, 0.50, 0.75], interpolation="linear"
        )
    return float(q1), float(median), float(q3)


def load_masked_corpus(
    output_dir: Path, corpus: str, expected_ids: np.ndarray
) -> tuple[np.ndarray, dict]:
    stem = "mimic3_5000" if corpus == "baseline" else "mimic4_5000"
    directory = output_dir / "embeddings"
    embedding_path = directory / f"embeddings_{stem}_masked_mean.npy"
    ids_path = directory / f"ids_{stem}_masked_mean.npy"
    metadata_path = directory / f"metadata_{stem}_masked_mean.json"
    for label, path in (
        ("masked embeddings", embedding_path),
        ("masked IDs", ids_path),
        ("masked metadata", metadata_path),
    ):
        if not path.is_file():
            raise ProtocolError(f"{corpus} {label} missing: {path}")

    metadata = read_json(metadata_path)
    if metadata.get("schema") != "ae5a-masked-mean-metadata-minimal-v3":
        raise ProtocolError(f"Unexpected metadata schema: {metadata_path}")
    if metadata.get("corpus") != corpus:
        raise ProtocolError(f"Metadata corpus mismatch: {metadata_path}")
    if metadata.get("model_id") != MODEL_ID:
        raise ProtocolError(f"Unexpected model in {metadata_path}")
    pooling = metadata.get("pooling", {})
    if pooling.get("l2_normalization") is not False:
        raise ProtocolError(f"No-L2 setting is not recorded: {metadata_path}")
    if pooling.get("dtype") != "float32":
        raise ProtocolError(f"Document float32 cast is not recorded: {metadata_path}")
    if pooling.get("across_windows") != "unweighted NumPy arithmetic mean":
        raise ProtocolError(f"Across-window rule changed: {metadata_path}")

    embeddings = np.load(embedding_path, allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=False)
    if embeddings.shape != (EXPECTED_N, EXPECTED_DIM) or embeddings.dtype != np.float32:
        raise ProtocolError(
            f"{corpus} masked embeddings have unexpected shape/dtype: "
            f"{embeddings.shape}, {embeddings.dtype}"
        )
    if not np.isfinite(embeddings).all():
        raise ProtocolError(f"{corpus} masked embeddings contain NaN or infinity")
    if ids.shape != (EXPECTED_N,) or ids.dtype != np.int64:
        raise ProtocolError(
            f"{corpus} masked IDs have unexpected shape/dtype: {ids.shape}, {ids.dtype}"
        )
    if not np.array_equal(ids.astype(np.int64), expected_ids.astype(np.int64)):
        raise ProtocolError(f"{corpus} masked IDs do not match the frozen order")
    return embeddings, metadata


def validate_embedding_pair(baseline_meta: dict, target_meta: dict) -> None:
    for field in ("model_id", "pooling", "windowing"):
        if baseline_meta.get(field) != target_meta.get(field):
            raise ProtocolError(f"Baseline and target metadata differ on {field}")


def validate_results(
    factor_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    exceedances: int,
    masked_p_text: str,
) -> None:
    expected_counts = {
        "pca_retention": 3,
        "rbf_bandwidth": 3,
        "window_stratified": 2,
        "embedding_construction": 1,
    }
    observed_counts = {
        factor: sum(row["factor"] == factor for row in factor_rows)
        for factor in expected_counts
    }
    if observed_counts != expected_counts:
        raise ProtocolError(
            f"Factor-cell counts changed: expected {expected_counts}, found {observed_counts}"
        )
    for row in factor_rows:
        factor = str(row["factor"])
        if factor in {"window_stratified", "embedding_construction"}:
            if row["percent_change_from_primary"] not in {"", None}:
                raise ProtocolError(f"Forbidden percentage comparison in {factor}")
        if factor != "embedding_construction" and row["permutation_p"]:
            raise ProtocolError(f"Unexpected p-value in {factor}")
    if masked_p_text != p_display(exceedances, N_PERMUTATIONS):
        raise ProtocolError("Masked-mean p-value text does not match exceedance count")
    zero_text = f"p < {1.0 / N_PERMUTATIONS:.3f}"
    if exceedances == 0 and masked_p_text != zero_text:
        raise ProtocolError(f"Zero exceedances must be rendered as {zero_text}")

    expected_pairs = {(size, seed) for size in SAMPLE_SIZES for seed in SAMPLE_SEEDS}
    observed_pairs = {
        (int(row["sample_size_per_corpus"]), int(row["seed"]))
        for row in sample_rows
    }
    if observed_pairs != expected_pairs or len(sample_rows) != len(expected_pairs):
        raise ProtocolError("Sample-size grid changed")
    for row in sample_rows:
        if row["nested_within_seed"] is not True:
            raise ProtocolError("A sample-size row is not marked nested")
    expected_summary_sizes = [*SAMPLE_SIZES, EXPECTED_N]
    if [
        int(row["sample_size_per_corpus"]) for row in summary_rows
    ] != expected_summary_sizes:
        raise ProtocolError("Sample-size summary rows changed")


def pct(value: float | str) -> str:
    if value == "" or value is None:
        return "not reported"
    return f"{float(value):+.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_reports(result: dict) -> tuple[str, str]:
    by_factor: dict[str, list[dict]] = {}
    for cell in result["factor_cells"]:
        by_factor.setdefault(cell["factor"], []).append(cell)

    primary = result["primary_reproduction"]
    pca_rows = [
        [
            cell["setting"],
            str(cell["pca_components"]),
            f"{cell['sigma']:.6f}",
            f"{cell['mmd2']:.6f}",
            pct(cell["percent_change_from_primary"]),
        ]
        for cell in by_factor["pca_retention"]
    ]
    bandwidth_rows = [
        [
            cell["setting"],
            str(cell["pca_components"]),
            f"{cell['sigma']:.6f}",
            f"{cell['mmd2']:.6f}",
            pct(cell["percent_change_from_primary"]),
        ]
        for cell in by_factor["rbf_bandwidth"]
    ]
    sample_rows = [
        [
            f"{row['sample_size_per_corpus']:,}",
            str(row["repetitions"]),
            (
                f"{row['pca_components_median']:g} "
                f"[{row['pca_components_min']}, {row['pca_components_max']}]"
            ),
            f"{row['sigma_median']:.6f} [{row['sigma_q1']:.6f}, {row['sigma_q3']:.6f}]",
            f"{row['mmd2_median']:.6f} [{row['mmd2_q1']:.6f}, {row['mmd2_q3']:.6f}]",
            pct(row["percent_change_from_primary_median"]),
        ]
        for row in result["sample_size_summary"]
    ]
    window_rows = [
        [
            cell["setting"],
            f"{cell['n_baseline']:,}",
            f"{cell['n_target']:,}",
            f"{cell['sigma']:.6f}",
            f"{cell['mmd2']:.6f}",
        ]
        for cell in by_factor["window_stratified"]
    ]
    masked = by_factor["embedding_construction"][0]

    markdown = f"""# AE.5a Minimal Sensitivity Results

## Primary reproduction

- Primary reproduction: PASS
- PCA components: {primary['pca_components']}
- Recomputed primary sigma: {primary['sigma']:.6f}
- Primary MMD squared: {primary['mmd2']:.6f}

## PCA retention

{markdown_table(['Retention', 'Components', 'Sigma', 'MMD squared', 'Change vs primary'], pca_rows)}

## RBF bandwidth

{markdown_table(['Bandwidth', 'Components', 'Sigma', 'MMD squared', 'Change vs primary'], bandwidth_rows)}

## Sample size

The 500, 1,000, and 2,000 rows summarize 20 nested draws from the independent primary baseline and target arrays. Their IQRs are stability summaries, not confidence intervals. The 5,000 row is the single frozen primary result.

{markdown_table(['n per corpus', 'Runs', 'Components median [range]', 'Sigma median [IQR]', 'MMD squared median [IQR]', 'Change vs primary'], sample_rows)}

## Window-stratified comparison

Each estimate uses its separate 2,500-note window pool, projected through the primary baseline-fitted PCA and evaluated with the fixed primary sigma. These window pools are not concatenated to create the primary target.

{markdown_table(['Target window', 'Baseline n', 'Target n', 'Fixed sigma', 'MMD squared'], window_rows)}

## Embedding-construction alternative

{markdown_table(['Pooling', 'Components', 'Recomputed sigma', 'MMD squared', 'Permutation result'], [[masked['setting'], str(masked['pca_components']), f"{masked['sigma']:.6f}", f"{masked['mmd2']:.6f}", masked['permutation_p']]])}

The embedding row varies within-window pooling only. Bio-ClinicalBERT, the final hidden layer, note set, tokenization, windowing, unweighted across-window aggregation, no L2 normalization, PCA rule, and bandwidth rule remain fixed. No percentage change is reported between CLS and masked-mean MMD squared because the representations induce different geometries.

## Interpretation guard

This report contains the prespecified numbers only. Add the interpretation after inspecting all cells. Do not assign small, medium, or large labels to MMD squared, and do not describe the window rows as a temporal trajectory.
"""

    latex_rows: list[str] = []
    for cell in by_factor["pca_retention"]:
        comparison = pct(cell["percent_change_from_primary"]).replace("%", "\\%")
        latex_rows.append(
            f"PCA retention & {cell['setting']} & {cell['pca_components']} & "
            f"{cell['sigma']:.6f} & {cell['mmd2']:.6f} & {comparison} \\\\"
        )
    for cell in by_factor["rbf_bandwidth"]:
        comparison = pct(cell["percent_change_from_primary"]).replace("%", "\\%")
        setting = cell["setting"].replace("*sigma0", " $\\times\\sigma_0$")
        latex_rows.append(
            f"RBF bandwidth & {setting} & {cell['pca_components']} & "
            f"{cell['sigma']:.6f} & {cell['mmd2']:.6f} & {comparison} \\\\"
        )
    masked_p_latex = "$" + masked["permutation_p"].replace(" ", "") + "$"
    latex_rows.append(
        f"Embedding pooling & masked mean & {masked['pca_components']} & "
        f"{masked['sigma']:.6f} & {masked['mmd2']:.6f} & {masked_p_latex} \\\\"
    )
    latex = (
        "% Auto-generated numeric table. Interpretation is intentionally omitted.\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "Factor & Setting & Components & $\\sigma$ & MMD$^2$ & Reported comparison \\\\\n"
        "\\midrule\n"
        + "\n".join(latex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return markdown, latex


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    report_dir = output_dir / "report"
    paths = {
        "factor_cells": results_dir / "ae5a_factor_cells.csv",
        "sample_runs": results_dir / "ae5a_sample_size_runs.csv",
        "sample_summary": results_dir / "ae5a_sample_size_summary.csv",
        "null": results_dir / "masked_mean_permutation_null.npy",
        "json": results_dir / "ae5a_results.json",
        "markdown": report_dir / "AE5A_RESULTS.md",
        "latex": report_dir / "ae5a_results_table.tex",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise ProtocolError(
            "Refusing to overwrite prior results: " + ", ".join(map(str, existing))
        )

    (
        baseline,
        baseline_ids,
        target,
        target_ids,
        window_a,
        _window_a_ids,
        window_b,
        _window_b_ids,
    ) = load_analysis_arrays(args)
    base_pca, target_pca, primary_pca, retained, sigma0, primary_mmd2 = (
        primary_geometry(baseline, target)
    )
    assert_primary_reproduction(int(primary_pca.n_components_), primary_mmd2)
    print(
        "PASS primary reproduction: "
        f"components={primary_pca.n_components_}, sigma={sigma0:.6f}, "
        f"MMD^2={primary_mmd2:.6f}"
    )

    masked_baseline, masked_baseline_meta = load_masked_corpus(
        output_dir, "baseline", baseline_ids
    )
    masked_target, masked_target_meta = load_masked_corpus(
        output_dir, "target", target_ids
    )
    validate_embedding_pair(masked_baseline_meta, masked_target_meta)

    factor_rows: list[dict[str, object]] = []
    for level in PCA_LEVELS:
        if level == 0.90:
            left, right, pca, level_retained, sigma, statistic = (
                base_pca,
                target_pca,
                primary_pca,
                retained,
                sigma0,
                primary_mmd2,
            )
        else:
            left, right, pca, level_retained = dynamic_pca(baseline, target, level)
            sigma = median_sigma(left, right, PRIMARY_SEED)
            statistic = mmd2_unbiased(left, right, sigma)
        factor_rows.append(
            {
                "factor": "pca_retention",
                "setting": f"{int(level * 100)}%",
                "n_baseline": len(left),
                "n_target": len(right),
                "pca_components": int(pca.n_components_),
                "retained_variance": float(level_retained),
                "sigma": sigma,
                "mmd2": statistic,
                "percent_change_from_primary": percent_change(
                    statistic, primary_mmd2
                ),
                "permutation_p": "",
                "note": "baseline-only PCA; median bandwidth recomputed",
            }
        )

    for multiplier in BANDWIDTH_MULTIPLIERS:
        sigma = sigma0 * multiplier
        statistic = mmd2_unbiased(base_pca, target_pca, sigma)
        factor_rows.append(
            {
                "factor": "rbf_bandwidth",
                "setting": f"{multiplier:g}*sigma0",
                "n_baseline": len(base_pca),
                "n_target": len(target_pca),
                "pca_components": int(primary_pca.n_components_),
                "retained_variance": retained,
                "sigma": sigma,
                "mmd2": statistic,
                "percent_change_from_primary": percent_change(
                    statistic, primary_mmd2
                ),
                "permutation_p": "",
                "note": "primary PCA and sigma0 fixed",
            }
        )

    window_a_pca = primary_pca.transform(window_a)
    window_b_pca = primary_pca.transform(window_b)
    for label, window_values in (
        (WINDOW_A, window_a_pca),
        (WINDOW_B, window_b_pca),
    ):
        statistic = mmd2_unbiased(base_pca, window_values, sigma0)
        factor_rows.append(
            {
                "factor": "window_stratified",
                "setting": label,
                "n_baseline": len(base_pca),
                "n_target": len(window_values),
                "pca_components": int(primary_pca.n_components_),
                "retained_variance": retained,
                "sigma": sigma0,
                "mmd2": statistic,
                "percent_change_from_primary": "",
                "permutation_p": "",
                "note": "separate window pool; primary PCA and sigma fixed",
            }
        )

    sample_rows: list[dict[str, object]] = []
    for seed in SAMPLE_SEEDS:
        baseline_order, target_order = fixed_nested_permutations(
            len(baseline), len(target), seed
        )
        for sample_size in SAMPLE_SIZES:
            baseline_indices = baseline_order[:sample_size]
            target_indices = target_order[:sample_size]
            left, right, pca, sample_retained = dynamic_pca(
                baseline[baseline_indices], target[target_indices], 0.90
            )
            sigma = median_sigma(left, right, PRIMARY_SEED)
            statistic = mmd2_unbiased(left, right, sigma)
            sample_rows.append(
                {
                    "sample_size_per_corpus": sample_size,
                    "seed": seed,
                    "pca_components": int(pca.n_components_),
                    "retained_variance": sample_retained,
                    "sigma": sigma,
                    "mmd2": statistic,
                    "percent_change_from_primary": percent_change(
                        statistic, primary_mmd2
                    ),
                    "nested_within_seed": True,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for sample_size in SAMPLE_SIZES:
        selected = [
            row for row in sample_rows if row["sample_size_per_corpus"] == sample_size
        ]
        mmd_q1, mmd_median, mmd_q3 = quantiles(
            [float(row["mmd2"]) for row in selected]
        )
        sigma_q1, sigma_median, sigma_q3 = quantiles(
            [float(row["sigma"]) for row in selected]
        )
        components = [int(row["pca_components"]) for row in selected]
        summary_rows.append(
            {
                "sample_size_per_corpus": sample_size,
                "repetitions": len(selected),
                "mmd2_median": mmd_median,
                "mmd2_q1": mmd_q1,
                "mmd2_q3": mmd_q3,
                "percent_change_from_primary_median": percent_change(
                    mmd_median, primary_mmd2
                ),
                "sigma_median": sigma_median,
                "sigma_q1": sigma_q1,
                "sigma_q3": sigma_q3,
                "pca_components_median": float(np.median(components)),
                "pca_components_min": min(components),
                "pca_components_max": max(components),
                "dependency_note": "nested within seed; IQR is not a confidence interval",
            }
        )
    summary_rows.append(
        {
            "sample_size_per_corpus": EXPECTED_N,
            "repetitions": 1,
            "mmd2_median": primary_mmd2,
            "mmd2_q1": primary_mmd2,
            "mmd2_q3": primary_mmd2,
            "percent_change_from_primary_median": 0.0,
            "sigma_median": sigma0,
            "sigma_q1": sigma0,
            "sigma_q3": sigma0,
            "pca_components_median": int(primary_pca.n_components_),
            "pca_components_min": int(primary_pca.n_components_),
            "pca_components_max": int(primary_pca.n_components_),
            "dependency_note": "single frozen primary result; not a repeated draw",
        }
    )

    masked_left, masked_right, masked_pca, masked_retained = dynamic_pca(
        masked_baseline, masked_target, 0.90
    )
    masked_sigma = median_sigma(masked_left, masked_right, PRIMARY_SEED)
    masked_direct = mmd2_unbiased(masked_left, masked_right, masked_sigma)
    masked_observed, null, exceedances, masked_threshold = permutation_test_precomputed(
        masked_left,
        masked_right,
        masked_sigma,
        n_permutations=N_PERMUTATIONS,
        seed=PRIMARY_SEED,
    )
    if not np.isclose(masked_direct, masked_observed, rtol=1e-7, atol=1e-8):
        raise ProtocolError(
            "Masked-mean observed MMD differs between direct and permutation paths"
        )
    masked_p_text = p_display(exceedances, N_PERMUTATIONS)
    factor_rows.append(
        {
            "factor": "embedding_construction",
            "setting": "masked_mean",
            "n_baseline": len(masked_left),
            "n_target": len(masked_right),
            "pca_components": int(masked_pca.n_components_),
            "retained_variance": masked_retained,
            "sigma": masked_sigma,
            "mmd2": masked_observed,
            "percent_change_from_primary": "",
            "permutation_p": masked_p_text,
            "note": (
                "pooling only; final layer; no L2; unweighted across windows; "
                "no percentage comparison across geometries"
            ),
        }
    )

    validate_results(
        factor_rows, sample_rows, summary_rows, exceedances, masked_p_text
    )
    result = {
        "schema": "ae5a-sensitivity-results-minimal-v3",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_inputs": analysis_input_paths(args),
        "primary_reproduction": {
            "passed": True,
            "pca_components": int(primary_pca.n_components_),
            "retained_variance": retained,
            "sigma": sigma0,
            "mmd2": primary_mmd2,
            "expected_mmd2": EXPECTED_MMD2,
        },
        "model": {
            "id": MODEL_ID,
            "baseline_device": masked_baseline_meta["device"],
            "target_device": masked_target_meta["device"],
        },
        "factor_cells": factor_rows,
        "sample_size_summary": summary_rows,
        "sample_size_runs_count": len(sample_rows),
        "masked_mean_permutation": {
            "n_permutations": N_PERMUTATIONS,
            "exceedances": exceedances,
            "exceedance_fraction": exceedances / N_PERMUTATIONS,
            "reported_p": masked_p_text,
            "threshold_alpha_0_05": masked_threshold,
        },
        "software_environment": environment_versions(),
    }
    markdown, latex = render_reports(result)

    factor_fields = [
        "factor",
        "setting",
        "n_baseline",
        "n_target",
        "pca_components",
        "retained_variance",
        "sigma",
        "mmd2",
        "percent_change_from_primary",
        "permutation_p",
        "note",
    ]
    atomic_write_csv(paths["factor_cells"], factor_fields, factor_rows)
    atomic_write_csv(paths["sample_runs"], list(sample_rows[0].keys()), sample_rows)
    atomic_write_csv(
        paths["sample_summary"], list(summary_rows[0].keys()), summary_rows
    )
    atomic_save_npy(paths["null"], null)
    atomic_write_json(paths["json"], result)
    atomic_write_text(paths["markdown"], markdown)
    atomic_write_text(paths["latex"], latex)

    print(f"PASS: all prespecified cells completed: {paths['json']}")
    print(f"Saved report: {paths['markdown']}")
    print(
        "Masked mean: "
        f"components={masked_pca.n_components_}, sigma={masked_sigma:.6f}, "
        f"MMD^2={masked_observed:.6f}, {masked_p_text}"
    )


if __name__ == "__main__":
    main()
