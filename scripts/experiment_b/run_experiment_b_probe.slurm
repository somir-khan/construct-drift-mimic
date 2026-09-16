#!/bin/bash

#SBATCH --job-name=expB_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=/work/somir49/drift-audit/logs/experiment_b_build_%j.out
#SBATCH --error=/work/somir49/drift-audit/logs/experiment_b_build_%j.err

set -euo pipefail
umask 027

# Submit exactly one phase at a time:
#
#   sbatch scripts/experiment_b/run_experiment_b_probe.slurm tune
#   sbatch scripts/experiment_b/run_experiment_b_probe.slurm evaluate
#
# Run "tune" first. Inspect and freeze its dev-only C-selection output before
# submitting "evaluate". This separation prevents test/M4 outcome access during
# model selection.

MODE="${1:-}"

if [[ "$MODE" != "tune" && "$MODE" != "evaluate" ]]; then
    echo "Usage: sbatch $0 {tune|evaluate}" >&2
    exit 2
fi

# This job is CPU-only. If your LONI cluster requires an explicit CPU
# partition, add the appropriate #SBATCH --partition=... line above.

# -----------------------------------------------------------------------------
# Edit paths only in this block.
# -----------------------------------------------------------------------------

ROOT="/work/somir49/drift-audit"
VENV="/work/somir49/venvs/drift-audit/bin/activate"

PREFLIGHT="$ROOT/outputs/experiment_b_preflight"
LABELS="$ROOT/outputs/experiment_b_labels"
REPRESENTATION="$ROOT/outputs/experiment_b_representation"
PROBE_OUTPUT="$ROOT/outputs/experiment_b_probe"

PROBE_SCRIPT="$ROOT/scripts/experiment_b/experiment_b_probe_frozen.py"
TOP50_CODES="$PREFLIGHT/experiment_b_top50_codes.txt"
MODEL_BUNDLE="$PROBE_OUTPUT/experiment_b_frozen_probe.joblib"

# -----------------------------------------------------------------------------
# Environment.
# -----------------------------------------------------------------------------

module purge
module load python/3.11.5-anaconda
cd "$ROOT"
source "$VENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$PROBE_OUTPUT"

echo "============================================================"
echo "EXPERIMENT B — PROBE: ${MODE^^}"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-not-set}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Repository:  $ROOT"
echo "Python:      $(command -v python)"
python --version

# -----------------------------------------------------------------------------
# Dev-only global-C selection. No test or M4 arguments exist in this command.
# -----------------------------------------------------------------------------

if [[ "$MODE" == "tune" ]]; then
    required_files=(
        "$PROBE_SCRIPT"
        "$TOP50_CODES"
        "$REPRESENTATION/pca_m3_train.npy"
        "$REPRESENTATION/row_ids_pca_m3_train.npy"
        "$LABELS/labels_m3_train_top50.npy"
        "$LABELS/row_ids_labels_m3_train.npy"
        "$REPRESENTATION/pca_m3_dev.npy"
        "$REPRESENTATION/row_ids_pca_m3_dev.npy"
        "$LABELS/labels_m3_dev_top50.npy"
        "$LABELS/row_ids_labels_m3_dev.npy"
    )

    for path in "${required_files[@]}"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: required file is missing: $path" >&2
            exit 3
        fi
    done

    if [[ -e "$MODEL_BUNDLE" ]]; then
        echo "ERROR: frozen model bundle already exists:" >&2
        echo "       $MODEL_BUNDLE" >&2
        echo "Refusing to overwrite a previous tune result." >&2
        exit 4
    fi

    python "$PROBE_SCRIPT" tune \
        --train-pca "$REPRESENTATION/pca_m3_train.npy" \
        --train-pca-ids "$REPRESENTATION/row_ids_pca_m3_train.npy" \
        --train-labels "$LABELS/labels_m3_train_top50.npy" \
        --train-label-ids "$LABELS/row_ids_labels_m3_train.npy" \
        --dev-pca "$REPRESENTATION/pca_m3_dev.npy" \
        --dev-pca-ids "$REPRESENTATION/row_ids_pca_m3_dev.npy" \
        --dev-labels "$LABELS/labels_m3_dev_top50.npy" \
        --dev-label-ids "$LABELS/row_ids_labels_m3_dev.npy" \
        --top50-codes "$TOP50_CODES" \
        --out-dir "$PROBE_OUTPUT"

    echo
    echo "Tune completed. STOP and inspect the dev-only C-selection output:"
    echo "$PROBE_OUTPUT/experiment_b_dev_c_selection.csv"
    echo
    echo "Do not submit evaluate until the selected C is accepted as frozen."

# -----------------------------------------------------------------------------
# Final fixed-model evaluation on M3 test and M4.
# -----------------------------------------------------------------------------

else
    required_files=(
        "$PROBE_SCRIPT"
        "$TOP50_CODES"
        "$MODEL_BUNDLE"
        "$REPRESENTATION/pca_m3_test.npy"
        "$REPRESENTATION/row_ids_pca_m3_test.npy"
        "$LABELS/labels_m3_test_top50.npy"
        "$LABELS/row_ids_labels_m3_test.npy"
        "$LABELS/subjects_m3_test.npy"
        "$LABELS/row_ids_subjects_m3_test.npy"
        "$REPRESENTATION/pca_m4_target.npy"
        "$REPRESENTATION/row_ids_pca_m4_target.npy"
        "$LABELS/labels_m4_target_top50.npy"
        "$LABELS/row_ids_labels_m4_target.npy"
        "$LABELS/subjects_m4_target.npy"
        "$LABELS/row_ids_subjects_m4_target.npy"
    )

    for path in "${required_files[@]}"; do
        if [[ ! -f "$path" ]]; then
            echo "ERROR: required file is missing: $path" >&2
            exit 5
        fi
    done

    python "$PROBE_SCRIPT" evaluate \
        --model-bundle "$MODEL_BUNDLE" \
        --test-pca "$REPRESENTATION/pca_m3_test.npy" \
        --test-pca-ids "$REPRESENTATION/row_ids_pca_m3_test.npy" \
        --test-labels "$LABELS/labels_m3_test_top50.npy" \
        --test-label-ids "$LABELS/row_ids_labels_m3_test.npy" \
        --test-subjects "$LABELS/subjects_m3_test.npy" \
        --test-subject-row-ids "$LABELS/row_ids_subjects_m3_test.npy" \
        --m4-pca "$REPRESENTATION/pca_m4_target.npy" \
        --m4-pca-ids "$REPRESENTATION/row_ids_pca_m4_target.npy" \
        --m4-labels "$LABELS/labels_m4_target_top50.npy" \
        --m4-label-ids "$LABELS/row_ids_labels_m4_target.npy" \
        --m4-subjects "$LABELS/subjects_m4_target.npy" \
        --m4-subject-row-ids "$LABELS/row_ids_subjects_m4_target.npy" \
        --top50-codes "$TOP50_CODES" \
        --out-dir "$PROBE_OUTPUT"

    echo
    echo "Final evaluation completed:"
    echo "$PROBE_OUTPUT/experiment_b_final_results.json"
fi

echo
echo "============================================================"
echo "EXPERIMENT B PROBE ${MODE^^} COMPLETED SUCCESSFULLY"
echo "End time: $(date)"
echo "============================================================"
