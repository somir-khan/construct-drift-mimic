#!/bin/bash

#SBATCH --job-name=expB_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=/work/somir49/drift-audit/logs/experiment_b_build_%j.out
#SBATCH --error=/work/somir49/drift-audit/logs/experiment_b_build_%j.err

set -euo pipefail
umask 027

# This job is CPU-only. If your LONI cluster requires an explicit CPU
# partition, add the appropriate #SBATCH --partition=... line above.

# -----------------------------------------------------------------------------
# Edit paths only in this block.
# -----------------------------------------------------------------------------

ROOT="/work/somir49/drift-audit"
VENV="/work/somir49/venvs/drift-audit/bin/activate"

PREFLIGHT="$ROOT/outputs/experiment_b_preflight"
EMBEDDINGS="$ROOT/data/experiment_b_embeddings"
LABELS="$ROOT/outputs/experiment_b_labels"
REPRESENTATION="$ROOT/outputs/experiment_b_representation"

MIMIC3_DB="/work/somir49/tmis-dataset/mimic3.db"
MIMIC4_DB="/work/somir49/tmis-dataset/mimic4.db"
PCA_MODEL="$ROOT/data/pca_model.pkl"

MATERIALIZE_SCRIPT="$ROOT/scripts/experiment_b/experiment_b_materialize_labels.py"
VALIDATE_SCRIPT="$ROOT/scripts/experiment_b/experiment_b_validate_embeddings.py"

M3_MANIFEST="$PREFLIGHT/experiment_b_m3_split_manifest.csv"
M4_MANIFEST="$PREFLIGHT/experiment_b_m4_icd9_only_manifest.csv"
TOP50_CODES="$PREFLIGHT/experiment_b_top50_codes.txt"
SUPPORT_CSV="$PREFLIGHT/experiment_b_top50_support.csv"

M3_TRAIN_IDS="$EMBEDDINGS/ids_m3_train_47723.npy"
M3_DEV_IDS="$EMBEDDINGS/ids_m3_dev_1631.npy"
M3_TEST_IDS="$EMBEDDINGS/ids_m3_test_3372.npy"
M4_IDS="$EMBEDDINGS/ids_m4_target_19667.npy"

M3_TRAIN_EMB="$EMBEDDINGS/embeddings_m3_train_47723.npy"
M3_DEV_EMB="$EMBEDDINGS/embeddings_m3_dev_1631.npy"
M3_TEST_EMB="$EMBEDDINGS/embeddings_m3_test_3372.npy"
M4_EMB="$EMBEDDINGS/embeddings_m4_target_19667.npy"

# -----------------------------------------------------------------------------
# Environment and early path validation.
# -----------------------------------------------------------------------------

module purge
module load python/3.11.5-anaconda
cd "$ROOT"
source "$VENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

required_files=(
    "$MATERIALIZE_SCRIPT"
    "$VALIDATE_SCRIPT"
    "$M3_MANIFEST"
    "$M4_MANIFEST"
    "$TOP50_CODES"
    "$SUPPORT_CSV"
    "$MIMIC3_DB"
    "$MIMIC4_DB"
    "$PCA_MODEL"
    "$M3_TRAIN_IDS"
    "$M3_DEV_IDS"
    "$M3_TEST_IDS"
    "$M4_IDS"
    "$M3_TRAIN_EMB"
    "$M3_DEV_EMB"
    "$M3_TEST_EMB"
    "$M4_EMB"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file is missing: $path" >&2
        exit 2
    fi
done

mkdir -p "$LABELS" "$REPRESENTATION"

echo "============================================================"
echo "EXPERIMENT B — MATERIALIZE + VALIDATE"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-not-set}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Repository:  $ROOT"
echo "Python:      $(command -v python)"
python --version

# -----------------------------------------------------------------------------
# Stage 1: labels, subject IDs, and independent row-ID witnesses.
# -----------------------------------------------------------------------------

echo
echo "[1/2] Materializing labels and subject vectors"

python "$MATERIALIZE_SCRIPT" \
    --m3-train-ids "$M3_TRAIN_IDS" \
    --m3-dev-ids "$M3_DEV_IDS" \
    --m3-test-ids "$M3_TEST_IDS" \
    --m4-target-ids "$M4_IDS" \
    --m3-split-manifest "$M3_MANIFEST" \
    --m4-manifest "$M4_MANIFEST" \
    --top50-codes "$TOP50_CODES" \
    --support-csv "$SUPPORT_CSV" \
    --mimic3-db "$MIMIC3_DB" \
    --mimic4-db "$MIMIC4_DB" \
    --out-dir "$LABELS"

# -----------------------------------------------------------------------------
# Stage 2: raw embedding validation and frozen float64 PCA transforms.
# -----------------------------------------------------------------------------

echo
echo "[2/2] Validating embeddings and applying the frozen PCA"

python "$VALIDATE_SCRIPT" \
    --m3-train-emb "$M3_TRAIN_EMB" \
    --m3-train-ids "$M3_TRAIN_IDS" \
    --m3-dev-emb "$M3_DEV_EMB" \
    --m3-dev-ids "$M3_DEV_IDS" \
    --m3-test-emb "$M3_TEST_EMB" \
    --m3-test-ids "$M3_TEST_IDS" \
    --m4-target-emb "$M4_EMB" \
    --m4-target-ids "$M4_IDS" \
    --m3-split-manifest "$M3_MANIFEST" \
    --m4-manifest "$M4_MANIFEST" \
    --pca-model "$PCA_MODEL" \
    --out-dir "$REPRESENTATION"

echo
echo "============================================================"
echo "MATERIALIZATION AND VALIDATION COMPLETED SUCCESSFULLY"
echo "End time: $(date)"
echo "============================================================"
echo "Labels:        $LABELS"
echo "Representation: $REPRESENTATION"
echo
echo "Next step: sbatch scripts/experiment_b/run_experiment_b_probe.slurm tune"
