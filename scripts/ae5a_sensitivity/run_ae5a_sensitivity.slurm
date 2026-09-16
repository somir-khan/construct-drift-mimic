#!/bin/bash
#SBATCH --job-name=ae5a_sensitivity
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=/work/somir49/drift-audit/logs/ae5a_sensitivity_%j.out
#SBATCH --error=/work/somir49/drift-audit/logs/ae5a_sensitivity_%j.err

set -euo pipefail
umask 027

# The logs directory must exist before sbatch is called.
# Submit from any directory with:
#   mkdir -p /work/somir49/drift-audit/logs
#   sbatch /work/somir49/drift-audit/scripts/ae5a_sensitivity/run_ae5a_sensitivity.slurm

ROOT="/work/somir49/drift-audit"
PACKAGE_DIR="$ROOT"
VENV="$ROOT/../venvs/drift-audit/bin/activate"
MIMIC3_DB="/work/somir49/tmis-dataset/mimic3.db"
MIMIC4_DB="/work/somir49/tmis-dataset/mimic4.db"
OUTPUT_DIR="$ROOT/outputs/ae5a_sensitivity"

module purge
module load python/3.11.5-anaconda
cd "$ROOT"
source "$VENV"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

required_files=(
    "$PACKAGE_DIR/scripts/ae5a_sensitivity/embed_masked_mean.py"
    "$PACKAGE_DIR/scripts/ae5a_sensitivity/run_ae5a_sensitivity.py"
    "$ROOT/data/embeddings/embeddings_mimic3_5000.npy"
    "$ROOT/data/embeddings/ids_mimic3_5000.npy"
    "$ROOT/data/embeddings/embeddings_mimic4_5000.npy"
    "$ROOT/data/embeddings/ids_mimic4_5000.npy"
    "$ROOT/data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy"
    "$ROOT/data/embeddings_windows/ids_mimic4_2500_2014_2016.npy"
    "$ROOT/data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy"
    "$ROOT/data/embeddings_windows/ids_mimic4_2500_2017_2019.npy"
    "$MIMIC3_DB"
    "$MIMIC4_DB"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file is missing: $path" >&2
        exit 3
    fi
done

echo "============================================================"
echo "AE.5a SENSITIVITY RUN"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-not-set}"
echo "Node:         $(hostname)"
echo "Start time:   $(date --iso-8601=seconds)"
echo "Repository:   $ROOT"
echo "Package:      $PACKAGE_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Python:       $(command -v python)"
python --version
nvidia-smi

python - <<'PY'
import torch

print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this SLURM job.")
print("GPU           :", torch.cuda.get_device_name(0))
print("CUDA runtime  :", torch.version.cuda)
PY

echo "============================================================"
echo "1/3: MIMIC-III masked-mean embeddings"
echo "============================================================"

python "$PACKAGE_DIR/scripts/ae5a_sensitivity/embed_masked_mean.py" \
    --corpus baseline \
    --mimic3-db "$MIMIC3_DB" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "2/3: MIMIC-IV masked-mean embeddings"
echo "============================================================"

python "$PACKAGE_DIR/scripts/ae5a_sensitivity/embed_masked_mean.py" \
    --corpus target \
    --mimic4-db "$MIMIC4_DB" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "3/3: AE.5a sensitivity calculations and reports"
echo "============================================================"

python "$PACKAGE_DIR/scripts/ae5a_sensitivity/run_ae5a_sensitivity.py" \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "AE.5a JOB COMPLETED SUCCESSFULLY"
echo "End time:     $(date --iso-8601=seconds)"
echo "Results:      $OUTPUT_DIR"
echo "============================================================"
