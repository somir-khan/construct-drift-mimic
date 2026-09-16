#!/bin/bash
#SBATCH --job-name=expc_mcd_final
#SBATCH --account=loni_driftaudit
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=expc_mcd_final-%j.out
#SBATCH --error=expc_mcd_final-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5-anaconda
source /work/somir49/venvs/drift-audit/bin/activate

python scripts/experiment_c/experiment_c_run_mcddd.py \
  --prepared-dir data/experiment_c/prepared \
  --mcddd-repo mcd-dd \
  --output-dir outputs/experiment_c/mcddd_final \
  --device cuda \
  --torch-threads 4 \
  --determinism strict