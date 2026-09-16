#!/bin/bash

#SBATCH --job-name=expB_embed
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=experiment_b_embed_%j.out
#SBATCH --error=experiment_b_embed_%j.err

set -euo pipefail

echo "============================================================"
echo "EXPERIMENT B — FROZEN EMBEDDING RUN"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "============================================================"

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

module purge
module load python/3.11.5-anaconda

cd /work/somir49/drift-audit

# If you created a virtual environment on LONI, activate it here.
# Uncomment/change this line if needed:
#
source /work/somir49/venvs/drift-audit/bin/activate

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi

echo
echo "CUDA validation:"
python - <<'PY'
import torch

print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available.")

print("GPU           :", torch.cuda.get_device_name(0))
print("CUDA runtime  :", torch.version.cuda)
PY

echo
echo "============================================================"
echo "STARTING FULL EXPERIMENT B EMBEDDING"
echo "============================================================"

python scripts/experiment_b/experiment_b_embed_frozen.py \
    --m3-manifest outputs/experiment_b_preflight/experiment_b_m3_split_manifest.csv \
    --m4-manifest outputs/experiment_b_preflight/experiment_b_m4_icd9_only_manifest.csv \
    --mimic3-db /work/somir49/tmis-dataset/mimic3.db \
    --mimic4-db /work/somir49/tmis-dataset/mimic4.db \
    --device cuda \
    --cohorts m3_train m3_dev m3_test m4_target \
    --output-dir data/experiment_b_embeddings \
    --checkpoint-every 100

echo
echo "============================================================"
echo "EMBEDDING COMMAND COMPLETED"
echo "End time: $(date)"
echo "============================================================"

python - <<'PY'
import os
import json
import numpy as np

root = "data/experiment_b_embeddings"

expected = {
    "m3_train": 47723,
    "m3_dev": 1631,
    "m3_test": 3372,
    "m4_target": 19667,
}

print()
print("=" * 72)
print("EXPERIMENT B FINAL EMBEDDING VALIDATION")
print("=" * 72)

all_ok = True

for cohort, n in expected.items():

    emb_path = os.path.join(
        root,
        f"embeddings_{cohort}_{n}.npy"
    )

    ids_path = os.path.join(
        root,
        f"ids_{cohort}_{n}.npy"
    )

    meta_path = os.path.join(
        root,
        f"metadata_{cohort}_{n}.json"
    )

    print()
    print(cohort)
    print("-" * 72)

    for p in [emb_path, ids_path, meta_path]:
        print(
            f"{os.path.basename(p):45s}",
            "FOUND" if os.path.exists(p) else "MISSING"
        )

    if not (
        os.path.exists(emb_path)
        and os.path.exists(ids_path)
        and os.path.exists(meta_path)
    ):
        all_ok = False
        continue

    X = np.load(emb_path, mmap_mode="r")
    ids = np.load(ids_path)

    print("embedding shape :", X.shape)
    print("ID shape        :", ids.shape)
    print("unique IDs      :", len(np.unique(ids)))

    shape_ok = X.shape == (n, 768)
    ids_ok = ids.shape == (n,)
    unique_ok = len(np.unique(ids)) == n

    finite_ok = bool(
        np.isfinite(X).all()
    )

    zero_rows = int(
        np.sum(
            np.all(X == 0, axis=1)
        )
    )

    print("shape OK        :", shape_ok)
    print("IDs OK          :", ids_ok)
    print("unique IDs OK   :", unique_ok)
    print("all finite      :", finite_ok)
    print("zero rows       :", zero_rows)

    if not (
        shape_ok
        and ids_ok
        and unique_ok
        and finite_ok
        and zero_rows == 0
    ):
        all_ok = False


print()
print("=" * 72)

if all_ok:
    print("RESULT: ALL FOUR EXPERIMENT B EMBEDDING COHORTS PASSED.")
else:
    raise RuntimeError(
        "One or more Experiment B embedding cohorts failed validation."
    )

print("=" * 72)
PY

echo
echo "============================================================"
echo "JOB SUCCESSFULLY FINISHED"
echo "$(date)"
echo "============================================================"