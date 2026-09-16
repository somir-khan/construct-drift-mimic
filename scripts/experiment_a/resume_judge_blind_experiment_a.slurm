#!/usr/bin/env bash
#SBATCH --job-name=judge_experiment_a_resume
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=logs/judge_experiment_a_resume_%j.out
#SBATCH --error=logs/judge_experiment_a_resume_%j.err

# Explicit resume entrypoint for the frozen blind Experiment A Judge.  It has
# no --requeue: submit it only after inspecting an interrupted primary run.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module load python/3.11.5-anaconda
source "$HOME/venvs/judge-llm/bin/activate"

python3 - <<'PY'
from importlib.metadata import version
from ollama import Client

print(f"ollama-python version: {version('ollama')}")
Client(host="http://127.0.0.1:1", timeout=1)
print("ollama.Client accepts timeout=...")
PY

export PATH="$HOME/ollama/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="/work/$USER/ollama-models"
export OLLAMA_KEEP_ALIVE="30m"
export OLLAMA_NUM_PARALLEL=1

INPUT="data/experiment_a_judge_input.csv"
EXEMPLARS="data/experiment_a_exemplars.csv"
VARIANTS="data/experiment_a_variants.csv"
RESULTS="data/experiment_a_results.csv"
INCOMPLETE="data/experiment_a_incomplete.csv"
RUNNER="scripts/experiment_a/judge_blind_run_experiment_a.py"
MODEL="gemma4:26b"

[[ -f "$RUNNER" && -f "$INPUT" && -f "$EXEMPLARS" && -f "$VARIANTS" ]] || {
    echo "ERROR: missing Experiment A Judge artifact or runner" >&2; exit 1;
}
# A first-note failure creates only the incomplete sidecar; an interruption
# between completed notes can leave only results.  Either is valid evidence of
# an interrupted run, but a wholly fresh invocation must use the primary job.
if [[ ! -s "$RESULTS" && ! -s "$INCOMPLETE" ]]; then
    echo "ERROR: resume requires a nonempty results file or incomplete sidecar." >&2
    echo "Use run_judge_blind_experiment_a.slurm for a new run." >&2
    exit 1
fi
if [[ -s "$INCOMPLETE" ]]; then
    echo "Preserving incomplete-run audit sidecar: $INCOMPLETE"
fi

# Verify the complete blinded manifest before starting the GPU-backed server.
python3 - "$INPUT" "$EXEMPLARS" "$VARIANTS" <<'PY'
import sys
import pandas as pd

input_path, exemplars_path, variants_path = sys.argv[1:]
judge_input = pd.read_csv(input_path)
if set(judge_input.columns) != {"record_id", "group", "note_text"}:
    raise SystemExit("ERROR: Judge input schema must be record_id, group, note_text only")
if len(judge_input) != 203 or judge_input["record_id"].duplicated().any():
    raise SystemExit("ERROR: require exactly 203 unique Judge-input records")
if judge_input["note_text"].isna().any() or judge_input["note_text"].astype(str).str.strip().eq("").any():
    raise SystemExit("ERROR: Judge input contains an empty note")
if judge_input["group"].value_counts().to_dict() != {
    "exemplar": 3, "top_positive": 100, "random_structural_negative": 100,
}:
    raise SystemExit("ERROR: require 3 exemplars and 100 records per Experiment A arm")

exemplars = pd.read_csv(exemplars_path)
if len(exemplars) != 3 or not {"record_id", "hadm_id", "normalized_text"}.issubset(exemplars.columns):
    raise SystemExit("ERROR: frozen exemplar manifest must contain exactly three normalized texts")
variants = pd.read_csv(variants_path)
required = {"variant_id", "cohort", "arm", "variant_text"}
if len(variants) != 200 or not required.issubset(variants.columns) or variants["variant_id"].duplicated().any():
    raise SystemExit("ERROR: frozen variant manifest must contain 200 unique variants")
for cohort in ("2014 - 2016", "2017 - 2019"):
    for arm in ("top_positive", "random_structural_negative"):
        if int(((variants["cohort"] == cohort) & (variants["arm"] == arm)).sum()) != 50:
            raise SystemExit(f"ERROR: require 50 {arm} variants in cohort {cohort}")
print("Resume manifest preflight passed: 3 exemplars + 200 frozen variants.")
PY

OLLAMA_PORT="$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
[[ -n "$OLLAMA_PORT" && "$OLLAMA_PORT" =~ ^[0-9]+$ ]] || {
    echo "ERROR: dynamic loopback-port selection failed" >&2; exit 1;
}
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
OLLAMA_API_HOST="http://${OLLAMA_HOST}"
echo "Ollama endpoint selected for this resume job: ${OLLAMA_API_HOST}"

ollama serve > "logs/ollama_experiment_a_resume_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ERROR: Ollama exited before readiness; see logs/ollama_experiment_a_resume_${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    if curl --fail --silent --show-error "${OLLAMA_API_HOST}/api/tags" >/dev/null; then break; fi
    sleep 1
done
curl --fail --silent --show-error "${OLLAMA_API_HOST}/api/tags" >/dev/null || {
    echo "ERROR: Ollama did not become ready within 90 seconds" >&2; exit 1;
}
if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
    echo "ERROR: required model is unavailable for this job: $MODEL" >&2
    exit 1
fi
echo "Model preflight passed: $MODEL"

python3 "$RUNNER" \
    --input "$INPUT" \
    --exemplars "$EXEMPLARS" \
    --variants "$VARIANTS" \
    --results "$RESULTS" \
    --incomplete "$INCOMPLETE" \
    --resume \
    --ollama-host "$OLLAMA_API_HOST" \
    --call-timeout 300
