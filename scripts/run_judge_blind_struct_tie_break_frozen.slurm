#!/usr/bin/env bash
#SBATCH --job-name=judge_struct_frozen
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --account=loni_driftaudit
#SBATCH --output=logs/judge_struct_frozen_%j.out
#SBATCH --error=logs/judge_struct_frozen_%j.err

# Frozen blind Judge run: one serial classifier process, one dedicated A100.
# Do not add --requeue: the script deliberately refuses a nonempty results CSV
# unless an investigator explicitly invokes it with --resume.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module load python/3.11.5-anaconda
source "$HOME/venvs/judge-llm/bin/activate"

# Fail before consuming GPU time if this environment cannot accept the only
# per-call hang protection used by the frozen script.
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
# The frozen Python script is serial. Higher server parallelism would not make
# this job faster and would only increase GPU-memory pressure.
export OLLAMA_NUM_PARALLEL=1

MODEL="gemma4:26b"
SAMPLES_CSV="data/judge_samples_300.csv"
RESULTS_CSV="data/judge_blind_struct_tie_break_results.csv"
JUDGE_SCRIPT="scripts/judge_blind_run_struct_tie_break_frozen.py"

[[ -f "$JUDGE_SCRIPT" ]] || { echo "ERROR: judge script not found: $JUDGE_SCRIPT" >&2; exit 1; }
[[ -f "$SAMPLES_CSV" ]] || { echo "ERROR: samples CSV not found: $SAMPLES_CSV" >&2; exit 1; }
if [[ -e "$RESULTS_CSV" ]]; then
    echo "ERROR: results path already exists: $RESULTS_CSV" >&2
    echo "Choose a new RESULTS_CSV before submission; do not enable blind requeue." >&2
    exit 1
fi

# This is intentionally before server startup: it verifies the judge's hard
# manifest gate, including literal cohort labels and text presence.
python3 - "$SAMPLES_CSV" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
required = {"hadm_id", "witness_score", "selection_group", "anchor_year_group", "text"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"ERROR: samples CSV missing required columns: {sorted(missing)}")
if len(df) != 303:
    raise SystemExit(f"ERROR: expected 303 total rows, found {len(df)}")

valid_groups = {"exemplar", "top", "bottom", "random"}
observed_groups = set(df["selection_group"].dropna())
unexpected = observed_groups - valid_groups
if unexpected:
    raise SystemExit(f"ERROR: unexpected selection_group values: {sorted(unexpected)}")

nonempty_text = df["text"].notna() & df["text"].astype(str).str.strip().ne("")
exemplars = df[df["selection_group"].eq("exemplar")]
audit = df[df["selection_group"].isin({"top", "bottom", "random"})]
if len(exemplars) != 3 or not nonempty_text.loc[exemplars.index].all():
    raise SystemExit("ERROR: require exactly 3 exemplars with nonempty text")
if len(audit) != 300 or audit["hadm_id"].duplicated().any() or not nonempty_text.loc[audit.index].all():
    raise SystemExit("ERROR: require 300 unique audit notes with nonempty text")

expected_cohorts = {"2014 - 2016", "2017 - 2019"}
labels = audit["anchor_year_group"].astype(str)
if not labels.eq(labels.str.strip()).all() or set(labels) != expected_cohorts:
    raise SystemExit("ERROR: cohort labels must be exactly '2014 - 2016' and '2017 - 2019'")
counts = audit.groupby(["anchor_year_group", "selection_group"]).size()
for cohort in expected_cohorts:
    for arm in ("top", "bottom", "random"):
        if int(counts.get((cohort, arm), 0)) != 50:
            raise SystemExit(f"ERROR: require 50 {arm} notes in cohort {cohort}")
print("Manifest preflight passed: 3 exemplars + 300 unique audit notes (50 per arm/cohort).")
PY

# Ask the kernel for an available loopback port. The process will release the
# probe socket before Ollama binds, so also require the spawned server to stay
# alive during readiness checking; a bind race fails safely instead of using
# an unrelated listener.
OLLAMA_PORT="$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
[[ -n "$OLLAMA_PORT" ]] || { echo "ERROR: port selection failed" >&2; exit 1; }
[[ "$OLLAMA_PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid selected port: $OLLAMA_PORT" >&2; exit 1; }

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
OLLAMA_API_HOST="http://${OLLAMA_HOST}"
echo "Ollama endpoint selected for this job: ${OLLAMA_API_HOST}"

ollama serve > "logs/ollama_serve_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ERROR: Ollama server exited before readiness; see logs/ollama_serve_${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    if curl --fail --silent --show-error "${OLLAMA_API_HOST}/api/tags" >/dev/null; then
        break
    fi
    sleep 1
done
curl --fail --silent --show-error "${OLLAMA_API_HOST}/api/tags" >/dev/null \
    || { echo "ERROR: Ollama did not become ready within 90 seconds" >&2; exit 1; }

# Fail fast if the job-local Ollama installation cannot serve the frozen model.
# OLLAMA_HOST is already set to this job's dynamic loopback endpoint.
if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
    echo "ERROR: Required model is not installed for this job: $MODEL" >&2
    exit 1
fi
echo "Model preflight passed: $MODEL"

# The required --ollama-host makes the endpoint visible in the submitted command
# and prevents any silent fallback to a default local port.
python3 "$JUDGE_SCRIPT" \
    --samples-csv "$SAMPLES_CSV" \
    --results-csv "$RESULTS_CSV" \
    --model "$MODEL" \
    --n-runs 5 \
    --temperature 0.7 \
    --max-retries 3 \
    --call-timeout 300 \
    --ollama-host "$OLLAMA_API_HOST"
