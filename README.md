# A Validity Audit Framework for Longitudinal AI Measurement

This repository contains the code accompanying *A Validity Audit Framework for
Longitudinal AI Measurement*. The framework evaluates whether a frozen clinical
NLP measurement process remains valid as the underlying data distribution
changes. MIMIC-III and MIMIC-IV provide the longitudinal validation setting.

The implementation combines:

- Bio_ClinicalBERT note embeddings;
- PCA-compressed maximum mean discrepancy (MMD);
- label-free surface-feature analysis;
- blinded LLM-based attribution of structural, lexical, and unresolved drift;
- a controlled positive experiment for escalation sensitivity;
- a frozen ICD-9 prediction task for downstream stability; and
- comparisons with DriftLens and MCD-DD.

## Repository structure

```text
scripts/
├── embed_and_save.py                         # Frozen note embedding pipeline
├── detect_drift.py                           # PCA and MMD drift test
├── select_judge_samples.py                   # Witness-based Judge sampling
├── judge_blind_run_struct_tie_break_frozen.py
├── surface_features.py                       # Label-free surface analysis
├── fig3_surface_features.py                  # Figure 3 renderer
├── mortality_base_rates.py                   # Mortality-rate validation
├── experiment_a/                             # Controlled positive experiment
├── experiment_b/                             # Frozen ICD-9 prediction task
├── experiment_c/                             # DriftLens and MCD-DD comparison
├── ae5a_sensitivity/                         # Representation/MMD sensitivity
└── addendum_corruption_mimic3/               # Report-preferred sensitivity
```

All commands below assume the current working directory is the repository root.

## Data requirements

The analyses require credentialed access to MIMIC-III and MIMIC-IV. The
repository does not distribute clinical notes, admission or subject identifiers,
embeddings derived from individual notes, database files, or prototype-note
exports.

Configure the local SQLite database paths in `.env`:

```dotenv
MIMIC3_DB_PATH=/path/to/mimic3.db
MIMIC4_DB_PATH=/path/to/mimic4.db
```

Generated files under `data/`, `outputs/`, and `logs/` may contain restricted or
derived MIMIC information and should remain outside public version control.
Exact cohort identifier manifests are therefore not distributed through this
repository. Any release of those derived files should use PhysioNet under the
same access agreement as the source data.

## Installation

Python 3.10 and 3.11 were used for the reported analyses.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for the local CPU or CUDA environment if it
differs from the build specified in `requirements.txt`.

The following external projects are not vendored:

- [CAML-MIMIC](https://github.com/jamesmullenbach/caml-mimic), for the published
  MIMIC-III train, development, and test admission splits used in Experiment B;
- [DriftLens](https://github.com/grecosalvatore/drift-lens), tag `v1.0.0`, for
  Experiment C; and
- [MCD-DD](https://github.com/LiangYiAnita/mcd-dd), commit
  `1139b29ac73c43886f267835fbf7a06125d63687`, for Experiment C.

The blinded Judge jobs also require Ollama and the frozen `gemma4:26b` model.

## Core audit pipeline

### 1. Generate the frozen embeddings

```bash
python scripts/embed_and_save.py --dataset mimic3 --sample-size 5000
python scripts/embed_and_save.py --dataset mimic4 --sample-size 5000

python scripts/embed_and_save.py --dataset mimic4 --sample-size 2500 \
    --anchor-year-group "2014 - 2016" \
    --output-suffix "_2014_2016" \
    --device cpu

python scripts/embed_and_save.py --dataset mimic4 --sample-size 2500 \
    --anchor-year-group "2017 - 2019" \
    --output-suffix "_2017_2019" \
    --device cpu
```

The unfiltered embeddings are written under `data/embeddings/`; the two
MIMIC-IV window embeddings are written under `data/embeddings_windows/`.

### 2. Run the PCA/MMD drift test

```bash
python scripts/detect_drift.py \
    --load-embeddings data/embeddings \
    --data-dir data \
    --pca-variance 0.90 \
    --pca-components-max 150 \
    --permutations 1000 \
    --alpha 0.05 \
    --rng-seed 42
```

This creates `data/baseline_pca.npy`, `data/target_pca.npy`, and the frozen
`data/pca_model.pkl` used by downstream analyses.

`detect_drift.py` selects the most recently modified matching embedding file
for each corpus. Keep only the intended 5,000-note baseline and target files in
`data/embeddings/` when reproducing the primary result.

### 3. Select blinded Judge samples

```bash
python scripts/select_judge_samples.py \
    --n-select 50 \
    --n-exemplars 3 \
    --subsample 1000 \
    --rng-seed 42 \
    --samples-csv data/judge_samples_300.csv
```

The manifest contains 300 selected MIMIC-IV notes: 50 Top, 50 Random, and 50
Bottom observations from each of the two patient-level anchor-year groups. It
also contains three MIMIC-III exemplars used by the Judge.

### 4. Run the frozen blinded Judge

Create the log directory before submitting any SLURM job:

```bash
mkdir -p logs
sbatch scripts/run_judge_blind_struct_tie_break_frozen.slurm
```

To resume an interrupted run after inspecting the partial output:

```bash
sbatch scripts/resume_judge_blind_struct_tie_break_frozen.slurm
```

The frozen protocol uses five calls at temperature 0.7, modal aggregation, and
the tie-break order Structural Drift > Lexical Drift > Unresolved.

### 5. Surface features and mortality validation

```bash
python scripts/surface_features.py
python scripts/fig3_surface_features.py
python scripts/mortality_base_rates.py
```

## Experiment A: controlled escalation test

Experiment A evaluates a semantic intervention and a matched structural
control under the blinded Judge protocol to test whether the audit escalates
under a known construct-relevant change.

```bash
python scripts/experiment_a/select_experiment_a_hosts.py --write-hosts
python scripts/experiment_a/generate_experiment_a_variants.py
python scripts/experiment_a/embed_experiment_a_diagnostics.py --device cuda
python scripts/calibrate_experiment_a_threshold_exploratory.py
python scripts/experiment_a/prepare_experiment_a_judge_input.py \
    --legacy-samples data/judge_samples_300.csv
python scripts/experiment_a/validate_experiment_a.py
sbatch scripts/experiment_a/run_judge_blind_experiment_a.slurm
```

Resume only after reviewing an interrupted primary run:

```bash
sbatch scripts/experiment_a/resume_judge_blind_experiment_a.slurm
```

The runner enforces the frozen Judge model and analysis configuration.

`export_experiment_a_selected_notes.py` is an optional local inspection
utility. Its output contains MIMIC note text and identifiers and must not be
committed or distributed.

## Experiment B: frozen ICD-9 prediction task

Experiment B trains a one-vs-rest logistic-regression probe on MIMIC-III and
evaluates the unchanged model on held-out MIMIC-III admissions and the native
ICD-9 MIMIC-IV 2014–2016 target cohort. The label set is selected from the
MIMIC-III training split only.

Create the frozen cohort manifests and Top-50 code list:

```bash
python scripts/experiment_b/experiment_b_preflight.py \
    --mullenbach-dir /path/to/caml-mimic/mimicdata/mimic3 \
    --mimic3-db /path/to/mimic3.db \
    --mimic4-db /path/to/mimic4.db \
    --out-dir outputs/experiment_b_preflight
```

Then execute the frozen embedding, validation, tuning, and evaluation stages:

```bash
sbatch scripts/experiment_b/experiment_b_embeddings.slurm
sbatch scripts/experiment_b/run_experiment_b_materialize_validate.slurm
sbatch scripts/experiment_b/run_experiment_b_probe.slurm tune
sbatch scripts/experiment_b/run_experiment_b_probe.slurm evaluate
```

Run `tune` first and freeze its development-set selection before submitting
`evaluate`. The packaged SLURM files contain site-specific paths and scheduler
settings that must be adjusted for another cluster.

The exploratory SQL feasibility probes are not required by this pipeline.

## AE.5a sensitivity analysis

The AE.5a analysis evaluates sensitivity to PCA dimension, sample size, and an
alternative masked-mean Bio_ClinicalBERT representation.

After updating the database, environment, and output paths in the launcher:

```bash
sbatch scripts/ae5a_sensitivity/run_ae5a_sensitivity.slurm
```

The launcher generates masked-mean embeddings for both corpora and then runs
the frozen sensitivity analysis.

## Experiment C: detector comparison

Experiment C compares the framework with DriftLens and MCD-DD using prepared
baseline, calibration, and target arrays.

Prepare the inputs:

```bash
python scripts/experiment_c/experiment_c_prepare.py \
    --baseline-emb data/embeddings/embeddings_mimic3_5000.npy \
    --baseline-ids data/embeddings/ids_mimic3_5000.npy \
    --target data/embeddings_windows/embeddings_mimic4_2500_2014_2016.npy data/embeddings_windows/ids_mimic4_2500_2014_2016.npy "2014 - 2016" \
    --target data/embeddings_windows/embeddings_mimic4_2500_2017_2019.npy data/embeddings_windows/ids_mimic4_2500_2017_2019.npy "2017 - 2019" \
    --m3-pool data/experiment_b_embeddings/embeddings_m3_train_47723.npy data/experiment_b_embeddings/ids_m3_train_47723.npy \
    --m3-pool data/experiment_b_embeddings/embeddings_m3_dev_1631.npy data/experiment_b_embeddings/ids_m3_dev_1631.npy \
    --m3-pool data/experiment_b_embeddings/embeddings_m3_test_3372.npy data/experiment_b_embeddings/ids_m3_test_3372.npy \
    --mimic3-db /path/to/mimic3.db \
    --output-dir data/experiment_c/prepared
```

Run the MCD-DD pilot and final jobs:

```bash
sbatch scripts/experiment_c/experiment_c_mcddd_pilot.slurm
sbatch scripts/experiment_c/experiment_c_mcddd.slurm
```

Run DriftLens:

```bash
mkdir -p data/experiment_c/driftlens logs

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/experiment_c/experiment_c_run_driftlens.py \
    --prepared-dir data/experiment_c/prepared \
    --driftlens-repo /path/to/drift-lens \
    --output-dir data/experiment_c/driftlens \
    --jobs 8 \
    --checkpoint-every 25
```

Combine the detector outputs:

```bash
python scripts/experiment_c/experiment_c_report.py \
    --driftlens-results data/experiment_c/driftlens/driftlens_results.json \
    --mcddd-runs outputs/experiment_c/mcddd_final/mcddd_runs.jsonl \
    --output-dir outputs/experiment_c/report
```

`extract_driftlens_prototype_notes.py` is an optional local inspection utility.
Its output contains MIMIC note text and admission identifiers and must not be
committed or distributed.

## Addendum/Report sensitivity analysis

This analysis tests whether replacing the latest-row Addendum with the latest
available Report for the same MIMIC-III admission changes the audit conclusion.
The public repository does not distribute the local overlap manifest because it
contains MIMIC admission identifiers.

Place the credentialed local prerequisites at:

```text
data/addendum_sensitivity/mimic3_baseline_addendum_overlap.csv
```

Expose the top-level modules to the nested scripts, then run:

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

python scripts/addendum_corruption_mimic3/paired_addendum_analysis.py
python scripts/addendum_corruption_mimic3/check_exemplar_representativeness.py
python scripts/addendum_corruption_mimic3/embed_addendum_report_sensitivity.py
python scripts/addendum_corruption_mimic3/build_report_sensitivity_baseline.py
python scripts/addendum_corruption_mimic3/run_addendum_mmd_sensitivity.py
python scripts/addendum_corruption_mimic3/run_addendum_permutation_sensitivity.py
python scripts/addendum_corruption_mimic3/run_addendum_surface_sensitivity.py
python scripts/addendum_corruption_mimic3/replay_judge_sample_selection.py
python scripts/addendum_corruption_mimic3/run_addendum_witness_sensitivity_v2.py
python scripts/addendum_corruption_mimic3/check_historical_random_rank_stability.py
```

## Reproducibility notes

- Do not change frozen random seeds, model names, PCA settings, sampling counts,
  or tie-break behavior when reproducing the reported analyses.
- Run the Experiment B `tune` and `evaluate` phases separately.
- Review and adapt the absolute paths and `#SBATCH` directives in the SLURM
  files before using them on a different cluster.
- Keep all note text, row-level identifiers, embeddings, and prototype exports
  inside a credentialed MIMIC environment.

## Citation

Please cite the accompanying manuscript when using this code. Final publication
metadata will be added after the article is published.
