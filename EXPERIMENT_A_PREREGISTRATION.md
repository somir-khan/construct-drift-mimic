# Experiment A — Preregistration: Primary vs. Secondary Analysis

**Date:** 2026-08-12
**Status:** Committed before `dose_probe_v6.py`'s judge stage has been run on
either configuration below. Neither `--fallback-sections` value has produced
judge results at the time this file is written.

## Decision

**Primary analysis: STRICT configuration** — `dose_probe_v6.py` run with
`--fallback-sections` unset (`config_label=strict`). Injection target is the
Hospital Course section only. The Section 3.4 escalation criterion (≥20
percentage-point difference in Unresolved rate between the top and random
strata) is evaluated against this configuration's results, and only this
configuration's results.

**Secondary analysis: FALLBACK configuration** — same run with
`--fallback-sections hpi` (`config_label=fallback_hpi`). History of Present
Illness (HPI) is used as a fallback injection target for hosts where Hospital
Course is absent or below the 500-character floor. Reported as a preregistered
robustness/sensitivity check. Its result does not substitute for the primary
analysis under any outcome, including if it clears 20pp and strict does not.

## Why strict is primary, decided now rather than after either result is known

- Uniform intervention across strata. Every host in the top-vs-random
  comparison is injected into the same kind of section. A difference in
  outcome can only be attributed to stratum, not to what was injected into.
- HPI's construct relevance relative to Hospital Course, and whether witness
  score responds to injection there the same way, is untested. Strict avoids
  resting the primary claim on an unverified assumption.
- Fallback's larger top-stratum n (~20 vs. ~13) comes disproportionately from
  the highest-margin, highest-rank hosts (verified: 5 of 7 rescued
  2017-2019 top-stratum hosts are ranks 1, 2, 3, 9, 10). That composition
  imbalance relative to random stratum's much lower fallback rate is a
  further reason to keep it out of the primary comparison.


## Reporting commitment

All four judge combinations (strict × blind/primed, fallback × blind/primed)
will be reported in the manuscript or response letter, regardless of which
configuration's Unresolved rate is higher and regardless of whether either
clears 20pp. `config_label` and `section_used` are recorded per row
specifically so which hosts came from which section, in which configuration,
remains auditable after the fact.
