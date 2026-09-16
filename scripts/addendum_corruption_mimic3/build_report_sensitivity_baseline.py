#!/usr/bin/env python3

import os
import json
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ORIGINAL_EMB = "data/embeddings/embeddings_mimic3_5000.npy"
ORIGINAL_IDS = "data/embeddings/ids_mimic3_5000.npy"

REPORT_EMB = (
    "data/addendum_sensitivity/"
    "mimic3_report_sensitivity_embeddings.npy"
)

REPORT_IDS = (
    "data/addendum_sensitivity/"
    "mimic3_report_sensitivity_ids.npy"
)

OUT_EMB = (
    "data/addendum_sensitivity/"
    "embeddings_mimic3_5000_report_sensitivity.npy"
)

OUT_IDS = (
    "data/addendum_sensitivity/"
    "ids_mimic3_5000_report_sensitivity.npy"
)

OUT_META = (
    "data/addendum_sensitivity/"
    "report_sensitivity_baseline_metadata.json"
)


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

for path in [
    ORIGINAL_EMB,
    ORIGINAL_IDS,
    REPORT_EMB,
    REPORT_IDS,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


X_original = np.load(ORIGINAL_EMB)
ids_original = np.load(ORIGINAL_IDS).astype(np.int64)

X_report = np.load(REPORT_EMB)
ids_report = np.load(REPORT_IDS).astype(np.int64)


print("=" * 72)
print("BUILD REPORT-PREFERRED MIMIC-III SENSITIVITY BASELINE")
print("=" * 72)

print()
print("Original baseline")
print("-----------------")
print("embeddings :", X_original.shape)
print("IDs        :", ids_original.shape)

print()
print("Replacement Reports")
print("-------------------")
print("embeddings :", X_report.shape)
print("IDs        :", ids_report.shape)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

assert X_original.shape == (5000, 768), X_original.shape
assert ids_original.shape == (5000,), ids_original.shape

assert X_report.shape == (328, 768), X_report.shape
assert ids_report.shape == (328,), ids_report.shape

assert len(np.unique(ids_original)) == 5000
assert len(np.unique(ids_report)) == 328

assert np.isfinite(X_original).all()
assert np.isfinite(X_report).all()


original_id_set = set(ids_original.tolist())
report_id_set = set(ids_report.tolist())

missing = sorted(report_id_set - original_id_set)

if missing:
    raise RuntimeError(
        f"{len(missing)} Report IDs are not present in original baseline: "
        f"{missing[:20]}"
    )

print()
print("All 328 Report IDs exist in original 5,000 baseline: PASS")


# ---------------------------------------------------------------------
# Build ID -> row index map
# ---------------------------------------------------------------------

row_by_id = {
    int(hadm_id): i
    for i, hadm_id in enumerate(ids_original)
}


# ---------------------------------------------------------------------
# Replace exact rows
# ---------------------------------------------------------------------

X_sensitivity = X_original.copy()

replaced_rows = []

for report_row, hadm_id in enumerate(ids_report):

    baseline_row = row_by_id[int(hadm_id)]

    X_sensitivity[baseline_row] = X_report[report_row]

    replaced_rows.append(baseline_row)


replaced_rows = np.array(replaced_rows, dtype=int)

assert len(np.unique(replaced_rows)) == 328


# ---------------------------------------------------------------------
# Strong validation:
#
# 1. exactly 328 rows should have been targeted
# 2. all 4,672 non-target rows must be byte/numerically unchanged
# ---------------------------------------------------------------------

target_mask = np.zeros(5000, dtype=bool)
target_mask[replaced_rows] = True

unchanged_rows = np.where(~target_mask)[0]

assert len(unchanged_rows) == 4672

if not np.array_equal(
    X_sensitivity[unchanged_rows],
    X_original[unchanged_rows],
):
    raise RuntimeError(
        "At least one supposedly unaffected baseline row changed."
    )


# ---------------------------------------------------------------------
# Diagnostic differences in replaced embeddings
# ---------------------------------------------------------------------

old_replaced = X_original[replaced_rows]
new_replaced = X_sensitivity[replaced_rows]

row_l2 = np.linalg.norm(
    new_replaced - old_replaced,
    axis=1,
)

changed_numerically = int(np.sum(row_l2 > 0))

print()
print("Replacement diagnostics")
print("-----------------------")

print(f"Rows targeted             : {len(replaced_rows)}")
print(f"Rows left untouched       : {len(unchanged_rows)}")
print(f"Numerically changed rows  : {changed_numerically}")

print(f"Median embedding L2 delta : {np.median(row_l2):.6f}")
print(f"Mean embedding L2 delta   : {np.mean(row_l2):.6f}")
print(f"Min embedding L2 delta    : {np.min(row_l2):.6f}")
print(f"Max embedding L2 delta    : {np.max(row_l2):.6f}")

if changed_numerically != 328:
    print(
        "WARNING: not all 328 replacement embeddings differ numerically "
        "from the original selected note."
    )


# ---------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------

assert X_sensitivity.shape == (5000, 768)
assert np.isfinite(X_sensitivity).all()

zero_rows = np.where(
    np.all(X_sensitivity == 0, axis=1)
)[0]

if len(zero_rows):
    raise RuntimeError(
        f"Sensitivity baseline contains {len(zero_rows)} zero vectors."
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUT_EMB),
    exist_ok=True,
)

np.save(
    OUT_EMB,
    X_sensitivity.astype(np.float32),
)

# IDs remain exactly identical and in exactly the same order.
np.save(
    OUT_IDS,
    ids_original.astype(np.int64),
)


metadata = {
    "construction": (
        "Original 5000-note MIMIC-III baseline with the 328 "
        "Addendum-first admissions that also had an eligible Report "
        "replaced by the most recent eligible Report embedding. "
        "The 3 Addendum-only admissions remain unchanged."
    ),
    "n_baseline": 5000,
    "n_addendum_first_exposed": 331,
    "n_report_replaced": 328,
    "n_addendum_only_retained": 3,
    "n_unchanged": 4672,
    "original_embeddings": ORIGINAL_EMB,
    "original_ids": ORIGINAL_IDS,
    "replacement_embeddings": REPORT_EMB,
    "replacement_ids": REPORT_IDS,
    "output_embeddings": OUT_EMB,
    "output_ids": OUT_IDS,
    "median_embedding_l2_delta": float(np.median(row_l2)),
    "mean_embedding_l2_delta": float(np.mean(row_l2)),
    "min_embedding_l2_delta": float(np.min(row_l2)),
    "max_embedding_l2_delta": float(np.max(row_l2)),
}

with open(
    OUT_META,
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Reload validation
# ---------------------------------------------------------------------

X_reload = np.load(OUT_EMB)
ids_reload = np.load(OUT_IDS)

assert np.array_equal(ids_reload, ids_original)

assert np.array_equal(
    X_reload[unchanged_rows],
    X_original[unchanged_rows],
)

assert np.allclose(
    X_reload[replaced_rows],
    X_sensitivity[replaced_rows],
    rtol=0,
    atol=0,
)


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(OUT_EMB)
print(OUT_IDS)
print(OUT_META)

print()
print(
    "PASS: sensitivity baseline contains "
    "328 Report replacements + 4,672 untouched rows."
)