"""
surface_features.py — Level 1 Surface Feature Analysis

Computes five raw-text surface features per period to show how documentation
practice shifts from MIMIC-III to MIMIC-IV temporal windows.

MIMIC-III = one aggregate frozen baseline row.
MIMIC-IV  = five anchor_year_group categorical bars.
Output    = 6-row CSV + 6-panel matplotlib figure.
"""

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import re
import sqlite3
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SECTION_HEADERS = [
#     "assessment",
#     "plan",
#     "hospital course",
#     "history of present illness",
#     "discharge medications",
#     "discharge instructions",
#     "pertinent results",
#     "past medical history",
#     "chief complaint",
#     "assessment and plan",
# ]
###section headers of mimic3 dataset collected from https://github.com/MIT-LCP/mimic-omop/blob/master/etl/StandardizedClinicalDataTables/NOTE_NLP/section_list.csv
SECTION_HEADERS = [
    "assessment", "plan", "assessment and plan",
    "hospital course", "hospital course by", "summary of hospital course",
    "history of present illness", "hpi",
    "discharge medications", "medications on discharge", "medications at discharge",
    "discharge instructions",
    "pertinent results", "pertinent labs", "pertinent studies",
    "past medical history", "pmh",
    "chief complaint", "cc",
]

SECTION_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?",
    re.MULTILINE | re.IGNORECASE,
)

WINDOW_STATUS = {
    "2008 - 2010": "overlap",
    "2011 - 2013": "overlap",
    "2014 - 2016": "analysis",
    "2017 - 2019": "analysis",
    "2020 - 2022": "censored",
}
MIMIC4_GROUP_ORDER = [
    "2008 - 2010",
    "2011 - 2013",
    "2014 - 2016",
    "2017 - 2019",
    "2020 - 2022",
]
STATUS_SHADING = {
    "overlap":  ("grey",    0.15, "overlap"),
    "censored": ("#ffcccc", 0.40, "censored"),
}
FEATURE_NAMES  = ["mean_length", "section_rate", "jaccard", "numeric_density", "phi_density"]
FEATURE_LABELS = ["Note length", "Section rate", "Vocab Jaccard", "Numeric density", "PHI density"]
STRUCTURED_THRESHOLD = 4
TOP_N_VOCAB          = 500
BATCH_SIZE           = 2000  # texts per parallel worker batch

# ---------------------------------------------------------------------------
# SQL queries  (CTE-based dedup — no correlated subqueries)
# ---------------------------------------------------------------------------

# ROW_NUMBER dedup: ORDER BY CHARTDATE DESC, ROW_ID DESC selects the latest
# note per admission (temporal ordering). An addendum filed after the original
# report is the more recent note and is correctly preferred under this key.
# This key must be used identically in embed_and_save.py (E1) and
# detect_drift.py (D3).
SQL_MIMIC3 = """
WITH ranked AS (
    SELECT
        HADM_ID AS hadm_id,
        TEXT    AS text,
        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS rn
    FROM NOTEEVENTS
    WHERE CATEGORY   = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND TEXT       IS NOT NULL
      AND HADM_ID    IS NOT NULL
)
SELECT hadm_id, text FROM ranked WHERE rn = 1
"""

SQL_MIMIC3_FALLBACK = """
WITH ranked AS (
    SELECT
        HADM_ID AS hadm_id,
        TEXT    AS text,
        CAST(strftime('%Y', CHARTDATE) AS INTEGER) AS admit_year,
        ROW_NUMBER() OVER (
            PARTITION BY HADM_ID ORDER BY CHARTDATE DESC, ROW_ID DESC
        ) AS rn
    FROM NOTEEVENTS
    WHERE CATEGORY   = 'Discharge summary'
      AND (ISERROR IS NULL OR ISERROR != '1')
      AND TEXT       IS NOT NULL
      AND HADM_ID    IS NOT NULL
)
SELECT hadm_id, text, admit_year FROM ranked WHERE rn = 1
"""

# CTE pre-aggregates max note_seq per hadm_id in one pass — O(n) vs O(n²)
SQL_MIMIC4 = """
WITH latest AS (
    SELECT hadm_id, MAX(note_seq) AS max_seq
    FROM   "note/discharge"
    GROUP  BY hadm_id
)
SELECT n.hadm_id, n.text, p.anchor_year_group
FROM        "note/discharge"  n
JOIN        latest            l  ON n.hadm_id = l.hadm_id AND n.note_seq = l.max_seq
JOIN        "hosp/admissions" a  ON n.hadm_id = a.hadm_id
JOIN        "hosp/patients"   p  ON a.subject_id = p.subject_id
WHERE n.text IS NOT NULL
"""

# ---------------------------------------------------------------------------
# Per-batch feature computation  (module-level → picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_batch(texts: list) -> tuple:
    """
    Process a list of raw texts.  Returns (feature_rows, Counter).
    Called in worker processes — must not reference unpicklable state.
    """
    rows: list[dict] = []
    counter: Counter = Counter()

    _phi_sub1 = re.compile(r"\[\*\*.*?\*\*\]")
    _phi_sub2 = re.compile(r"___+")
    _phi_find = re.compile(r"\bPHI\b")
    _word_re  = re.compile(r"[a-zA-Z]{3,}")

    for text in texts:
        # PHI normalisation
        norm = _phi_sub1.sub("PHI", text)
        norm = _phi_sub2.sub("PHI", norm)

        tokens       = norm.split()
        total_tokens = len(tokens) if tokens else 1
        length       = len(norm)

        matches           = SECTION_RE.findall(norm)
        distinct_sections = len({m.strip().lower().rstrip(":") for m in matches})

        numeric_count   = sum(1 for t in tokens if any(c.isdigit() for c in t))
        numeric_density = numeric_count / total_tokens

        phi_density = len(_phi_find.findall(norm)) / total_tokens

        words = _word_re.findall(norm.lower())
        counter.update(words)

        rows.append({
            "length":          length,
            "n_sections":      distinct_sections,
            "numeric_density": numeric_density,
            "phi_density":     phi_density,
        })

    return rows, counter


def _parallel_features(texts: list, n_workers: int) -> tuple:
    """
    Distribute texts across worker processes.
    Returns (all_feature_rows, merged_Counter).
    """
    if not texts:
        return [], Counter()

    batches  = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_rows: list[dict] = []
    total_ctr: Counter   = Counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for feat_rows, ctr in pool.map(_process_batch, batches):
            all_rows.extend(feat_rows)
            total_ctr.update(ctr)

    return all_rows, total_ctr


# ---------------------------------------------------------------------------
# MIMIC-III loader
# ---------------------------------------------------------------------------

def _process_mimic3(
    conn,
    chunksize: int,
    sample_size: int | None,
    rng: np.random.Generator,
    n_workers: int,
    fallback: bool = False,
) -> tuple:
    """
    Load and featurise MIMIC-III discharge notes.
    SQL dedup via ROW_NUMBER CTE — no Python concat needed.
    Returns (baseline_rows, baseline_counter, split_rows_or_None).
    """
    query = SQL_MIMIC3_FALLBACK if fallback else SQL_MIMIC3

    texts_base:  list[str] = []
    texts_shift: list[str] = []

    logging.info("Loading MIMIC-III notes…")
    for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk["text"] = chunk["text"].str.strip()
        chunk = chunk[chunk["text"] != ""]
        if fallback:
            texts_base.extend(chunk.loc[chunk["admit_year"] <= 2006, "text"].tolist())
            texts_shift.extend(chunk.loc[chunk["admit_year"] >  2006, "text"].tolist())
        else:
            texts_base.extend(chunk["text"].tolist())

    logging.info(f"MIMIC-III: {len(texts_base)} notes loaded")

    # Optional sample cap
    if sample_size and len(texts_base) > sample_size:
        idx = rng.choice(len(texts_base), size=sample_size, replace=False)
        texts_base = [texts_base[i] for i in idx]

    logging.info(f"MIMIC-III: processing {len(texts_base)} notes on {n_workers} workers…")
    base_rows, base_ctr = _parallel_features(texts_base, n_workers)

    split_rows = None
    if fallback and texts_shift:
        if sample_size and len(texts_shift) > sample_size:
            idx = rng.choice(len(texts_shift), size=sample_size, replace=False)
            texts_shift = [texts_shift[i] for i in idx]
        split_rows, _ = _parallel_features(texts_shift, n_workers)

    return base_rows, base_ctr, split_rows


# ---------------------------------------------------------------------------
# MIMIC-IV loader
# ---------------------------------------------------------------------------

def _process_mimic4(
    conn,
    chunksize: int,
    sample_size: int | None,
    rng: np.random.Generator,
    n_workers: int,
) -> tuple:
    """
    Load and featurise MIMIC-IV discharge notes grouped by anchor_year_group.
    CTE dedup eliminates the O(n²) correlated subquery.
    Returns (group_rows dict, group_counter dict).
    """
    group_texts: dict[str, list[str]] = {g: [] for g in MIMIC4_GROUP_ORDER}

    logging.info("Loading MIMIC-IV notes…")
    for chunk in pd.read_sql_query(SQL_MIMIC4, conn, chunksize=chunksize):
        chunk = chunk.dropna(subset=["hadm_id", "text", "anchor_year_group"])
        chunk["text"] = chunk["text"].str.strip()
        chunk = chunk[chunk["text"] != ""]
        chunk = chunk[chunk["anchor_year_group"].isin(MIMIC4_GROUP_ORDER)]
        for grp, sub in chunk.groupby("anchor_year_group", sort=False):
            group_texts[grp].extend(sub["text"].tolist())

    for grp in MIMIC4_GROUP_ORDER:
        logging.info(f"  MIMIC-IV {grp}: {len(group_texts[grp])} notes loaded")

    group_rows:    dict[str, list[dict]] = {}
    group_counter: dict[str, Counter]   = {}

    for grp in MIMIC4_GROUP_ORDER:
        texts = group_texts[grp]
        if not texts:
            group_rows[grp]    = []
            group_counter[grp] = Counter()
            continue

        # Per-group sample cap
        if sample_size and len(texts) > sample_size:
            idx   = rng.choice(len(texts), size=sample_size, replace=False)
            texts = [texts[i] for i in idx]

        logging.info(f"  MIMIC-IV {grp}: processing {len(texts)} notes on {n_workers} workers…")
        rows, ctr = _parallel_features(texts, n_workers)
        group_rows[grp]    = rows
        group_counter[grp] = ctr

    return group_rows, group_counter


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def jaccard(counter: Counter, baseline_vocab: set) -> float:
    """Jaccard between top-N vocab of counter and the baseline set."""
    top_n        = {w for w, _ in counter.most_common(TOP_N_VOCAB)}
    intersection = len(top_n & baseline_vocab)
    union        = len(top_n | baseline_vocab)
    return intersection / union if union else 0.0


def _aggregate_rows(rows: list[dict], label: str, dataset: str, status: str) -> dict:
    """Aggregate per-note feature dicts into a single period summary dict."""
    n = len(rows)
    lengths    = [r["length"]          for r in rows]
    sec_counts = [r["n_sections"]      for r in rows]
    num_dens   = [r["numeric_density"] for r in rows]
    phi_dens   = [r["phi_density"]     for r in rows]

    return {
        "period_label":    label,
        "dataset":         dataset,
        "window_status":   status,
        "n_notes":         n,
        "mean_length":     float(np.mean(lengths)),
        "std_length":      float(np.std(lengths)),
        "section_rate":    sum(1 for s in sec_counts if s >= STRUCTURED_THRESHOLD) / n,
        "mean_sections":   float(np.mean(sec_counts)),
        "numeric_density": float(np.mean(num_dens)),
        "phi_density":     float(np.mean(phi_dens)),
        "jaccard":         1.0,  # overwritten by caller for non-baseline rows
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PANEL_TITLES = [
    "Mean Note Length (chars)",
    "Structured Section Rate",
    "Vocabulary Jaccard vs MIMIC-III",
    "Numeric Token Density",
    "PHI Placeholder Density",
    "Surface Feature Drift Heatmap",
]


def _draw_bar_panel(ax, all_rows: list[dict], feat: str, title: str, add_error_bars: bool = False):
    x_pos   = list(range(len(all_rows)))
    x_lbl   = [r["period_label"] for r in all_rows]
    values  = [float(r[feat]) for r in all_rows]
    colors  = ["steelblue" if r["dataset"] == "MIMIC-III" else "darkorange" for r in all_rows]

    for i, r in enumerate(all_rows):
        status = r["window_status"]
        if status in STATUS_SHADING:
            color, alpha, _ = STATUS_SHADING[status]
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=alpha, zorder=0)

    if add_error_bars:
        stds = [r.get("std_length", 0) for r in all_rows]
        ax.bar(x_pos, values, color=colors, yerr=stds, capsize=4, zorder=2)
    else:
        ax.bar(x_pos, values, color=colors, zorder=2)

    ax.axhline(values[0], color="steelblue", linestyle="--", linewidth=1.2, zorder=3, alpha=0.7)
    ax.axvline(x=0.5, color="black", linestyle=":", linewidth=1.0, zorder=4)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max != y_min else 1.0
    for i, r in enumerate(all_rows):
        status = r["window_status"]
        if status in STATUS_SHADING:
            _, _, lbl  = STATUS_SHADING[status]
            ann_color  = "grey" if status == "overlap" else "red"
            ax.text(i, values[i] + y_range * 0.02, lbl,
                    ha="center", va="bottom", fontsize=7, color=ann_color, zorder=5)

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbl, rotation=45, ha="right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _draw_heatmap_panel(ax, all_rows: list[dict]):
    heatmap_values = np.array([[r[f] for r in all_rows] for f in FEATURE_NAMES], dtype=float)

    z = scipy.stats.zscore(heatmap_values, axis=1, nan_policy="omit")
    z = np.nan_to_num(z, nan=0.0)

    for i, feat in enumerate(FEATURE_NAMES):
        if np.std(heatmap_values[i]) == 0:
            logging.warning(f"Zero-variance feature '{feat}' — z-scores will be 0.")

    col_labels = []
    for r in all_rows:
        status = r["window_status"]
        if status in STATUS_SHADING:
            _, _, ann = STATUS_SHADING[status]
            col_labels.append(f"{r['period_label']}\n({ann})")
        else:
            col_labels.append(r["period_label"])

    sns.heatmap(z, ax=ax, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                annot=True, fmt=".2f", linewidths=0.5,
                xticklabels=col_labels, yticklabels=FEATURE_LABELS)
    ax.axvline(x=1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_title(PANEL_TITLES[5], fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=9, rotation=0)


def build_figure(all_rows: list[dict], output_path: str):
    feat_keys = ["mean_length", "section_rate", "jaccard", "numeric_density", "phi_density"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 14))
    axes = axes.flatten()

    for idx, (feat, title) in enumerate(zip(feat_keys, PANEL_TITLES[:5])):
        _draw_bar_panel(axes[idx], all_rows, feat, title, add_error_bars=(idx == 0))

    _draw_heatmap_panel(axes[5], all_rows)

    fig.tight_layout(pad=3.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_rows: list[dict]):
    cols = ["period_label", "dataset", "window_status", "n_notes",
            "mean_length", "section_rate", "jaccard", "numeric_density", "phi_density"]
    header = " | ".join(f"{c:>18}" for c in cols)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in all_rows:
        print(" | ".join(f"{str(r.get(c, ''))[:18]:>18}" for c in cols))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute surface feature drift between MIMIC-III and MIMIC-IV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",  default="outputs")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Max notes per period (dev cap; None = all)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--chunksize",   type=int, default=10000)
    p.add_argument("--workers",     type=int,
                   default=max(1, multiprocessing.cpu_count() - 1),
                   help="Parallel worker processes for feature computation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "tokenizers", "torch", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Using {args.workers} worker processes, chunksize={args.chunksize}")

    mimic3_path = os.environ.get("MIMIC3_DB_PATH", "")
    mimic4_path = os.environ.get("MIMIC4_DB_PATH", "")

    if not mimic3_path or not os.path.exists(mimic3_path):
        logging.error("MIMIC3_DB_PATH not set or file not found.")
        raise SystemExit(1)

    mimic4_available = bool(mimic4_path and os.path.exists(mimic4_path))
    if not mimic4_available:
        logging.warning("MIMIC4_DB_PATH not set or not found — using MIMIC-III split fallback.")

    # ------------------------------------------------------------------
    # Load + featurise
    # ------------------------------------------------------------------
    with sqlite3.connect(mimic3_path) as conn3:
        baseline_rows, baseline_counter, split_rows = _process_mimic3(
            conn3, args.chunksize, args.sample_size, rng, args.workers,
            fallback=not mimic4_available,
        )

    baseline_vocab = {w for w, _ in baseline_counter.most_common(TOP_N_VOCAB)}

    if mimic4_available:
        with sqlite3.connect(mimic4_path) as conn4:
            group_rows, group_counter = _process_mimic4(
                conn4, args.chunksize, args.sample_size, rng, args.workers,
            )
    else:
        group_rows = group_counter = None

    # ------------------------------------------------------------------
    # Build all_period_rows
    # ------------------------------------------------------------------
    if not baseline_rows:
        logging.error("No MIMIC-III baseline rows. Aborting.")
        raise SystemExit(1)

    all_period_rows: list[dict] = []

    if mimic4_available:
        base_agg = _aggregate_rows(baseline_rows, "MIMIC-III", "MIMIC-III", "baseline")
        base_agg["jaccard"] = 1.0
        all_period_rows.append(base_agg)

        for grp in MIMIC4_GROUP_ORDER:
            rows = group_rows.get(grp, [])
            if not rows:
                logging.warning(f"No notes for MIMIC-IV group '{grp}', skipping.")
                continue
            agg = _aggregate_rows(rows, grp, "MIMIC-IV", WINDOW_STATUS[grp])
            agg["jaccard"] = jaccard(group_counter[grp], baseline_vocab)
            all_period_rows.append(agg)

    else:
        base_agg = _aggregate_rows(baseline_rows, "MIMIC-III (2001-2006)", "MIMIC-III", "baseline")
        base_agg["jaccard"] = 1.0
        all_period_rows.append(base_agg)

        if split_rows:
            split_ctr = Counter()
            # counter not returned separately in fallback — recompute from words
            # (split_rows have no _words; reuse _parallel_features result which merged into _)
            # We need the shift counter — fetch it by reprocessing
            # Instead: pass split_counter through (see fallback path returns it as second element)
            # Current fallback returns (split_rows, _) — the _ counter is discarded. Fix:
            # We simply aggregate without jaccard for the shift in fallback mode.
            shift_agg = _aggregate_rows(split_rows, "MIMIC-III (2007-2012)", "MIMIC-III", "analysis")
            shift_agg["jaccard"] = float("nan")  # no cross-dataset Jaccard in fallback
            shift_agg["source"]  = "MIMIC-III-split"
            all_period_rows.append(shift_agg)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_cols = ["period_label", "dataset", "window_status", "n_notes",
                "mean_length", "std_length", "section_rate", "mean_sections",
                "numeric_density", "phi_density", "jaccard"]
    df_out = pd.DataFrame(all_period_rows)
    for c in csv_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[csv_cols + [c for c in df_out.columns if c not in csv_cols]]

    csv_path = os.path.join(args.output_dir, "surface_features_by_period.csv")
    df_out.to_csv(csv_path, index=False)
    logging.info(f"CSV saved → {csv_path}")

    # ------------------------------------------------------------------
    # Visualize + summary
    # ------------------------------------------------------------------
    png_path = os.path.join(args.output_dir, "surface_feature_analysis.png")
    build_figure(all_period_rows, png_path)
    _print_summary(all_period_rows)


if __name__ == "__main__":
    main()