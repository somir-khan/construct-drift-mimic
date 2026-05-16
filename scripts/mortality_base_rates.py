"""
scripts/mortality_base_rates.py
Appendix B.2 — Population-Level Audit Findings (Mortality Base Rates)

PURPOSE
-------
Reproduce the five mortality base-rate statistics reported in Appendix B.2 of
the TMIS paper.  These numbers are NOT produced by evaluate.py or
detect_drift.py; they come from standalone SQL queries against the SQLite
MIMIC-III/IV databases and were previously run ad-hoc.  This file is the
canonical provenance record so the numbers can be re-verified at any time.

LAST VERIFIED: 2026-04-20

PAPER TARGETS
-------------
  MIMIC-III  note-linked mortality  : 10.5%  n = 49,038  (NEWBORN excluded)
  MIMIC-IV   note-linked mortality  :  2.4%  n ≈ 256,341
  MIMIC-IV   population mortality   :  2.2%  = 11,801 / 546,028
  MIMIC-III  high-acuity restricted : 12.0%  (EMERGENCY + URGENT)
  MIMIC-IV   high-acuity restricted :  3.6%  (EW EMER. + DIRECT EMER. + URGENT)

  Note: an earlier draft of the paper reported 12.7% and 3.7% for the
  high-acuity cohorts from an ad-hoc query whose exact cohort definition
  could not be recovered. B.2 and the PAPER_RATE constants here were
  updated on 2026-04-20 to match the reproducible query below.

METHODOLOGICAL CHOICES
----------------------
  - Unit of analysis: distinct admissions (HADM_ID), NOT raw note rows.
  - NEWBORN excluded from all MIMIC-III cohorts: NEWBORN admissions represent
    obstetric/neonatal episodes with near-zero in-hospital mortality and would
    deflate the headline rate; MIMIC-IV has no equivalent single NEWBORN
    category (those cases are spread across other admission types).
  - ISERROR filter uses the SQLite empty-string-safe pattern:
      (ISERROR IS NULL OR ISERROR != '1')
    MIMIC-III SQLite exports store the field as '' rather than NULL, so a plain
    IS NULL check silently drops all notes.
  - CTE pattern (WITH note_linked_hadm AS (...)) is used instead of EXISTS
    subqueries.  EXISTS-based rewrites require a full NOTEEVENTS scan (no index
    on NOTEEVENTS.HADM_ID in these exports) and take 10+ minutes; the CTE
    aggregates distinct HADM_IDs once and joins in O(admissions).
  - MIMIC-IV table names use slash-format identifiers that must be
    double-quoted in SQL: "hosp/admissions", "note/discharge".

Usage:
    python scripts/mortality_base_rates.py
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()
MIMIC3_DB_PATH = os.getenv("MIMIC3_DB_PATH")
MIMIC4_DB_PATH = os.getenv("MIMIC4_DB_PATH")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def mimic3_note_linked_mortality(conn: sqlite3.Connection) -> dict:
    """
    MIMIC-III headline mortality: admissions that have at least one discharge
    summary note (after ISERROR filter), NEWBORN excluded.
    Paper target: 10.5% at n = 49,083.
    """
    query = """
    -- Step 1: collect distinct HADM_IDs that have a usable discharge summary.
    -- The ISERROR guard uses IS NULL OR != '1' because the SQLite export stores
    -- error-flagged notes as '' (empty string) rather than NULL.
    WITH note_linked_hadm AS (
        SELECT DISTINCT HADM_ID
        FROM NOTEEVENTS
        WHERE CATEGORY = 'Discharge summary'
          AND (ISERROR IS NULL OR ISERROR != '1')
          AND HADM_ID IS NOT NULL
    )
    SELECT
        COUNT(*)                                          AS n_admissions,
        SUM(a.HOSPITAL_EXPIRE_FLAG)                      AS n_deaths,
        ROUND(
            CAST(SUM(a.HOSPITAL_EXPIRE_FLAG) AS REAL)
            / COUNT(*), 4
        )                                                AS rate
    FROM ADMISSIONS a
    JOIN note_linked_hadm nl ON a.HADM_ID = nl.HADM_ID
    -- Exclude NEWBORN: neonatal/obstetric episodes with near-zero mortality
    -- would deflate the headline rate.  No equivalent single category exists
    -- in MIMIC-IV, so this filter is MIMIC-III-specific.
    WHERE a.ADMISSION_TYPE != 'NEWBORN'
    """
    row = conn.execute(query).fetchone()
    n_admissions, n_deaths, rate = row
    log.info(
        "MIMIC-III note-linked (NEWBORN excl.) | n=%d | deaths=%d | rate=%.4f",
        n_admissions, n_deaths, rate,
    )
    return {
        "rate":         rate,
        "n_admissions": n_admissions,
        "n_deaths":     n_deaths,
        "query_label":  "mimic3_note_linked",
    }


def mimic4_note_linked_mortality(conn: sqlite3.Connection) -> dict:
    """
    MIMIC-IV mortality for admissions that have at least one discharge note.
    Joins "hosp/admissions" to "note/discharge" on hadm_id.
    Paper target: 2.4% at n ≈ 256,341.
    """
    query = """
    -- Collect distinct hadm_ids from the MIMIC-IV discharge note table.
    -- MIMIC-IV table names use slash-format identifiers and must be
    -- double-quoted; "note/discharge" is the canonical note table here.
    -- MIMIC-IV discharge notes do not carry an ISERROR column; no filter needed.
    WITH note_linked_hadm AS (
        SELECT DISTINCT hadm_id
        FROM "note/discharge"
        WHERE hadm_id IS NOT NULL
    )
    SELECT
        COUNT(*)                                          AS n_admissions,
        SUM(a.hospital_expire_flag)                      AS n_deaths,
        ROUND(
            CAST(SUM(a.hospital_expire_flag) AS REAL)
            / COUNT(*), 4
        )                                                AS rate
    FROM "hosp/admissions" a
    JOIN note_linked_hadm nl ON a.hadm_id = nl.hadm_id
    """
    row = conn.execute(query).fetchone()
    n_admissions, n_deaths, rate = row
    log.info(
        "MIMIC-IV note-linked | n=%d | deaths=%d | rate=%.4f",
        n_admissions, n_deaths, rate,
    )
    return {
        "rate":         rate,
        "n_admissions": n_admissions,
        "n_deaths":     n_deaths,
        "query_label":  "mimic4_note_linked",
    }


def mimic4_population_mortality(conn: sqlite3.Connection) -> dict:
    """
    MIMIC-IV population-level mortality: all rows in "hosp/admissions",
    no note-linkage filter.
    Paper target: 2.2% = 11,801 / 546,028.
    """
    query = """
    -- No note join — full admissions table to get the population denominator.
    SELECT
        COUNT(*)                                          AS n_admissions,
        SUM(hospital_expire_flag)                        AS n_deaths,
        ROUND(
            CAST(SUM(hospital_expire_flag) AS REAL)
            / COUNT(*), 4
        )                                                AS rate
    FROM "hosp/admissions"
    """
    row = conn.execute(query).fetchone()
    n_admissions, n_deaths, rate = row
    log.info(
        "MIMIC-IV population | n=%d | deaths=%d | rate=%.4f",
        n_admissions, n_deaths, rate,
    )
    return {
        "rate":         rate,
        "n_admissions": n_admissions,
        "n_deaths":     n_deaths,
        "query_label":  "mimic4_population",
    }


def mimic3_high_acuity_mortality(conn: sqlite3.Connection) -> dict:
    """
    MIMIC-III mortality restricted to high-acuity admission types:
    EMERGENCY and URGENT only (NEWBORN still excluded by the type filter).
    Paper target: 12.0%.
    """
    query = """
    WITH note_linked_hadm AS (
        SELECT DISTINCT HADM_ID
        FROM NOTEEVENTS
        WHERE CATEGORY = 'Discharge summary'
          AND (ISERROR IS NULL OR ISERROR != '1')
          AND HADM_ID IS NOT NULL
    )
    SELECT
        COUNT(*)                                          AS n_admissions,
        SUM(a.HOSPITAL_EXPIRE_FLAG)                      AS n_deaths,
        ROUND(
            CAST(SUM(a.HOSPITAL_EXPIRE_FLAG) AS REAL)
            / COUNT(*), 4
        )                                                AS rate
    FROM ADMISSIONS a
    JOIN note_linked_hadm nl ON a.HADM_ID = nl.HADM_ID
    -- Restrict to emergency/urgent only; NEWBORN is implicitly excluded because
    -- it is not in this list.  ELECTIVE is excluded to isolate high-acuity risk.
    WHERE a.ADMISSION_TYPE IN ('EMERGENCY', 'URGENT')
    """
    row = conn.execute(query).fetchone()
    n_admissions, n_deaths, rate = row
    log.info(
        "MIMIC-III high-acuity (EMERGENCY+URGENT) | n=%d | deaths=%d | rate=%.4f",
        n_admissions, n_deaths, rate,
    )
    return {
        "rate":         rate,
        "n_admissions": n_admissions,
        "n_deaths":     n_deaths,
        "query_label":  "mimic3_high_acuity",
    }


def mimic4_high_acuity_mortality(conn: sqlite3.Connection) -> dict:
    """
    MIMIC-IV mortality restricted to high-acuity admission types.
    MIMIC-IV uses more granular type labels; the nearest equivalents to
    MIMIC-III's EMERGENCY+URGENT are 'EW EMER.', 'DIRECT EMER.', and 'URGENT'.
    Paper target: 3.6%.
    """
    query = """
    WITH note_linked_hadm AS (
        SELECT DISTINCT hadm_id
        FROM "note/discharge"
        WHERE hadm_id IS NOT NULL
    )
    SELECT
        COUNT(*)                                          AS n_admissions,
        SUM(a.hospital_expire_flag)                      AS n_deaths,
        ROUND(
            CAST(SUM(a.hospital_expire_flag) AS REAL)
            / COUNT(*), 4
        )                                                AS rate
    FROM "hosp/admissions" a
    JOIN note_linked_hadm nl ON a.hadm_id = nl.hadm_id
    -- MIMIC-IV admission_type vocabulary differs from MIMIC-III.
    -- 'EW EMER.' = emergency-ward emergency (closest to MIMIC-III EMERGENCY).
    -- 'DIRECT EMER.' = direct-admit emergency.
    -- 'URGENT' = urgent (same label as MIMIC-III URGENT).
    -- 'ELECTIVE' and 'OBSERVATION ADMIT' are intentionally excluded.
    WHERE a.admission_type IN ('EW EMER.', 'DIRECT EMER.', 'URGENT')
    """
    row = conn.execute(query).fetchone()
    n_admissions, n_deaths, rate = row
    log.info(
        "MIMIC-IV high-acuity (EW EMER.+DIRECT EMER.+URGENT) | n=%d | deaths=%d | rate=%.4f",
        n_admissions, n_deaths, rate,
    )
    return {
        "rate":         rate,
        "n_admissions": n_admissions,
        "n_deaths":     n_deaths,
        "query_label":  "mimic4_high_acuity",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MIMIC3_DB_PATH or not MIMIC4_DB_PATH:
        log.error("MIMIC3_DB_PATH and MIMIC4_DB_PATH must both be set in .env")
        sys.exit(1)

    conn3 = sqlite3.connect(MIMIC3_DB_PATH)
    conn4 = sqlite3.connect(MIMIC4_DB_PATH)
    log.info("Connected to MIMIC-III: %s", MIMIC3_DB_PATH)
    log.info("Connected to MIMIC-IV:  %s", MIMIC4_DB_PATH)

    try:
        r_m3_nl  = mimic3_note_linked_mortality(conn3)
        r_m4_nl  = mimic4_note_linked_mortality(conn4)
        r_m4_pop = mimic4_population_mortality(conn4)
        r_m3_ha  = mimic3_high_acuity_mortality(conn3)
        r_m4_ha  = mimic4_high_acuity_mortality(conn4)
    finally:
        conn3.close()
        conn4.close()

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    PAPER_RATE = {
    "mimic3_note_linked": 0.106,
    "mimic4_note_linked": 0.025,   
    "mimic4_population":  0.022,
    "mimic3_high_acuity": 0.120,  
    "mimic4_high_acuity": 0.036,   
    }
    # Only the two cohorts whose n is explicitly stated in the paper.
    PAPER_N = {
        "mimic3_note_linked": 49083,
        "mimic4_population":  546028,
    }

    rows = [r_m3_nl, r_m4_nl, r_m4_pop, r_m3_ha, r_m4_ha]

    print()
    print("=" * 76)
    print("  APPENDIX B.2  MORTALITY BASE RATES")
    print("=" * 76)
    fmt = "  {:<35s}  {:>8s}  {:>8s}  {:>9s}  {}"
    print(fmt.format("Cohort", "Computed", "Paper", "n_adm", "n_deaths  flags"))
    print("  " + "-" * 72)
    for r in rows:
        label    = r["query_label"]
        computed = r["rate"]
        paper    = PAPER_RATE[label]
        flags    = ""

        if abs(computed - paper) > 0.005:
            flags += "!"  # rate deviates by more than 0.5 pp

        if label in PAPER_N:
            paper_n   = PAPER_N[label]
            tolerance = max(100, paper_n * 0.01)
            if abs(r["n_admissions"] - paper_n) > tolerance:
                flags += "#"  # n_admissions deviates by more than 1% or 100 rows

        print(fmt.format(
            label,
            f"{computed:.3f}",
            f"{paper:.3f}",
            f"{r['n_admissions']:,}",
            f"{r['n_deaths']:,}  {flags}".strip(),
        ))

    print("=" * 76)
    print("  Flags:  ! rate delta > 0.005   # n_admissions delta > max(100, 1% of paper n)")
    print()

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    results_path = os.path.join(outputs_dir, "mortality_base_rates.json")

    payload = {
        "verified_at":            datetime.now(timezone.utc).isoformat(),
        "mimic3_note_linked":     {k: v for k, v in r_m3_nl.items()  if k != "query_label"},
        "mimic4_note_linked":     {k: v for k, v in r_m4_nl.items()  if k != "query_label"},
        "mimic4_population":      {k: v for k, v in r_m4_pop.items() if k != "query_label"},
        "mimic3_high_acuity":     {k: v for k, v in r_m3_ha.items()  if k != "query_label"},
        "mimic4_high_acuity":     {k: v for k, v in r_m4_ha.items()  if k != "query_label"},
        "paper_appendix_section": "B.2 (Population-Level Audit Findings)",
        "paper_latex_label":      "appendix:mortality",
    }

    with open(results_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Results saved -> %s", results_path)


if __name__ == "__main__":
    main()
