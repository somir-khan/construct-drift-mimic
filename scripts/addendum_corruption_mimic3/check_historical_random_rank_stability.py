#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

MANIFEST = Path(
    "data/judge_samples_300.csv"
)

DETAIL = {
    "2014 - 2016": Path(
        "data/addendum_sensitivity/"
        "witness_geometry_2014_2016.csv"
    ),
    "2017 - 2019": Path(
        "data/addendum_sensitivity/"
        "witness_geometry_2017_2019.csv"
    ),
}


def summarize_group(
    manifest,
    detail,
    window,
    group,
):

    historical_ids = set(
        manifest.loc[
            (
                manifest["anchor_year_group"].astype(str)
                == window
            )
            &
            (
                manifest["selection_group"]
                == group
            ),
            "hadm_id",
        ].astype(int)
    )

    rows = detail[
        detail["hadm_id"]
        .astype(int)
        .isin(historical_ids)
    ].copy()

    if len(rows) != 50:
        raise RuntimeError(
            f"{window} {group}: "
            f"expected 50 rows, got {len(rows)}"
        )

    ranks = rows[
        "rank_report_preferred"
    ]

    print()
    print(f"{group.upper()}")
    print(
        f"  n                         : "
        f"{len(rows)}"
    )

    print(
        f"  sensitivity rank min      : "
        f"{int(ranks.min())}"
    )

    print(
        f"  sensitivity rank median   : "
        f"{ranks.median():.1f}"
    )

    print(
        f"  sensitivity rank max      : "
        f"{int(ranks.max())}"
    )

    print(
        f"  in sensitivity Top-50     : "
        f"{int((ranks <= 50).sum())}/50"
    )

    print(
        f"  in sensitivity Top-100    : "
        f"{int((ranks <= 100).sum())}/50"
    )

    print(
        f"  in sensitivity Bottom-50  : "
        f"{int((ranks >= 2451).sum())}/50"
    )

    print(
        f"  in sensitivity Bottom-100 : "
        f"{int((ranks >= 2401).sum())}/50"
    )


def main():

    manifest = pd.read_csv(MANIFEST)

    print("=" * 72)
    print("HISTORICAL SAMPLE RANK STABILITY")
    print("=" * 72)

    for window, path in DETAIL.items():

        detail = pd.read_csv(path)

        print()
        print("=" * 72)
        print(window)
        print("=" * 72)

        for group in [
            "top",
            "random",
            "bottom",
        ]:
            summarize_group(
                manifest,
                detail,
                window,
                group,
            )


if __name__ == "__main__":
    main()