"""
Chronologically split data/processed/panel.parquet into train / validation /
holdout, and freeze the resulting date boundaries in results/split_dates.json
so the split can't silently drift on a future re-run.

Split order (row-count based, not calendar-day based):
  1. holdout   = last 20% of rows (most recent dates) -- written to
                 data/processed/DO_NOT_TOUCH_holdout.parquet and never
                 used again by anything else in this pipeline.
  2. in-sample = the remaining first 80% of rows.
  3. train     = first 60% of the in-sample rows.
  4. validation = remaining 40% of the in-sample rows.

Net shares of the full panel: train 48%, validation 32%, holdout 20%.
"""

import json
from pathlib import Path

import pandas as pd

PANEL_PARQUET_PATH = Path("data/processed/panel.parquet")
TRAIN_PARQUET_PATH = Path("data/processed/train.parquet")
VALIDATION_PARQUET_PATH = Path("data/processed/validation.parquet")
HOLDOUT_PARQUET_PATH = Path("data/processed/DO_NOT_TOUCH_holdout.parquet")
SPLIT_DATES_JSON_PATH = Path("results/split_dates.json")

HOLDOUT_FRACTION = 0.20
TRAIN_FRACTION_OF_IN_SAMPLE = 0.60


def describe(name, df):
    start = df.index.min().date().isoformat()
    end = df.index.max().date().isoformat()
    print(f"  {name}: {len(df)} rows, {start} to {end}")
    return {"start_date": start, "end_date": end, "n_rows": len(df)}


def main():
    panel = pd.read_parquet(PANEL_PARQUET_PATH).sort_index()
    n_total = len(panel)

    n_in_sample = round(n_total * (1 - HOLDOUT_FRACTION))
    in_sample = panel.iloc[:n_in_sample]
    holdout = panel.iloc[n_in_sample:]

    n_train = round(len(in_sample) * TRAIN_FRACTION_OF_IN_SAMPLE)
    train = in_sample.iloc[:n_train]
    validation = in_sample.iloc[n_train:]

    for path, df in [
        (TRAIN_PARQUET_PATH, train),
        (VALIDATION_PARQUET_PATH, validation),
        (HOLDOUT_PARQUET_PATH, holdout),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    print(f"Full panel: {n_total} rows, {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"\nSplit boundaries (row-count based: "
          f"holdout={HOLDOUT_FRACTION:.0%} of total, "
          f"train={TRAIN_FRACTION_OF_IN_SAMPLE:.0%} of the remaining in-sample portion):")
    train_info = describe("train      ", train)
    val_info = describe("validation ", validation)
    holdout_info = describe("holdout    ", holdout)

    split_dates = {
        "source_panel": str(PANEL_PARQUET_PATH),
        "split_method": "chronological_row_count",
        "holdout_fraction_of_total": HOLDOUT_FRACTION,
        "train_fraction_of_in_sample": TRAIN_FRACTION_OF_IN_SAMPLE,
        "n_total_rows": n_total,
        "train": {**train_info, "path": str(TRAIN_PARQUET_PATH)},
        "validation": {**val_info, "path": str(VALIDATION_PARQUET_PATH)},
        "holdout": {
            **holdout_info,
            "path": str(HOLDOUT_PARQUET_PATH),
            "note": "DO NOT TOUCH: reserved out-of-sample evaluation set. "
                    "Never load, inspect, or fit anything on this file until "
                    "the final, single out-of-sample evaluation.",
        },
    }

    SPLIT_DATES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DATES_JSON_PATH, "w") as f:
        json.dump(split_dates, f, indent=2)

    print(f"\nWrote frozen split boundaries to {SPLIT_DATES_JSON_PATH}")


if __name__ == "__main__":
    main()
