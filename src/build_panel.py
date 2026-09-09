"""
Build data/processed/panel.parquet from data/raw/gdx_data.xlsx.

Reads the single 'GDX' sheet (header row at row 7, data from row 9 on --
see src/load_xlsx.py for how this layout was discovered), and writes a
tidy date-indexed wide table of close prices for SPY, IAU, GLD, TLT, GDX,
GDXJ. Missing markers ('#N/A' etc.) become NaN rather than being dropped,
so the row index stays a single continuous trading-day calendar shared by
all tickers.
"""

from pathlib import Path

import pandas as pd

RAW_XLSX_PATH = Path("data/raw/gdx_data.xlsx")
PANEL_PARQUET_PATH = Path("data/processed/panel.parquet")

MISSING_MARKERS = {"#N/A", "N/A", "NA", "n/a", ""}


def main():
    raw = pd.read_excel(RAW_XLSX_PATH, sheet_name="GDX", header=6)
    raw = raw.dropna(how="all")

    date_col = raw.columns[0]
    raw = raw.rename(columns={date_col: "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    panel = raw.set_index("date").sort_index()
    panel = panel.replace(list(MISSING_MARKERS), pd.NA).apply(pd.to_numeric)
    panel.index.name = "date"

    PANEL_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PARQUET_PATH)

    print(f"Wrote {PANEL_PARQUET_PATH}")
    print(f"  shape: {panel.shape}")
    print(f"  columns: {list(panel.columns)}")
    print(f"  date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"  missing values per column:\n{panel.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
