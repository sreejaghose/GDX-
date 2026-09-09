"""
Exploratory data analysis on the TRAIN split only (data/processed/train.parquet).

Produces, in results/eda/:
  01_prices.png              -- price level, one small-multiple panel per ticker
  02_cumulative_returns.png  -- cumulative daily return, all six tickers overlaid
  03_correlation_matrix.png  -- correlation matrix of daily returns (heatmap)
  04_acf_gdx_gdxj.png        -- autocorrelation of GDX and GDXJ daily returns, 20 lags
  05_rolling_corr_gdx_gld.png -- rolling 60-day correlation, GDX vs GLD

Also prints the summary statistics used to write results/eda_summary.md, so the
prose in that file is traceable back to numbers computed here.

Never touches validation.parquet or DO_NOT_TOUCH_holdout.parquet.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

TRAIN_PARQUET_PATH = Path("data/processed/train.parquet")
EDA_DIR = Path("results/eda")

TICKERS = ["SPY", "IAU", "GLD", "TLT", "GDX", "GDXJ"]

# Fixed-order categorical palette (dataviz skill reference palette, light mode).
SERIES_COLORS = {
    "SPY": "#2a78d6",   # slot 1 blue
    "IAU": "#eb6834",   # slot 2 orange
    "GLD": "#1baf7a",   # slot 3 aqua
    "TLT": "#eda100",   # slot 4 yellow
    "GDX": "#e87ba4",   # slot 5 magenta
    "GDXJ": "#008300",  # slot 6 green
}
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#e34948", "#f0efec", "#2a78d6"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_prices(prices: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, ticker in zip(axes.flat, TICKERS):
        s = prices[ticker].dropna()
        ax.plot(s.index, s.values, color=SERIES_COLORS[ticker], linewidth=1.5)
        ax.set_title(ticker, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.tick_params(labelsize=8)
    fig.suptitle("Train split: daily close price by ticker (2006-05-22 to 2013-11-18)",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = EDA_DIR / "01_prices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cumulative_returns(returns: pd.DataFrame):
    cum = (1 + returns.fillna(0)).cumprod() - 1
    for ticker in TICKERS:
        first_valid = returns[ticker].first_valid_index()
        cum.loc[:first_valid, ticker] = np.nan

    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker in TICKERS:
        s = cum[ticker].dropna()
        ax.plot(s.index, s.values * 100, color=SERIES_COLORS[ticker], linewidth=1.6, label=ticker)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Train split: cumulative daily return by ticker", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()
    out = EDA_DIR / "02_cumulative_returns.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return cum


def plot_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(TICKERS)))
    ax.set_yticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS)
    ax.set_yticklabels(TICKERS)
    ax.grid(False)
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            val = corr.values[i, j]
            text_color = INK_PRIMARY if abs(val) < 0.6 else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Train split: daily return correlation matrix", color=INK_PRIMARY, fontsize=13, loc="left")
    fig.tight_layout()
    out = EDA_DIR / "03_correlation_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return corr


def plot_acf_gdx_gdxj(returns: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    acf_values = {}
    for ax, ticker in zip(axes, ["GDX", "GDXJ"]):
        s = returns[ticker].dropna()
        plot_acf(s, lags=20, ax=ax, color=SERIES_COLORS[ticker], vlines_kwargs={"colors": SERIES_COLORS[ticker]}, title="")
        ax.set_title(f"{ticker} daily return ACF (n={len(s)})", color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("Lag (trading days)")
        acf_values[ticker] = acf(s, nlags=20, fft=True)
    fig.suptitle("Train split: autocorrelation of daily returns, 20 lags (shaded = 95% CI)",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = EDA_DIR / "04_acf_gdx_gdxj.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return acf_values


def plot_rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    roll = returns["GDX"].rolling(window).corr(returns["GLD"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(roll.index, roll.values, color=SERIES_COLORS["GDX"], linewidth=1.3)
    ax.axhline(roll.mean(), color=INK_MUTED, linewidth=1, linestyle="--",
               label=f"mean = {roll.mean():.2f}")
    ax.axhline(0, color=GRIDLINE, linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"Rolling {window}-day correlation")
    ax.set_title(f"Train split: rolling {window}-day correlation, GDX vs GLD daily returns",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    out = EDA_DIR / "05_rolling_corr_gdx_gld.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return roll


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(TRAIN_PARQUET_PATH).sort_index()
    returns = prices.pct_change()

    plot_prices(prices)
    cum = plot_cumulative_returns(returns)
    corr = plot_correlation_matrix(returns)
    acf_values = plot_acf_gdx_gdxj(returns)
    roll = plot_rolling_correlation(returns)

    print("\n=== Summary stats for results/eda_summary.md ===")
    print("\nFinal cumulative return by ticker:")
    for ticker in TICKERS:
        print(f"  {ticker}: {cum[ticker].dropna().iloc[-1] * 100:.1f}%")

    print("\nDaily return correlation matrix:")
    print(corr.round(3).to_string())

    print("\nGDX/GLD pairwise correlation:", corr.loc["GDX", "GLD"].round(3))

    print("\nDaily return std (annualized, %):")
    for ticker in TICKERS:
        s = returns[ticker].dropna()
        print(f"  {ticker}: {s.std() * np.sqrt(252) * 100:.1f}%")

    for ticker in ["GDX", "GDXJ"]:
        vals = acf_values[ticker]
        n = returns[ticker].dropna().shape[0]
        ci = 1.96 / np.sqrt(n)
        sig_lags = [lag for lag in range(1, 21) if abs(vals[lag]) > ci]
        print(f"\n{ticker} ACF: lag-1 = {vals[1]:.3f}, 95% CI = +/-{ci:.3f}, "
              f"significant lags (1-20) = {sig_lags}")

    print(f"\nGDX/GLD rolling {60}-day correlation: "
          f"mean={roll.mean():.3f}, min={roll.min():.3f}, max={roll.max():.3f}, "
          f"std={roll.std():.3f}")
    print(f"  min at {roll.idxmin().date()}, max at {roll.idxmax().date()}")


if __name__ == "__main__":
    main()
