"""
Baseline strategy performance on TRAIN + VALIDATION (never the holdout).

Strategies:
  1. Buy-and-hold GDX
  2. Buy-and-hold SPY
  3. Naive reversal on GDX: position_t = -sign(return_{t-1}), i.e. short GDX
     for day t if yesterday's return was positive, long if it was negative,
     flat if yesterday's return was exactly zero. Held for exactly 1 day,
     no transaction costs.

Stats reported: CAGR, annualized Sharpe (rf=0), max drawdown, win rate.
Writes the comparison table to results/baselines.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_PARQUET_PATH = Path("data/processed/train.parquet")
VALIDATION_PARQUET_PATH = Path("data/processed/validation.parquet")
OUTPUT_MD_PATH = Path("results/baselines.md")

TRADING_DAYS_PER_YEAR = 252


def load_in_sample() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PARQUET_PATH)
    val = pd.read_parquet(VALIDATION_PARQUET_PATH)
    panel = pd.concat([train, val]).sort_index()
    assert panel.index.is_unique, "train/validation date ranges overlap"
    return panel


def compute_stats(returns: pd.Series) -> dict:
    """returns: daily strategy returns (may contain NaN/0 for flat days)."""
    r = returns.dropna()
    n_days = len(r)
    wealth = (1 + r).cumprod()

    total_return = wealth.iloc[-1] - 1
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (wealth.iloc[-1]) ** (1 / years) - 1

    sharpe = (r.mean() / r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if r.std() > 0 else np.nan

    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    max_drawdown = drawdown.min()

    nonzero = r[r != 0]
    win_rate = (nonzero > 0).sum() / len(nonzero) if len(nonzero) > 0 else np.nan

    return {
        "n_days": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": r.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def main():
    panel = load_in_sample()
    daily_returns = panel.pct_change()

    gdx_returns = daily_returns["GDX"].dropna()
    spy_returns = daily_returns["SPY"].dropna()

    gdx_prev_return = gdx_returns.shift(1)
    naive_position = -np.sign(gdx_prev_return)
    naive_returns = (naive_position * gdx_returns).dropna()

    strategies = {
        "Buy-and-hold GDX": gdx_returns,
        "Buy-and-hold SPY": spy_returns,
        "Naive reversal on GDX (short if r_{t-1}>0, long if r_{t-1}<0, 1-day hold, no costs)": naive_returns,
    }

    stats = {name: compute_stats(r) for name, r in strategies.items()}

    print(f"In-sample period: {panel.index.min().date()} to {panel.index.max().date()} "
          f"({len(panel)} rows: train + validation)")
    for name, s in stats.items():
        print(f"\n{name}")
        for k, v in s.items():
            print(f"  {k}: {v}")

    short_names = {
        "Buy-and-hold GDX": "Buy-and-hold GDX",
        "Buy-and-hold SPY": "Buy-and-hold SPY",
        "Naive reversal on GDX (short if r_{t-1}>0, long if r_{t-1}<0, 1-day hold, no costs)": "Naive reversal on GDX",
    }

    lines = []
    lines.append("# Baseline Strategy Comparison — Train + Validation\n")
    lines.append(
        f"In-sample period: **{panel.index.min().date()} to {panel.index.max().date()}** "
        f"(train + validation combined, per `results/split_dates.json`; holdout untouched).\n"
    )
    lines.append(
        "No transaction costs, slippage, or borrow costs are modeled anywhere in this table.\n"
    )
    lines.append("| Strategy | N days | Total return | CAGR | Ann. vol | Sharpe | Max drawdown | Win rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        short = short_names[name]
        lines.append(
            f"| {short} | {s['n_days']} | {s['total_return']*100:.1f}% | {s['cagr']*100:.1f}% | "
            f"{s['annualized_vol']*100:.1f}% | {s['sharpe']:.2f} | {s['max_drawdown']*100:.1f}% | "
            f"{s['win_rate']*100:.1f}% |"
        )
    lines.append("")
    lines.append("**Naive reversal rule definition:** position for day *t* is "
                  "`-sign(return_{t-1})` on GDX — short 1 unit if GDX's prior-day return was "
                  "positive, long 1 unit if negative, flat if exactly zero. Position is held for "
                  "exactly one day (re-evaluated daily) and reset each day; no leverage, no costs.")
    lines.append("")
    lines.append(f"CAGR/Sharpe/vol annualized using {TRADING_DAYS_PER_YEAR} trading days/year. "
                  "Win rate excludes flat (exactly zero-return) days from the denominator.")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
