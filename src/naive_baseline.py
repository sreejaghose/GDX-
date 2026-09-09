"""
Naive reversal baseline, built strictly against results/timing_spec.md.

Signal:  reversal (contrarian) on the prior-period close-to-close return.
  prior_return_t = P[t] / P[t-1] - 1        (known at close t)
  signal_t       = -sign(prior_return_t)    (known at close t: +1 long,
                                              -1 short, 0 flat)

Timing (per results/timing_spec.md, not close-to-close by default):
  entry_date_t = t+1  (close of day after the signal is known)
  exit_date_t  = t+2  (one trading day later)
  target_return_t = P[t+2] / P[t+1] - 1     (the return the position
                                              actually earns)

  strategy_return_t = signal_t * target_return_t

Runs on data/processed/train.parquet only. Verifies, row by row, that the
signal depends only on prices dated <= the signal date, and that the entry
timestamp is strictly later than the signal date (i.e. zero lookahead) --
this is not just asserted, it's checked against the actual date index.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_PARQUET_PATH = Path("data/processed/train.parquet")
OUT_DIR = Path("results/naive_baseline")
RETURNS_PARQUET_PATH = OUT_DIR / "baseline_returns.parquet"
VERIFICATION_MD_PATH = OUT_DIR / "verification.md"

TICKERS = ["SPY", "IAU", "GLD", "TLT", "GDX", "GDXJ"]


def build_baseline(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One frame per ticker, indexed by signal date t, columns:
    signal_date, entry_date, exit_date, prior_return, signal,
    target_return, strategy_return.
    """
    dates = prices.index
    n = len(dates)
    frames = {}

    for ticker in TICKERS:
        p = prices[ticker]
        prior_return = p.pct_change()               # r_t, known at close t
        signal = -np.sign(prior_return)              # known at close t

        entry_price = p.shift(-1)                    # P[t+1]
        exit_price = p.shift(-2)                      # P[t+2]
        target_return = exit_price / entry_price - 1  # realized [t+1 -> t+2]

        strategy_return = signal * target_return

        entry_date = pd.Series(dates, index=dates).shift(-1)
        exit_date = pd.Series(dates, index=dates).shift(-2)

        frames[ticker] = pd.DataFrame({
            "signal_date": dates,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "prior_return": prior_return,
            "signal": signal,
            "target_return": target_return,
            "strategy_return": strategy_return,
        }, index=dates)

    return frames


def verify_zero_lookahead(frames: dict[str, pd.DataFrame], prices: pd.DataFrame) -> list[str]:
    """Programmatic checks, not just assertions on construction. Returns a
    list of human-readable pass/fail lines for the verification report.
    """
    lines = []
    dates = list(prices.index)
    date_pos = {d: i for i, d in enumerate(dates)}

    all_ok = True
    for ticker, df in frames.items():
        valid = df.dropna(subset=["entry_date", "exit_date", "strategy_return"])

        # Check 1: entry_date is strictly later than signal_date, exit_date
        # strictly later than entry_date -- for every single row.
        entry_after_signal = (valid["entry_date"] > valid["signal_date"]).all()
        exit_after_entry = (valid["exit_date"] > valid["entry_date"]).all()

        # Check 2: entry_date/exit_date are exactly the next two rows in the
        # actual trading-day index (t+1, t+2 by position, not by calendar
        # arithmetic) -- catches off-by-one errors around missing data.
        positions_ok = True
        for signal_date, entry_date, exit_date in zip(
            valid["signal_date"], valid["entry_date"], valid["exit_date"]
        ):
            i = date_pos[signal_date]
            if i + 1 >= len(dates) or i + 2 >= len(dates):
                positions_ok = False
                break
            if dates[i + 1] != entry_date or dates[i + 2] != exit_date:
                positions_ok = False
                break

        # Check 3: prior_return (the signal input) uses only P[t] and
        # P[t-1] -- recompute independently from raw prices and compare.
        recomputed_prior = prices[ticker].pct_change()
        prior_matches = np.allclose(
            valid["prior_return"].values,
            recomputed_prior.loc[valid.index].values,
            equal_nan=True,
        )

        # Check 4: target_return uses only P[t+1] and P[t+2] -- recompute
        # independently by looking up those exact calendar dates in the raw
        # price series and compare.
        recomputed_target = (
            prices[ticker].reindex(valid["exit_date"]).values
            / prices[ticker].reindex(valid["entry_date"]).values
            - 1
        )
        target_matches = np.allclose(
            valid["target_return"].values, recomputed_target, equal_nan=True
        )

        ok = entry_after_signal and exit_after_entry and positions_ok and prior_matches and target_matches
        all_ok = all_ok and ok

        lines.append(f"### {ticker}")
        lines.append(f"- entry_date > signal_date for all rows: {entry_after_signal}")
        lines.append(f"- exit_date > entry_date for all rows: {exit_after_entry}")
        lines.append(f"- entry/exit dates are exactly t+1/t+2 by row position: {positions_ok}")
        lines.append(f"- signal recomputed independently from P[t], P[t-1] matches: {prior_matches}")
        lines.append(f"- target_return recomputed independently from P[t+1], P[t+2] matches: {target_matches}")
        lines.append(f"- **{'PASS' if ok else 'FAIL'}** ({len(valid)} usable rows)")
        lines.append("")

    lines.insert(0, f"## Zero-lookahead verification: {'ALL PASS' if all_ok else 'FAILURE DETECTED'}\n")
    return lines


def summarize_performance(frames: dict[str, pd.DataFrame]) -> list[str]:
    lines = ["## Naive baseline performance (train split, informational only)\n"]
    lines.append("| Ticker | n trades | hit rate | mean return | ann. return | ann. vol | Sharpe |")
    lines.append("|---|---|---|---|---|---|---|")
    for ticker, df in frames.items():
        r = df["strategy_return"].dropna()
        r = r[df.loc[r.index, "signal"] != 0]
        if len(r) == 0:
            continue
        hit_rate = (r > 0).mean()
        mean_r = r.mean()
        ann_r = mean_r * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_r / ann_vol if ann_vol > 0 else float("nan")
        lines.append(
            f"| {ticker} | {len(r)} | {hit_rate:.1%} | {mean_r*100:.3f}% "
            f"| {ann_r*100:.1f}% | {ann_vol*100:.1f}% | {sharpe:.2f} |"
        )
    lines.append("")
    lines.append(
        "No costs, slippage, or sizing applied -- this is a signal-timing sanity "
        "check, not a performance claim."
    )
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(TRAIN_PARQUET_PATH).sort_index()

    frames = build_baseline(prices)
    verification_lines = verify_zero_lookahead(frames, prices)
    performance_lines = summarize_performance(frames)

    combined = pd.concat(
        {ticker: df for ticker, df in frames.items()}, names=["ticker", "row"]
    ).reset_index(level="ticker")
    combined.to_parquet(RETURNS_PARQUET_PATH)
    print(f"Wrote {RETURNS_PARQUET_PATH} ({len(combined)} rows)")

    report = ["# Naive Baseline: Lookahead Verification & Performance\n",
              "Built against results/timing_spec.md. Train split only.\n"]
    report += verification_lines
    report += performance_lines

    with open(VERIFICATION_MD_PATH, "w") as f:
        f.write("\n".join(report))
    print(f"Wrote {VERIFICATION_MD_PATH}")

    print("\n".join(verification_lines))
    print("\n".join(performance_lines))


if __name__ == "__main__":
    main()
