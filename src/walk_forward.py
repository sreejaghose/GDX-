"""
Walk-forward validation of the GDX trailing-return threshold rule, on
train+validation combined (the holdout split is never touched here or
anywhere else in this project).

Design decisions (the request left these underspecified; see
results/walk_forward/methodology.md for the full rationale):

  - Rule form: at date t, trailing_return_N[t] = GDX[t]/GDX[t-N] - 1
    (reusing the exact same causal calculation as src/features.py). The
    threshold T is a DEADZONE: no position is taken unless
    |trailing_return_N[t]| >= T. When triggered, DIRECTION (momentum:
    bet with the move, or reversal: bet against it) decides the sign.
  - What gets "refit" each quarter: direction. A fixed rule with no free
    parameter wouldn't need refitting; direction is the one thing that can
    plausibly change quarter to quarter, and Step 9's in-sample fit found
    reversal dominant on the full train split -- refitting lets that be
    checked and potentially overridden as more data arrives.
  - Refit criterion: whichever direction has the higher realized Sharpe
    ratio within the expanding fit window (using only causal, in-window
    data) wins for the next quarter. Falls back to reversal (the Step 9
    in-sample finding) if neither direction has enough triggered trades
    in-window to estimate a Sharpe ratio.
  - Realized returns: uses the SAME entry/exit timing as
    results/timing_spec.md and src/naive_baseline.py -- signal known at
    close t, entry at close t+1, exit at close t+2 -- not the Step 7
    same-day-to-next-day ML label. This keeps the backtest consistent
    with the project's one established, lookahead-verified execution
    convention.
  - Expanding window, calendar quarters: quarter i's prediction uses only
    data from quarters strictly before quarter i. The first one or two
    quarters are pure warm-up (skipped -- not enough history to compute
    even the shortest lookback, let alone fit a direction).

Outputs:
  results/walk_forward_grid.csv          -- one row per (lookback, threshold)
                                             combo, sorted by walk-forward Sharpe
  results/walk_forward/equity_curves.csv -- stitched daily OOS returns per combo
  results/walk_forward/direction_log.csv -- refit direction per combo per quarter
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import load_train_val_panel

LOOKBACKS = [5, 10, 20, 60]
THRESHOLDS = [0.005, 0.01, 0.015, 0.02]

MIN_FIT_TRADES = 10        # min triggered signals in-window to trust a direction's Sharpe
WARMUP_ROWS_BUFFER = 20    # extra rows beyond the lookback before the first fit is attempted

OUT_DIR = Path("results")
WF_DIR = OUT_DIR / "walk_forward"
GRID_CSV_PATH = OUT_DIR / "walk_forward_grid.csv"
EQUITY_CURVES_PATH = WF_DIR / "equity_curves.csv"
DIRECTION_LOG_PATH = WF_DIR / "direction_log.csv"

TRADING_DAYS_PER_YEAR = 252


def build_signal_frame(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    gdx = panel["GDX"]
    trailing_return = gdx / gdx.shift(lookback) - 1  # causal: t, t-lookback only

    entry_price = gdx.shift(-1)   # P[t+1], per results/timing_spec.md
    exit_price = gdx.shift(-2)    # P[t+2]
    target_return = exit_price / entry_price - 1

    return pd.DataFrame({
        "trailing_return": trailing_return,
        "target_return": target_return,
    }, index=panel.index)


def raw_signal_for_threshold(trailing_return: pd.Series, threshold: float) -> pd.Series:
    """+1/-1 by sign of the move once it clears the deadzone, else 0 (flat)."""
    triggered = trailing_return.abs() >= threshold
    return np.sign(trailing_return).where(triggered, 0.0)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def score_direction(fit_df: pd.DataFrame, threshold: float, direction: int) -> tuple[float, int]:
    raw_signal = raw_signal_for_threshold(fit_df["trailing_return"], threshold)
    strat_return = direction * raw_signal * fit_df["target_return"]
    traded = strat_return[raw_signal != 0].dropna()
    if len(traded) < MIN_FIT_TRADES:
        return np.nan, len(traded)
    return sharpe_ratio(traded), len(traded)


def choose_direction(fit_df: pd.DataFrame, threshold: float) -> tuple[int, dict]:
    sharpe_momentum, n_momentum = score_direction(fit_df, threshold, +1)
    sharpe_reversal, n_reversal = score_direction(fit_df, threshold, -1)

    info = {
        "sharpe_momentum": sharpe_momentum, "n_trades_momentum": n_momentum,
        "sharpe_reversal": sharpe_reversal, "n_trades_reversal": n_reversal,
    }

    if np.isnan(sharpe_momentum) and np.isnan(sharpe_reversal):
        info["fallback_used"] = True
        return -1, info  # default to reversal, per Step 9's in-sample finding

    info["fallback_used"] = False
    if np.isnan(sharpe_reversal) or (not np.isnan(sharpe_momentum) and sharpe_momentum > sharpe_reversal):
        return +1, info
    return -1, info


def run_walk_forward(panel: pd.DataFrame, lookback: int, threshold: float) -> tuple[pd.Series, list[dict]]:
    signal_frame = build_signal_frame(panel, lookback)
    quarters = sorted(signal_frame.index.to_period("Q").unique())
    quarter_of = signal_frame.index.to_period("Q")

    min_fit_rows = lookback + WARMUP_ROWS_BUFFER

    oos_returns = []
    direction_log = []

    for i in range(1, len(quarters)):
        predict_quarter = quarters[i]
        fit_mask = quarter_of < predict_quarter
        predict_mask = quarter_of == predict_quarter

        fit_df = signal_frame.loc[fit_mask].dropna(subset=["trailing_return", "target_return"])
        predict_df = signal_frame.loc[predict_mask]

        if len(fit_df) < min_fit_rows:
            continue  # not enough expanding-window history yet -- warm-up

        # Zero-lookahead check: every fit-window date must be strictly
        # earlier than every predict-window date.
        assert fit_df.index.max() < predict_df.index.min(), (
            f"lookahead: fit window reaches into predict quarter {predict_quarter}"
        )

        direction, info = choose_direction(fit_df, threshold)

        raw_signal = raw_signal_for_threshold(predict_df["trailing_return"], threshold)
        strat_return = direction * raw_signal * predict_df["target_return"]

        oos_returns.append(strat_return)
        direction_log.append({
            "lookback": lookback, "threshold": threshold, "quarter": str(predict_quarter),
            "direction": "momentum" if direction == 1 else "reversal",
            "fit_window_rows": len(fit_df), **info,
        })

    if not oos_returns:
        return pd.Series(dtype=float), direction_log

    return pd.concat(oos_returns).sort_index(), direction_log


def summarize(oos_returns: pd.Series, lookback: int, threshold: float) -> dict:
    r = oos_returns.dropna()
    traded = r[r != 0]

    cum_return = (1 + r.fillna(0)).prod() - 1 if len(r) else np.nan
    max_drawdown = np.nan
    if len(r):
        equity = (1 + r.fillna(0)).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_drawdown = drawdown.min()

    return {
        "lookback": lookback,
        "threshold": threshold,
        "walk_forward_sharpe": sharpe_ratio(r),
        "n_oos_days": len(r),
        "n_trades": len(traded),
        "pct_days_in_market": len(traded) / len(r) if len(r) else np.nan,
        "hit_rate": (traded > 0).mean() if len(traded) else np.nan,
        "cumulative_return": cum_return,
        "annualized_return": r.mean() * TRADING_DAYS_PER_YEAR if len(r) else np.nan,
        "annualized_vol": r.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(r) else np.nan,
        "max_drawdown": max_drawdown,
        "oos_start": r.index.min().date().isoformat() if len(r) else None,
        "oos_end": r.index.max().date().isoformat() if len(r) else None,
    }


def main():
    panel = load_train_val_panel()
    print(f"train+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    grid_rows = []
    all_equity_curves = {}
    all_direction_logs = []

    for lookback in LOOKBACKS:
        for threshold in THRESHOLDS:
            oos_returns, direction_log = run_walk_forward(panel, lookback, threshold)
            summary = summarize(oos_returns, lookback, threshold)
            grid_rows.append(summary)
            all_direction_logs.extend(direction_log)

            combo_key = f"lookback_{lookback}d_threshold_{threshold*100:.1f}pct"
            all_equity_curves[combo_key] = oos_returns

            print(f"lookback={lookback:>2}d threshold={threshold:.1%}: "
                  f"Sharpe={summary['walk_forward_sharpe']:.3f}, "
                  f"n_oos_days={summary['n_oos_days']}, n_trades={summary['n_trades']}, "
                  f"cum_return={summary['cumulative_return']:.1%}")

    grid_df = pd.DataFrame(grid_rows).sort_values("walk_forward_sharpe", ascending=False, na_position="last")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(GRID_CSV_PATH, index=False)
    print(f"\nWrote {GRID_CSV_PATH}")

    WF_DIR.mkdir(parents=True, exist_ok=True)
    equity_df = pd.DataFrame(all_equity_curves)
    equity_df.to_csv(EQUITY_CURVES_PATH)
    print(f"Wrote {EQUITY_CURVES_PATH}")

    direction_log_df = pd.DataFrame(all_direction_logs)
    direction_log_df.to_csv(DIRECTION_LOG_PATH, index=False)
    print(f"Wrote {DIRECTION_LOG_PATH}")

    print(f"\nTop 5 by walk-forward Sharpe:\n{grid_df.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
