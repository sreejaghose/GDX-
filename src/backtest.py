"""
Custom, transparent backtest engine for the GDX trailing-return threshold
strategy (vectorbt's import is broken in this environment -- a plotly
version-compatibility crash on `import vectorbt` -- and a custom engine
keeps every accounting step inspectable, consistent with the rest of this
project's lookahead-verification discipline).

Signal: reversal (default) or momentum on trailing_return_N, with a
threshold deadzone -- the same rule used in src/naive_baseline.py and
src/walk_forward.py. `lookback`, `threshold`, and `exit_rule` are all
function arguments to run_backtest(), never hardcoded.

Timing -- market-on-close, zero lookahead (matches results/timing_spec.md):
  A decision (enter, or evaluate an exit rule) at close t is always
  EXECUTED at the close of day t+1, never at close t itself. Fixed-holding
  exits go through the exact same decide-at-t/execute-at-t+1 pattern as
  data-dependent exits (stop/take, signal-flip) -- one uniform rule for
  every exit type rather than special-casing schedule-based exits. A
  useful consequence: exit_after_n_days(0) exactly reproduces the original
  1-day-hold convention from results/timing_spec.md / src/naive_baseline.py
  (entry at close t+1, exit at close t+2, relative to the signal date t).

Costs and sizing:
  - commission_bps + commission_fixed charged on both entry and exit.
  - slippage_bps applied unfavorably to the execution price (buys pay up,
    sells/shorts receive less), on both legs.
  - Position size = position_fraction * current equity, converted to
    INTEGER shares by default (allow_fractional_shares=False) -- this
    matters a lot across the $1,000-$1,000,000 account-size range the
    engine is meant to support: small accounts round down more, leaving
    idle cash drag that this engine actually models rather than assumes
    away.

Outputs (via run_backtest -> BacktestResult): equity_curve, trade_log,
turnover (total $ and annualized ratio), and trades_per_month.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.features import load_train_val_panel

TRADING_DAYS_PER_YEAR = 252

OUT_DIR = Path("results/backtest")


# --------------------------------------------------------------------------
# Exit rules -- pluggable, passed into run_backtest() as `exit_rule`
# --------------------------------------------------------------------------

@dataclass
class ExitContext:
    entry_date: pd.Timestamp
    entry_price: float
    direction: int            # +1 long, -1 short
    current_date: pd.Timestamp
    current_price: float
    days_held: int
    current_signal: int       # today's freshly computed signal, -1/0/+1


def exit_after_n_days(n: int) -> Callable[[ExitContext], bool]:
    """Fixed holding period. n=0 reproduces results/timing_spec.md's
    original 1-day-hold convention exactly (see module docstring)."""
    def rule(ctx: ExitContext) -> bool:
        return ctx.days_held >= n
    return rule


def exit_on_signal_change() -> Callable[[ExitContext], bool]:
    """Exit as soon as today's fresh signal is no longer the same
    direction as the open position (covers both a flip and going flat)."""
    def rule(ctx: ExitContext) -> bool:
        return ctx.current_signal != ctx.direction
    return rule


def exit_on_stop_take(stop_pct: float, take_pct: float) -> Callable[[ExitContext], bool]:
    """Close-to-close stop-loss / take-profit (no intraday data available,
    so this can only be checked once per day at the close)."""
    def rule(ctx: ExitContext) -> bool:
        move = ctx.direction * (ctx.current_price / ctx.entry_price - 1)
        return move <= -abs(stop_pct) or move >= abs(take_pct)
    return rule


def exit_any(*rules: Callable[[ExitContext], bool]) -> Callable[[ExitContext], bool]:
    """Combinator: exit as soon as ANY given rule triggers."""
    def rule(ctx: ExitContext) -> bool:
        return any(r(ctx) for r in rules)
    return rule


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------

def compute_strategy_signal(prices: pd.Series, lookback: int, threshold: float,
                             direction_sign: int = -1) -> pd.Series:
    """direction_sign: -1 = reversal (default, matches the project's naive
    baseline finding), +1 = momentum. Threshold is a deadzone: no signal
    unless |trailing_return_N| >= threshold. Causal: trailing_return[t]
    uses only prices[t] and prices[t-lookback]."""
    trailing_return = prices / prices.shift(lookback) - 1
    triggered = trailing_return.abs() >= threshold
    raw_signal = np.sign(trailing_return).where(triggered, 0.0)
    return direction_sign * raw_signal


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    shares: float
    entry_notional: float
    exit_notional: float
    commission_paid: float
    slippage_cost: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    turnover_total_dollars: float
    turnover_annualized_ratio: float
    trades_per_month: float
    n_trades: int
    n_open_at_end: int
    params: dict = field(default_factory=dict)


def run_backtest(
    panel: pd.DataFrame,
    lookback: int,
    threshold: float,
    exit_rule: Callable[[ExitContext], bool],
    ticker: str = "GDX",
    account_size: float = 100_000.0,
    position_fraction: float = 1.0,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    commission_fixed: float = 0.0,
    allow_fractional_shares: bool = False,
    direction_sign: int = -1,
) -> BacktestResult:
    prices = panel[ticker].dropna().sort_index()
    dates = prices.index
    n = len(dates)

    strategy_signal = compute_strategy_signal(prices, lookback, threshold, direction_sign)

    cash = account_size
    in_position = False
    direction = 0
    shares = 0.0
    entry_price = None
    entry_date = None
    entry_index = None
    entry_commission = 0.0
    entry_notional = 0.0
    entry_slippage_cost = 0.0

    equity_records = []
    trades: list[Trade] = []

    for i in range(n - 1):  # need dates[i+1] to exist for execution
        t = dates[i]
        t_next = dates[i + 1]
        price_t = prices.loc[t]
        price_next = prices.loc[t_next]

        # Mark-to-market equity as of close t, using only price_t (today's
        # own close) -- valuation, not a trading decision, so this is not
        # a lookahead concern.
        equity_t = cash + direction * shares * price_t if in_position else cash
        equity_records.append((t, equity_t))

        if not in_position:
            sig = strategy_signal.loc[t]
            if pd.notna(sig) and sig != 0:
                d = int(sig)
                transaction_direction = d  # entering long = buy (+1), entering short = sell (-1)
                execution_price = price_next * (1 + slippage_bps / 1e4 * transaction_direction)
                quoted_price = price_next
                target_notional = position_fraction * equity_t
                sh = (target_notional / execution_price if allow_fractional_shares
                      else np.floor(target_notional / execution_price))
                if sh > 0:
                    notional = sh * execution_price
                    commission = commission_fixed + commission_bps / 1e4 * notional
                    cash += -d * notional - commission

                    direction = d
                    shares = sh
                    entry_price = execution_price
                    entry_date = t_next
                    entry_index = i + 1
                    entry_commission = commission
                    entry_notional = notional
                    entry_slippage_cost = abs(execution_price - quoted_price) * sh
                    in_position = True
        else:
            days_held = i - entry_index
            current_sig = strategy_signal.loc[t]
            ctx = ExitContext(
                entry_date=entry_date, entry_price=entry_price, direction=direction,
                current_date=t, current_price=price_t, days_held=days_held,
                current_signal=int(current_sig) if pd.notna(current_sig) else 0,
            )
            if exit_rule(ctx):
                transaction_direction = -direction  # closing = opposite trade
                execution_price = price_next * (1 + slippage_bps / 1e4 * transaction_direction)
                quoted_price = price_next
                notional = shares * execution_price
                commission = commission_fixed + commission_bps / 1e4 * notional
                cash += direction * notional - commission

                gross_pnl = direction * shares * (execution_price - entry_price)
                total_commission = entry_commission + commission
                exit_slippage_cost = abs(execution_price - quoted_price) * shares
                total_slippage_cost = entry_slippage_cost + exit_slippage_cost
                net_pnl = gross_pnl - total_commission

                trades.append(Trade(
                    entry_date=entry_date, exit_date=t_next, direction=direction,
                    entry_price=entry_price, exit_price=execution_price, shares=shares,
                    entry_notional=entry_notional, exit_notional=notional,
                    commission_paid=total_commission, slippage_cost=total_slippage_cost,
                    gross_pnl=gross_pnl, net_pnl=net_pnl,
                    return_pct=net_pnl / entry_notional if entry_notional else np.nan,
                    holding_days=(i + 1 - entry_index),
                ))

                in_position = False
                direction = 0
                shares = 0.0
                entry_price = None
                entry_date = None
                entry_index = None

    # Final mark-to-market (last date has no t+1 to execute against).
    t_last = dates[-1]
    price_last = prices.loc[t_last]
    equity_last = cash + direction * shares * price_last if in_position else cash
    equity_records.append((t_last, equity_last))
    n_open_at_end = 1 if in_position else 0

    equity_curve = pd.Series(dict(equity_records)).sort_index()
    equity_curve.index.name = "date"
    equity_curve.name = "equity"

    trade_log = pd.DataFrame([t.__dict__ for t in trades])

    total_turnover = trade_log[["entry_notional", "exit_notional"]].sum().sum() if len(trade_log) else 0.0
    if in_position:
        total_turnover += entry_notional  # the still-open position's entry leg did turn over capital

    avg_equity = equity_curve.mean()
    n_years = (dates[-1] - dates[0]).days / 365.25
    turnover_annualized_ratio = (total_turnover / avg_equity / n_years) if (avg_equity > 0 and n_years > 0) else np.nan

    n_months = (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month) + 1
    trades_per_month = len(trade_log) / n_months if n_months > 0 else np.nan

    params = dict(
        lookback=lookback, threshold=threshold, ticker=ticker, account_size=account_size,
        position_fraction=position_fraction, commission_bps=commission_bps,
        slippage_bps=slippage_bps, commission_fixed=commission_fixed,
        allow_fractional_shares=allow_fractional_shares, direction_sign=direction_sign,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        turnover_total_dollars=total_turnover,
        turnover_annualized_ratio=turnover_annualized_ratio,
        trades_per_month=trades_per_month,
        n_trades=len(trade_log),
        n_open_at_end=n_open_at_end,
        params=params,
    )


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def performance_summary(result: BacktestResult) -> dict:
    equity = result.equity_curve
    daily_returns = equity.pct_change().dropna()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    return {
        **result.params,
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annualized_return": float(daily_returns.mean() * TRADING_DAYS_PER_YEAR),
        "annualized_vol": float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe": float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                  if daily_returns.std() > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "n_trades": result.n_trades,
        "n_open_at_end": result.n_open_at_end,
        "trades_per_month": result.trades_per_month,
        "turnover_total_dollars": result.turnover_total_dollars,
        "turnover_annualized_ratio": result.turnover_annualized_ratio,
        "total_commission_paid": float(result.trade_log["commission_paid"].sum()) if result.n_trades else 0.0,
        "total_slippage_cost": float(result.trade_log["slippage_cost"].sum()) if result.n_trades else 0.0,
        "win_rate": float((result.trade_log["net_pnl"] > 0).mean()) if result.n_trades else float("nan"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_train_val_panel()
    print(f"train+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    # Default demo run: best combo from results/walk_forward_grid.csv
    # (lookback=20d, threshold=2.0%), reversal direction, 1-day hold
    # (exit_after_n_days(0) -- see module docstring for why 0, not 1).
    default_result = run_backtest(
        panel, lookback=20, threshold=0.02, exit_rule=exit_after_n_days(0),
        account_size=100_000.0, position_fraction=1.0,
        commission_bps=10.0, slippage_bps=5.0,
    )
    default_result.equity_curve.to_csv(OUT_DIR / "equity_curve.csv")
    default_result.trade_log.to_csv(OUT_DIR / "trade_log.csv", index=False)
    print(f"Wrote {OUT_DIR / 'equity_curve.csv'} and {OUT_DIR / 'trade_log.csv'}")

    summary = performance_summary(default_result)
    print("\nDefault run summary (lookback=20d, threshold=2.0%, $100,000 account):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    pd.Series(summary).to_frame("value").to_csv(OUT_DIR / "summary.csv")
    print(f"Wrote {OUT_DIR / 'summary.csv'}")

    # Demonstrate exit_rule really is a pluggable argument.
    exit_rule_variants = {
        "1-day hold (timing_spec default)": exit_after_n_days(0),
        "5-day hold": exit_after_n_days(4),
        "20-day hold": exit_after_n_days(19),
        "exit on signal change": exit_on_signal_change(),
        "stop 3% / take 5%": exit_on_stop_take(0.03, 0.05),
    }
    exit_rows = []
    for name, rule in exit_rule_variants.items():
        r = run_backtest(panel, lookback=20, threshold=0.02, exit_rule=rule,
                          account_size=100_000.0, commission_bps=10.0, slippage_bps=5.0)
        exit_rows.append({"exit_rule": name, **performance_summary(r)})
    exit_comparison = pd.DataFrame(exit_rows)
    exit_comparison.to_csv(OUT_DIR / "exit_rule_comparison.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'exit_rule_comparison.csv'}")
    print(exit_comparison[["exit_rule", "sharpe", "total_return", "n_trades", "trades_per_month"]].to_string(index=False))

    # Demonstrate account-size parameterization from $1,000 to $1,000,000.
    account_sizes = [1_000, 10_000, 100_000, 1_000_000]
    account_rows = []
    for acct in account_sizes:
        r = run_backtest(panel, lookback=20, threshold=0.02, exit_rule=exit_after_n_days(0),
                          account_size=acct, position_fraction=1.0,
                          commission_bps=10.0, slippage_bps=5.0)
        account_rows.append({"account_size": acct, **performance_summary(r)})
    account_comparison = pd.DataFrame(account_rows)
    account_comparison.to_csv(OUT_DIR / "account_size_sensitivity.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'account_size_sensitivity.csv'}")
    print(account_comparison[["account_size", "total_return", "n_trades", "total_commission_paid", "total_slippage_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
