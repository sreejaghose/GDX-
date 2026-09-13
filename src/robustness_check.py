"""
Robustness checks on the single best config from Step 11
(results/backtest_results.csv rank 1: lookback=20d, threshold=2.0%,
fixed reversal direction, 1-day hold, $0.005/share commission + 0.05%
slippage, $100,000 account) -- read from that CSV rather than hardcoded,
so this script can't silently drift from what Step 11 actually found best.

All four checks run on train+validation (holdout untouched):

(1) Perturb lookback and threshold by +/-20% (3x3 grid including the
    corners where both are perturbed together) and re-run the backtest
    with identical costs/settings otherwise.
(2) Split train+validation into 4 equal (by row count) chronological
    sub-periods and report performance in each. Sliced from ONE
    continuous baseline backtest (not four independently re-run/re-warmed
    backtests) so sub-period results reflect the actual single trade
    sequence Step 11 evaluated, not an artifact of restarting the engine
    mid-history.
(3) Re-run the baseline config with 2x costs (commission_per_share
    $0.01, slippage 10bps).
(4) Using the SAME sub-periods from (2), confirm trades-per-month >= 1
    in every one individually, not just in the 9.17/month aggregate
    Step 11 reported.

Outputs:
  results/robustness/perturbation_grid.csv
  results/robustness/subperiod_performance.csv
  results/robustness/cost_sensitivity.csv
  results/robustness_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import exit_after_n_days, performance_summary, run_backtest
from src.features import load_train_val_panel

BACKTEST_RESULTS_PATH = Path("results/backtest_results.csv")
OUT_DIR = Path("results/robustness")
REPORT_PATH = Path("results/robustness_report.md")

TRADING_DAYS_PER_YEAR = 252
N_SUBPERIODS = 4
PERTURBATION_PCT = 0.20


def load_best_config() -> dict:
    df = pd.read_csv(BACKTEST_RESULTS_PATH)
    best = df.sort_values("rank").iloc[0]
    return {
        "lookback": int(best["lookback"]),
        "threshold": float(best["threshold"]),
        "account_size": float(best["account_size"]),
        "commission_per_share": float(best["commission_per_share"]),
        "slippage_bps": float(best["slippage_bps"]),
        "backtest_cagr": float(best["backtest_cagr"]),
        "backtest_sharpe": float(best["backtest_sharpe"]),
    }


def run_config(panel, lookback, threshold, commission_per_share, slippage_bps, account_size=100_000.0):
    result = run_backtest(
        panel, lookback=lookback, threshold=threshold,
        exit_rule=exit_after_n_days(0),
        account_size=account_size, position_fraction=1.0,
        commission_bps=0.0, commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
    )
    return result, performance_summary(result)


# --------------------------------------------------------------------------
# (1) Parameter perturbation
# --------------------------------------------------------------------------

def run_perturbation_grid(panel, base: dict) -> pd.DataFrame:
    lookback_base = base["lookback"]
    threshold_base = base["threshold"]

    lookback_variants = {
        "-20%": max(1, round(lookback_base * (1 - PERTURBATION_PCT))),
        "baseline": lookback_base,
        "+20%": round(lookback_base * (1 + PERTURBATION_PCT)),
    }
    threshold_variants = {
        "-20%": threshold_base * (1 - PERTURBATION_PCT),
        "baseline": threshold_base,
        "+20%": threshold_base * (1 + PERTURBATION_PCT),
    }

    rows = []
    for lb_label, lb in lookback_variants.items():
        for th_label, th in threshold_variants.items():
            _, summary = run_config(panel, lb, th, base["commission_per_share"], base["slippage_bps"], base["account_size"])
            rows.append({
                "lookback_perturbation": lb_label, "threshold_perturbation": th_label,
                "lookback": lb, "threshold": th,
                "cagr": summary["cagr"], "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"], "trades_per_month": summary["trades_per_month"],
                "n_trades": summary["n_trades"], "total_return": summary["total_return"],
            })

    grid = pd.DataFrame(rows)
    baseline_row = grid[(grid["lookback_perturbation"] == "baseline") & (grid["threshold_perturbation"] == "baseline")].iloc[0]
    grid["cagr_degradation_pp"] = (grid["cagr"] - baseline_row["cagr"]) * 100
    grid["sharpe_degradation"] = grid["sharpe"] - baseline_row["sharpe"]
    return grid


# --------------------------------------------------------------------------
# (2) Sub-period performance (sliced from one continuous baseline backtest)
# --------------------------------------------------------------------------

def slice_performance(equity_curve: pd.Series, trade_log: pd.DataFrame,
                       chunk_dates: pd.DatetimeIndex) -> dict:
    chunk_start, chunk_end = chunk_dates[0], chunk_dates[-1]
    chunk_equity = equity_curve.loc[chunk_start:chunk_end]

    start_equity = float(chunk_equity.iloc[0])
    end_equity = float(chunk_equity.iloc[-1])
    n_years = (chunk_end - chunk_start).days / 365.25
    cagr = float((end_equity / start_equity) ** (1 / n_years) - 1) if n_years > 0 and start_equity > 0 else float("nan")

    # Daily returns computed strictly WITHIN this chunk (drops the chunk's
    # own first day, since there's no in-chunk prior value to compare it
    # to -- avoids mixing in the cross-boundary transition day).
    chunk_returns = chunk_equity.pct_change().dropna()
    sharpe = (float(chunk_returns.mean() / chunk_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
              if len(chunk_returns) > 1 and chunk_returns.std() > 0 else float("nan"))

    # Chunk-LOCAL drawdown: peak resets at the start of the chunk rather
    # than carrying an all-time high in from a previous chunk.
    running_max = chunk_equity.cummax()
    drawdown = chunk_equity / running_max - 1
    max_drawdown = float(drawdown.min())

    n_months = (chunk_end.year - chunk_start.year) * 12 + (chunk_end.month - chunk_start.month) + 1
    if len(trade_log):
        in_chunk = trade_log[(trade_log["entry_date"] >= chunk_start) & (trade_log["entry_date"] <= chunk_end)]
    else:
        in_chunk = trade_log
    n_trades = len(in_chunk)
    trades_per_month = n_trades / n_months if n_months > 0 else float("nan")

    return {
        "start_date": chunk_start.date().isoformat(), "end_date": chunk_end.date().isoformat(),
        "n_days": len(chunk_dates), "start_equity": start_equity, "end_equity": end_equity,
        "cagr": cagr, "sharpe": sharpe, "max_drawdown": max_drawdown,
        "n_trades": n_trades, "trades_per_month": trades_per_month,
        "profitable": end_equity > start_equity,
        "trades_per_month_gte_1": trades_per_month >= 1,
    }


def run_subperiod_analysis(panel, base: dict) -> pd.DataFrame:
    result, _ = run_config(panel, base["lookback"], base["threshold"],
                            base["commission_per_share"], base["slippage_bps"], base["account_size"])
    trade_log = result.trade_log.copy()
    trade_log["entry_date"] = pd.to_datetime(trade_log["entry_date"])

    date_chunks = np.array_split(result.equity_curve.index, N_SUBPERIODS)

    rows = []
    for i, chunk_dates in enumerate(date_chunks, start=1):
        row = slice_performance(result.equity_curve, trade_log, chunk_dates)
        rows.append({"subperiod": i, **row})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (3) Cost sensitivity (2x costs)
# --------------------------------------------------------------------------

def run_cost_sensitivity(panel, base: dict) -> pd.DataFrame:
    _, summary_1x = run_config(panel, base["lookback"], base["threshold"],
                                base["commission_per_share"], base["slippage_bps"], base["account_size"])
    _, summary_2x = run_config(panel, base["lookback"], base["threshold"],
                                base["commission_per_share"] * 2, base["slippage_bps"] * 2, base["account_size"])

    rows = [
        {"cost_scenario": "1x (baseline)", "commission_per_share": base["commission_per_share"],
         "slippage_bps": base["slippage_bps"], **{k: summary_1x[k] for k in
         ["cagr", "sharpe", "max_drawdown", "trades_per_month", "n_trades", "total_return"]}},
        {"cost_scenario": "2x", "commission_per_share": base["commission_per_share"] * 2,
         "slippage_bps": base["slippage_bps"] * 2, **{k: summary_2x[k] for k in
         ["cagr", "sharpe", "max_drawdown", "trades_per_month", "n_trades", "total_return"]}},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(base: dict, perturbation_grid: pd.DataFrame, subperiods: pd.DataFrame,
                  cost_sensitivity: pd.DataFrame):
    lines = ["# Robustness Report: Best Config from Step 11\n"]
    lines.append(
        f"Config under test: **lookback={base['lookback']}d, threshold={base['threshold']:.1%}** "
        f"(the only profitable-after-costs config in `results/backtest_results.csv`: "
        f"baseline CAGR {base['backtest_cagr']:.2%}, Sharpe {base['backtest_sharpe']:.3f}). "
        f"Fixed reversal direction, 1-day hold, $100,000 account -- same settings as Step 11 "
        f"throughout unless a check explicitly varies one. Train+validation only; holdout untouched.\n"
    )
    lines.append(
        "**Headline**: the baseline's edge was already thin (CAGR well under 1%, Sharpe 0.157). "
        "None of the checks below should be read as \"does a strong strategy survive stress\" -- "
        "they're better read as \"how much of this thin, already-marginal result is left once "
        "reasonable uncertainty is applied.\"\n"
    )

    # --- (1) Perturbation ---
    lines.append("## 1. Parameter perturbation (+/-20% on lookback and threshold)\n")
    lines.append(
        f"3x3 grid: lookback in {{{perturbation_grid[perturbation_grid.lookback_perturbation!='baseline'].lookback.min()}, "
        f"{base['lookback']}, {perturbation_grid[perturbation_grid.lookback_perturbation!='baseline'].lookback.max()}}} "
        f"x threshold in {{{perturbation_grid.threshold.min():.1%}, {base['threshold']:.1%}, {perturbation_grid.threshold.max():.1%}}}, "
        "same costs/exit/direction as baseline throughout.\n"
    )
    lines.append("| Lookback | Threshold | CAGR | Sharpe | Max DD | Trades/mo | CAGR vs. baseline (pp) |")
    lines.append("|---|---|---|---|---|---|---|")
    grid_sorted = perturbation_grid.sort_values(["lookback", "threshold"])
    for _, row in grid_sorted.iterrows():
        marker = " **(baseline)**" if row["lookback_perturbation"] == "baseline" and row["threshold_perturbation"] == "baseline" else ""
        lines.append(
            f"| {int(row['lookback'])}d{marker} | {row['threshold']:.1%} | {row['cagr']:.2%} | {row['sharpe']:.3f} "
            f"| {row['max_drawdown']:.1%} | {row['trades_per_month']:.2f} | {row['cagr_degradation_pp']:+.2f} |"
        )
    lines.append("")

    n_positive = (perturbation_grid["cagr"] > 0).sum()
    n_total = len(perturbation_grid)
    worst = perturbation_grid.loc[perturbation_grid["sharpe"].idxmin()]
    lines.append(
        f"**{n_positive} of {n_total}** grid points (including the baseline itself) have positive CAGR. "
        f"Worst case: lookback={int(worst['lookback'])}d, threshold={worst['threshold']:.1%} "
        f"-> Sharpe {worst['sharpe']:.3f}, CAGR {worst['cagr']:.2%}. A config whose neighbors in "
        "parameter space are mostly unprofitable is a config that was likely selected by the walk-"
        "forward grid search finding a narrow, not-very-robust local optimum rather than a broad, "
        "stable edge.\n"
    )

    # --- (2) Sub-periods ---
    lines.append("## 2. Performance by sub-period (4 equal chronological chunks)\n")
    lines.append(
        "Sliced from the single continuous baseline backtest (same trade sequence Step 11 "
        "evaluated), not four independently re-run backtests -- so results below reflect the "
        "actual realized path, not a re-warmed restart at each boundary.\n"
    )
    lines.append("| Sub-period | Dates | CAGR | Sharpe | Max DD (local) | Trades | Trades/mo | Profitable |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in subperiods.iterrows():
        lines.append(
            f"| {int(row['subperiod'])} | {row['start_date']} to {row['end_date']} | {row['cagr']:.2%} "
            f"| {row['sharpe']:.3f} | {row['max_drawdown']:.1%} | {int(row['n_trades'])} "
            f"| {row['trades_per_month']:.2f} | {'Yes' if row['profitable'] else 'No'} |"
        )
    lines.append("")
    n_profitable_periods = subperiods["profitable"].sum()
    lines.append(
        f"**{n_profitable_periods} of {N_SUBPERIODS} sub-periods are profitable.** "
        f"{'Performance is concentrated in specific periods rather than persistent throughout the sample.' if n_profitable_periods < N_SUBPERIODS else 'Performance holds up across every sub-period.'}\n"
    )

    # --- (3) Cost sensitivity ---
    lines.append("## 3. Cost sensitivity (2x commission and slippage)\n")
    lines.append("| Scenario | Commission/share | Slippage | CAGR | Sharpe | Max DD | Total return |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in cost_sensitivity.iterrows():
        lines.append(
            f"| {row['cost_scenario']} | ${row['commission_per_share']:.3f} | {row['slippage_bps']:.0f}bps "
            f"| {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.1%} | {row['total_return']:.1%} |"
        )
    lines.append("")
    cagr_1x = cost_sensitivity.iloc[0]["cagr"]
    cagr_2x = cost_sensitivity.iloc[1]["cagr"]
    lines.append(
        f"Doubling costs moves CAGR from {cagr_1x:.2%} to {cagr_2x:.2%} "
        f"({'flips it negative' if cagr_1x > 0 and cagr_2x <= 0 else 'stays positive' if cagr_2x > 0 else 'stays negative'}). "
        f"At {cost_sensitivity.iloc[0]['trades_per_month']:.1f} trades/month, this strategy trades "
        "often enough that cost assumptions matter a lot -- this is not a low-turnover strategy "
        "where a cost-doubling would be a rounding error.\n"
    )

    # --- (4) Trades-per-month check ---
    lines.append("## 4. Trades-per-month >= 1 check, per sub-period (not just in aggregate)\n")
    lines.append(
        f"Step 11 reported {base['backtest_cagr']:+.2%} CAGR at an aggregate 9.17 trades/month. "
        "That aggregate number could in principle hide a sub-period with almost no trades at all "
        "(e.g. if the signal simply stopped firing for a year) -- checked explicitly per sub-period below:\n"
    )
    lines.append("| Sub-period | Trades/month | >= 1? |")
    lines.append("|---|---|---|")
    for _, row in subperiods.iterrows():
        lines.append(f"| {int(row['subperiod'])} | {row['trades_per_month']:.2f} | {'PASS' if row['trades_per_month_gte_1'] else 'FAIL'} |")
    lines.append("")
    all_pass = subperiods["trades_per_month_gte_1"].all()
    lines.append(
        f"**{'All 4 sub-periods pass' if all_pass else str((~subperiods['trades_per_month_gte_1']).sum()) + ' of 4 sub-periods FAIL'} "
        f"the >= 1 trade/month bar.** "
        + ("Trade frequency is not an artifact of a few active months hiding inside an inactive "
           "aggregate -- the strategy fires consistently throughout the sample."
           if all_pass else
           "Trade frequency is NOT consistent across the sample -- at least one sub-period is "
           "meaningfully less active than the aggregate figure suggests.")
        + "\n"
    )

    lines.append("## Bottom line\n")
    lines.append(
        f"- Parameter sensitivity: {n_positive}/{n_total} nearby configs profitable -- "
        f"{'a fragile, narrow optimum' if n_positive <= n_total // 2 + 1 else 'reasonably broad'}.\n"
        f"- Sub-period consistency: {n_profitable_periods}/{N_SUBPERIODS} periods profitable.\n"
        f"- Cost sensitivity: {'survives' if cagr_2x > 0 else 'does not survive'} a 2x cost assumption.\n"
        f"- Trade frequency: {'consistent' if all_pass else 'inconsistent'} across sub-periods.\n\n"
        "Given the baseline's CAGR was already under half a percent before any of these checks, "
        "this config should not be read as a validated edge -- it should be read as what's left "
        "of the walk-forward grid's optimism after realistic execution, and these checks show how "
        "little of that margin was left to give. This is still train+validation; none of this "
        "touches the holdout split.\n"
    )

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_best_config()
    print(f"Best config (from {BACKTEST_RESULTS_PATH}): lookback={base['lookback']}d, "
          f"threshold={base['threshold']:.1%}, baseline CAGR={base['backtest_cagr']:.2%}, "
          f"Sharpe={base['backtest_sharpe']:.3f}")

    panel = load_train_val_panel()
    print(f"train+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    print("\n(1) Running perturbation grid...")
    perturbation_grid = run_perturbation_grid(panel, base)
    perturbation_grid.to_csv(OUT_DIR / "perturbation_grid.csv", index=False)
    print(perturbation_grid[["lookback", "threshold", "cagr", "sharpe", "cagr_degradation_pp"]].to_string(index=False))

    print("\n(2) Running sub-period analysis...")
    subperiods = run_subperiod_analysis(panel, base)
    subperiods.to_csv(OUT_DIR / "subperiod_performance.csv", index=False)
    print(subperiods[["subperiod", "start_date", "end_date", "cagr", "sharpe", "trades_per_month", "profitable"]].to_string(index=False))

    print("\n(3) Running cost sensitivity...")
    cost_sensitivity = run_cost_sensitivity(panel, base)
    cost_sensitivity.to_csv(OUT_DIR / "cost_sensitivity.csv", index=False)
    print(cost_sensitivity.to_string(index=False))

    print("\n(4) Trades-per-month >= 1 per sub-period:")
    print(subperiods[["subperiod", "trades_per_month", "trades_per_month_gte_1"]].to_string(index=False))

    write_report(base, perturbation_grid, subperiods, cost_sensitivity)


if __name__ == "__main__":
    main()
