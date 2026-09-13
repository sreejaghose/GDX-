"""
Take the top 5 (lookback, threshold) configs from results/walk_forward_grid.csv
(ranked there by walk-forward Sharpe -- see src/walk_forward.py) and run each
through the Step 10 backtest engine (src/backtest.py) on train+validation
(holdout untouched), with realistic per-share costs.

Costs: commission $0.005/share + 0.05% (5bps) slippage per leg. The walk-
forward grid didn't model costs at all, so this checks whether its ranking
survives contact with them.

Everything else about the run is exactly src/backtest.py's default
behavior, not re-litigated here: fixed reversal direction (direction_sign=-1)
and a 1-day hold (exit_after_n_days(0), reproducing the original
results/timing_spec.md convention). Both are worth stating plainly: the
walk-forward Sharpes being reproduced here were earned under an ADAPTIVE,
quarterly-refit direction (see results/walk_forward/direction_log.csv) --
this run uses one fixed direction for the whole period, which
results/backtest/README.md already showed can matter more than costs do
for the top grid combo. Any change here from the grid's ranking should be
read with that in mind, not assumed to be a costs effect alone.

Outputs results/backtest_results.csv: CAGR, Sharpe, max drawdown, trade
frequency (trades/month), and profitability after costs, per config.
"""

from pathlib import Path

import pandas as pd

from src.backtest import exit_after_n_days, performance_summary, run_backtest
from src.features import load_train_val_panel

WALK_FORWARD_GRID_PATH = Path("results/walk_forward_grid.csv")
OUT_PATH = Path("results/backtest_results.csv")

TOP_N = 5
ACCOUNT_SIZE = 100_000.0
POSITION_FRACTION = 1.0
COMMISSION_PER_SHARE = 0.005   # $/share, both legs
SLIPPAGE_BPS = 5.0             # 0.05%, both legs
COMMISSION_BPS = 0.0           # using per-share commission instead of bps-of-notional


def main():
    grid = pd.read_csv(WALK_FORWARD_GRID_PATH)
    top5 = grid.sort_values("walk_forward_sharpe", ascending=False).head(TOP_N).reset_index(drop=True)
    print(f"Top {TOP_N} configs from {WALK_FORWARD_GRID_PATH}:")
    print(top5[["lookback", "threshold", "walk_forward_sharpe"]].to_string(index=False))

    panel = load_train_val_panel()
    print(f"\ntrain+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    rows = []
    for rank, grid_row in top5.iterrows():
        lookback = int(grid_row["lookback"])
        threshold = float(grid_row["threshold"])

        result = run_backtest(
            panel, lookback=lookback, threshold=threshold,
            exit_rule=exit_after_n_days(0),
            account_size=ACCOUNT_SIZE, position_fraction=POSITION_FRACTION,
            commission_bps=COMMISSION_BPS, commission_per_share=COMMISSION_PER_SHARE,
            slippage_bps=SLIPPAGE_BPS,
        )
        summary = performance_summary(result)

        # gross_pnl (from the trade log) is computed from SLIPPAGE-ADJUSTED
        # execution prices, so slippage is already embedded in it -- adding
        # total_slippage_cost to it again would double-count. Recover the
        # frictionless (quoted-price) gross P&L by adding slippage back,
        # so the full cost waterfall is explicit and each number is what
        # it says it is:
        #   gross_pnl_at_quoted_prices - total_slippage - total_commission
        #     == gross_pnl_total (execution-price-based) - total_commission
        #     == net_pnl_after_costs
        gross_pnl_total = float(result.trade_log["gross_pnl"].sum()) if result.n_trades else 0.0
        total_commission = float(result.trade_log["commission_paid"].sum()) if result.n_trades else 0.0
        total_slippage = float(result.trade_log["slippage_cost"].sum()) if result.n_trades else 0.0
        gross_pnl_at_quoted_prices = gross_pnl_total + total_slippage
        net_pnl_total = gross_pnl_total - total_commission

        rows.append({
            "rank": rank + 1,
            "lookback": lookback,
            "threshold": threshold,
            "walk_forward_sharpe": float(grid_row["walk_forward_sharpe"]),
            "backtest_cagr": summary["cagr"],
            "backtest_sharpe": summary["sharpe"],
            "max_drawdown": summary["max_drawdown"],
            "trades_per_month": summary["trades_per_month"],
            "n_trades": summary["n_trades"],
            "total_return": summary["total_return"],
            "gross_pnl_at_quoted_prices": gross_pnl_at_quoted_prices,
            "total_slippage_cost": total_slippage,
            "total_commission_paid": total_commission,
            "net_pnl_after_costs": net_pnl_total,
            "profitable_after_costs": bool(net_pnl_total > 0),
            "win_rate": summary["win_rate"],
            "turnover_annualized_ratio": summary["turnover_annualized_ratio"],
            "account_size": ACCOUNT_SIZE,
            "commission_per_share": COMMISSION_PER_SHARE,
            "slippage_bps": SLIPPAGE_BPS,
            "exit_rule": "exit_after_n_days(0) [1-day hold, timing_spec default]",
            "direction": "reversal (fixed)",
        })

        print(f"\nrank {rank+1}: lookback={lookback}d threshold={threshold:.1%} "
              f"(walk-forward Sharpe {grid_row['walk_forward_sharpe']:.3f})")
        print(f"  backtest CAGR={summary['cagr']:.2%}  Sharpe={summary['sharpe']:.3f}  "
              f"max DD={summary['max_drawdown']:.2%}  trades/mo={summary['trades_per_month']:.2f}  "
              f"profitable after costs={net_pnl_total > 0} (net ${net_pnl_total:,.0f})")

    results_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    n_profitable = results_df["profitable_after_costs"].sum()
    print(f"\n{n_profitable} of {TOP_N} top walk-forward configs are profitable after realistic costs "
          f"under a fixed-direction, 1-day-hold deployment.")


if __name__ == "__main__":
    main()
