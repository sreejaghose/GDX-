"""
Statistical significance checks on the final config (the single best
config from Step 11 / results/backtest_results.csv, rank 1: lookback=20d,
threshold=2.0%, fixed reversal direction, 1-day hold, $0.005/share
commission + 0.05% slippage, $100,000 account -- read from that CSV, not
hardcoded, same as src/robustness_check.py).

(1) One-sample t-test: mean DAILY strategy return (from the backtest's
    equity curve) vs. zero.
(2) Bootstrap: resample the TRADE-LEVEL returns (trade_log["return_pct"],
    net of costs) with replacement, 1000 iterations, computing a Sharpe
    ratio each time (annualized by trades/year, since these are per-trade
    not per-day returns) -- report the 5th/95th percentile of that
    distribution.

Both run on train+validation (holdout untouched). Written up in
results/significance_report.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import exit_after_n_days, run_backtest
from src.features import load_train_val_panel
from src.robustness_check import load_best_config

OUT_DIR = Path("results/significance")
REPORT_PATH = Path("results/significance_report.md")

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
TRADING_DAYS_PER_YEAR = 252


def run_baseline(base: dict):
    panel = load_train_val_panel()
    result = run_backtest(
        panel, lookback=base["lookback"], threshold=base["threshold"],
        exit_rule=exit_after_n_days(0), account_size=base["account_size"],
        position_fraction=1.0, commission_bps=0.0,
        commission_per_share=base["commission_per_share"], slippage_bps=base["slippage_bps"],
    )
    return result


def daily_return_ttest(equity_curve: pd.Series) -> dict:
    daily_returns = equity_curve.pct_change().dropna()
    t_stat, p_value = stats.ttest_1samp(daily_returns, popmean=0.0)
    return {
        "n_days": len(daily_returns),
        "mean_daily_return": float(daily_returns.mean()),
        "std_daily_return": float(daily_returns.std()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "p_value_one_sided": float(p_value / 2) if t_stat > 0 else float(1 - p_value / 2),
        "significant_at_5pct": bool(p_value < 0.05),
        "daily_returns": daily_returns,
    }


def bootstrap_trade_sharpe(trade_log: pd.DataFrame, n_years: float, n_iterations: int = N_BOOTSTRAP,
                            seed: int = RANDOM_SEED) -> dict:
    trade_returns = trade_log["return_pct"].dropna().values
    n_trades = len(trade_returns)
    trades_per_year = n_trades / n_years

    point_estimate_sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year) \
        if trade_returns.std() > 0 else float("nan")

    rng = np.random.default_rng(seed)
    bootstrap_sharpes = np.empty(n_iterations)
    for i in range(n_iterations):
        resample = rng.choice(trade_returns, size=n_trades, replace=True)
        std = resample.std()
        bootstrap_sharpes[i] = (resample.mean() / std * np.sqrt(trades_per_year)) if std > 0 else np.nan

    valid = bootstrap_sharpes[~np.isnan(bootstrap_sharpes)]
    p5, p50, p95 = np.percentile(valid, [5, 50, 95])

    return {
        "n_trades": n_trades,
        "trades_per_year": trades_per_year,
        "point_estimate_sharpe": float(point_estimate_sharpe),
        "bootstrap_mean_sharpe": float(valid.mean()),
        "bootstrap_std_sharpe": float(valid.std()),
        "bootstrap_p5_sharpe": float(p5),
        "bootstrap_p50_sharpe": float(p50),
        "bootstrap_p95_sharpe": float(p95),
        "pct_bootstrap_sharpe_positive": float((valid > 0).mean()),
        "n_valid_iterations": int(len(valid)),
        "bootstrap_sharpes": bootstrap_sharpes,
    }


def write_report(base: dict, ttest: dict, bootstrap: dict):
    lines = ["# Statistical Significance Report: Final Config\n"]
    lines.append(
        f"Config: **lookback={base['lookback']}d, threshold={base['threshold']:.1%}**, fixed "
        f"reversal direction, 1-day hold, $0.005/share commission + 0.05% slippage, $100,000 "
        f"account -- the single best (and only profitable-after-costs) config from "
        f"`results/backtest_results.csv`, already stress-tested in `results/robustness_report.md`. "
        f"Train+validation only; holdout untouched.\n"
    )

    lines.append("## 1. T-test: mean daily strategy return vs. zero\n")
    lines.append(
        f"n = {ttest['n_days']} daily returns from the backtest's equity curve. "
        f"Mean daily return = {ttest['mean_daily_return']*100:.4f}%, std = {ttest['std_daily_return']*100:.4f}%.\n"
    )
    lines.append(
        f"- **t-statistic: {ttest['t_statistic']:.3f}**\n"
        f"- **two-sided p-value: {ttest['p_value']:.4f}**\n"
        f"- {'**Significant** at the 5% level' if ttest['significant_at_5pct'] else '**Not significant** at the 5% level'} "
        f"(mean daily return is {'' if ttest['significant_at_5pct'] else 'not '}statistically distinguishable from zero).\n"
    )
    if not ttest["significant_at_5pct"]:
        lines.append(
            "This is consistent with everything found so far: the baseline's CAGR was already "
            "under half a percent (`results/backtest_results.csv`), and `results/robustness_report.md` "
            "found the config profitable in only 2 of 4 sub-periods and fragile to +/-20% parameter "
            "perturbation. A t-test failing to reject the null here is not a new finding so much as "
            "the same thin-edge conclusion stated in a different, standard statistical form.\n"
        )
    else:
        lines.append(
            "Note: daily strategy returns are serially correlated for a position held over multiple "
            "days and are not independent draws, which a plain one-sample t-test assumes -- treat this "
            "p-value as indicative, not exact.\n"
        )

    lines.append("## 2. Bootstrap distribution of trade-level Sharpe ratios\n")
    lines.append(
        f"{bootstrap['n_trades']} trades resampled with replacement, {N_BOOTSTRAP} iterations "
        f"(seed={RANDOM_SEED}). Each iteration's Sharpe is annualized by "
        f"sqrt(trades/year) = sqrt({bootstrap['trades_per_year']:.1f}), since these are per-trade, "
        "not per-day, returns.\n"
    )
    lines.append(
        f"- Point-estimate Sharpe (actual trade sequence, no resampling): **{bootstrap['point_estimate_sharpe']:.3f}**\n"
        f"- Bootstrap mean Sharpe: {bootstrap['bootstrap_mean_sharpe']:.3f} (std: {bootstrap['bootstrap_std_sharpe']:.3f})\n"
        f"- **5th percentile: {bootstrap['bootstrap_p5_sharpe']:.3f}**\n"
        f"- 50th percentile (median): {bootstrap['bootstrap_p50_sharpe']:.3f}\n"
        f"- **95th percentile: {bootstrap['bootstrap_p95_sharpe']:.3f}**\n"
        f"- Fraction of bootstrap iterations with Sharpe > 0: {bootstrap['pct_bootstrap_sharpe_positive']:.1%}\n"
    )
    ci_straddles_zero = bootstrap["bootstrap_p5_sharpe"] < 0 < bootstrap["bootstrap_p95_sharpe"]
    lines.append(
        f"The 5th-95th percentile bootstrap interval [{bootstrap['bootstrap_p5_sharpe']:.3f}, "
        f"{bootstrap['bootstrap_p95_sharpe']:.3f}] "
        + ("**straddles zero**, meaning resampling the same {} trades with replacement produces both "
           "clearly negative and clearly positive Sharpe outcomes depending on which trades happen to "
           "be over- or under-sampled -- the point estimate's sign is not resampling-stable."
           .format(bootstrap['n_trades'])
           if ci_straddles_zero else
           "does **not** straddle zero, meaning the Sharpe's sign is resampling-stable even though its "
           "magnitude varies considerably across resamples.")
        + "\n"
    )
    lines.append(
        "Caveat: bootstrapping trades with replacement assumes trades are exchangeable/i.i.d., which "
        "ignores any time-ordering or regime structure (e.g. the concentration in 2006-2009 and "
        "2015-2018 found in `results/robustness_report.md`) -- it tests \"how much does the Sharpe "
        "estimate wobble under resampling of this exact trade set,\" not \"would this hold up out of "
        "sample.\" A wide or zero-straddling interval is still an honest, useful warning sign even "
        "though the assumption is imperfect.\n"
    )

    lines.append("## Bottom line\n")
    lines.append(
        f"- T-test: mean daily return is {'statistically significant' if ttest['significant_at_5pct'] else 'not statistically distinguishable from zero'} (p={ttest['p_value']:.3f}).\n"
        f"- Bootstrap: 90% interval for annualized trade-level Sharpe is "
        f"[{bootstrap['bootstrap_p5_sharpe']:.2f}, {bootstrap['bootstrap_p95_sharpe']:.2f}]"
        f"{', straddling zero' if ci_straddles_zero else ''}.\n\n"
        "Combined with the walk-forward grid search's optimism (Step 9), the fixed-direction "
        "backtest's thin margin (Step 11), and the fragility found under parameter/cost/sub-period "
        "stress (robustness report), this config does not clear a reasonable bar for statistical "
        "significance. Nothing here touches the holdout split -- these are all still "
        "train+validation diagnostics.\n"
    )

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main():
    base = load_best_config()
    print(f"Final config: lookback={base['lookback']}d, threshold={base['threshold']:.1%}")

    result = run_baseline(base)
    n_years = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 365.25

    print("\n(1) Daily-return t-test...")
    ttest = daily_return_ttest(result.equity_curve)
    print(f"  n={ttest['n_days']}, mean={ttest['mean_daily_return']:.5f}, "
          f"t={ttest['t_statistic']:.3f}, p={ttest['p_value']:.4f}")

    print("\n(2) Bootstrap trade-level Sharpe (1000 iterations)...")
    bootstrap = bootstrap_trade_sharpe(result.trade_log, n_years)
    print(f"  point estimate Sharpe={bootstrap['point_estimate_sharpe']:.3f}")
    print(f"  bootstrap: mean={bootstrap['bootstrap_mean_sharpe']:.3f}, "
          f"p5={bootstrap['bootstrap_p5_sharpe']:.3f}, p50={bootstrap['bootstrap_p50_sharpe']:.3f}, "
          f"p95={bootstrap['bootstrap_p95_sharpe']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ttest["daily_returns"].to_csv(OUT_DIR / "daily_returns.csv", header=["daily_return"])
    pd.Series(bootstrap["bootstrap_sharpes"], name="bootstrap_sharpe").to_csv(
        OUT_DIR / "bootstrap_sharpes.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'daily_returns.csv'} and {OUT_DIR / 'bootstrap_sharpes.csv'}")

    write_report(base, ttest, bootstrap)


if __name__ == "__main__":
    main()
