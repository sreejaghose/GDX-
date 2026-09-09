# Baseline Strategy Comparison — Train + Validation

In-sample period: **2006-05-22 to 2018-11-15** (train + validation combined, per `results/split_dates.json`; holdout untouched).

No transaction costs, slippage, or borrow costs are modeled anywhere in this table.

| Strategy | N days | Total return | CAGR | Ann. vol | Sharpe | Max drawdown | Win rate |
|---|---|---|---|---|---|---|---|
| Buy-and-hold GDX | 3145 | -44.6% | -4.6% | 42.4% | 0.10 | -80.5% | 49.6% |
| Buy-and-hold SPY | 3145 | 178.6% | 8.6% | 19.0% | 0.53 | -55.2% | 54.8% |
| Naive reversal on GDX | 3144 | 1007.1% | 21.3% | 42.3% | 0.67 | -68.2% | 54.1% |

**Naive reversal rule definition:** position for day *t* is `-sign(return_{t-1})` on GDX — short 1 unit if GDX's prior-day return was positive, long 1 unit if negative, flat if exactly zero. Position is held for exactly one day (re-evaluated daily) and reset each day; no leverage, no costs.

CAGR/Sharpe/vol annualized using 252 trading days/year. Win rate excludes flat (exactly zero-return) days from the denominator.

## Caveats

- **The naive reversal number is almost certainly not investable as shown.**
  It trades every single day (100% daily turnover) with zero transaction
  costs, slippage, or borrow cost assumed. It is exploiting a small
  negative lag-1 autocorrelation in GDX daily returns over this combined
  window (-0.027; note this is *stronger* than the -0.007 measured on the
  train split alone in `results/eda_summary.md`, so part of the edge is
  concentrated in the validation-period regime). Even a few basis points
  of round-trip cost per day, applied 3,144 times, would materially erode
  or eliminate this result — that's the natural next check before reading
  anything into this rule.
- All three series here trade through the 2008 financial crisis and 2011
  gold peak/selloff; GDX's -80.5% max drawdown and near-zero long-run CAGR
  reflect that this is a genuinely hard period to hold gold miners through
  buy-and-hold.
- Sharpe ratios use a 0% risk-free rate, not the actual T-bill rate over
  2006-2018.
