# Robustness Report: Best Config from Step 11

Config under test: **lookback=20d, threshold=2.0%** (the only profitable-after-costs config in `results/backtest_results.csv`: baseline CAGR 0.44%, Sharpe 0.157). Fixed reversal direction, 1-day hold, $100,000 account -- same settings as Step 11 throughout unless a check explicitly varies one. Train+validation only; holdout untouched.

**Headline**: the baseline's edge was already thin (CAGR well under 1%, Sharpe 0.157). None of the checks below should be read as "does a strong strategy survive stress" -- they're better read as "how much of this thin, already-marginal result is left once reasonable uncertainty is applied."

## 1. Parameter perturbation (+/-20% on lookback and threshold)

3x3 grid: lookback in {16, 20, 24} x threshold in {1.6%, 2.0%, 2.4%}, same costs/exit/direction as baseline throughout.

| Lookback | Threshold | CAGR | Sharpe | Max DD | Trades/mo | CAGR vs. baseline (pp) |
|---|---|---|---|---|---|---|
| 16d | 1.6% | -9.56% | -0.213 | -85.4% | 9.40 | -10.00 |
| 16d | 2.0% | -8.97% | -0.200 | -82.2% | 9.01 | -9.41 |
| 16d | 2.4% | -3.36% | 0.018 | -79.0% | 8.71 | -3.80 |
| 20d | 1.6% | 1.34% | 0.189 | -71.4% | 9.40 | +0.90 |
| 20d **(baseline)** | 2.0% | 0.44% | 0.157 | -69.5% | 9.17 | +0.00 |
| 20d | 2.4% | -1.55% | 0.085 | -68.8% | 8.89 | -1.99 |
| 24d | 1.6% | -9.23% | -0.181 | -80.7% | 9.46 | -9.67 |
| 24d | 2.0% | -6.36% | -0.069 | -70.1% | 9.20 | -6.80 |
| 24d | 2.4% | -7.66% | -0.120 | -75.8% | 8.90 | -8.10 |

**2 of 9** grid points (including the baseline itself) have positive CAGR. Worst case: lookback=16d, threshold=1.6% -> Sharpe -0.213, CAGR -9.56%. A config whose neighbors in parameter space are mostly unprofitable is a config that was likely selected by the walk-forward grid search finding a narrow, not-very-robust local optimum rather than a broad, stable edge.

## 2. Performance by sub-period (4 equal chronological chunks)

Sliced from the single continuous baseline backtest (same trade sequence Step 11 evaluated), not four independently re-run backtests -- so results below reflect the actual realized path, not a re-warmed restart at each boundary.

| Sub-period | Dates | CAGR | Sharpe | Max DD (local) | Trades | Trades/mo | Profitable |
|---|---|---|---|---|---|---|---|
| 1 | 2006-05-22 to 2009-07-07 | 8.86% | 0.413 | -33.7% | 351 | 9.00 | Yes |
| 2 | 2009-07-08 to 2012-08-17 | -1.43% | 0.052 | -48.0% | 344 | 9.05 | No |
| 3 | 2012-08-20 to 2015-10-05 | -4.48% | -0.011 | -48.4% | 358 | 9.18 | No |
| 4 | 2015-10-06 to 2018-11-15 | 0.50% | 0.139 | -21.3% | 331 | 8.71 | Yes |

**2 of 4 sub-periods are profitable.** Performance is concentrated in specific periods rather than persistent throughout the sample.

## 3. Cost sensitivity (2x commission and slippage)

| Scenario | Commission/share | Slippage | CAGR | Sharpe | Max DD | Total return |
|---|---|---|---|---|---|---|
| 1x (baseline) | $0.005 | 5bps | 0.44% | 0.157 | -69.5% | 5.6% |
| 2x | $0.010 | 10bps | -13.82% | -0.376 | -88.2% | -84.4% |

Doubling costs moves CAGR from 0.44% to -13.82% (flips it negative). At 9.2 trades/month, this strategy trades often enough that cost assumptions matter a lot -- this is not a low-turnover strategy where a cost-doubling would be a rounding error.

## 4. Trades-per-month >= 1 check, per sub-period (not just in aggregate)

Step 11 reported +0.44% CAGR at an aggregate 9.17 trades/month. That aggregate number could in principle hide a sub-period with almost no trades at all (e.g. if the signal simply stopped firing for a year) -- checked explicitly per sub-period below:

| Sub-period | Trades/month | >= 1? |
|---|---|---|
| 1 | 9.00 | PASS |
| 2 | 9.05 | PASS |
| 3 | 9.18 | PASS |
| 4 | 8.71 | PASS |

**All 4 sub-periods pass the >= 1 trade/month bar.** Trade frequency is not an artifact of a few active months hiding inside an inactive aggregate -- the strategy fires consistently throughout the sample.

## Bottom line

- Parameter sensitivity: 2/9 nearby configs profitable -- a fragile, narrow optimum.
- Sub-period consistency: 2/4 periods profitable.
- Cost sensitivity: does not survive a 2x cost assumption.
- Trade frequency: consistent across sub-periods.

Given the baseline's CAGR was already under half a percent before any of these checks, this config should not be read as a validated edge -- it should be read as what's left of the walk-forward grid's optimism after realistic execution, and these checks show how little of that margin was left to give. This is still train+validation; none of this touches the holdout split.
