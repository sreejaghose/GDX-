# Top-5 Walk-Forward Configs Under Realistic Costs

`src/run_top5_backtest.py` takes the top 5 (lookback, threshold) rows from
`results/walk_forward_grid.csv` (ranked by walk-forward Sharpe) and runs
each through `src/backtest.py` on train+validation, with $0.005/share
commission + 0.05% slippage per leg, $100,000 account, 100% position
sizing, fixed reversal direction, 1-day hold. Output:
`results/backtest_results.csv`.

## Headline: only 1 of 5 is profitable after costs

| Rank | Lookback | Threshold | Walk-forward Sharpe | Backtest CAGR | Backtest Sharpe | Max DD | Profitable after costs |
|---|---|---|---|---|---|---|---|
| 1 | 20d | 2.0% | 0.379 | +0.4% | 0.157 | -69% | **Yes** ($5,618 net) |
| 2 | 60d | 0.5% | 0.251 | -18.2% | -0.556 | -94% | No (-$91,866) |
| 3 | 60d | 2.0% | 0.232 | -8.0% | -0.129 | -77% | No (-$64,818) |
| 4 | 60d | 1.5% | 0.200 | -16.4% | -0.485 | -91% | No (-$89,282) |
| 5 | 60d | 1.0% | 0.197 | -15.5% | -0.445 | -92% | No (-$87,833) |

## Why the ranking doesn't hold up -- and it isn't mainly about costs

This mirrors what `results/backtest/README.md` already found for the #1
combo alone, now confirmed across all top 5: **the dominant effect is
fixed vs. adaptive direction, not transaction costs.**

`walk_forward.py` earned these Sharpe ratios by **re-picking direction
(momentum vs. reversal) every quarter** based on in-window performance
(`results/walk_forward/direction_log.csv`) -- and with zero costs modeled.
This run uses backtest.py's default: **one fixed direction (reversal) for
the entire 12.5-year period**, now with real costs on top. Both changes
point the same way, but the direction lock-in is the bigger one: rank 1's
gross P&L at quoted prices was ~$233K (recoverable from the CSV: gross +
slippage + commission), and total commission was only ~$59K on 1,384
trades -- costs alone would not have erased that edge. What erases it is
that reversal was the *wrong* direction for large stretches of this period
that the adaptive walk-forward process was free to trade around by
switching to momentum, and a fixed-direction deployment cannot.

The four lookback=60d configs all lose badly (CAGR -8% to -18%, max
drawdown 77-94%) -- notably worse than rank 1's lookback=20d, even though
their walk-forward Sharpes were all positive and reasonably close together
(0.197-0.251). That similarity in the walk-forward metric did not carry
over to fixed-direction performance at all, which is itself informative:
**walk-forward Sharpe alone is not a reliable guide to how a config will
behave once direction is no longer allowed to adapt.**

## What this means for next steps

- A config that ranks well in an adaptive walk-forward evaluation should
  not be assumed to be a good candidate for a *fixed*-direction, static
  deployment -- these are different questions, and this table is the
  first place in this project where that gap is made concrete rather
  than theoretical.
- If a fixed-direction strategy is genuinely the target design, direction
  should be selected (or the adaptive quarterly-refit mechanism from
  `walk_forward.py` should be ported into `backtest.py`) as part of the
  evaluation, not decided once at the top of a script and left there.
- Rank 1's result, while technically "profitable after costs," is a CAGR
  of 0.44% against a -69% max drawdown and 1,384 trades -- not something
  to read as a validated strategy on the strength of this table alone.
- All of this is still on train+validation; the holdout split remains
  untouched.
