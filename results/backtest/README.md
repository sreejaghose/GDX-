# Backtest Engine: Results & Reconciliation

`src/backtest.py`, run on train+validation (holdout untouched). Default
demo uses the walk-forward grid's top combo (lookback=20d, threshold=2.0%),
$100,000 account, 10bps commission + 5bps slippage per leg, 100% position
sizing, reversal direction, 1-day hold (`exit_after_n_days(0)`).

## The headline number will look wrong at first glance -- it isn't a bug

`results/walk_forward_grid.csv` reported Sharpe **+0.379** for this exact
(lookback, threshold) combo. This engine's default run on the same combo
gives Sharpe **-0.469**, total return **-88.8%**. Both numbers are correct;
they answer different questions, for three concrete reasons (verified by
inspecting `trade_log.csv` directly -- commission/slippage/holding-period
all match their configured values exactly):

1. **Direction is fixed here, adaptive there.** `walk_forward.py` re-picks
   momentum vs. reversal every quarter based on in-window Sharpe (see
   `results/walk_forward/direction_log.csv`); this engine's default run
   uses a single fixed direction (reversal) for the entire 12.5-year
   period. At 100% position sizing, being on the wrong side of a
   volatile, trending asset for extended stretches is costly -- this is
   the dominant effect, not costs.
2. **Costs are now real.** 10bps commission + 5bps slippage per leg
   (~30bps round-trip) across 1,384 trades is a genuine, if secondary,
   drag that `walk_forward.py` did not model at all.
3. **Trade cadence differs mechanically.** `walk_forward.py` computes an
   independent t+1->t+2 return every day the signal is triggered (a daily
   rebalance with no notion of "already in a position"). This engine
   models a *persistent* position: once entered, the strategy is flat and
   unable to re-enter until the exit executes, so it inherently trades
   less often than the triggering-day count alone would suggest (1,384
   trades here vs. ~2,541 signal-triggering days in the walk-forward
   version of the same combo).

**This is the point of building this engine**: it demonstrates that a
literal, fully-invested, fixed-direction implementation of the walk-forward
grid's best combo does not survive contact with realistic execution and
persistent position accounting. That's a real finding, not a discrepancy
to explain away.

## Exit rule matters a lot (`exit_rule_comparison.csv`)

| Exit rule | Sharpe | Total return | Trades | Trades/month |
|---|---|---|---|---|
| 1-day hold (timing_spec default) | -0.469 | -88.8% | 1,384 | 9.17 |
| 5-day hold | -0.135 | -78.8% | 491 | 3.25 |
| 20-day hold | 0.143 | -28.3% | 145 | 0.96 |
| **exit on signal change** | **0.271** | **+41.2%** | 279 | 1.85 |
| stop 3% / take 5% | 0.089 | -34.9% | 387 | 2.56 |

Holding longer, or exiting only when the signal itself changes rather than
on a fixed 1-day clock, meaningfully improves risk-adjusted performance
here -- fewer, more deliberate trades reduce the cost drag and let the
reversal thesis actually play out. `exit_on_signal_change()` is the best
of the five tried, still far short of walk-forward's adaptive-direction
result, but a large improvement over the naive 1-day hold.

## Account size ($1,000 to $1,000,000): `account_size_sensitivity.csv`

| Account size | Total return | Trades | Commission paid | Slippage cost |
|---|---|---|---|---|
| $1,000 | -87.9% | 1,384 | $1,295 | $647 |
| $10,000 | -88.7% | 1,384 | $13,182 | $6,591 |
| $100,000 | -88.8% | 1,384 | $131,984 | $65,992 |
| $1,000,000 | -88.8% | 1,384 | $1,319,924 | $659,962 |

Costs scale linearly with account size as expected (commission/slippage
are both bps-of-notional). Total return is *slightly* less negative at
$1,000 than at $1,000,000 -- integer share rounding leaves more idle cash
on the sidelines for a small account with a ~$30 GDX share price (e.g. a
$1,000 account buying 100% of equity in $30 shares can only round to
whole shares, occasionally leaving a few dollars uninvested), which
mechanically shields a small account very slightly during a losing
stretch. This is a real, if minor, effect of the integer-share-lot
sizing (`allow_fractional_shares=False` by default) and is exactly the
kind of account-size-dependent behavior this parameterization exists to
surface.

## Caveats

- Single-asset, no diversification, no volatility targeting, no margin
  limits -- 100% of equity in a ~45%-annualized-vol asset every trade is
  aggressive by construction in the default demo (this is intentional,
  to keep the demo parameters identical to the walk-forward grid's
  winning combo; a real allocation would almost certainly use a smaller
  `position_fraction`).
- No look-ahead: entry/exit decisions at close t always execute at close
  t+1, uniformly for every exit rule type (see `src/backtest.py`
  module docstring for why fixed-holding-period exits use the same
  decide/execute lag as data-dependent ones).
- This is still train+validation data -- no claim about the untouched
  holdout split is made anywhere in this report.
