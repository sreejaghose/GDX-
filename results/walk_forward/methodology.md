# Walk-Forward Validation: Methodology

Implements `src/walk_forward.py`. Grid: lookback ∈ {5, 10, 20, 60} trading
days × threshold ∈ {0.5%, 1%, 1.5%, 2%}, 16 combos total. Data: train +
validation combined (2006-05-22 to 2018-11-15); the holdout split is never
touched.

The request ("refit the model every quarter... grid of lookback periods and
thresholds") left several things underspecified. Decisions made, and why:

## What "the model" is, and what gets refit

A fixed rule (direction + lookback + threshold, all constant) has no free
parameter and so nothing to meaningfully "refit." **Direction** —
momentum (bet with the N-day move) vs. reversal (bet against it) — is the
one parameter that plausibly changes over time and is what gets refit each
quarter. This directly extends the threshold-direction search from the
Step 9 in-sample fit (`results/naive_models/train_fit_report.md`, which
found reversal dominant on the full train split) into a proper walk-forward
setting where that choice can be revisited, and potentially overridden, as
more data arrives.

**Refit rule**: at each quarterly refit point, compute the realized Sharpe
ratio of both directions using only trades triggered within the expanding
fit window (data strictly before the predict quarter); pick whichever
direction has the higher in-window Sharpe. If neither direction has at
least 10 triggered trades in-window, default to reversal (the Step 9
finding) rather than fit noise.

## What the threshold means

Applied as a **deadzone** on `trailing_return_N`: no position is taken
unless `|trailing_return_N| >= threshold`; when triggered, position sign
follows the refit direction. This is a standard way to give a percentage
threshold operational meaning in a trading rule (avoid trading on moves
too small to be a real signal) and is the natural reading of "threshold"
paired with "lookback" in a grid — a pure decision-boundary value (as
Step 9's percentile search used) wouldn't naturally live on a fixed
{0.5%, 1%, 1.5%, 2%} grid.

## What return each trade earns

Reuses the **exact same entry/exit timing** established in
`results/timing_spec.md` and implemented in `src/naive_baseline.py`:
signal known at close *t* → entry at close *t+1* → exit at close *t+2*.
This is deliberate: it's the one execution convention in this project
that has been explicitly specified and independently verified for zero
lookahead, so the walk-forward backtest reuses it rather than inventing a
second, unverified convention (e.g. Step 7's same-day-to-next-day ML
label, which was built for a different purpose).

## Expanding window / quarterly mechanics

Calendar quarters (`pandas` `Q` periods). Quarter *i*'s prediction uses
only rows from quarters strictly before quarter *i* (`fit_window.index.max()
< predict_window.index.min()` is asserted in code on every iteration, not
just assumed). The first two quarters are pure warm-up: quarters are
skipped until the fit window has at least `lookback + 20` valid rows, so
the actual first predicted quarter differs by lookback (2006-10-02 for
lookback ≤ 20d; 2007-01-03 for lookback = 60d, needing one more quarter of
history) — see `oos_start` in `results/walk_forward_grid.csv`.

## Stitching the equity curve

Each quarter's out-of-window predictions are concatenated in date order
into one continuous daily return series per (lookback, threshold) combo —
this is what "walk-forward Sharpe" and the other summary stats in
`results/walk_forward_grid.csv` are computed from. Flat (non-triggered)
days contribute 0 to that series (no position, no return), consistent
with an equity curve for a strategy that isn't always in the market.
Per-combo daily series are in `results/walk_forward/equity_curves.csv`;
the quarter-by-quarter refit direction and in-window Sharpes used to pick
it are in `results/walk_forward/direction_log.csv`.

## Reading the results

No transaction costs, slippage, or position sizing (each triggered signal
is a full, un-levered position) — same caveat as `src/naive_baseline.py`.
The best combo (lookback=20d, threshold=2.0%, Sharpe 0.379) is a plausible
but not dramatic edge over the weakest (Sharpe −0.245); given how weak the
in-sample fit was in Step 9 (logistic regression AUC 0.543, 0/10
coefficients significant) and that this is the best of 16 combos searched,
**this grid should be read as an evaluation methodology check, not
evidence of a real trading edge** — the same overfitting-from-search
caveat that applied to Step 9's threshold search applies here, compounded
across lookback × threshold rather than threshold alone.
