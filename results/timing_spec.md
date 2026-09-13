# Trading Timing Specification

Written *before* any baseline is computed, per the constraint that all
downstream signal/entry/exit logic must implement this spec exactly — not
default to close-to-close because it's convenient.

## Data constraint that drives this spec

`data/processed/panel.parquet` (and its train/validation/holdout splits)
contains **daily close prices only** — six columns (SPY, IAU, GLD, TLT, GDX,
GDXJ), one row per trading day, no open/high/low/intraday data (confirmed:
`src/build_panel.py` reads a single close-price column per ticker from
`data/raw/gdx_data.xlsx`; there is no open-price series anywhere in this
project). This means "enter at the open" is **not implementable** against
this dataset — there is no open price to enter at. Every decision below is
constrained to use only the close-price series that actually exists.

## 1. Entry time

**Close of day *t+1***, where *t* is the day the signal is fully known (see
§3). Not the close of day *t* itself.

- If entry were at the close of day *t* — the same close used to compute
  the signal — that would require knowing the day-*t* close *before* it
  happens in order to decide to trade into it, which is lookahead by
  construction (the signal literally cannot exist before the print it
  claims to trade on).
- The earliest close that is *both* (a) a real, tradable price and (b)
  strictly later in time than the moment the signal is known is the next
  day's close, day *t+1*.
- In a dataset with open prices, the conventional choice would be the open
  of day *t+1* (minimizes the gap between signal and execution). That price
  doesn't exist here, so close of day *t+1* is the earliest honest
  substitute. This is a real limitation, noted explicitly rather than
  silently assumed: this spec pays for one extra day of latency (the
  day-*t*-close-to-day-*(t+1)*-close move is missed entirely) relative to
  a same-open-day execution.

## 2. Exit time

**Close of day *t+2*** — one trading day after entry. This fixes the
holding period at exactly one trading day, matching the horizon of the
naive reversal baseline (a bet on the next single day's move). Any
model built later against this spec must hold positions for the same
[entry, exit) = [close *t+1*, close *t+2*) window unless a different
horizon is declared and this file is updated to match.

## 3. Step-ahead prediction target

For each ticker and each signal date *t*:

```
target_return_t = P[t+2] / P[t+1] - 1
```

i.e. the close-to-close return realized strictly *between* the entry
timestamp (close *t+1*) and the exit timestamp (close *t+2*). This is the
exact, fully-executable return any strategy sized by a signal known at day
*t* will actually earn — not the return over [t, t+1], and not the
same-index return `pct_change()` would give you by default.

## Signal-to-execution timeline

```
day:            t-1  ---------  t  ---------  t+1  ---------  t+2
price known:    P[t-1]          P[t]          P[t+1]          P[t+2]
                                  |
signal formed --------------------              (uses P[t], P[t-1] only)
                                  |               |
                                  |   entry -------                exit ------
                                  |   close t+1                    close t+2
                                  |___signal-to-entry gap___|
                                       (>= 1 full day, zero lookahead)
```

The signal is a function of `P[t]` and `P[t-1]` only — both already printed
by the moment the signal is "formed." Entry happens at `P[t+1]`, which does
not exist yet at the moment the signal is formed. This ordering is what
`src/naive_baseline.py` builds and verifies programmatically (see
`results/naive_baseline/verification.md`).
