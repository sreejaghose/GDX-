# Naive Baseline: Lookahead Verification & Performance

Built against results/timing_spec.md. Train split only.

## Zero-lookahead verification: ALL PASS

### SPY
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1885 usable rows)

### IAU
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1885 usable rows)

### GLD
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1885 usable rows)

### TLT
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1885 usable rows)

### GDX
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1885 usable rows)

### GDXJ
- entry_date > signal_date for all rows: True
- exit_date > entry_date for all rows: True
- entry/exit dates are exactly t+1/t+2 by row position: True
- signal recomputed independently from P[t], P[t-1] matches: True
- target_return recomputed independently from P[t+1], P[t+2] matches: True
- **PASS** (1009 usable rows)

## Naive baseline performance (train split, informational only)

| Ticker | n trades | hit rate | mean return | ann. return | ann. vol | Sharpe |
|---|---|---|---|---|---|---|
| SPY | 1876 | 49.3% | 0.038% | 9.6% | 22.3% | 0.43 |
| IAU | 1856 | 49.0% | 0.033% | 8.3% | 21.7% | 0.38 |
| GLD | 1877 | 49.9% | 0.036% | 9.1% | 21.4% | 0.43 |
| TLT | 1878 | 51.0% | 0.018% | 4.4% | 15.5% | 0.29 |
| GDX | 1878 | 50.9% | 0.092% | 23.1% | 45.0% | 0.51 |
| GDXJ | 1007 | 49.0% | -0.028% | -7.1% | 44.1% | -0.16 |

No costs, slippage, or sizing applied -- this is a signal-timing sanity check, not a performance claim.