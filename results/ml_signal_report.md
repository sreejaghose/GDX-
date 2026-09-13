# ML-Driven Signal: Logistic Regression, Walk-Forward Refit

Replaces the hand-specified reversal-threshold rule with an actual predictive model: a logistic regression on the ten causal features from `src/features.py` (trailing returns, realized vol, RSI, GLD/IAU/TLT momentum, GDX/GDXJ relative momentum), refit every quarter on an expanding window and evaluated only on the next, unseen quarter -- so every prediction used to trade is genuinely out-of-sample relative to the data that fit it. Signal: +1 (long) if the model's predicted P(next-day GDX up) > 0.5, else -1 (short) -- always in the market, no deadzone. Fed through the exact same execution engine (market-on-close, entry t+1/exit t+2 per `results/timing_spec.md`, $0.005/share commission + 0.05% slippage, $100,000 account) as the frozen reversal-rule config, for a fair comparison. Train+validation only; holdout untouched.

## Walk-forward refit summary

34 quarterly refits, each on an expanding window (first refit uses >= 120 rows). Mean in-window (fit) accuracy across all refits: 54.4% (this is TRAIN accuracy for that quarter's fit, not out-of-sample -- shown only to track whether the model is fitting anything at all).

Last 8 quarterly refits:

| Quarter | Fit rows | Train accuracy | Predicted rows |
|---|---|---|---|
| 2017Q1 | 1777 | 54.4% | 62 |
| 2017Q2 | 1839 | 53.9% | 63 |
| 2017Q3 | 1902 | 54.0% | 63 |
| 2017Q4 | 1965 | 53.4% | 63 |
| 2018Q1 | 2028 | 53.4% | 61 |
| 2018Q2 | 2089 | 53.3% | 64 |
| 2018Q3 | 2153 | 53.2% | 63 |
| 2018Q4 | 2216 | 53.2% | 34 |

Signal distribution over the full period: 980 long days, 1131 short days, 1035 flat days (warm-up quarters before the first refit, or days with a missing feature -- e.g. GDX/GDXJ relative momentum before GDXJ's 2009 launch).

## Backtest performance (same costs/timing as the frozen reversal-rule config)

| Metric | ML signal | Frozen reversal-rule config |
|---|---|---|
| CAGR | -13.14% | +0.44% |
| Sharpe | -0.564 | 0.157 |
| Max drawdown | -85.5% | -69.5% |
| Trades/month | 6.99 | 9.17 |
| Win rate | 48.3% | 48.8% |

The ML-driven signal's in-sample CAGR is **worse than** the frozen reversal rule's. This comparison is informational, not a replacement for the frozen config -- `results/FROZEN_CONFIG.json` still reflects the process's actual final decision. Applying the same robustness (parameter/cost/sub-period perturbation) and significance (t-test, bootstrap) checks used on the reversal rule to this signal before drawing any conclusion from it is the natural next step, not done here.
