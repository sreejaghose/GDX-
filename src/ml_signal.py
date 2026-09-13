"""
An actual predictive-model trading signal, replacing the hand-specified
reversal rule used everywhere else in this project (src/naive_baseline.py,
src/walk_forward.py, src/backtest.py's default). This module:

  1. Fits a logistic regression on the causal features from src/features.py
     to predict sign(next-day GDX return).
  2. Refits it every quarter on an EXPANDING window (same quarterly
     mechanics as src/walk_forward.py), and predicts only on the next,
     not-yet-seen quarter -- so every prediction is genuinely out-of-
     sample relative to the data that fit it. This is the walk-forward
     discipline that makes "predict t+1 using data through t" a real
     claim rather than an in-sample fit dressed up as one.
  3. Converts each prediction into a trading signal: +1 (long) if
     predicted P(up) > 0.5, else -1 (short) -- always in the market, no
     flat/deadzone state, since a real classifier gives a directional
     call every day (unlike the threshold rule's magnitude-based deadzone).
  4. Feeds the resulting signal through the EXACT SAME execution engine
     (src/backtest.py's run_backtest_on_signal: market-on-close, the same
     entry-t+1/exit-t+2 timing as results/timing_spec.md, the same costs)
     used for the reversal rule, so the two are a fair, apples-to-apples
     comparison.

Train+validation only; holdout untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.backtest import exit_after_n_days, performance_summary, run_backtest_on_signal
from src.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features, load_train_val_panel

OUT_DIR = Path("results/ml_signal")
REPORT_PATH = Path("results/ml_signal_report.md")

MIN_FIT_ROWS = 120  # ~ half a year of daily rows, well past the longest 60-day feature warmup


def walk_forward_ml_signal(features: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Quarterly-refit logistic regression signal. Returns (signal, fit_log)
    where fit_log records each quarter's fit-window size and train accuracy,
    for auditability."""
    quarters = sorted(features.index.to_period("Q").unique())
    quarter_of = features.index.to_period("Q")

    signal_parts = []
    fit_log = []

    for i in range(1, len(quarters)):
        predict_quarter = quarters[i]
        fit_mask = quarter_of < predict_quarter
        predict_mask = quarter_of == predict_quarter

        fit_df = features.loc[fit_mask].dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])
        predict_df = features.loc[predict_mask]

        if len(fit_df) < MIN_FIT_ROWS:
            continue  # not enough expanding-window history yet -- warm-up

        # Zero-lookahead check: every fit-window date strictly earlier than
        # every predict-window date (mirrors src/walk_forward.py's check).
        assert fit_df.index.max() < predict_df.index.min(), (
            f"lookahead: fit window reaches into predict quarter {predict_quarter}"
        )

        X_fit = fit_df[FEATURE_COLUMNS].values
        y_fit = (fit_df[LABEL_COLUMN] > 0).astype(int).values

        scaler = StandardScaler()
        X_fit_scaled = scaler.fit_transform(X_fit)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_fit_scaled, y_fit)
        train_accuracy = model.score(X_fit_scaled, y_fit)

        # Predict on the next quarter. Rows with any missing feature
        # (e.g. GDX/GDXJ relative momentum before GDXJ's 2009 launch)
        # can't be scored -- they get signal 0 (flat), not a guess.
        predict_complete = predict_df.dropna(subset=FEATURE_COLUMNS)
        if len(predict_complete):
            X_predict_scaled = scaler.transform(predict_complete[FEATURE_COLUMNS].values)
            predicted_prob_up = model.predict_proba(X_predict_scaled)[:, 1]
            quarter_signal = pd.Series(
                np.where(predicted_prob_up > 0.5, 1, -1), index=predict_complete.index
            )
            signal_parts.append(quarter_signal)

        fit_log.append({
            "quarter": str(predict_quarter), "fit_window_rows": len(fit_df),
            "train_accuracy": train_accuracy, "n_predicted": len(predict_complete),
        })

    if not signal_parts:
        return pd.Series(dtype=float), fit_log

    signal = pd.concat(signal_parts).sort_index()
    # Reindex to the full feature date range so downstream code sees an
    # aligned series; unpredicted rows (warm-up quarters, missing features)
    # are flat (0), not NaN, so the backtest engine treats them as no-trade
    # rather than erroring.
    signal = signal.reindex(features.index, fill_value=0)
    return signal, fit_log


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_train_val_panel()
    features = build_features(panel)
    print(f"train+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    signal, fit_log = walk_forward_ml_signal(features)
    fit_log_df = pd.DataFrame(fit_log)
    fit_log_df.to_csv(OUT_DIR / "quarterly_fit_log.csv", index=False)
    print(f"\nWalk-forward refit: {len(fit_log_df)} quarters, "
          f"mean train accuracy={fit_log_df['train_accuracy'].mean():.3f}")
    print(fit_log_df.tail(8).to_string(index=False))

    n_long = (signal == 1).sum()
    n_short = (signal == -1).sum()
    n_flat = (signal == 0).sum()
    print(f"\nSignal distribution: {n_long} long, {n_short} short, {n_flat} flat (warm-up/missing-feature days)")

    result = run_backtest_on_signal(
        panel, signal, exit_rule=exit_after_n_days(0),
        account_size=100_000.0, position_fraction=1.0,
        commission_bps=0.0, commission_per_share=0.005, slippage_bps=5.0,
    )
    result.equity_curve.to_csv(OUT_DIR / "equity_curve.csv")
    result.trade_log.to_csv(OUT_DIR / "trade_log.csv", index=False)

    summary = performance_summary(result)
    print(f"\nML-signal backtest (same costs/timing as the frozen reversal-rule config):")
    for k in ["cagr", "sharpe", "max_drawdown", "trades_per_month", "n_trades", "total_return", "win_rate"]:
        print(f"  {k}: {summary[k]}")

    pd.Series(summary).to_frame("value").to_csv(OUT_DIR / "summary.csv")

    write_report(fit_log_df, signal, summary)


def write_report(fit_log_df: pd.DataFrame, signal: pd.Series, summary: dict):
    lines = ["# ML-Driven Signal: Logistic Regression, Walk-Forward Refit\n"]
    lines.append(
        "Replaces the hand-specified reversal-threshold rule with an actual predictive model: "
        "a logistic regression on the ten causal features from `src/features.py` "
        "(trailing returns, realized vol, RSI, GLD/IAU/TLT momentum, GDX/GDXJ relative momentum), "
        "refit every quarter on an expanding window and evaluated only on the next, unseen quarter -- "
        "so every prediction used to trade is genuinely out-of-sample relative to the data that fit it. "
        "Signal: +1 (long) if the model's predicted P(next-day GDX up) > 0.5, else -1 (short) -- always "
        "in the market, no deadzone. Fed through the exact same execution engine (market-on-close, "
        "entry t+1/exit t+2 per `results/timing_spec.md`, $0.005/share commission + 0.05% slippage, "
        "$100,000 account) as the frozen reversal-rule config, for a fair comparison. "
        "Train+validation only; holdout untouched.\n"
    )

    lines.append("## Walk-forward refit summary\n")
    lines.append(
        f"{len(fit_log_df)} quarterly refits, each on an expanding window "
        f"(first refit uses >= {MIN_FIT_ROWS} rows). Mean in-window (fit) accuracy across all refits: "
        f"{fit_log_df['train_accuracy'].mean():.1%} (this is TRAIN accuracy for that quarter's fit, "
        "not out-of-sample -- shown only to track whether the model is fitting anything at all).\n"
    )
    lines.append("Last 8 quarterly refits:\n")
    lines.append("| Quarter | Fit rows | Train accuracy | Predicted rows |")
    lines.append("|---|---|---|---|")
    for _, row in fit_log_df.tail(8).iterrows():
        lines.append(f"| {row['quarter']} | {int(row['fit_window_rows'])} | {row['train_accuracy']:.1%} | {int(row['n_predicted'])} |")
    lines.append("")

    n_long, n_short, n_flat = (signal == 1).sum(), (signal == -1).sum(), (signal == 0).sum()
    lines.append(
        f"Signal distribution over the full period: {n_long} long days, {n_short} short days, "
        f"{n_flat} flat days (warm-up quarters before the first refit, or days with a missing "
        "feature -- e.g. GDX/GDXJ relative momentum before GDXJ's 2009 launch).\n"
    )

    lines.append("## Backtest performance (same costs/timing as the frozen reversal-rule config)\n")
    lines.append("| Metric | ML signal | Frozen reversal-rule config |")
    lines.append("|---|---|---|")
    lines.append(f"| CAGR | {summary['cagr']:.2%} | +0.44% |")
    lines.append(f"| Sharpe | {summary['sharpe']:.3f} | 0.157 |")
    lines.append(f"| Max drawdown | {summary['max_drawdown']:.1%} | -69.5% |")
    lines.append(f"| Trades/month | {summary['trades_per_month']:.2f} | 9.17 |")
    lines.append(f"| Win rate | {summary['win_rate']:.1%} | 48.8% |")
    lines.append("")

    verdict = "better than" if summary["cagr"] > 0.0044 else ("comparable to" if abs(summary["cagr"] - 0.0044) < 0.005 else "worse than")
    lines.append(
        f"The ML-driven signal's in-sample CAGR is **{verdict}** the frozen reversal rule's. "
        "This comparison is informational, not a replacement for the frozen config -- "
        "`results/FROZEN_CONFIG.json` still reflects the process's actual final decision. "
        "Applying the same robustness (parameter/cost/sub-period perturbation) and significance "
        "(t-test, bootstrap) checks used on the reversal rule to this signal before drawing any "
        "conclusion from it is the natural next step, not done here.\n"
    )

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
