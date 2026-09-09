"""
Plots for results/robustness_report.md. Reads the CSVs already written by
src/robustness_check.py (results/robustness/*.csv) plus one re-run of the
baseline config (for the equity curve) rather than duplicating any of that
script's analysis logic.

Writes four PNGs to results/robustness/:
  plot_perturbation_heatmap.png  -- Sharpe across the lookback x threshold grid
  plot_subperiod_equity.png      -- baseline equity curve with sub-period
                                     boundaries and profitable/unprofitable shading
  plot_cost_sensitivity.png      -- 1x vs 2x costs, CAGR and Sharpe
  plot_trades_per_month.png      -- trades/month per sub-period vs. the >=1 bar
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import exit_after_n_days, run_backtest
from src.features import load_train_val_panel
from src.robustness_check import load_best_config

OUT_DIR = Path("results/robustness")

# Dataviz skill reference palette (light mode) -- same constants used in
# src/eda.py and src/backtest.py's equity_curve_comparison.png.
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#e34948", "#f0efec", "#2a78d6"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "grid.color": GRIDLINE,
    "font.family": "sans-serif", "axes.grid": True, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_perturbation_heatmap():
    grid = pd.read_csv(OUT_DIR / "perturbation_grid.csv")
    lookbacks = sorted(grid["lookback"].unique())
    thresholds = sorted(grid["threshold"].unique())

    sharpe_matrix = grid.pivot(index="lookback", columns="threshold", values="sharpe").loc[lookbacks, thresholds]
    cagr_matrix = grid.pivot(index="lookback", columns="threshold", values="cagr").loc[lookbacks, thresholds]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])
    vmax = np.abs(sharpe_matrix.values).max()
    im = ax.imshow(sharpe_matrix.values, cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(thresholds)))
    ax.set_yticks(range(len(lookbacks)))
    ax.set_xticklabels([f"{t:.1%}" for t in thresholds])
    ax.set_yticklabels([f"{lb:.0f}d" for lb in lookbacks])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Lookback")
    ax.grid(False)

    baseline_lb, baseline_th = 20, 0.02
    for i, lb in enumerate(lookbacks):
        for j, th in enumerate(thresholds):
            sharpe_val = sharpe_matrix.loc[lb, th]
            cagr_val = cagr_matrix.loc[lb, th]
            is_baseline = (lb == baseline_lb and abs(th - baseline_th) < 1e-9)
            text_color = "#ffffff" if abs(sharpe_val) > vmax * 0.6 else INK_PRIMARY
            label = f"{sharpe_val:.2f}\n({cagr_val*100:+.1f}%)"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color,
                    fontweight="bold" if is_baseline else "normal")
            if is_baseline:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK_PRIMARY, linewidth=2.5))

    fig.colorbar(im, ax=ax, label="Sharpe ratio")
    ax.set_title("Perturbation grid: Sharpe (CAGR in parens), baseline outlined",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    out = OUT_DIR / "plot_perturbation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_subperiod_equity(base: dict):
    panel = load_train_val_panel()
    result = run_backtest(
        panel, lookback=base["lookback"], threshold=base["threshold"],
        exit_rule=exit_after_n_days(0), account_size=base["account_size"],
        position_fraction=1.0, commission_bps=0.0,
        commission_per_share=base["commission_per_share"], slippage_bps=base["slippage_bps"],
    )
    equity = result.equity_curve
    subperiods = pd.read_csv(OUT_DIR / "subperiod_performance.csv", parse_dates=["start_date", "end_date"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for _, row in subperiods.iterrows():
        color = GOOD if row["profitable"] else CRITICAL
        ax.axvspan(row["start_date"], row["end_date"], color=color, alpha=0.08, linewidth=0)

    ax.plot(equity.index, equity.values, color=BLUE, linewidth=1.4)
    ax.axhline(base["account_size"], color=INK_MUTED, linewidth=0.8, linestyle="--")

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.10)  # headroom for the labels below
    label_y = ax.get_ylim()[1] * 0.97
    for _, row in subperiods.iterrows():
        mid = row["start_date"] + (row["end_date"] - row["start_date"]) / 2
        status = "profitable" if row["profitable"] else "unprofitable"
        status_color = GOOD if row["profitable"] else CRITICAL
        ax.text(mid, label_y, f"#{int(row['subperiod'])}: {status}\n{row['cagr']*100:+.1f}% CAGR",
                ha="center", va="top", fontsize=8.5, color=status_color, fontweight="bold")

    ax.set_ylabel("Equity ($)")
    ax.set_title(f"Baseline equity curve by sub-period (lookback={base['lookback']}d, threshold={base['threshold']:.1%})",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    out = OUT_DIR / "plot_subperiod_equity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cost_sensitivity():
    cost = pd.read_csv(OUT_DIR / "cost_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(cost))
    colors = [BLUE, ORANGE]

    ax = axes[0]
    bars = ax.bar(x, cost["cagr"] * 100, color=colors, width=0.55)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cost["cost_scenario"])
    ax.set_ylabel("CAGR (%)")
    ax.set_title("CAGR", color=INK_PRIMARY, fontsize=11, loc="left")
    y_min, y_max = (cost["cagr"] * 100).min(), (cost["cagr"] * 100).max()
    pad = max(y_max - y_min, 1) * 0.25
    ax.set_ylim(y_min - pad, y_max + pad)
    for bar, val in zip(bars, cost["cagr"]):
        offset = pad * 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + (offset if val >= 0 else -offset),
                f"{val*100:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color=INK_PRIMARY)

    ax = axes[1]
    bars = ax.bar(x, cost["sharpe"], color=colors, width=0.55)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cost["cost_scenario"])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe", color=INK_PRIMARY, fontsize=11, loc="left")
    y_min, y_max = cost["sharpe"].min(), cost["sharpe"].max()
    pad = max(y_max - y_min, 0.1) * 0.25
    ax.set_ylim(y_min - pad, y_max + pad)
    for bar, val in zip(bars, cost["sharpe"]):
        offset = pad * 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, val + (offset if val >= 0 else -offset),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color=INK_PRIMARY)

    fig.suptitle("Cost sensitivity: 1x vs. 2x commission and slippage", color=INK_PRIMARY, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "plot_cost_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_trades_per_month():
    subperiods = pd.read_csv(OUT_DIR / "subperiod_performance.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = subperiods["subperiod"]
    colors = [GOOD if v else CRITICAL for v in subperiods["trades_per_month_gte_1"]]
    bars = ax.bar(x, subperiods["trades_per_month"], color=colors, width=0.55)
    ax.axhline(1, color=INK_MUTED, linewidth=1.2, linestyle="--", label="Minimum bar (1/month)")

    ax.set_ylim(0, subperiods["trades_per_month"].max() * 1.18)
    for bar, val in zip(bars, subperiods["trades_per_month"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}", ha="center", fontsize=9, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Sub-period {i}" for i in x])
    ax.set_ylabel("Trades / month")
    ax.set_title("Trade frequency by sub-period", color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "plot_trades_per_month.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    base = load_best_config()
    plot_perturbation_heatmap()
    plot_subperiod_equity(base)
    plot_cost_sensitivity()
    plot_trades_per_month()


if __name__ == "__main__":
    main()
