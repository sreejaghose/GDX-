"""
Fit two naive models predicting next-day GDX direction, on the TRAIN split
only. Validation is not touched -- boundaries come from results/split_dates.json.

(a) Threshold rule: for each trailing-return window in
    src.features.TRAILING_RETURN_WINDOWS (5/10/20/60d), search over
    thresholds and both possible directions (momentum: predict up if
    return > threshold; reversal: predict up if return < threshold) for
    the training-accuracy-maximizing rule. This is a single-feature
    decision stump, fit exhaustively rather than via a library, so every
    number in the report is traceable.

(b) Logistic regression: all features from src/features.py, standardized,
    predicting sign(next-day GDX return). Reports coefficients on the
    standardized scale (comparable magnitudes = feature importance) with
    signs, plus training accuracy/AUC.

Both are pure in-sample fit diagnostics -- no claim of out-of-sample skill.
Writes results/naive_models/train_fit_report.md.

Run as a module from the repo root (imports src.features):
    python3 -m src.fit_naive_models
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, LABEL_COLUMN, TRAILING_RETURN_WINDOWS

FEATURES_PARQUET_PATH = Path("data/processed/features_train_val.parquet")
SPLIT_DATES_JSON_PATH = Path("results/split_dates.json")
OUT_DIR = Path("results/naive_models")
REPORT_PATH = OUT_DIR / "train_fit_report.md"

# Rough priors from earlier EDA/spec work, used only for the "does this make
# economic sense" commentary -- not used anywhere in the fitting itself.
EXPECTED_SIGN_NOTES = {
    "gdx_trailing_return_5d": "ambiguous a priori; EDA found ~zero ACF in GDX daily returns, so expect a small/insignificant coefficient",
    "gdx_trailing_return_10d": "ambiguous a priori; same ACF-based expectation of weak signal",
    "gdx_trailing_return_20d": "ambiguous a priori; same ACF-based expectation of weak signal",
    "gdx_trailing_return_60d": "medium-term trend; if anything, weak momentum (positive) is more plausible than at daily horizons, but still expected weak",
    "gdx_realized_vol_20d": "negative expected (leverage/vol-feedback effect: rising realized vol tends to coincide with falling prices)",
    "gdx_rsi_14": "negative expected under mean-reversion/overbought-oversold reasoning, though daily-frequency evidence for this is typically weak",
    "gld_momentum_20d": "weakly positive plausible, but caution: EDA's 0.76 GDX/GLD correlation was CONTEMPORANEOUS (same-day), not predictive -- it says nothing about whether trailing gold momentum forecasts GDX's *next-day* move, so a negative or near-zero coefficient here is not a contradiction of the EDA finding",
    "iau_momentum_20d": "same caveat as GLD; IAU/GLD correlation 0.995 (EDA) means this pair is collinear with gld_momentum_20d, so their individual coefficients (signs included) are not reliably attributable",
    "tlt_momentum_20d": "weakly positive plausible (falling real yields historically support gold/miners), but this is a lagged/predictive claim, not the same as any contemporaneous correlation measured in EDA",
    "gdx_gdxj_relative_momentum_20d": "no strong prior; reflects senior vs junior miner outperformance, not directly gold-price-linked",
}


def load_train_features() -> pd.DataFrame:
    with open(SPLIT_DATES_JSON_PATH) as f:
        split_dates = json.load(f)
    train_end = pd.Timestamp(split_dates["train"]["end_date"])

    features = pd.read_parquet(FEATURES_PARQUET_PATH)
    train_features = features.loc[features.index <= train_end]
    if train_features.index.max() != train_end:
        raise ValueError("train feature slice does not reach the expected train end date")
    return train_features


def fit_threshold_rules(features: pd.DataFrame) -> list[dict]:
    """Each window is fit on its own maximal complete subset (that feature
    column + label, non-null) -- NOT on the intersection with every other
    feature, so a single feature's shorter history (e.g. the GDXJ-derived
    relative-momentum feature) doesn't needlessly shrink these single-
    feature fits.
    """
    results = []
    for window in TRAILING_RETURN_WINDOWS:
        col = f"gdx_trailing_return_{window}d"
        subset = features[[col, LABEL_COLUMN]].dropna()
        x = subset[col].values
        y = (subset[LABEL_COLUMN] > 0).astype(int).values

        base_rate = y.mean()
        majority_baseline = max(base_rate, 1 - base_rate)
        candidate_thresholds = np.unique(np.percentile(x, np.linspace(1, 99, 197)))

        best = {"accuracy": -1.0}
        for threshold in candidate_thresholds:
            for direction, pred in [
                ("momentum (predict up if return > threshold)", (x > threshold).astype(int)),
                ("reversal (predict up if return < threshold)", (x < threshold).astype(int)),
            ]:
                acc = (pred == y).mean()
                if acc > best["accuracy"]:
                    best = {"window": window, "threshold": float(threshold),
                            "direction": direction, "accuracy": float(acc)}

        results.append({**best, "n_obs": len(subset), "majority_baseline": float(majority_baseline),
                         "lift_over_baseline": float(best["accuracy"] - majority_baseline)})
    return results


def fit_logistic_regression(features: pd.DataFrame) -> dict:
    """Uses the intersection of all features + label being non-null --
    necessarily the most restrictive subset, since it's a multivariate fit.
    """
    df = features[FEATURE_COLUMNS + [LABEL_COLUMN]].dropna()
    X = df[FEATURE_COLUMNS].values
    y = (df[LABEL_COLUMN] > 0).astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    pred_proba = model.predict_proba(X_scaled)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    accuracy = (pred == y).mean()
    auc = roc_auc_score(y, pred_proba)

    # statsmodels.Logit gives p-values/std errors on the same standardized
    # features (unpenalized MLE) -- sklearn's LogisticRegression doesn't
    # expose these directly, so this is a second, complementary fit purely
    # for significance testing, not a different model.
    X_sm = sm.add_constant(X_scaled)
    sm_result = sm.Logit(y, X_sm).fit(disp=0)
    p_values = sm_result.pvalues[1:]  # drop the intercept p-value

    coef_table = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "standardized_coef": model.coef_[0],
        "raw_coef": model.coef_[0] / scaler.scale_,
        "odds_ratio_per_1sd": np.exp(model.coef_[0]),
        "p_value": p_values,
    }).sort_values("standardized_coef", key=np.abs, ascending=False).reset_index(drop=True)

    return {
        "n_obs": len(y),
        "base_rate_up": float(y.mean()),
        "intercept": float(model.intercept_[0]),
        "train_accuracy": float(accuracy),
        "train_auc": float(auc),
        "coef_table": coef_table,
    }


def write_report(threshold_results: list[dict], logit_results: dict, n_total: int):
    lines = ["# Naive Model Fits: Train Split Only (In-Sample Diagnostics)\n"]
    lines.append(
        f"Fit on `data/processed/features_train_val.parquet`, filtered to the train "
        f"date range only (per `results/split_dates.json`, {n_total} train rows). "
        f"Validation is not touched.\n"
    )
    lines.append(
        "Each threshold rule (a) is fit on its own maximal complete subset (its one "
        "feature + label, non-null) -- **not** the intersection with every other "
        "feature. The logistic regression (b) necessarily uses the intersection of "
        "all ten features + label, which is smaller: `gdx_gdxj_relative_momentum_20d` "
        "is undefined until ~40 trading days after GDXJ's 2009-11-11 launch, so the "
        "joint model effectively only sees the train split from late 2009 onward. "
        "This is called out explicitly below rather than silently shrinking every "
        "model's sample to match the most-restrictive feature.\n"
    )
    lines.append(
        "**These are in-sample fit diagnostics only** -- accuracy/AUC here reflect "
        "fit to the same data the models were trained on and are expected to look "
        "optimistic. No claim of out-of-sample skill is made; validation-split "
        "evaluation is a separate, later step.\n"
    )

    lines.append("## (a) Threshold rule on trailing-N-day return\n")
    lines.append(
        "For each window, every training-accuracy-maximizing (threshold, direction) "
        "pair was found by exhaustive search over 1st-99th percentile thresholds, in "
        "both a momentum and a reversal direction.\n"
    )
    lines.append("| Window | n obs | Best direction | Threshold | Train accuracy | Majority-class baseline | Lift |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in threshold_results:
        lines.append(
            f"| {r['window']}d | {r['n_obs']} | {r['direction']} | {r['threshold']*100:.2f}% "
            f"| {r['accuracy']:.1%} | {r['majority_baseline']:.1%} | "
            f"{r['lift_over_baseline']*100:+.2f} pp |"
        )
    lines.append("")
    best_overall = max(threshold_results, key=lambda r: r["lift_over_baseline"])
    lines.append(
        f"**Best window: {best_overall['window']}d**, lift of "
        f"{best_overall['lift_over_baseline']*100:+.2f} percentage points over always "
        f"predicting the majority class. "
    )
    if best_overall["lift_over_baseline"] < 0.02:
        lines.append(
            "That lift is tiny (well within what an exhaustive threshold search would "
            "find by chance alone, given ~1,800 training days and hundreds of candidate "
            "thresholds tried) -- consistent with the near-zero autocorrelation found "
            "in the EDA step (`results/eda_summary.md`): trailing return alone does not "
            "carry meaningful next-day directional signal for GDX at these horizons.\n"
        )
    else:
        lines.append(
            "This is a larger lift than the EDA's near-zero ACF finding would suggest; "
            "treat with suspicion pending out-of-sample validation -- an exhaustive "
            "threshold search over many candidates is prone to in-sample overfitting.\n"
        )

    lines.append("## (b) Logistic regression on all Step 7 features\n")
    lines.append(
        f"n = {logit_results['n_obs']}, base rate (fraction of up days) = "
        f"{logit_results['base_rate_up']:.1%}, intercept = {logit_results['intercept']:.4f}\n"
    )
    lines.append(
        f"Train accuracy: {logit_results['train_accuracy']:.1%} "
        f"(majority baseline: {max(logit_results['base_rate_up'], 1 - logit_results['base_rate_up']):.1%}). "
        f"Train AUC: {logit_results['train_auc']:.3f}\n"
    )
    lines.append(
        "Coefficients are reported on standardized features (mean 0, unit variance), "
        "so magnitude is directly comparable across features and doubles as an "
        "importance ranking. Raw (unstandardized) coefficients and per-1-SD odds "
        "ratios are also given. P-values come from an unpenalized `statsmodels.Logit` "
        "fit on the same standardized features, purely for significance testing "
        "(not a different model from the sklearn fit above).\n"
    )
    n_significant = int((logit_results["coef_table"]["p_value"] < 0.05).sum())
    lines.append(
        f"**{n_significant} of {len(FEATURE_COLUMNS)} feature coefficients are "
        f"significant at p < 0.05** (no multiple-testing correction applied; with "
        f"10 tests, roughly 0.5 significant results would be expected by chance "
        f"alone at that threshold).\n"
    )
    lines.append("| Rank | Feature | Standardized coef | Sign | p-value | Odds ratio / 1 SD | Raw coef | Economic expectation |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, row in logit_results["coef_table"].iterrows():
        sign = "+" if row["standardized_coef"] > 0 else "-"
        note = EXPECTED_SIGN_NOTES.get(row["feature"], "")
        sig_marker = "**" if row["p_value"] < 0.05 else ""
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['standardized_coef']:+.4f} | {sign} "
            f"| {sig_marker}{row['p_value']:.3f}{sig_marker} "
            f"| {row['odds_ratio_per_1sd']:.3f} | {row['raw_coef']:+.4f} | {note} |"
        )
    lines.append("")

    lines.append("### Does the sign make economic sense?\n")
    lines.append(
        "**Important distinction before reading signs below**: the EDA's 0.76 "
        "GDX/GLD correlation (`results/eda_summary.md`) is a *contemporaneous* "
        "(same-day) return correlation. The coefficients here answer a different "
        "question -- whether the *trailing* 20-day move in gold *predicts* GDX's "
        "*next-day* return. These are not the same relationship, and a coefficient "
        "sign that disagrees with the contemporaneous correlation is not by itself "
        "a red flag.\n"
    )
    coef_by_feature = dict(zip(logit_results["coef_table"]["feature"], logit_results["coef_table"]["standardized_coef"]))

    gld_sign = coef_by_feature["gld_momentum_20d"]
    iau_sign = coef_by_feature["iau_momentum_20d"]
    lines.append(
        f"- **Gold momentum**: `gld_momentum_20d` coefficient is {gld_sign:+.4f}, "
        f"`iau_momentum_20d` is {iau_sign:+.4f} -- {'both negative' if gld_sign < 0 and iau_sign < 0 else 'mixed signs'}, "
        f"small in magnitude, and **not directly interpretable individually**: IAU "
        f"and GLD are near-duplicate series (0.995 correlation, per EDA), so this "
        f"pair is collinear and logistic regression arbitrarily splits (and can "
        f"flip the sign of) credit between them. The combined direction "
        f"({gld_sign + iau_sign:+.4f} summed) is the more trustworthy quantity, and "
        f"it says trailing gold momentum had, if anything, a weak negative "
        f"association with next-day GDX direction in this sample -- plausible as "
        f"short-horizon overreaction/reversal, but given the model's near-random "
        f"AUC ({logit_results['train_auc']:.3f}) this should not be over-read."
    )

    tlt_sign = coef_by_feature["tlt_momentum_20d"]
    lines.append(
        f"- **TLT momentum**: {tlt_sign:+.4f} "
        f"({'positive, consistent with falling yields supporting gold miners' if tlt_sign > 0 else 'negative -- opposite of the falling-yields-support-gold prior'}), "
        "though again this is a small-magnitude coefficient in a low-AUC model."
    )

    vol_sign = coef_by_feature["gdx_realized_vol_20d"]
    lines.append(
        f"- **Realized volatility**: {vol_sign:+.4f} "
        f"({'negative, consistent with the leverage/vol-feedback effect' if vol_sign < 0 else 'positive -- opposite of the typical vol-feedback prior'})."
    )

    rsi_sign = coef_by_feature["gdx_rsi_14"]
    lines.append(
        f"- **RSI(14)**: {rsi_sign:+.4f} "
        f"({'negative, consistent with mean-reversion/overbought-oversold reasoning' if rsi_sign < 0 else 'positive -- momentum-consistent rather than mean-reversion-consistent'})."
    )

    trailing_coefs = {w: coef_by_feature[f"gdx_trailing_return_{w}d"] for w in TRAILING_RETURN_WINDOWS}
    max_abs_trailing = max(abs(v) for v in trailing_coefs.values())
    lines.append(
        f"- **Own trailing returns (5/10/20/60d)**: standardized coefficients are "
        f"{', '.join(f'{w}d={v:+.4f}' for w, v in trailing_coefs.items())} -- "
        f"all small relative to the top-ranked features"
        f"{' (largest magnitude ' + f'{max_abs_trailing:.4f}' + ')' if True else ''}. "
        "This is consistent with the EDA's near-zero return autocorrelation finding: "
        "GDX's own past return is a weak predictor of its next-day direction, and any "
        "signal here should be treated skeptically pending validation."
    )

    lines.append(
        f"\n**Caveat**: with train accuracy of {logit_results['train_accuracy']:.1%} "
        f"against a {max(logit_results['base_rate_up'], 1 - logit_results['base_rate_up']):.1%} "
        f"majority baseline and AUC of {logit_results['train_auc']:.3f} (0.5 = "
        "coin flip), this model has very little in-sample explanatory power to "
        "begin with. Sign and significance are useful for a sanity check against "
        "priors, but no economically meaningful predictive relationship should be "
        "claimed from this fit alone -- and importantly, none of this has been "
        "checked against validation data yet.\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main():
    features = load_train_features()
    n_total = len(features)
    print(f"Train rows: {n_total}")

    threshold_results = fit_threshold_rules(features)
    print("\nThreshold rule results:")
    for r in threshold_results:
        print(f"  {r['window']}d (n={r['n_obs']}): {r['direction']}, threshold={r['threshold']:.4f}, "
              f"accuracy={r['accuracy']:.3f} (baseline {r['majority_baseline']:.3f})")

    logit_results = fit_logistic_regression(features)
    print(f"\nLogistic regression: n={logit_results['n_obs']}, "
          f"train accuracy={logit_results['train_accuracy']:.3f}, "
          f"train AUC={logit_results['train_auc']:.3f}")
    print(logit_results["coef_table"].to_string())

    write_report(threshold_results, logit_results, n_total)


if __name__ == "__main__":
    main()
