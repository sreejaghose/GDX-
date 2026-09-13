# Naive Model Fits: Train Split Only (In-Sample Diagnostics)

Fit on `data/processed/features_train_val.parquet`, filtered to the train date range only (per `results/split_dates.json`, 1888 train rows). Validation is not touched.

Each threshold rule (a) is fit on its own maximal complete subset (its one feature + label, non-null) -- **not** the intersection with every other feature. The logistic regression (b) necessarily uses the intersection of all ten features + label, which is smaller: `gdx_gdxj_relative_momentum_20d` is undefined until ~40 trading days after GDXJ's 2009-11-11 launch, so the joint model effectively only sees the train split from late 2009 onward. This is called out explicitly below rather than silently shrinking every model's sample to match the most-restrictive feature.

**These are in-sample fit diagnostics only** -- accuracy/AUC here reflect fit to the same data the models were trained on and are expected to look optimistic. No claim of out-of-sample skill is made; validation-split evaluation is a separate, later step.

## (a) Threshold rule on trailing-N-day return

For each window, every training-accuracy-maximizing (threshold, direction) pair was found by exhaustive search over 1st-99th percentile thresholds, in both a momentum and a reversal direction.

| Window | n obs | Best direction | Threshold | Train accuracy | Majority-class baseline | Lift |
|---|---|---|---|---|---|---|
| 5d | 1883 | reversal (predict up if return < threshold) | -0.24% | 52.9% | 50.8% | +2.12 pp |
| 10d | 1878 | reversal (predict up if return < threshold) | -3.61% | 52.4% | 50.8% | +1.60 pp |
| 20d | 1868 | reversal (predict up if return < threshold) | -11.10% | 52.9% | 50.7% | +2.19 pp |
| 60d | 1828 | reversal (predict up if return < threshold) | -5.14% | 52.8% | 50.7% | +2.13 pp |

**Best window: 20d**, lift of +2.19 percentage points over always predicting the majority class. 
This is a larger lift than the EDA's near-zero ACF finding would suggest; treat with suspicion pending out-of-sample validation -- an exhaustive threshold search over many candidates is prone to in-sample overfitting.

## (b) Logistic regression on all Step 7 features

n = 992, base rate (fraction of up days) = 49.1%, intercept = -0.0361

Train accuracy: 53.2% (majority baseline: 50.9%). Train AUC: 0.543

Coefficients are reported on standardized features (mean 0, unit variance), so magnitude is directly comparable across features and doubles as an importance ranking. Raw (unstandardized) coefficients and per-1-SD odds ratios are also given. P-values come from an unpenalized `statsmodels.Logit` fit on the same standardized features, purely for significance testing (not a different model from the sklearn fit above).

**0 of 10 feature coefficients are significant at p < 0.05** (no multiple-testing correction applied; with 10 tests, roughly 0.5 significant results would be expected by chance alone at that threshold).

| Rank | Feature | Standardized coef | Sign | p-value | Odds ratio / 1 SD | Raw coef | Economic expectation |
|---|---|---|---|---|---|---|---|
| 1 | `gdx_rsi_14` | -0.1380 | - | 0.261 | 0.871 | -0.0090 | negative expected under mean-reversion/overbought-oversold reasoning, though daily-frequency evidence for this is typically weak |
| 2 | `gdx_trailing_return_5d` | -0.1092 | - | 0.199 | 0.897 | -2.2460 | ambiguous a priori; EDA found ~zero ACF in GDX daily returns, so expect a small/insignificant coefficient |
| 3 | `gdx_gdxj_relative_momentum_20d` | -0.1041 | - | 0.221 | 0.901 | -2.1937 | no strong prior; reflects senior vs junior miner outperformance, not directly gold-price-linked |
| 4 | `gdx_trailing_return_10d` | +0.0922 | + | 0.413 | 1.097 | +1.3907 | ambiguous a priori; same ACF-based expectation of weak signal |
| 5 | `tlt_momentum_20d` | +0.0771 | + | 0.364 | 1.080 | +1.9790 | weakly positive plausible (falling real yields historically support gold/miners), but this is a lagged/predictive claim, not the same as any contemporaneous correlation measured in EDA |
| 6 | `gld_momentum_20d` | -0.0477 | - | 0.518 | 0.953 | -0.9736 | weakly positive plausible, but caution: EDA's 0.76 GDX/GLD correlation was CONTEMPORANEOUS (same-day), not predictive -- it says nothing about whether trailing gold momentum forecasts GDX's *next-day* move, so a negative or near-zero coefficient here is not a contradiction of the EDA finding |
| 7 | `iau_momentum_20d` | -0.0369 | - | 0.528 | 0.964 | -0.7532 | same caveat as GLD; IAU/GLD correlation 0.995 (EDA) means this pair is collinear with gld_momentum_20d, so their individual coefficients (signs included) are not reliably attributable |
| 8 | `gdx_trailing_return_20d` | +0.0330 | + | 0.843 | 1.034 | +0.3847 | ambiguous a priori; same ACF-based expectation of weak signal |
| 9 | `gdx_trailing_return_60d` | +0.0091 | + | 0.877 | 1.009 | +0.0693 | medium-term trend; if anything, weak momentum (positive) is more plausible than at daily horizons, but still expected weak |
| 10 | `gdx_realized_vol_20d` | +0.0066 | + | 0.896 | 1.007 | +0.0647 | negative expected (leverage/vol-feedback effect: rising realized vol tends to coincide with falling prices) |

### Does the sign make economic sense?

**Important distinction before reading signs below**: the EDA's 0.76 GDX/GLD correlation (`results/eda_summary.md`) is a *contemporaneous* (same-day) return correlation. The coefficients here answer a different question -- whether the *trailing* 20-day move in gold *predicts* GDX's *next-day* return. These are not the same relationship, and a coefficient sign that disagrees with the contemporaneous correlation is not by itself a red flag.

- **Gold momentum**: `gld_momentum_20d` coefficient is -0.0477, `iau_momentum_20d` is -0.0369 -- both negative, small in magnitude, and **not directly interpretable individually**: IAU and GLD are near-duplicate series (0.995 correlation, per EDA), so this pair is collinear and logistic regression arbitrarily splits (and can flip the sign of) credit between them. The combined direction (-0.0846 summed) is the more trustworthy quantity, and it says trailing gold momentum had, if anything, a weak negative association with next-day GDX direction in this sample -- plausible as short-horizon overreaction/reversal, but given the model's near-random AUC (0.543) this should not be over-read.
- **TLT momentum**: +0.0771 (positive, consistent with falling yields supporting gold miners), though again this is a small-magnitude coefficient in a low-AUC model.
- **Realized volatility**: +0.0066 (positive -- opposite of the typical vol-feedback prior).
- **RSI(14)**: -0.1380 (negative, consistent with mean-reversion/overbought-oversold reasoning).
- **Own trailing returns (5/10/20/60d)**: standardized coefficients are 5d=-0.1092, 10d=+0.0922, 20d=+0.0330, 60d=+0.0091 -- all small relative to the top-ranked features (largest magnitude 0.1092). This is consistent with the EDA's near-zero return autocorrelation finding: GDX's own past return is a weak predictor of its next-day direction, and any signal here should be treated skeptically pending validation.

**Caveat**: with train accuracy of 53.2% against a 50.9% majority baseline and AUC of 0.543 (0.5 = coin flip), this model has very little in-sample explanatory power to begin with. Sign and significance are useful for a sanity check against priors, but no economically meaningful predictive relationship should be claimed from this fit alone -- and importantly, none of this has been checked against validation data yet.
