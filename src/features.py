"""
Feature engineering for next-day GDX return prediction.

Uses ONLY data/processed/train.parquet + data/processed/validation.parquet
(concatenated, chronological, contiguous) -- never
data/processed/DO_NOT_TOUCH_holdout.parquet.

Every feature is computed with a strictly causal rolling window: the value
at row t is a function of prices at index positions <= t only (.shift(N)
with N >= 0, .rolling(W) with the window ending at t, never centered and
never .shift(-N)). The label is the one deliberate exception: it is
next-day GDX return, i.e. a function of prices at t and t+1, assigned to
row t.

Features (all on close prices unless noted):
  gdx_trailing_return_{5,10,20,60}d  -- GDX[t]/GDX[t-N] - 1
  gdx_realized_vol_20d               -- rolling 20d std of GDX daily returns,
                                         annualized, window ending at t
  gdx_rsi_14                         -- 14-period RSI on GDX, window ending at t
  gld_momentum_20d                   -- GLD[t]/GLD[t-20] - 1
  iau_momentum_20d                   -- IAU[t]/IAU[t-20] - 1
  tlt_momentum_20d                   -- TLT[t]/TLT[t-20] - 1
  gdx_gdxj_relative_momentum_20d     -- momentum of the GDX/GDXJ price ratio
                                         over 20 days: uses only GDX[t],
                                         GDX[t-20], GDXJ[t], GDXJ[t-20]

Label:
  label_gdx_next_return              -- GDX[t+1]/GDX[t] - 1, assigned to row t

verify_no_lookahead() proves causality programmatically (not by inspection)
via truncation invariance: recomputing every feature on data truncated at
a cutoff must reproduce, exactly, the values the full-data computation
produced up to and including that cutoff. If any feature secretly depended
on data after t, truncating the input would change (or NaN-out) its value
at or before the cutoff, and the check would fail.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_PARQUET_PATH = Path("data/processed/train.parquet")
VALIDATION_PARQUET_PATH = Path("data/processed/validation.parquet")
FEATURES_PARQUET_PATH = Path("data/processed/features_train_val.parquet")

TRAILING_RETURN_WINDOWS = (5, 10, 20, 60)
VOL_WINDOW = 20
RSI_WINDOW = 14
CROSS_ASSET_MOMENTUM_WINDOW = 20

FEATURE_COLUMNS = (
    [f"gdx_trailing_return_{w}d" for w in TRAILING_RETURN_WINDOWS]
    + [
        "gdx_realized_vol_20d",
        "gdx_rsi_14",
        "gld_momentum_20d",
        "iau_momentum_20d",
        "tlt_momentum_20d",
        "gdx_gdxj_relative_momentum_20d",
    ]
)
LABEL_COLUMN = "label_gdx_next_return"


def load_train_val_panel() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PARQUET_PATH)
    val = pd.read_parquet(VALIDATION_PARQUET_PATH)
    if not (train.index.max() < val.index.min()):
        raise ValueError("train and validation splits are not strictly chronological/non-overlapping")
    return pd.concat([train, val]).sort_index()


def trailing_return(prices: pd.Series, window: int) -> pd.Series:
    """prices[t] / prices[t-window] - 1. Causal: uses index t and t-window only."""
    return prices / prices.shift(window) - 1


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized std of daily returns, window ending at t (causal)."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def rsi(prices: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Simple-moving-average RSI. delta[t] = prices[t]-prices[t-1]; the
    rolling mean of gains/losses over [t-window+1, t] uses only data up to
    t, so RSI[t] is causal.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def relative_momentum(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """Momentum of the price_a/price_b ratio over `window` days. Uses only
    price_a[t], price_a[t-window], price_b[t], price_b[t-window].
    """
    ratio = price_a / price_b
    return ratio / ratio.shift(window) - 1


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    gdx = panel["GDX"]
    gdx_returns = gdx.pct_change()

    features = pd.DataFrame(index=panel.index)

    for w in TRAILING_RETURN_WINDOWS:
        features[f"gdx_trailing_return_{w}d"] = trailing_return(gdx, w)

    features["gdx_realized_vol_20d"] = realized_vol(gdx_returns, VOL_WINDOW)
    features["gdx_rsi_14"] = rsi(gdx, RSI_WINDOW)

    features["gld_momentum_20d"] = trailing_return(panel["GLD"], CROSS_ASSET_MOMENTUM_WINDOW)
    features["iau_momentum_20d"] = trailing_return(panel["IAU"], CROSS_ASSET_MOMENTUM_WINDOW)
    features["tlt_momentum_20d"] = trailing_return(panel["TLT"], CROSS_ASSET_MOMENTUM_WINDOW)
    features["gdx_gdxj_relative_momentum_20d"] = relative_momentum(
        panel["GDX"], panel["GDXJ"], CROSS_ASSET_MOMENTUM_WINDOW
    )

    # label_gdx_next_return[t] = GDX[t+1]/GDX[t] - 1.
    # gdx.pct_change() gives r[t] = GDX[t]/GDX[t-1] - 1 at index t;
    # shifting by -1 moves r[t+1] back onto row t.
    features[LABEL_COLUMN] = gdx_returns.shift(-1)

    return features


def verify_no_lookahead(panel: pd.DataFrame, n_checkpoints: int = 6) -> None:
    """Truncation-invariance check. Raises AssertionError on any violation."""
    full = build_features(panel)

    n = len(panel)
    min_cutoff = max(TRAILING_RETURN_WINDOWS) + VOL_WINDOW + 5
    checkpoints = np.linspace(min_cutoff, n - 2, n_checkpoints, dtype=int)

    for cutoff in checkpoints:
        truncated_panel = panel.iloc[: cutoff + 1]  # rows 0..cutoff inclusive
        truncated = build_features(truncated_panel)

        for col in FEATURE_COLUMNS:
            full_vals = full[col].iloc[: cutoff + 1]
            trunc_vals = truncated[col]
            if not np.allclose(full_vals.values, trunc_vals.values, equal_nan=True):
                raise AssertionError(
                    f"Lookahead detected in feature '{col}': values up to cutoff "
                    f"{panel.index[cutoff]} changed when future data was removed."
                )

        # The label is expected to differ/NaN at the cutoff row once future
        # data is removed (that's exactly its t+1 dependency) -- verify that
        # dependency directly instead of asserting invariance.
        assert pd.isna(truncated[LABEL_COLUMN].iloc[-1]), (
            f"label at truncation cutoff {panel.index[cutoff]} should be NaN "
            "once GDX[t+1] is unavailable, but a value was produced -- this "
            "means the label is not using t+1 data as expected."
        )
        # Every row strictly before the cutoff still has its correct label,
        # since GDX[t+1] for those rows is still present in the truncated panel.
        if not np.allclose(
            full[LABEL_COLUMN].iloc[:cutoff].values,
            truncated[LABEL_COLUMN].iloc[:cutoff].values,
            equal_nan=True,
        ):
            raise AssertionError(
                f"label values before cutoff {panel.index[cutoff]} changed under truncation."
            )

    # Direct, independent recomputation of the label from raw prices.
    recomputed_label = panel["GDX"].shift(-1) / panel["GDX"] - 1
    if not np.allclose(full[LABEL_COLUMN].values, recomputed_label.values, equal_nan=True):
        raise AssertionError("label does not match an independent GDX[t+1]/GDX[t]-1 recomputation.")

    # Label at row t must equal same-day return recomputed at row t+1 --
    # i.e. it is exactly one step ahead, not t or t+2.
    same_day_return = panel["GDX"].pct_change()
    shifted_label = full[LABEL_COLUMN].shift(1)
    valid = shifted_label.notna() & same_day_return.notna()
    if not np.allclose(shifted_label[valid].values, same_day_return[valid].values):
        raise AssertionError("label is not aligned to exactly t+1 (off-by-one detected).")


def main():
    panel = load_train_val_panel()
    print(f"train+validation panel: {len(panel)} rows, {panel.index.min().date()} to {panel.index.max().date()}")

    verify_no_lookahead(panel)
    print("Zero-lookahead verification (truncation invariance): PASS for all features and label")

    features = build_features(panel)
    FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PARQUET_PATH)
    print(f"Wrote {FEATURES_PARQUET_PATH}  shape={features.shape}")
    print(f"\nNaN counts per column:\n{features.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
