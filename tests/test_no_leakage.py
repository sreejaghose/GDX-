"""
Programmatic (not by-inspection) proof that src/features.py has zero
lookahead: every feature at row t must be unaffected by anything that
happens to the price data after t, and the label at row t must depend on
exactly t+1 -- no more, no less.

Uses synthetic, deterministic price data (not the real project data) so
these tests are fast, self-contained, and don't silently pass/fail based on
what happens to live data files. A GDXJ pre-launch NaN gap is included to
mirror the real dataset's shape, since that's exactly the kind of edge case
that breaks naive rolling-window code.

Two independent techniques are used against every feature column:
  1. Truncation invariance -- recompute on data cut off at t; values up to
     and including t must be identical to the full-data computation.
  2. Future-perturbation invariance -- corrupt all prices strictly after t
     (same array length, so this isn't just "ran out of data"); values up
     to and including t must still be identical.
A feature that secretly used future data would fail at least one of these.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CROSS_ASSET_MOMENTUM_WINDOW,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RSI_WINDOW,
    TRAILING_RETURN_WINDOWS,
    VOL_WINDOW,
    build_features,
    verify_no_lookahead,
)

N_ROWS = 400
GDXJ_LAUNCH_ROW = 60
MIN_WARMUP = max(TRAILING_RETURN_WINDOWS) + VOL_WINDOW + 5


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2015-01-01", periods=N_ROWS)
    tickers = ["SPY", "IAU", "GLD", "TLT", "GDX", "GDXJ"]

    panel = pd.DataFrame(index=dates)
    for ticker in tickers:
        log_returns = rng.normal(loc=0.0002, scale=0.015, size=N_ROWS)
        prices = 100 * np.exp(np.cumsum(log_returns))
        panel[ticker] = prices

    # Mirror the real dataset: GDXJ has no data before some launch date.
    panel.loc[panel.index[:GDXJ_LAUNCH_ROW], "GDXJ"] = np.nan

    return panel


def _assert_features_equal_up_to(full: pd.DataFrame, other: pd.DataFrame, upto_idx: int, columns):
    for col in columns:
        full_vals = full[col].iloc[: upto_idx + 1].values
        other_vals = other[col].iloc[: upto_idx + 1].values
        assert np.allclose(full_vals, other_vals, equal_nan=True), (
            f"Feature '{col}' differs on or before row {upto_idx} after a change "
            f"to data strictly after that row -- this means it used future data."
        )


class TestTruncationInvariance:
    """Recomputing on truncated data must reproduce identical past values."""

    @pytest.mark.parametrize("cutoff", [MIN_WARMUP, MIN_WARMUP + 20, 150, 250, N_ROWS - 2])
    def test_features_unchanged_when_future_rows_removed(self, synthetic_panel, cutoff):
        full = build_features(synthetic_panel)
        truncated_panel = synthetic_panel.iloc[: cutoff + 1]
        truncated = build_features(truncated_panel)
        _assert_features_equal_up_to(full, truncated, cutoff, FEATURE_COLUMNS)

    def test_verify_no_lookahead_passes_on_synthetic_data(self, synthetic_panel):
        # This should not raise.
        verify_no_lookahead(synthetic_panel)


class TestFuturePerturbationInvariance:
    """Corrupting future prices (without changing series length) must not
    change any feature value at or before the corruption point."""

    @pytest.mark.parametrize("cutoff", [MIN_WARMUP + 10, 150, 250])
    def test_features_unchanged_when_future_prices_corrupted(self, synthetic_panel, cutoff):
        full = build_features(synthetic_panel)

        corrupted_panel = synthetic_panel.copy()
        rng = np.random.default_rng(seed=123)
        future_rows = corrupted_panel.index[cutoff + 1 :]
        for ticker in corrupted_panel.columns:
            corrupted_panel.loc[future_rows, ticker] = rng.uniform(1, 10_000, size=len(future_rows))

        corrupted = build_features(corrupted_panel)
        _assert_features_equal_up_to(full, corrupted, cutoff, FEATURE_COLUMNS)

    def test_features_unchanged_when_future_prices_are_nan(self, synthetic_panel):
        cutoff = 200
        full = build_features(synthetic_panel)

        corrupted_panel = synthetic_panel.copy()
        future_rows = corrupted_panel.index[cutoff + 1 :]
        corrupted_panel.loc[future_rows, :] = np.nan

        corrupted = build_features(corrupted_panel)
        _assert_features_equal_up_to(full, corrupted, cutoff, FEATURE_COLUMNS)


class TestLabelIsStrictlyNextDay:
    """The label is the one deliberate t+1 dependency -- verify it's
    exactly t+1, not t (leakage) or t+2 (wrong target)."""

    def test_label_equals_independent_next_day_return_recomputation(self, synthetic_panel):
        features = build_features(synthetic_panel)
        recomputed = synthetic_panel["GDX"].shift(-1) / synthetic_panel["GDX"] - 1
        assert np.allclose(features[LABEL_COLUMN].values, recomputed.values, equal_nan=True)

    def test_label_shifted_forward_one_step_equals_same_day_return(self, synthetic_panel):
        """label[t] shifted to sit at t+1 must equal the ordinary same-day
        return computed at t+1 -- i.e. label[t] is GDX[t+1]/GDX[t]-1, not
        GDX[t]/GDX[t-1]-1 (t, leakage) or GDX[t+2]/GDX[t+1]-1 (t+2)."""
        features = build_features(synthetic_panel)
        same_day_return = synthetic_panel["GDX"].pct_change()
        shifted_label = features[LABEL_COLUMN].shift(1)
        valid = shifted_label.notna() & same_day_return.notna()
        assert valid.sum() > 0
        assert np.allclose(shifted_label[valid].values, same_day_return[valid].values)

    def test_label_at_row_t_does_not_equal_same_day_or_two_day_return(self, synthetic_panel):
        """Sanity check that the label is a genuinely different series from
        the t-return and t+2-return, i.e. this isn't a no-op comparison."""
        features = build_features(synthetic_panel)
        same_day_return = synthetic_panel["GDX"].pct_change()
        two_day_ahead_return = synthetic_panel["GDX"].pct_change().shift(-2)

        valid_vs_same_day = features[LABEL_COLUMN].notna() & same_day_return.notna()
        assert not np.allclose(
            features[LABEL_COLUMN][valid_vs_same_day].values,
            same_day_return[valid_vs_same_day].values,
        )

        valid_vs_two_day = features[LABEL_COLUMN].notna() & two_day_ahead_return.notna()
        assert not np.allclose(
            features[LABEL_COLUMN][valid_vs_two_day].values,
            two_day_ahead_return[valid_vs_two_day].values,
        )

    def test_last_row_label_is_nan(self, synthetic_panel):
        """No data exists for t+1 at the final row -- the label must be NaN,
        never fabricated (e.g. by wraparound or forward-fill)."""
        features = build_features(synthetic_panel)
        assert pd.isna(features[LABEL_COLUMN].iloc[-1])


class TestFeatureColumnsPresent:
    def test_all_expected_feature_columns_exist(self, synthetic_panel):
        features = build_features(synthetic_panel)
        for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
            assert col in features.columns

    def test_relative_momentum_uses_only_matching_window_prices(self, synthetic_panel):
        """gdx_gdxj_relative_momentum_20d[t] must equal an independent
        recomputation from GDX/GDXJ at t and t-window only."""
        features = build_features(synthetic_panel)
        ratio = synthetic_panel["GDX"] / synthetic_panel["GDXJ"]
        recomputed = ratio / ratio.shift(CROSS_ASSET_MOMENTUM_WINDOW) - 1
        assert np.allclose(
            features["gdx_gdxj_relative_momentum_20d"].values,
            recomputed.values,
            equal_nan=True,
        )

    def test_rsi_bounded_between_0_and_100(self, synthetic_panel):
        features = build_features(synthetic_panel)
        rsi_vals = features["gdx_rsi_14"].dropna()
        assert (rsi_vals >= 0).all() and (rsi_vals <= 100).all()
