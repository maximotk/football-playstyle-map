import pytest
import pandas as pd
import numpy as np

from project_code.src.tab1 import preprocessing


@pytest.fixture
def toy_df_with_missing():
    """Create a toy dataframe with missing values and one zero-variance column."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "team": ["A", "B", "C", "D"],
        "feat1": [1.0, 2.0, np.nan, 4.0],
        "feat2": [np.nan, np.nan, 3.0, 4.0],
        "feat3": [5.0, 5.0, 5.0, 5.0],   # <-- zero variance
    })


def test_analyze_and_handle_missing_values_drops_and_imputes(toy_df_with_missing):
    """Test that features with many missing values are dropped and others imputed."""
    cleaned = preprocessing.analyze_and_handle_missing_values(
        toy_df_with_missing,
        context_features=["id", "team"],
        threshold_drop=50
    )
    assert "feat2" not in cleaned.columns  # dropped (>=50% missing)
    assert cleaned.isna().sum().sum() == 0  # all imputed


def test_analyze_and_handle_constants_removes_low_info(toy_df_with_missing):
    """Test that constant features are removed."""
    cleaned = preprocessing.analyze_and_handle_constants(
        toy_df_with_missing,
        context_features=["id", "team"]
    )
    assert "feat3" not in cleaned.columns  # removed because constant


def test_standardize_features_raises_on_zero_variance(toy_df_with_missing):
    """Test that zero-variance features raise an AssertionError."""
    df = toy_df_with_missing.drop(columns=["feat2"])  # leave feat3 constant
    with pytest.raises(AssertionError, match="Zero-variance columns"):
        preprocessing.standardize_features(df, context_features=["id", "team"], scaler="standard")


def test_standardize_features_invalid_scaler_raises(toy_df_with_missing):
    """Test invalid scaler name raises ValueError."""
    # Remove zero-variance col to avoid triggering AssertionError first
    df = toy_df_with_missing.drop(columns=["feat2", "feat3"])
    with pytest.raises(ValueError, match="invalid"):
        preprocessing.standardize_features(df, context_features=["id", "team"], scaler="invalid")


def test_standardize_features_standard_scaler(toy_df_with_missing):
    """Test features are standardized with StandardScaler."""
    # Remove zero-variance col to allow scaling
    df = toy_df_with_missing.drop(columns=["feat2", "feat3"])
    cleaned = preprocessing.standardize_features(df, context_features=["id", "team"], scaler="standard")
    # check mean ~ 0, std ~ 1
    features = [c for c in cleaned.columns if c not in ["id", "team"]]
    means = cleaned[features].mean().round()
    stds = cleaned[features].std(ddof=0).round()
    assert all(means.abs() <= 1)  # near 0
    assert all((stds - 1).abs() <= 1)  # near 1


def test_standardize_features_minmax_scaler(toy_df_with_missing):
    """Test features are scaled with MinMaxScaler."""
    # Remove zero-variance col to allow scaling
    df = toy_df_with_missing.drop(columns=["feat2", "feat3"])
    cleaned = preprocessing.standardize_features(df, context_features=["id", "team"], scaler="minmax")
    features = [c for c in cleaned.columns if c not in ["id", "team"]]
    assert cleaned[features].min().min() == 0.0
    assert cleaned[features].max().max() == 1.0
