import pytest
import pandas as pd
import numpy as np
from project_code.src.tab1 import preprocessing


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "team": ["A", "B", "C", "D"],
    })


def test_missing_values_drop_impute_and_copy(base_df, capsys):
    """Cover dropping, imputing, and verbose/copy branches."""
    df = base_df.assign(
        f1=[1, None, None, None],   # 75% missing → dropped
        f2=[1, 2, np.nan, 4],       # 25% missing → imputed
    )
    out = preprocessing.analyze_and_handle_missing_values(
        df, context_features=["id", "team"], threshold_drop=50, verbose=True
    )
    captured = capsys.readouterr()
    assert "Dropped:" in captured.out and "Imputed:" in captured.out
    assert "f1" not in out and "f2" in out

    # branch: no missing values → direct copy
    df2 = base_df.assign(f1=[1, 2, 3, 4])
    out2 = preprocessing.analyze_and_handle_missing_values(df2, ["id", "team"], verbose=True)
    assert "f1" in out2.columns


def test_missing_values_all_dropped(base_df):
    """Cover branch where all features dropped → empty frame returned."""
    df = base_df.assign(f1=[np.nan, np.nan, np.nan, np.nan])
    out = preprocessing.analyze_and_handle_missing_values(df, ["id", "team"], threshold_drop=0)
    assert list(out.columns) == ["id", "team"]


def test_constants_dropped_and_verbose(base_df, capsys):
    """Cover dropping constants + flagged features with verbose=True."""
    df = base_df.assign(
        const=[5, 5, 5, 5],
        low_var=[1, 1, 2, 2],
        ok=[1, 2, 3, 4],
    )
    out = preprocessing.analyze_and_handle_constants(df, ["id", "team"], verbose=True)
    captured = capsys.readouterr()
    assert "Dropping" in captured.out and "feature" in captured.out
    assert "const" not in out and "ok" in out


def test_standardize_scalers_and_errors(base_df):
    """Cover standard scaler, minmax scaler, zero variance, and invalid scaler."""
    df = base_df.assign(f1=[1, 2, 3, 4], f2=[10, 20, 30, 40])
    # standard
    out_std = preprocessing.standardize_features(df, ["id", "team"], scaler="standard")
    assert out_std[["f1", "f2"]].mean().abs().max() < 1
    # minmax
    out_mm = preprocessing.standardize_features(df, ["id", "team"], scaler="minmax")
    assert out_mm[["f1", "f2"]].min().min() == 0.0
    assert out_mm[["f1", "f2"]].max().max() == 1.0
    # zero variance
    df_zero = base_df.assign(f1=[1, 1, 1, 1])
    with pytest.raises(AssertionError):
        preprocessing.standardize_features(df_zero, ["id", "team"])
    # invalid
    with pytest.raises(ValueError):
        preprocessing.standardize_features(df, ["id", "team"], scaler="invalid")
