import pandas as pd
import numpy as np
import pytest

from project_code.src.tab1 import utils

def test_n_bins_fd_small_series():
    """Test _n_bins_fd returns at least 2 bins for small non-constant series."""
    s = pd.Series([1, 2])  # only 2 values
    bins = utils._n_bins_fd(s)
    assert bins >= 2  # small but non-constant series should give >= 2


def test_n_bins_fd_constant_series():
    """Test _n_bins_fd handles constant series correctly."""
    s = pd.Series([5, 5, 5, 5])
    bins = utils._n_bins_fd(s)
    # for constant input, 1 bin makes sense
    assert bins == 1


def test_shannon_entropy_zero_variance():
    """Test shannon_entropy returns 0 for constant input."""
    s = pd.Series([7, 7, 7, 7])
    result = utils.shannon_entropy(s)
    assert result == 0.0  # constant input → entropy 0


def test_shannon_entropy_random_values():
    """Test shannon_entropy returns positive value for varied input."""
    s = pd.Series([1, 2, 3, 4, 5])
    result = utils.shannon_entropy(s)
    assert result > 0.0  # should have some entropy


def test_gini_impurity_zero_variance():
    """Test gini_impurity returns 0 for constant input."""
    s = pd.Series([10, 10, 10, 10])
    result = utils.gini_impurity(s)
    assert result == 0.0  # no diversity → gini 0


def test_gini_impurity_random_values():
    """Test gini_impurity returns valid value in (0,1] for varied input."""
    s = pd.Series([1, 2, 3, 4, 5])
    result = utils.gini_impurity(s)
    assert 0.0 < result <= 1.0  # valid range
