import pandas as pd
import numpy as np
import pytest

from project_code.src.tab1 import utils

def test_n_bins_fd_iqr_zero_branch_many_uniques():
    """Hit the IQR==0 fallback when uniques > 20 (so FD path is used)."""
    zeros = [0] * 76
    extras = list(range(1, 25))  # 24 unique values
    s = pd.Series(zeros + extras)
    bins = utils._n_bins_fd(s)
    assert bins == max(2, int(round(np.sqrt(len(s)))))

def test_n_bins_fd_too_few_values():
    """If series has fewer than 2 values, return 2 bins."""
    s = pd.Series([42])
    bins = utils._n_bins_fd(s)
    assert bins == 2


def test_n_bins_fd_unique_values_shortcut():
    """If <= 20 unique values, bins equals number of uniques."""
    s = pd.Series([1, 1, 2, 2, 3])  # only 3 uniques
    bins = utils._n_bins_fd(s)
    assert bins == 3


def test_n_bins_fd_iqr_zero_branch():
    """If IQR=0 but n>2, fall back to sqrt(n)."""
    s = pd.Series([5, 5, 5, 6, 6, 6])  # IQR=0
    bins = utils._n_bins_fd(s)
    assert bins == max(2, int(round(np.sqrt(len(s)))))


def test_n_bins_fd_freedman_diaconis_normal_case():
    """Normal FD calculation produces reasonable number of bins."""
    s = pd.Series(np.arange(100))  # large spread, IQR > 0
    bins = utils._n_bins_fd(s)
    assert bins >= 2
    # sanity: shouldn't just equal n
    assert bins < len(s)


def test_shannon_entropy_constant_and_random():
    """Shannon entropy returns 0 for constants, >0 for varied input."""
    s_const = pd.Series([7, 7, 7, 7])
    s_rand = pd.Series([1, 2, 3, 4, 5])
    assert utils.shannon_entropy(s_const) == 0.0
    assert utils.shannon_entropy(s_rand) > 0.0


def test_gini_impurity_constant_and_random():
    """Gini impurity returns 0 for constants, >0 for varied input."""
    s_const = pd.Series([10, 10, 10, 10])
    s_rand = pd.Series([1, 2, 3, 4, 5])
    assert utils.gini_impurity(s_const) == 0.0
    assert 0.0 < utils.gini_impurity(s_rand) <= 1.0

