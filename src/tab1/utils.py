import pandas as pd
import numpy as np


def _n_bins_fd(series: pd.Series) -> int:
    """
    Compute the number of bins using the Freedman–Diaconis rule.

    Falls back to alternative rules for small sample sizes or low-variance data.

    :param series: Input numeric series.
    :return: Number of bins (at least 2).
    """
    x = series.dropna().to_numpy()
    n = x.size
    if n < 2:
        return 2
    unique_vals = np.unique(x)
    if len(unique_vals) <= 20:
        return len(unique_vals)
    # Freedman–Diaconis
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    if iqr == 0:
        return max(2, int(round(np.sqrt(n))))
    bin_width = 2 * iqr / (n ** (1/3))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(2, bins)


def shannon_entropy(series: pd.Series) -> float:
    """
    Compute Shannon entropy for a numeric series.

    The series is binned using the Freedman–Diaconis rule, and entropy is
    calculated from the normalized bin counts.

    :param series: Input numeric series.
    :return: Shannon entropy value.
    """
    x = series.to_numpy()
    n = x.size
    # guard degenerate cases (all same or <2 obs)
    if n < 2 or np.nanstd(x) == 0:
        return 0.0
    bins = _n_bins_fd(series)              # <-- integer
    # pd.cut can still error if min==max; we're guarded above
    counts = pd.Series(pd.cut(x, bins=bins)).value_counts(dropna=False, normalize=True)

    probs = counts[counts > 0]
    return float(-(probs * np.log2(probs)).sum())

def gini_impurity(series: pd.Series) -> float:
    """
    Compute Gini impurity for a numeric series.

    The series is binned using the Freedman–Diaconis rule, and impurity is
    calculated from the normalized bin counts.

    :param series: Input numeric series.
    :return: Gini impurity value.
    """
    x = series.to_numpy()
    n = x.size
    if n < 2 or np.nanstd(x) == 0:
        return 0.0
    bins = _n_bins_fd(series)
    counts = pd.Series(pd.cut(x, bins=bins)).value_counts(dropna=False, normalize=True)
    probs = counts[counts > 0]
    return float(1 - (probs ** 2).sum())
