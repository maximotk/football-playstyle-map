import pandas as pd
import numpy as np


def _n_bins_fd(series):
    """Number of bins using Freedman–Diaconis rule, with fallbacks."""
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


def shannon_entropy(series):
    """Compute Shannon entropy for numeric series by binning."""
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

def gini_impurity(series):
    """Compute Gini impurity for numeric series by binning."""
    x = series.to_numpy()
    n = x.size
    if n < 2 or np.nanstd(x) == 0:
        return 0.0
    bins = _n_bins_fd(series)
    counts = pd.Series(pd.cut(x, bins=bins)).value_counts(dropna=False, normalize=True)
    probs = counts[counts > 0]
    return float(1 - (probs ** 2).sum())
