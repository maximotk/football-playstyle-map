import pandas as pd
from statsbombpy import sb
import pickle
import numpy as np
import math
from scipy.stats import entropy
import warnings
from collections import defaultdict
import os
from rapidfuzz import fuzz, process
import sys
warnings.filterwarnings("ignore")
import re

import matplotlib.pyplot as plt
import seaborn as sns
import math, textwrap
from matplotlib.ticker import PercentFormatter

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype
from matplotlib.gridspec import GridSpec

from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE
import matplotlib.patheffects as pe
from umap import UMAP

from .utils import *









def analyze_and_handle_missing_values(df, context_features, threshold_drop=30,  verbose=False):
    

    metadata = df.copy()[context_features]
    assert metadata.isna().sum().sum() == 0, "Metadata contains missing values."

    df_to_clean = df.drop(columns=context_features).copy()
    assert all(is_numeric_dtype(df_to_clean[c]) for c in df_to_clean.columns), \
        "Non-numeric columns found in df_to_clean."


    n = len(df)
    
    missing_by_feature = (
        df_to_clean
        .isna()
        .sum()
        .reset_index(name="n_missing")
        .rename(columns={"index": "feature"})
        .query("n_missing > 0")
        .assign(percent_missing=lambda x: 100 * x["n_missing"] / n)
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(missing_by_feature)

    features_to_drop = (
        missing_by_feature
        .query("percent_missing >= @threshold_drop")["feature"]
        .to_list()
    )
    df_to_clean.drop(columns=features_to_drop, inplace=True)
    if verbose:
        features_to_impute = (
            missing_by_feature
            .query("percent_missing < @threshold_drop")["feature"]
            .to_list()
        )
        print(f"Dropped: {features_to_drop}")
        print(f"Imputed: {features_to_impute}")

    if df_to_clean.shape[1] == 0:
        features_imputed = pd.DataFrame(index=df.index)
    elif not df_to_clean.isna().any().any():
        features_imputed = df_to_clean.copy()
    else:
        br = BayesianRidge(tol=1e-3)
        imputer = IterativeImputer(
            estimator=br,
            max_iter=5,                 # try 5 first; bump to 10 if needed
            imputation_order="ascending",
            initial_strategy="median",
            n_nearest_features=min(50, max(1, df_to_clean.shape[1] - 1)),
            sample_posterior=False,
            tol=1e-3,
            random_state=1,
        )
        features_imputed = pd.DataFrame(
            imputer.fit_transform(df_to_clean),
            columns=df_to_clean.columns,
            index=df_to_clean.index
        )
    assert features_imputed.isna().sum().sum() == 0

    df_imputed = pd.concat([metadata, features_imputed], axis=1)
    kept_cols = [c for c in df.columns if c not in features_to_drop]
    df_cleaned = df_imputed[kept_cols]

    return df_cleaned







def analyze_and_handle_constants(
    df,
    context_features,
    eps_mean=1e-8,           # prevent div-by-zero in CV
    std_thresh=1e-12,        # auto-drop if std <= this
    cv_thresh=0.01,          # flag if CV <= this (conservative)
    entropy_thresh=0.1,      # bits; flag if entropy <= this
    gini_thresh=0.02,        # flag if gini <= this
    verbose=False
):

    metadata = df.copy()[context_features]
    df_to_clean = df.drop(columns=context_features).copy()
    assert all(is_numeric_dtype(df_to_clean[c]) for c in df_to_clean.columns), \
        "Non-numeric columns found in df_to_clean."

    means = df_to_clean.mean()
    stds = df_to_clean.std(ddof=0)
    cvs = stds / means.abs().clip(lower=eps_mean)
    entropies = df_to_clean.apply(shannon_entropy)
    ginis = df_to_clean.apply(gini_impurity)

    constants_mask = (stds <= std_thresh) | (df_to_clean.nunique(dropna=False) <= 1)
    dropped_constants = df_to_clean.columns[constants_mask].tolist()

    flag_low_cv = (cvs <= cv_thresh) & (~constants_mask)
    flag_low_entropy = (entropies <= entropy_thresh) & (~constants_mask)
    flag_low_gini = (ginis <= gini_thresh) & (~constants_mask)

    report = (
        pd.DataFrame({
        "feature": df_to_clean.columns,
        "mean": means,
        "std": stds,
        "cv": cvs,
        "entropy": entropies,
        "gini": ginis,
        "is_constant": constants_mask,
        "flag_low_cv": flag_low_cv,
        "flag_low_entropy": flag_low_entropy,
        "flag_low_gini": flag_low_gini
    })
    .assign(n_flags=lambda x: x[["is_constant", "flag_low_cv", "flag_low_entropy", "flag_low_gini"]].sum(axis=1))
    .sort_values(
        ["n_flags","is_constant", "flag_low_cv", "flag_low_entropy", "flag_low_gini", "std"],
        ascending=[False, False, False, False, False, True]
    )
    .query("n_flags > 0")
    .reset_index(drop=True)
    )
    if verbose:
        print(report)


    # drop constants
    if dropped_constants:
        df_to_clean.drop(columns=dropped_constants, inplace=True)
        if verbose:
             print(f"Dropping {len(dropped_constants)} features: {dropped_constants} - constant features.")


    
    # drop flags
    flagged_features = (
        report
        .query("is_constant == False")
        .query("n_flags >= 2")["feature"]
        .to_list()
    )
    df_to_clean.drop(columns=flagged_features, inplace=True)
    dropped_constants = dropped_constants + flagged_features
    if verbose:
        print(f"Dropping {len(flagged_features)} features: {flagged_features} - >= 2 flags")

    df_dropped = pd.concat([metadata, df_to_clean], axis=1)
    df_cleaned = df_dropped[[c for c in df.columns if c not in dropped_constants]]

    return df_cleaned



def standardize_features(df, context_features, verbose=False, scaler="standard"):    

    metadata = df[context_features].copy()
    df_to_clean = df[[c for c in df.columns if c not in context_features]].copy()
    assert all(is_numeric_dtype(df_to_clean[c]) for c in df_to_clean.columns), \
        "Non-numeric columns found in df_to_clean."
    
    stds = df_to_clean.std(ddof=0)
    zeros = stds[stds == 0].index.tolist()
    assert not zeros, f"Zero-variance columns present (remove before scaling): {zeros}"

    if scaler.lower() == "standard":
    
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(df_to_clean),
            columns=df_to_clean.columns,
            index=df_to_clean.index
        )
    elif scaler.lower() == "minmax":
        scaler = MinMaxScaler()
        features_scaled = pd.DataFrame(
                    scaler.fit_transform(df_to_clean),
                    columns=df_to_clean.columns,
                    index=df_to_clean.index
                )
    
    else:
        raise ValueError(f"{scaler} is an invalid choice. Expected 'standard' or 'minmax'")

    df_scaled = pd.concat([metadata, features_scaled], axis=1)
    df_cleaned = df_scaled[list(df.columns)]

    return df_cleaned