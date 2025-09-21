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

def run_viz(
        df,
        context_features,
        color_col="cluster_dominant",
        model="tsne",
        n_components=2,
        metric="euclidean",
        learning_rate=1,
        init="pca",
        random_state=1,
        perplexity=30,
        max_iter=2000,
        early_exaggeration=12,
        method="exact",
        n_neighbors=15,
        min_dist=0.25,
        n_epochs=2000,
        set_op_mix_ratio=1,
        local_connectivity=1,
        repulsion_strength=1
    ):
    cols_to_drop = list(set(context_features + [color_col]))
    metadata = df.copy()[cols_to_drop]
    data = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])

    if model == "tsne":
        tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            init=init,
            metric=metric,
            early_exaggeration=early_exaggeration,
            method=method,
            random_state=random_state
        )
        tsne_data = pd.DataFrame(tsne_model.fit_transform(data), columns=["X", "Y"])
        viz_df = pd.concat([metadata, tsne_data], axis=1)

    elif model == "umap":
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            random_state=random_state
        )
        umap_data = pd.DataFrame(umap_model.fit_transform(data), columns=["X", "Y"])
        viz_df = pd.concat([metadata, umap_data], axis=1)

    # unique cluster labels
    unique_vals = viz_df[color_col].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_vals))

    # assign Mixed to grey if it exists
    color_map = {val: palette[i] for i, val in enumerate(unique_vals)}
    if "Mixed" in color_map:
        color_map["Mixed"] = (0.6, 0.6, 0.6)

    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    sns.scatterplot(
        data=viz_df,
        x="X", y="Y",
        hue=color_col,
        palette=color_map,
        linewidth=0,
        s=200, alpha=0.9,
        ax=ax
    )


    for _, row in viz_df.iterrows():
        ax.text(
            row["X"] + 0.05,
            row["Y"] + 0.05,
            row["team"],
            fontsize=7,
            color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    ax.set_title(f"{model.upper()} of Teams, colored by Playstyle", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.3)
    sns.despine(ax=ax)
    ax.set_aspect('equal', 'box')

    ax.legend(title="Playstyle", loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=8)
    ax.set_title(f"{model.upper()} of Teams, colored by Playstyle", fontweight="bold")


    return viz_df, fig




def team_axes_heatmap(
    df,
    name_map,
    figsize=(7, None),
    cmap="magma_r",        # reversed so dark = high intensity
    annot=False,
    max_height=10,
):
    cols_in  = list(name_map.keys())
    cols_out = list(name_map.values())
    tmp = df[['team'] + cols_in].copy().rename(columns=name_map)
    X = tmp.groupby('team', as_index=True)[cols_out].mean()

    # spectral seriation
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    A = cosine_similarity(Xn)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    row_order = np.argsort(fiedler)

    Xo = X.iloc[row_order]

    # figure height scaling
    n_rows = len(Xo)
    height = min(max_height, 0.3 * n_rows + 2.5)

    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # heatmap with smaller colorbar
    sns.heatmap(
        Xo, cmap=cmap, annot=annot,
        cbar_kws={'label': 'Intensity', 'shrink': 0.4, 'aspect': 15},
        ax=ax
    )

        # after sns.heatmap(...)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Intensity", fontsize=10)   # smaller label font
    cbar.ax.tick_params(labelsize=8)              # smaller tick labels too


    # axis labels and ticks
    ax.set_title("Team Playstyle Dimension Mix", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Team", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha="right")

    fig.tight_layout()

    return fig