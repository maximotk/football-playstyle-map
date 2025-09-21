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
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
import matplotlib.patheffects as pe
from umap import UMAP


def process_components(components, names, n_components=4, n_features=10):
    """
    Compact visualization of top-N features per component.
    - Category-based color coding (with horizontal legend).
    - Subplots more compact, labels wrapped tighter.
    """
    top_components = components[:n_components]

    comp_dfs = []
    for i, component in enumerate(top_components, start=1):
        importances = np.abs(component) / np.abs(component).sum()
        directions = np.where(component >= 0, "Positive", "Negative")
        df_info = (
            pd.DataFrame({
                "Feature": names,
                "Importance": importances,
                "Direction": directions
            })
            .assign(Category=lambda x: x["Feature"].apply(
                lambda s: s[s.find("_")+1 : s.find("_", s.find("_")+1)]
            ))
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
            .iloc[:n_features]
        )
        df_info["Component"] = f"Dimension {i}"
        comp_dfs.append(df_info)

    all_top = pd.concat(comp_dfs, ignore_index=True)

    # --- Color palette for categories
    cats = all_top["Category"].astype(str).unique().tolist()
    base = sns.color_palette("tab20", n_colors=20)
    cat2color = {c: base[j % len(base)] for j, c in enumerate(cats)}

    # --- Layout (reduce spacing between subplots)
    cols = n_components
    fig, axes = plt.subplots(
        1, cols, figsize=(4.8 * cols, 6), dpi=150,
        gridspec_kw={"wspace": 0.2}  # tighter spacing
    )
    if cols == 1:
        axes = [axes]

    for ax, (comp, df_comp) in zip(axes, all_top.groupby("Component")):
        feat = df_comp.reset_index(drop=True)

        n = len(feat)
        ncols = 2
        nrows = math.ceil(n / ncols)

        for idx, r in feat.iterrows():
            rrow = idx // ncols
            rcol = idx % ncols
            rect = plt.Rectangle(
                (rcol, nrows - 1 - rrow), 1, 1,
                facecolor=cat2color.get(str(r["Category"]), (0.7, 0.7, 0.7)),
                edgecolor="white"
            )
            ax.add_patch(rect)

            # Wrap text more aggressively (so labels fit)
            label = textwrap.fill(str(r["Feature"]), width=15)
            ax.text(
                rcol + 0.5, nrows - 1 - rrow + 0.5, label,
                ha="center", va="center", fontsize=10, fontweight="medium",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2")
            )

        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(comp, fontsize=13, fontweight="bold")

    # --- Horizontal legend
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=cat2color[c], label=c)
        for c in cats
    ]
    fig.legend(
        handles, [h.get_label() for h in handles],
        title="Category",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False, fontsize=11, title_fontsize=13,
        ncol=len(cats)
    )

    fig.suptitle(f"Top {n_features} Features per Dimension",
                 fontsize=18, fontweight="bold", y=1.10)

    return fig


def get_cosine_similarity_nmf(input_list):

    n_components = input_list[0].shape[0]
    
    comp_dict = defaultdict(list)
    for array in input_list:
        for i in range(n_components):
            comp_dict[f"Comp{i}"].append(array[i, :])
    
    mean_cos_sim_list = []
    for _, list_comp in comp_dict.items():
        tuples = []
        for i, comp in enumerate(list_comp):
            norm = np.linalg.norm(comp)
            tuples.append((i, comp, norm))
        cos_sim_list = [a.dot(b) / (a_norm * b_norm) if a_norm * b_norm != 0 else 0.0 for index_a, a, a_norm in tuples for index_b, b, b_norm in tuples if index_a > index_b]
        mean_cos_sim = np.array(cos_sim_list).mean()
        mean_cos_sim_list.append(mean_cos_sim)
    
    global_mean = np.mean(np.array(mean_cos_sim_list))

    return global_mean


def run_nmf(df, context_features, negative_columns):
    metadata = df.copy()[context_features]
    data = df.drop(columns=context_features + negative_columns)
    feature_names = data.columns.to_list()

    ks = []
    errors = []
    rel_errors = []
    cosine_similarities = []
    min_weights = []
    avg_weights = []
    for k in range(2, 11):

        model = NMF(
            n_components=k,
            random_state=1,
            init="nndsvda"
        )
        model.fit(data)

        # Reconstruction error
        error = model.reconstruction_err_
        norm_X = np.linalg.norm(data, 'fro')
        relative_error = error / norm_X


        ks.append(k)
        errors.append(error)
        rel_errors.append(relative_error)

        # No degenerate features
        W = model.transform(data)
        W_normalized = W / np.sum(W, axis=1)[:, None]
        avg_weight_feature = W_normalized.mean(axis=0)
        min_avg_weight = avg_weight_feature.min()
        avg_avg_weight = avg_weight_feature.mean()

        min_weights.append(min_avg_weight)
        avg_weights.append(avg_avg_weight)

        # Stability Check
        component_weights = []
        ref_H = None
        for s in range(30):

            b_sample = data.sample(frac=1, replace=True, random_state=s)

            model_bootstrap = NMF(
                n_components=k,
                random_state=1,
                init="nndsvda"
            )
            model_bootstrap.fit(b_sample)

            H = model_bootstrap.components_

            if ref_H is None:
                ref_H = H.copy()
            else:
                S = cosine_similarity(ref_H, H)
                r, c = linear_sum_assignment(-S)
                perm = np.empty_like(r)
                perm[r] = c
                H = H[perm, :]


            component_weights.append(H)

        global_cosine_similarity = get_cosine_similarity_nmf(component_weights)
        cosine_similarities.append(global_cosine_similarity)


    k_choice_df = pd.DataFrame({
        "k": ks,
        "reconstruction_error": rel_errors,
        "avg_cosine_similarity": cosine_similarities,
        "min_avg_feature_importance": min_weights
    })

    best_k = int(
        k_choice_df
        .query("avg_cosine_similarity >= 0.85 & min_avg_feature_importance >= 0.15")
        .sort_values("reconstruction_error")
        .iloc[0]["k"]
    )

    model_chosen = NMF(
        n_components=best_k,
        random_state=1,
        init="nndsvda"
    )

    W_chosen = model_chosen.fit_transform(data)

    W_chosen_standardized = W_chosen / np.sum(W_chosen, axis=1)[:, None]
    soft_clusters = pd.DataFrame(
        W_chosen_standardized,
        columns=[f"score_cluster_{i}" for i in range(1, best_k + 1)]
    )
    
    soft_clusters = pd.concat([metadata, soft_clusters], axis=1)
    components_chosen = model_chosen.components_

    n_components_chosen = len(components_chosen)

    return soft_clusters, components_chosen, n_components_chosen, feature_names


