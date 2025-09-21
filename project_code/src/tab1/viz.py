import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.patheffects as pe
from umap import UMAP
warnings.filterwarnings("ignore")

def run_viz(
        df: pd.DataFrame,
        context_features: list[str],
        color_col: str = "cluster_dominant",
        model: str = "tsne",
        n_components: int = 2,
        metric: str = "euclidean",
        learning_rate: float = 1,
        init: str = "pca",
        random_state: int = 1,
        perplexity: int = 30,
        max_iter: int = 2000,
        early_exaggeration: float = 12,
        method: str = "exact",
        n_neighbors: int = 15,
        min_dist: float = 0.25,
        n_epochs: int = 2000,
        set_op_mix_ratio: float = 1,
        local_connectivity: int = 1,
        repulsion_strength: float = 1
    ) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Run dimensionality reduction (t-SNE or UMAP) and visualize teams.

    :param df: Input dataframe with numeric features and metadata.
    :param context_features: Columns to keep as metadata (e.g., IDs, team names).
    :param color_col: Column used for coloring points in the scatter plot.
    :param model: Dimensionality reduction algorithm ("tsne" or "umap").
    :param n_components: Target dimensionality (usually 2 for visualization).
    :param metric: Distance metric for t-SNE.
    :param learning_rate: Learning rate for t-SNE.
    :param init: Initialization method for t-SNE.
    :param random_state: Random seed for reproducibility.
    :param perplexity: Perplexity parameter for t-SNE.
    :param max_iter: Maximum number of iterations for t-SNE.
    :param early_exaggeration: Early exaggeration factor for t-SNE.
    :param method: Optimization method for t-SNE ("exact" or "barnes_hut").
    :param n_neighbors: Number of neighbors for UMAP.
    :param min_dist: Minimum distance between embedded points for UMAP.
    :param n_epochs: Number of training epochs for UMAP.
    :param set_op_mix_ratio: Balance between local and global structure in UMAP.
    :param local_connectivity: Local connectivity for UMAP.
    :param repulsion_strength: Repulsion strength in UMAP layout.
    :return: Tuple of (dataframe with reduced coordinates, matplotlib figure).
    """
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

    unique_vals = viz_df[color_col].unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_vals))

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
    df: pd.DataFrame,
    name_map: dict[str, str],
    figsize: tuple[int, float | None] = (7, None),
    cmap: str = "magma_r",
    annot: bool = False,
    max_height: int = 10,
) -> plt.Figure:
    """
    Plot a heatmap of average playstyle dimensions for each team.

    Uses spectral seriation to order teams for better visualization.

    :param df: Input dataframe containing team-level data.
    :param name_map: Mapping of original feature names to display names.
    :param figsize: Width and height of the figure (height auto-scales if None).
    :param cmap: Colormap used for the heatmap.
    :param annot: Whether to annotate heatmap cells with values.
    :param max_height: Maximum figure height (prevents oversized plots).
    :return: Matplotlib figure containing the heatmap.
    """
    cols_in  = list(name_map.keys())
    cols_out = list(name_map.values())
    tmp = df[['team'] + cols_in].copy().rename(columns=name_map)
    X = tmp.groupby('team', as_index=True)[cols_out].mean()

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    A = cosine_similarity(Xn)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    row_order = np.argsort(fiedler)

    Xo = X.iloc[row_order]

    n_rows = len(Xo)
    height = min(max_height, 0.3 * n_rows + 2.5)

    fig, ax = plt.subplots(figsize=(figsize[0], height))

    sns.heatmap(
        Xo, cmap=cmap, annot=annot,
        cbar_kws={'label': 'Intensity', 'shrink': 0.4, 'aspect': 15},
        ax=ax
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Intensity", fontsize=10)   
    cbar.ax.tick_params(labelsize=8)             


    ax.set_title("Team Playstyle Dimension Mix", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Team", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha="right")

    fig.tight_layout()

    return fig