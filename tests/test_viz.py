import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.tab1 import viz


@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "match_id": [1, 2, 3, 4, 5, 6],
        "team": ["A", "B", "C", "D", "E", "F"],
        "competition": ["Test"] * 6,
        "score_cluster_1": [0.7, 0.2, 0.1, 0.5, 0.6, 0.3],
        "score_cluster_2": [0.3, 0.8, 0.9, 0.5, 0.4, 0.7],
        "cluster_dominant": ["Dimension 1", "Dimension 2", "Dimension 2", "Dimension 1", "Dimension 1", "Dimension 2"]
    })


def test_run_viz_returns_viz_df_and_fig(toy_df):
    viz_df, fig = viz.run_viz(
        toy_df,
        context_features=["match_id", "team", "competition"],
        color_col="cluster_dominant",
        model="tsne",
        n_components=2,
        perplexity=2,   # keep tiny for test speed
        max_iter=250
    )
    assert isinstance(viz_df, pd.DataFrame)
    assert "X" in viz_df.columns and "Y" in viz_df.columns
    assert isinstance(fig, plt.Figure)


def test_run_viz_umap_mode(toy_df):
    viz_df, fig = viz.run_viz(
        toy_df,
        context_features=["match_id", "team", "competition"],
        color_col="cluster_dominant",
        model="umap",
        n_components=2,
        n_neighbors=2, 
        n_epochs=50
    )
    assert isinstance(viz_df, pd.DataFrame)
    assert "X" in viz_df.columns and "Y" in viz_df.columns
    assert isinstance(fig, plt.Figure)


def test_team_axes_heatmap_returns_figure(toy_df):
    name_map = {
        "score_cluster_1": "Dimension 1",
        "score_cluster_2": "Dimension 2"
    }
    fig = viz.team_axes_heatmap(toy_df, name_map)
    assert isinstance(fig, plt.Figure)
