import pytest
import pandas as pd

from src.tab1 import clustering, nmf, viz


def make_toy_pipeline_df():
    """Helper to create a toy dataframe for full pipeline testing."""
    return pd.DataFrame({
        "match_id": range(1, 7),
        "team": list("ABCDEF"),
        "competition": ["Test"] * 6,
        "c1": [0.6, 0.55, 0.42, 0.2, 0.5, 0.7],
        "c2": [0.3, 0.30, 0.30, 0.5, 0.2, 0.1],
        "c3": [0.1, 0.15, 0.28, 0.3, 0.3, 0.2],
    })


def test_full_pipeline_runs():
    """Test the full pipeline: clustering → NMF → visualization runs successfully."""
    df = make_toy_pipeline_df()

    # Step 1: Clustering
    name_map = {"c1": "score_cluster_1", "c2": "score_cluster_2", "c3": "score_cluster_3"}
    clustered = clustering.create_clusters(df, name_map)

    assert "cluster_dominant" in clustered.columns
    assert "cluster" in clustered.columns

    # Step 2: NMF
    soft_clusters, components, n_components, feature_names = nmf.run_nmf(
        clustered,
        context_features=["match_id", "team", "competition", "cluster_dominant", "cluster"],
        negative_columns=[],
    )

    assert soft_clusters.shape[0] == len(df)
    assert components.shape[0] == n_components

    # Step 3: Visualization
    viz_df, fig = viz.run_viz(
        soft_clusters,
        context_features=["team", "competition", "cluster_dominant"],
        color_col="cluster_dominant",
        model="tsne",
        n_components=2,
        perplexity=2,
        max_iter=250
    )

    assert "X" in viz_df.columns and "Y" in viz_df.columns
    assert fig is not None
