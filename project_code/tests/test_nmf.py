import pandas as pd
import numpy as np
import pytest
import matplotlib.figure as mpl_fig
import matplotlib.pyplot as plt
from project_code.src.tab1 import nmf


def make_toy_df(n_samples=15, n_features=12):
    """Helper to create toy dataframe with metadata and numeric features."""
    data = {
        "match_id": range(1, n_samples + 1),
        "team": [f"T{i}" for i in range(1, n_samples + 1)],
        "competition": ["X"] * n_samples,
    }
    for j in range(1, n_features + 1):
        data[f"feat{j}"] = np.arange(1, n_samples + 1) + j
    return pd.DataFrame(data)


def test_process_components_returns_figure():
    """Test process_components returns a matplotlib Figure."""
    components = np.array([
        [0.2, 0.3, 0.5],
        [0.1, 0.7, 0.2]
    ])
    names = ["feat1", "feat2", "feat3"]

    fig = nmf.process_components(components, names, n_components=2, n_features=3)
    assert isinstance(fig, mpl_fig.Figure)


def test_get_cosine_similarity_nmf_returns_between_minus1_and1():
    """Test get_cosine_similarity_nmf outputs value between -1 and 1."""
    arrays = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0.9, 0.1], [0.2, 0.8]])
    ]
    result = nmf.get_cosine_similarity_nmf(arrays)
    assert -1.0 <= result <= 1.0


def test_run_nmf_outputs_shapes():
    """Test run_nmf returns correct outputs and shapes."""
    df = make_toy_df()  # bigger df
    soft_clusters, components, n_components_chosen, feature_names = nmf.run_nmf(
        df,
        context_features=["match_id", "team", "competition"],
        negative_columns=[]
    )

    assert "match_id" in soft_clusters.columns
    assert "team" in soft_clusters.columns
    assert any(col.startswith("score_cluster_") for col in soft_clusters.columns)

    assert isinstance(components, np.ndarray)
    assert components.shape[0] == n_components_chosen
    assert all(isinstance(n, str) for n in feature_names)



def test_process_components_single_component():
    """Test process_components works when only one component is passed."""
    # one component with 3 features
    components = np.array([[0.5, 0.3, 0.2]])
    names = ["feat_a", "feat_b", "feat_c"]

    fig = nmf.process_components(
        components,
        names,
        n_components=1,
        n_features=3,
        component_names=["Custom Dimension"]
    )

    assert isinstance(fig, plt.Figure)


def test_run_nmf_triggers_fallback_branch():
    """Test run_nmf triggers the fallback when no candidate meets thresholds."""
    df = pd.DataFrame({
        "match_id": [1, 2, 3, 4],
        "team": ["A", "B", "C", "D"],
        "competition": ["X"] * 4,
        "feat1": [1, 1, 1, 1.1],
        "feat2": [2, 2, 2, 2.1],
    })

    soft_clusters, components, n_components_chosen, feature_names = nmf.run_nmf(
        df,
        context_features=["match_id", "team", "competition"],
        negative_columns=[]
    )

    assert "match_id" in soft_clusters.columns
    assert isinstance(components, np.ndarray)
    assert n_components_chosen >= 2
    assert all(isinstance(n, str) for n in feature_names)
