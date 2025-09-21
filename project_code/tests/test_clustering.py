import pandas as pd
import pytest
from project_code.src.tab1 import clustering


def make_toy_cluster_df():
    """Create synthetic data for cluster.py tests."""
    return pd.DataFrame({
        "c1": [0.7, 0.55, 0.42, 0.2],
        "c2": [0.2, 0.30, 0.30, 0.3],
        "c3": [0.1, 0.15, 0.28, 0.5],
    })


def test_rule_based_cluster_dominant():
    """Test that rule_based_cluster assigns correct dominant and mixed clusters."""
    df = make_toy_cluster_df()
    result = clustering.rule_based_cluster(df, ["c1", "c2", "c3"])

    # row 0 → c1 is >= 0.6, so c1 dominates
    assert result.loc[0, "cluster_dominant"] == "c1"

    # row 1 → c1=0.55 and c2=0.30 → still c1
    assert result.loc[1, "cluster_dominant"] == "c1"

    # row 2 → c1=0.42, c2=0.30, c3=0.28 → too close → Mixed
    assert result.loc[2, "cluster"] == "Mixed"

    # row 3 → c3=0.50, c2=0.30 → combo rule fires
    assert "+" in result.loc[3, "cluster"]
    assert result.loc[3, "cluster_dominant"] == "c3"
    

def test_create_clusters_applies_name_map():
    """Test that create_clusters renames columns and applies clustering correctly."""
    df = make_toy_cluster_df()
    name_map = {"c1": "score_cluster_1", "c2": "score_cluster_2", "c3": "score_cluster_3"}

    result = clustering.create_clusters(df, name_map)

    # renamed columns exist
    assert "score_cluster_1" in result.columns
    # cluster cols created
    assert "cluster_dominant" in result.columns
    assert "cluster" in result.columns
    # cluster_dominant must align with renamed version or Mixed
    assert set(result["cluster_dominant"]).issubset(
        {"score_cluster_1", "score_cluster_2", "score_cluster_3", "Mixed"}
    )
