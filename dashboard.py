import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from project_code.src.tab1.preprocessing import *
from project_code.src.tab1.nmf import *
from project_code.src.tab1.viz import *
from project_code.src.tab1.clustering import *


st.set_page_config(page_title="Team Playstyles", layout="wide")
st.title("⚽ Team Playstyle Dashboard")

# --- Sidebar
st.sidebar.header("Settings")

competition = st.sidebar.selectbox(
    "Choose Competition",
    [
        "FIFA World Cup 2022",
        "FIFA World Cup 2018",
        "UEFA Euro 2024",
        "UEFA Euro 2020",
        "Copa America 2024",
        "African Cup of Nations 2023"
    ]
)

data_type = st.sidebar.radio(
    "Data Type",
    ["Unilateral (Team Only)", "With Opponent Stats"]
)

# --- Load data depending on selection
folder = f"data/processed/all_tournaments"

if data_type == "Unilateral (Team Only)":
    df = pd.read_parquet(f"{folder}/team_match_features_unilateral.parquet")
else:
    df = pd.read_parquet(f"{folder}/team_match_features.parquet")

# --- Preprocessing workflow

context_features = ["match_id", "team", "competition"]

# unique cache key depending on data_type
cache_key = f"nmf_results_{data_type}"

status_placeholder = st.empty()
if cache_key not in st.session_state:    

    # Show the message inside the placeholder
    status_placeholder.write("⏳ Running preprocessing + NMF across all tournaments...")
    df_imputed = analyze_and_handle_missing_values(df, context_features=context_features, verbose=False)
    df_cleaned = analyze_and_handle_constants(
        df_imputed,
        context_features=context_features,
        gini_thresh=0.8,
        entropy_thresh=0.2,
        cv_thresh=1
    )
    df_standardized_positive = standardize_features(
        df_cleaned,
        context_features=context_features,
        scaler="minmax"
    )

    negative_columns = df_cleaned.drop(columns=context_features).loc[:, lambda df: (df < 0).any()].columns.to_list()

    soft_clusters, components, n_components_chosen, feature_names = run_nmf(
        df_standardized_positive,
        context_features=context_features,
        negative_columns=negative_columns
    )
    
    # -- Apply Clustering
    renaming = {f"score_cluster_{i}": f"Dimension {i}" for i in range(1, n_components_chosen + 1)}
    team_clustering = create_clusters(soft_clusters, renaming)

    context_features_extended = context_features + ["cluster", "cluster_dominant"]

    # save in session_state
    st.session_state[cache_key] = {
        "soft_clusters_global": soft_clusters,
        "components": components,
        "n_components_chosen": n_components_chosen,
        "feature_names": feature_names,
        "renaming": renaming,
        "team_clustering_global": team_clustering,
        "context_features": context_features_extended,
    }

    # Clear the message once done
status_placeholder.empty()

results = st.session_state[cache_key]

soft_clusters_global = results["soft_clusters_global"]
components = results["components"]
n_components_chosen = results["n_components_chosen"]

feature_names = results["feature_names"]
renaming = results["renaming"]
team_clustering_global = results["team_clustering_global"]
context_features = results["context_features"]

# --- Filter for competition
soft_clusters_selected = soft_clusters_global.query("competition == @competition").reset_index(drop=True)
team_clustering_selected= team_clustering_global.query("competition == @competition").reset_index(drop=True)


# --- Tabs & Results
tab1, tab2 = st.tabs(["Playstyle Analysis", "Further Exploration"])

with tab1:

    st.subheader("Major Playstyle Dimensions Detected")
    st.write("Breakdown of features per NMF component:")
    fig_compact = process_components(
        components,
        feature_names,
        n_components=n_components_chosen,
        n_features=10
    )
    st.pyplot(fig_compact)

    st.subheader("Map of Matches by Dominant Playstyle Dimension")
    viz_df, fig = run_viz(
        team_clustering_selected,
        context_features=context_features,
        model="tsne",
       # perplexity=min(30, max(5, len(team_clustering_selected) // 3)),
        learning_rate=2,
        color_col="cluster_dominant"
    )
    st.pyplot(fig)

    st.subheader("Team Playstyle Mix Overview")
    st.write("Vertical order matters too - teams further away from each other are more different.")
    fig2 = team_axes_heatmap(soft_clusters_selected, renaming)
    st.pyplot(fig2)

with tab2:
    st.subheader("Additional Analysis (coming soon)")
    st.info("This tab will be used for more advanced analysis later.")
