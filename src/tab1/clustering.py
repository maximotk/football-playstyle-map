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


def rule_based_cluster(df, cluster_cols):
    n_clusters = len(cluster_cols)

    computation = (
        df.copy()[cluster_cols]
        .assign(
            sorting=lambda x: x.apply(
                lambda row: row.nlargest(n_clusters - 1).index.to_list(), axis=1),
            cluster=lambda x: x.apply(
                lambda row:
                    (
                        row["sorting"][0]) if row[row["sorting"][0]] >= 0.6 or (row[row["sorting"][0]] >= 0.55 and row[row["sorting"][1]] <= 0.35)
                        else f'{row["sorting"][0]} + {row["sorting"][1]}' if row[row["sorting"][0]] >= 0.40 and row[row["sorting"][1]] >= 0.25 and row[row["sorting"][:2]].sum() > 0.70 and row[row["sorting"][1]] - row[row["sorting"][2]] >= 0.05
                        else "Mixed",
                axis=1
            ),
            cluster_dominant=lambda x: x.apply(lambda row: row["sorting"][0] if row["cluster"] != "Mixed" else "Mixed", axis=1)
        )

    )

    return computation[["cluster_dominant", "cluster"]]

def create_clusters(df, name_map):

    output = df.copy()
    output.rename(columns=name_map, inplace=True)

    output[["cluster_dominant", "cluster"]] = rule_based_cluster(output, list(name_map.values()))

    return output 

