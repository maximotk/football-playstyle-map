import pandas as pd

def rule_based_cluster(df: pd.DataFrame, cluster_cols: list[str]) -> pd.DataFrame:
    """
    Assign clusters to players using a rule-based approach.

    The function selects the dominant and secondary features from the given
    cluster columns and applies thresholds to decide whether a player belongs
    to a single cluster, a mixed cluster, or a combination of two clusters.

    :param df: Input dataframe with cluster feature columns.
    :param cluster_cols: Names of the columns used for clustering.
    :return: Dataframe with added 'cluster_dominant' and 'cluster' columns.
    """
    n_clusters = len(cluster_cols)

    computation = (
        df.copy()[cluster_cols]
        .assign(
            sorting=lambda x: x.apply(
                lambda row: row.nlargest(n_clusters).index.to_list(), axis=1),
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

def create_clusters(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """
    Create cluster assignments from a dataframe using a mapping of column names.

    The function renames the dataframe columns according to the provided mapping
    and applies rule-based clustering to determine dominant and mixed clusters.

    :param df: Input dataframe with original feature columns.
    :param name_map: Mapping of original column names to standardized cluster names.
    :return: Dataframe with added 'cluster_dominant' and 'cluster' columns.
    """

    output = df.copy()
    output.rename(columns=name_map, inplace=True)

    output[["cluster_dominant", "cluster"]] = rule_based_cluster(output, list(name_map.values()))

    return output 

