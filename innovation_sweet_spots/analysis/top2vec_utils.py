"""

"""
from innovation_sweet_spots import logging
from innovation_sweet_spots.analysis import top2vec
from innovation_sweet_spots.utils.io import (
    save_text_items,
    read_text_items,
)

import pandas as pd
from typing import Iterator, Dict, Union
from os import PathLike
from functools import lru_cache
from numpy import sort


@lru_cache()
def load_top2vec_model(model_name: str, folder: PathLike) -> top2vec.Top2Vec:
    """Loads a top2vec model"""
    filepath = folder / f"{model_name}.p"
    logging.info(f"Loading a top2vec model from {filepath}")
    return top2vec.Top2Vec.load(filepath)


def load_cluster_descriptions(model_name: str, folder: PathLike) -> pd.DataFrame:
    """Loads a table of the top2vec cluster descriptions"""
    return pd.read_csv(folder / f"{model_name}_cluster_descriptions.csv")


def load_document_ids(model_name: str, folder: PathLike) -> Iterator[str]:
    """Loads a list with the document identifiers"""
    return read_text_items(folder / f"{model_name}_document_ids.txt")


def load_top2vec_model_data(
    model_name: str, folder: PathLike, only_metadata: bool = False
) -> dict:
    """
    Loads files pertaining to a top2vec model

    Returns:
        Dictionary with the following keys:
            - model_name: The provided name of the model
            - model: A Top2Vec object
            - document_ids: A list of document ids
            - cluster_descriptions: A table with cluster descriptions
    """
    model_dict = {
        "model_name": model_name,
        "model": load_top2vec_model(model_name, folder)
        if only_metadata is False
        else None,
        "document_ids": load_document_ids(model_name, folder),
        "cluster_descriptions": load_cluster_descriptions(model_name, folder),
    }
    # Data consistency checks
    if not only_metadata:
        assert len(model_dict["document_ids"]) == len(
            model_dict["model"].documents
        ), "The number of documents in the model is different from the number of document identifiers"
        assert (
            len(model_dict["cluster_descriptions"])
            == model_dict["model"].get_num_topics()
        ), "The number of topics in the model is different from the number of described topics"
        logging.info(
            f"Returning a top2vec model data dictionary with the items: {list(model_dict.keys())}"
        )
    return model_dict


def save_document_clusters(document_clusters, model_name: str, folder: PathLike):
    """Save document cluster table following the naming convention"""
    document_clusters.to_csv(
        folder / f"{model_name}_document_clusters.csv", index=False
    )


def load_document_clusters(model_name: str, folder: PathLike):
    """Load document cluster table following the naming convention"""
    return pd.read_csv(folder / f"{model_name}_document_clusters.csv")


def load_document_cluster_data(model_name: str, folder: PathLike):
    """Load document cluster table and associated model data"""
    model_dict = load_top2vec_model_data(model_name, folder, only_metadata=True)
    model_dict["document_clusters"] = load_document_clusters(model_name, folder)
    del model_dict["model"]
    # Data consistency checks
    assert (
        sort(model_dict["document_clusters"].cluster_id.unique())
        != model_dict["cluster_descriptions"].cluster_id
    ).sum() == 0, (
        "Described clusters and document clusters do not have a perfect correspondence"
    )
    logging.info(
        f"Returning a top2vec document cluster dictionary with the items: {list(model_dict.keys())}"
    )
    return model_dict


def cluster_name(cluster_id: int):
    return f"cluster_{cluster_id}"


def cluster_probability_column(cluster_id: int):
    return f"cluster_{cluster_id}_prob"


def query_clusters(
    clusters: Iterator[int], document_clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns indicators whether the documents are in the specified clusters, and their probabilities

    Args:
        clusters: A list of cluster numbers
        document_clusters: A table of document cluster assignments and probabilities

    Returns:
        A dataframe with the following columns:
            - a column for document identifiers
            - a column with the probability of the assigned cluster (each document is assigned to only one cluster)
            - a column for each of the cluster, indicating if document is in the cluster
            - a column 'any_cluster' indicating if a document has been in any of the specified clusters
    """
    # Initialise the output dataframe
    df = document_clusters[["id", "cluster_probability"]].copy()
    df[[cluster_name(c) for c in clusters]] = False
    # Save ids that have been assigned to any of the clusters
    any_cluster = set()
    # Check each cluster
    for c in clusters:
        ids_in_cluster = document_clusters.query("cluster_id == @c")["id"].to_list()
        df.loc[df["id"].isin(ids_in_cluster), cluster_name(c)] = True
        any_cluster = any_cluster | set(ids_in_cluster)
    # Create a column to indicate if the document has been assigned to any of the specified clusters
    df["any_cluster"] = False
    df.loc[df["id"].isin(any_cluster), "any_cluster"] = True
    return df


# TO - DO
# Function exporting the cluster description table


class QueryClusters:
    """
    This class helps to query documents based on their top2vec clusters
    """

    def __init__(self, document_clusters: Union[pd.DataFrame, dict]):
        if type(document_clusters) is dict:
            self.document_clusters = document_clusters["document_clusters"]
            self.cluster_descriptions = document_clusters["cluster_descriptions"]
        else:
            self.document_clusters = document_clusters
            self.cluster_descriptions = None

    def get_cluster_documents(self, cluster: int):
        """Finds all doucments assigned to the specified cluster and returns a table with their ids and sorted probabilities"""
        return (
            self.document_clusters.query("cluster_id == @cluster").sort_values(
                "cluster_probability", ascending=False
            )
        )[["id", "cluster_id", "cluster_probability"]]

    def check_clusters(self, clusters: Iterator[int]) -> pd.DataFrame:
        """Returns indicators whether the documents are in the specified clusters, and their probabilities"""
        return query_clusters(clusters, self.document_clusters)

    def get_cluster_description(self, cluster: int):
        """Return cluster labels"""
        return self.cluster_descriptions.query("cluster_id == @cluster").to_dict(
            orient="records"
        )[0]
