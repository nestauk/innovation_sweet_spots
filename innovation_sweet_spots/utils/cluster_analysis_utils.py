"""
innovation_sweet_spots.utils.cluster_analysis_utils

Module for various cluster analysis (eg extracting cluster-specific keywords)
"""
import nltk

nltk.download("stopwords")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from typing import Iterator, Dict
from collections import defaultdict
from tqdm import tqdm

import hdbscan
import umap
from innovation_sweet_spots import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()

DEFAULT_STOPWORDS = stopwords.words("english")

umap_def_params = {
    "n_components": 50,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

hdbscan_def_params = {
    "min_cluster_size": 15,
    "min_samples": 1,
    "cluster_selection_method": "leaf",
    "prediction_data": True,
}


def umap_reducer(
    vectors: np.typing.ArrayLike,
    umap_params: dict = umap_def_params,
    random_umap_state: int = 1,
) -> np.typing.ArrayLike:
    """"Reduce dimensions of the input array using UMAP"""
    logging.info(
        f"Generating {umap_def_params['n_components']}-d UMAP embbedings for {len(vectors)} vectors"
    )
    reducer = umap.UMAP(random_state=random_umap_state, **umap_params)
    return reducer.fit_transform(vectors)


def hdbscan_clustering(
    vectors: np.typing.ArrayLike,
    hdbscan_params: dict = hdbscan_def_params,
    random_hdbscan_state: int = 3333,
) -> np.typing.ArrayLike:
    """Cluster vectors using HDBSCAN.

    Returns a dataframe with columns for highest probability cluster index
    and probability
    """
    logging.info(f"Clustering {len(vectors)} vectors with HDBSCAN.")
    np.random.seed(random_hdbscan_state)
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusterer.fit(vectors)
    cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
    highest_cluster_prob = np.array([(np.argmax(x), np.max(x)) for x in cluster_probs])
    return pd.DataFrame(
        {
            "labels": highest_cluster_prob[:, 0],
            "probability": highest_cluster_prob[:, 1],
        }
    ).astype({"labels": int})


def kmeans_clustering(
    vectors: np.typing.ArrayLike, kmeans_params: dict
) -> pd.DataFrame:
    """Cluster vectors using K-Means clustering"""
    logging.info(f"Clustering {len(vectors)} vectors with K-Means clustering")
    kmeans = KMeans(**kmeans_params).fit(vectors)
    return kmeans.labels_


def param_grid_search(
    vectors: np.typing.ArrayLike, search_params: dict, cluster_with_hdbscan: bool
) -> pd.DataFrame:
    """Perform grid search over search parameters and calculate
    mean silhouette score.

    Args:
        vectors: Embedding vectors.
        search_params: Dictionary with keys as parameter names
            and values as a list of parameters to search through.
        cluster_with_hdbscan: True to cluster with HDBSCAN and
            False to cluster with K-Means.

    Returns:
        Dataframe with information on clustering method, parameters,
            mean silhouette scoure.
    """
    parameters_record = []
    silhouette_score_record = []
    method_record = []
    for parameters in tqdm(ParameterGrid(search_params)):
        parameters_record.append(parameters)
        reduced_dims_vectors = umap_reducer(vectors)
        if cluster_with_hdbscan:
            clusters = hdbscan_clustering(
                reduced_dims_vectors, hdbscan_params=parameters
            )
            silhouette = silhouette_score(reduced_dims_vectors, clusters.labels.values)
            method_record.append("HDBSCAN")
        else:
            clusters = kmeans_clustering(reduced_dims_vectors, kmeans_params=parameters)
            silhouette = silhouette_score(reduced_dims_vectors, clusters)
            method_record.append("K-Means clustering")
        silhouette_score_record.append(silhouette)

    return pd.DataFrame.from_dict(
        {
            "method": method_record,
            "model_params": parameters_record,
            "silhouette_score": silhouette_score_record,
        }
    ).sort_values("silhouette_score", ascending=False)


def simple_preprocessing(text: str, stopwords=DEFAULT_STOPWORDS) -> str:
    """Simple preprocessing for cluster texts"""
    text = re.sub(r"[^a-zA-Z ]+", "", text).lower()
    text = simple_tokenizer(text)
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [t for t in text if ((t not in stopwords) and (len(t) > 1))]
    return " ".join(text)


def simple_tokenizer(text: str) -> Iterator[str]:
    return [token.strip() for token in text.split(" ") if len(token) > 0]


def cluster_texts(documents: Iterator[str], cluster_labels: Iterator) -> Dict:
    """
    Creates a large text string for each cluster, by joining up the
    text strings (documents) belonging to the same cluster

    Args:
        documents: A list of text strings
        cluster_labels: A list of cluster labels, indicating the membership of the text strings

    Returns:
        A dictionary where keys are cluster labels, and values are cluster text documents
    """

    assert len(documents) == len(cluster_labels)
    doc_type = type(documents[0])

    cluster_text_dict = defaultdict(doc_type)
    for i, doc in enumerate(documents):
        if doc_type is str:
            cluster_text_dict[cluster_labels[i]] += doc + " "
        elif doc_type is list:
            cluster_text_dict[cluster_labels[i]] += doc
    return cluster_text_dict


def cluster_keywords(
    documents: Iterator[str],
    cluster_labels: Iterator[int],
    n: int = 10,
    tokenizer=simple_tokenizer,
    max_df: float = 0.90,
    min_df: float = 0.01,
    Vectorizer=TfidfVectorizer,
) -> Dict:
    """
    Generates keywords that characterise the cluster, using the specified Vectorizer

    Args:
        documents: List of (preprocessed) text documents
        cluster_labels: List of integer cluster labels
        n: Number of top keywords to return
        Vectorizer: Vectorizer object to use (eg, TfidfVectorizer, CountVectorizer)
        tokenizer: Function to use to tokenise the input documents; by default splits the document into words

    Returns:
        Dictionary that maps cluster integer labels to a list of keywords
    """

    # Define vectorizer
    vectorizer = Vectorizer(
        analyzer="word",
        tokenizer=tokenizer,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_df=max_df,
        min_df=min_df,
        max_features=10000,
    )

    # Create cluster text documents
    cluster_documents = cluster_texts(documents, cluster_labels)
    unique_cluster_labels = list(cluster_documents.keys())

    # Apply the vectorizer
    token_score_matrix = vectorizer.fit_transform(list(cluster_documents.values()))

    # Create a token lookup dictionary
    id_to_token = dict(
        zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
    )

    # For each cluster, check the top n tokens
    top_cluster_tokens = {}
    for i in range(token_score_matrix.shape[0]):
        # Get the cluster feature vector
        x = token_score_matrix[i, :].todense()
        # Find the indices of the top n tokens
        x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
        # Find the tokens corresponding to the top n indices
        top_cluster_tokens[unique_cluster_labels[i]] = [id_to_token[j] for j in x]

    return top_cluster_tokens
