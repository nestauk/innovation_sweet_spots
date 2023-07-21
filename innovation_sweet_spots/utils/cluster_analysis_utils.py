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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction import text
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
    """ "Reduce dimensions of the input array using UMAP"""
    logging.info(
        f"Generating {umap_def_params['n_components']}-d UMAP embbedings for {len(vectors)} vectors"
    )
    reducer = umap.UMAP(random_state=random_umap_state, **umap_params)
    return reducer.fit_transform(vectors)


def hdbscan_clustering(
    vectors: np.typing.ArrayLike, hdbscan_params: dict, have_noise_labels: bool = False
) -> np.typing.ArrayLike:
    """Cluster vectors using HDBSCAN.

    Args:
        vectors: Vectors to cluster.
        hdbscan_params: Clustering parameters.
        have_noise_labels: If True, HDBSCAN will label vectors with
            noise as -1. If False, no vectors will be labelled as noise
            but vectors with 0 probability of being assigned to any cluster
            will be labelled as -2.


    Returns:
        Dataframe with label assignment and probability of
            belonging to that cluster.
    """
    logging.info(f"Clustering {len(vectors)} vectors with HDBSCAN.")
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusterer.fit(vectors)
    if have_noise_labels:
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
    else:
        cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
        probabilities = []
        labels = []
        for probs in cluster_probs:
            probability = np.max(probs)
            label = (
                -2 if probability == 0 or np.isnan(probability) else np.argmax(probs)
            )
            probabilities.append(probability)
            labels.append(label)
        probabilities = np.array(probabilities)
        labels = np.array(labels)

    return pd.DataFrame({"labels": labels, "probability": probabilities}).astype(
        {"labels": int}
    )


def kmeans_clustering(
    vectors: np.typing.ArrayLike, kmeans_params: dict
) -> pd.DataFrame:
    """Cluster vectors using K-Means clustering"""
    logging.info(f"Clustering {len(vectors)} vectors with K-Means clustering")
    kmeans = KMeans(**kmeans_params).fit(vectors)
    return kmeans.labels_


def kmeans_param_grid_search(
    vectors: np.typing.ArrayLike, search_params: dict, random_seeds: list
) -> pd.DataFrame:
    """Perform grid search over search parameters and calculate
    mean silhouette score for K-means clustering

    Args:
        vectors: Embedding vectors.
        search_params: Dictionary with keys as parameter names
            and values as a list of parameters to search through.
        random_seeds: Param search will be performed for each
            random seed specified and then the results averaged.

    Returns:
        Dataframe with information on clustering method,
        parameters, random seed, mean silhouette score.
    """
    parameters_record = []
    silhouette_score_record = []
    method_record = []
    random_seed_record = []
    reduced_dims_vectors = umap_reducer(vectors)
    distances = euclidean_distances(reduced_dims_vectors)
    for random_seed in tqdm(random_seeds):
        for parameters in tqdm(ParameterGrid(search_params)):
            parameters["random_state"] = random_seed
            clusters = kmeans_clustering(reduced_dims_vectors, kmeans_params=parameters)
            silhouette = silhouette_score(distances, clusters, metric="precomputed")
            method_record.append("K-Means clustering")
            parameters.pop("random_state")
            parameters_record.append(str(parameters))
            silhouette_score_record.append(silhouette)
            random_seed_record.append(random_seed)

    return (
        pd.DataFrame.from_dict(
            {
                "method": method_record,
                "model_params": parameters_record,
                "random_seed": random_seed_record,
                "silhouette_score": silhouette_score_record,
            }
        )
        .groupby(["method", "model_params"])["silhouette_score"]
        .mean()
        .reset_index()
        .sort_values("silhouette_score", ascending=False)
    )


def hdbscan_param_grid_search(
    vectors: np.typing.ArrayLike, search_params: dict, have_noise_labels: bool = False
) -> pd.DataFrame:
    """Perform grid search over search parameters and calculate
    mean silhouette score for HDBSCAN

    Args:
        vectors: Embedding vectors.
        search_params: Dictionary with keys as parameter names
            and values as a list of parameters to search through.
        have_noise_labels: If True, HDBSCAN will label vectors with
            noise as -1. If False, no vectors will be labelled as noise
            but vectors with 0 probability of being assigned to any cluster
            will be labelled as -2.

    Returns:
        Dataframe with information on clustering method, parameters,
            mean silhouette scoure.
    """
    parameters_record = []
    silhouette_score_record = []
    method_record = []
    reduced_dims_vectors = umap_reducer(vectors)
    distances = euclidean_distances(reduced_dims_vectors)
    for parameters in tqdm(ParameterGrid(search_params)):
        clusters = hdbscan_clustering(
            reduced_dims_vectors,
            hdbscan_params=parameters,
            have_noise_labels=have_noise_labels,
        )
        silhouette = silhouette_score(
            distances, clusters.labels.values, metric="precomputed"
        )
        method_record.append("HDBSCAN")
        parameters_record.append(parameters)
        silhouette_score_record.append(silhouette)

    return pd.DataFrame.from_dict(
        {
            "method": method_record,
            "model_params": parameters_record,
            "silhouette_score": silhouette_score_record,
        }
    ).sort_values("silhouette_score", ascending=False)


def highest_silhouette_model_params(param_search_results: pd.DataFrame) -> dict:
    """Return dictionary of model params with the highest
    scoring mean silhouette score"""
    return param_search_results.query(
        "silhouette_score == silhouette_score.max()"
    ).model_params.values[0]


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
    my_stop_words = text.ENGLISH_STOP_WORDS

    # Define vectorizer
    vectorizer = Vectorizer(
        analyzer="word",
        tokenizer=tokenizer,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_df=max_df,
        min_df=min_df,
        max_features=10000,
        stop_words=my_stop_words,
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


## kk additions
import altair as alt

alt.data_transformers.disable_max_rows()
import innovation_sweet_spots.utils.embeddings_utils as eu
import innovation_sweet_spots.utils.plotting_utils as pu


def cluster_visualisation(
    vectors,
    cluster_labels: list,
    width=600,
    height=600,
    random_state=1,
    extra_data: pd.DataFrame = None,
):
    """Reduces the vectors to 2D and plots them with altair"""
    if vectors.shape[1] != 2:
        vectors_2d = eu.reduce_to_2D(vectors, random_state)
    else:
        vectors_2d = vectors
    # Create a dataframe
    data = pd.DataFrame(
        data={
            "x": vectors_2d[:, 0],
            "y": vectors_2d[:, 1],
            "cluster_label": [str(c) for c in cluster_labels],
        }
    )
    # Add extra data
    if extra_data is not None:
        extra_columns = list(extra_data.columns)
        for col in extra_columns:
            data[col] = extra_data[col].to_list()
    else:
        extra_columns = []
    # Plot the clusters
    fig = (
        alt.Chart(data, width=width, height=height)
        .mark_circle()
        .encode(
            x=alt.X(
                "x", axis=alt.Axis(labels=False, title="", ticks=False, domain=False)
            ),
            y=alt.Y(
                "y", axis=alt.Axis(labels=False, title="", ticks=False, domain=False)
            ),
            color=alt.Color("cluster_label"),
            tooltip=["cluster_label"] + extra_columns,
        )
    )
    fig = (
        pu.configure_plots(fig)
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .interactive()
    )
    return vectors_2d, fig
