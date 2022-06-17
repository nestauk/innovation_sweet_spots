"""
innovation_sweet_spots.utils.cluster_analysis_utils
Module for various cluster analysis (eg extracting cluster-specific keywords)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterator, Dict
from collections import defaultdict

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


def hdbscan_clustering(
    vectors,
    umap_params=umap_def_params,
    hdbscan_params=hdbscan_def_params,
    random_umap_state=1,
    random_hdbscan_state=3333,
    return_only_labels=True,
):
    """
    Helper function for quickly getting some clusters

    Outputs an array of shape (n_vectors, 2) with columns for best cluster index and probability
    """
    # UMAP
    logging.info(
        f"Generating {umap_def_params['n_components']}-d UMAP embbedings for {len(vectors)} vectors"
    )
    reducer_low_dim = umap.UMAP(random_state=random_umap_state, **umap_params)
    embedding = reducer_low_dim.fit_transform(vectors)
    logging.info(f"Clustering {len(embedding)} vectors")
    # HDBSCAN
    np.random.seed(random_hdbscan_state)
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusterer.fit(embedding)
    # Cluster probabilities
    cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
    best_cluster_prob = np.array([(np.argmax(x), np.max(x)) for x in cluster_probs])

    return best_cluster_prob


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
