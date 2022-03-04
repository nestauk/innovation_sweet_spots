"""
innovation_sweet_spots.utils.cluster_analysis_utils

Module for various cluster analysis (eg extracting cluster-specific keywords)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterator, Dict
from collections import defaultdict
from innovation_sweet_spots.utils.text_processing_utils import simple_tokenizer


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
    cluster_text_dict = defaultdict(str)
    for i, doc in enumerate(documents):
        cluster_text_dict[cluster_labels[i]] += doc + " "
    return cluster_text_dict


def cluster_keywords(
    documents: Iterator[str],
    cluster_labels: Iterator[int],
    n: int = 10,
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
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=simple_tokenizer,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_df=0.90,
        min_df=0.01,
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
