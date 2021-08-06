"""
Module for applying guided LDA on the data
"""
from innovation_sweet_spots import PROJECT_DIR, logging
import pandas as pd
import numpy as np
from typing import Iterator
import warnings
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis

warnings.filterwarnings("ignore", category=DeprecationWarning)
import guidedlda


DEF_PARAMETERS = {"n_topics": 50, "n_iter": 20, "random_state": 7, "refresh": 20}
DEF_SEED_CONFIDENCE = 0.95
DEF_OUTPUTS_DIR = PROJECT_DIR / "outputs/data"


def identity_function(x):
    return x


def prepare_model_inputs(
    selected_doc_lists: Iterator[Iterator[str]], topic_seeds: Iterator[Iterator[str]]
):
    """
    Takes a tokenised list of documents and generates a
    document-token matrix and vocabulary

    Parameters
    ----------
    selected_doc_lists :
        Tokenised documents (a list of lists of tokens)
    """

    # Define a vectorizer for preprocessed tokenised documents
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=identity_function,
        preprocessor=identity_function,
        token_pattern=None,
        max_features=50000,
        binary=False,
    )
    # Document (rows) x token (columns) matrix
    doc_token_matrix = vectorizer.fit_transform(selected_doc_lists)
    # Vocabulary
    vocab = list(np.asarray(vectorizer.get_feature_names()))
    # Prepare topic seeds
    topic_seed_dict = get_topic_seed_ids(topic_seeds, vocab)

    return doc_token_matrix, topic_seed_dict, vocab, vectorizer


def get_topic_seed_ids(topic_seeds: Iterator[Iterator[str]], vocab: Iterator[str]):
    """
    Converts tokens to vectorizer matrix indices

    Parameters
    ----------
    topic_seeds :
        List of lists of tokens for seeding topics (one token list per topic)
    """
    # Tokens to index dictionary
    token2id = dict((v, idx) for idx, v in enumerate(vocab))
    # Topic seeds need to be provided by their indices
    topic_seed_dict = {}
    for topic_id, topic_seed_tokens in enumerate(topic_seeds):
        for token in topic_seed_tokens:
            if token in token2id:
                topic_seed_dict[token2id[token]] = topic_id
    return topic_seed_dict


def run_model(
    doc_token_matrix,
    topic_seed_dict,
    parameters=DEF_PARAMETERS,
    seed_confidence=DEF_SEED_CONFIDENCE,
):
    """Fit an LDA model with seeded topic keywords"""
    model = guidedlda.GuidedLDA(**parameters)
    model.fit(
        doc_token_matrix, seed_topics=topic_seed_dict, seed_confidence=seed_confidence
    )
    return model


def get_topic_keywords(model, vocab, n_top_words=8):
    """Collect top n most probable tokens for each topic"""
    topic_word = model.topic_word_
    top_topic_keywords = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][: -(n_top_words + 1) : -1]
        top_topic_keywords.append("Topic {}: {}".format(i, " ".join(topic_words)))
    return top_topic_keywords


def plot_pyLDAvis(
    model, vocab, doc_token_matrix, fpath=DEF_OUTPUTS_DIR / "LDAvis.html"
):
    """Plot visualisation of the topics"""
    # Transform the matrix into a dataframe
    tef_dtm = pd.DataFrame(doc_token_matrix.todense())
    # Calculate doc lengths as the sum of each row of the dtm
    doc_lengths = tef_dtm.sum(axis=1, skipna=True)
    # Transpose the dtm and get a sum of the overall term frequency
    dtm_trans = tef_dtm.T
    dtm_trans["total"] = dtm_trans.sum(axis=1, skipna=True)
    # Create a data dictionary as per this tutorial https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/Movie%20Reviews%2C%20AP%20News%2C%20and%20Jeopardy.ipynb
    data = {
        "topic_term_dists": model.topic_word_,
        "doc_topic_dists": model.doc_topic_,
        "doc_lengths": doc_lengths,
        "vocab": vocab,
        "term_frequency": list(dtm_trans["total"]),
        "sort_topics": False,
    }
    # Prepare the data
    tef_vis_data = pyLDAvis.prepare(**data)
    # Save to HTML
    pyLDAvis.save_html(tef_vis_data, str(fpath))
