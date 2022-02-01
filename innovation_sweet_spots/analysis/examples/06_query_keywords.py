# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import load_pickle
from typing import Iterator
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# %%
def token_list_to_string(token_list: Iterator[str]) -> str:
    """Converts a list of tokens to a string"""
    s_list = [s for s_list in [tok.split("_") for tok in token_list] for s in s_list]
    # Return a string padded with spaces
    return f" {' '.join(s_list)} "


def is_search_term_present(
    search_term: str, documents: Iterator[str]
) -> Iterator[bool]:
    """
    Simple method to check if a keyword or keyphrase is in the set of string documents.
    Note that this method does not perform any text cleaning. Therefore, the text should be already
    preprocessed, and the formulation of search terms should be congruent with the preprocessing approach.

    Args:
        search_term: The string to search for in the documents
        documents: List of text strings to check

    Returns:
        List of booleans, with True for each document that contains the search term
    """
    return [search_term in doc for doc in documents]


def get_matching_documents(
    documents: Iterator[str], matches=Iterator[bool]
) -> Iterator[str]:
    """Returns the subset of documents whose corresponding elements in the list of matches is True"""
    assert len(documents) == len(
        matches
    ), "The number of documents must equal the number of elements in matches"
    return [doc for i, doc in enumerate(documents) if matches[i] is np.bool_(True)]


def find_documents_with_terms(
    terms: Iterator[str],
    corpus: Iterator[str],
) -> ArrayLike:
    """
    Indicates the documents in corpus that contain all provided terms

    For example, if terms = ['heat', 'hydrogen'] then the method will return True
    for documents that contain both 'heat' AND 'hydrogen' strings

    Args:
        terms: A list of string terms
        corpus: A list of string documents

    Returns:
        Array of booleans, with True elements indicating presence of all provided terms
        in the corresponding corpus' documents
    """
    # Initiate a list with all elements equal to True
    bool_list = np.array([True] * len(corpus), dtype=bool)
    # Check each term in terms list
    for term in terms:
        bool_list = bool_list & np.array(is_search_term_present(term, corpus))
    return bool_list


def find_documents_with_set_of_terms(
    set_of_terms: Iterator[Iterator[str]],
    corpus: Iterator[str],
    verbose: bool = False,
) -> ArrayLike:
    """
    Indicates the documents in corpus that contain any of the provided set of terms.

    For example, if set_of_terms = [['heat', 'hydrogen'], ['heat pump']], then
    the method will return True for all documents in the corpus that have either
    both the strings 'heat' AND 'hydrogen', OR that have the string 'heat pump'

    Args:
        terms: A list of a list of string terms
        corpus: A list of string documents

    Returns:
        A dictionary with terms as keys, and array of booleans as values where True elements
            indicate presence of the provided terms in the corresponding corpus' documents.
            The key "any_terms" indicates documents with any of the provided terms.
    """
    # Initiate a list with all elements equal to False
    bool_list = np.array([False] * len(corpus), dtype=bool)
    terms_matches = dict()
    for terms in set_of_terms:
        # Check if terms are in the documents, and store output list in the dictionary
        matches = find_documents_with_terms(terms, corpus)
        if verbose:
            logging.info(f"Found {matches.sum()} documents with search terms {terms}")
        # Contribute to the summary output list
        bool_list = bool_list | matches
        terms_matches[str(terms)] = matches
    # Save the summary output list
    terms_matches["has_any_terms"] = bool_list
    return terms_matches


# %%
DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/tokenised_data/"
CORPUS_FILEPATH = DIR / "gtr_docs_tokenised_full.p"
SEARCH_TERMS = [["construction"], ["education"]]

# %%
# TODO: Need to create getter functions for loading and preprocessing the corpus file

# %%
# Load the tokenised corpus file (a dict id: tokenised text)
corpus_gtr = load_pickle(CORPUS_FILEPATH)

# %%
# Select a test corpus
tokenised_corpus = {key: corpus_gtr[key] for i, key in enumerate(corpus_gtr) if i < 100}

# %%
# Prepare text corpus
text_corpus = [token_list_to_string(s) for s in tokenised_corpus.values()]

# %%
# Find matches and prepare output
matches = find_documents_with_set_of_terms(search_terms, text_corpus, verbose=True)
matches["id"] = list(tokenised_corpus.keys())
df = pd.DataFrame(matches)

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib

importlib.reload(wu)
GTR = wu.GtrWrangler()

# %%
GTR.add_project_data(df.query("has_any_terms==True"), id_column="id", columns=["title"])

# %%
