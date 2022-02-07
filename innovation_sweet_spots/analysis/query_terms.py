"""
innovation_sweet_spots.analysis.query_terms

Module for searching texts using search terms (keywords and key phrases)
"""
from innovation_sweet_spots import logging
import numpy as np
import pandas as pd
from typing import Iterator, Dict, Union
from numpy.typing import ArrayLike


def token_list_to_string(
    token_list: Iterator[str], within_token_separator: str = "_", padding: str = " "
) -> str:
    """
    Converts a list of tokens to a string. For example: ['best', 'heat_pump'] -> ' best heat pump '

    Args:
        token_list: List of tokens
        within_token_separator: Symbol used to indicate spaces within tokens that are phrases
            (eg, in 'heat_pump'); only relevant if the tokens are bigrams, trigrams etc.

    Returns:
        String of words, where tokens have been separated by spaces and within token separators are
        also converted to spaces. The string is also padded with spaces to facilitate simple search term querying.
    """
    s_list = [
        s
        for s_list in [tok.split(within_token_separator) for tok in token_list]
        for s in s_list
    ]
    # Return a padded string
    return f"{padding}{' '.join(s_list)}{padding}"


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
    text_corpus: Iterator[str],
    verbose: bool = False,
) -> ArrayLike:
    """
    Indicates the documents in corpus that contain any of the provided set of terms.

    For example, if set_of_terms = [['heat', 'hydrogen'], ['heat pump']], then
    the method will return True for all documents in the corpus that have either
    both the strings 'heat' AND 'hydrogen', OR that have the string 'heat pump'

    Args:
        terms: A list of a list of string terms
        text_corpus: A list of string documents

    Returns:
        A dictionary with terms as keys, and array of booleans as values where True elements
            indicate presence of the provided terms in the corresponding text_corpus's documents.
            The key "any_terms" indicates documents with any of the provided terms.
    """
    # Initiate a list with all elements equal to False
    bool_list = np.array([False] * len(text_corpus), dtype=bool)
    terms_matches = dict()
    for terms in set_of_terms:
        # Check if terms are in the documents, and store output list in the dictionary
        matches = find_documents_with_terms(terms, text_corpus)
        if verbose:
            logging.info(f"Found {matches.sum()} documents with search terms {terms}")
        # Contribute to the summary output list
        bool_list = bool_list | matches
        terms_matches[str(terms)] = matches
    # Save the summary output list
    terms_matches["has_any_terms"] = bool_list
    return terms_matches


class QueryTerms:
    """
    This class helps to find documents in a corpus of text 'documents'
    that contain search terms. Note that the documents should be preprocessed.
    """

    def __init__(
        self, corpus: Dict[str, Union[Iterator[str], str]], verbose: bool = True
    ):
        """
        Args:
            corpus: Should be a dictionary following the format {document_id: document}
                The document data type might be a string or a list of tokens
        """
        # Check the document type (assuming that all documents have the same type):
        if type(next(iter(corpus.values()))) is list:
            # If the document type is list, convert the list of tokens into a single string
            self.text_corpus = [token_list_to_string(s) for s in corpus.values()]
        else:
            self.text_corpus = list(corpus.values())
        self.document_ids = list(corpus.keys())
        self.verbose = verbose

    def find_matches(
        self, search_queries: Iterator[Iterator[str]], return_only_matches: bool = False
    ) -> pd.DataFrame:
        """
        Finds documents containing the provided set of search queries. Each search
        query must be specified as a list of strings, where all strings need to be
        present in the document.

        For example, if search_queries = [['heat', 'hydrogen'], ['heat pump']], then
        the method will return True for all documents in the corpus that have either
        both the strings 'heat' AND 'hydrogen', OR that have the string 'heat pump'

        Args:
            search_terms: A list of a search queries, where a query is a list of strings
            return_only_matches: If True, will only return the documents that are matches

        Returns:
            A dataframe with the following columns:
                - a column for document identifiers
                - a boolean column for each of the search queries, where True indicates a match
                - a column 'has_any_terms' which indicates if any search queries where matched
        """
        matches = find_documents_with_set_of_terms(
            search_queries, self.text_corpus, verbose=self.verbose
        )
        matches["id"] = list(self.document_ids)
        if return_only_matches:
            return pd.DataFrame(matches).query("has_any_terms==True")
        else:
            return pd.DataFrame(matches)
