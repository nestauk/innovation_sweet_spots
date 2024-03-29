"""
innovation_sweet_spots.utils.text_processing_utils

Utils for text preprocessing and tokenising
"""
import spacy
import logging
import re
import spacy
from gensim import models
from typing import Iterator
from os import PathLike
import pandas as pd
import html
from innovation_sweet_spots.utils.text_cleaning_utils import clean_text

# DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["ner"]}
DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["tok2vec"]}
DROP_NERS = ["ORG", "DATE", "QUANTITY", "PERSON", "CARDINAL", "ORDINAL", "GPE", "LOC"]
DROP_NERS_MIN = []


def preprocess_nothing(text: str) -> str:
    """Dummy function for doing no preprocessing"""
    return text


def preprocess_minimal(text: str) -> str:
    """Converts to lower case and strips whitespace"""
    return text.lower().strip()


def preprocess_clean_text(text: str) -> str:
    """More involved preprocessing (punctation, lemmatising etc.)"""
    return clean_text(text)


def create_documents_from_dataframe(
    df: pd.DataFrame, columns: Iterator[str], preprocessor=preprocess_nothing
) -> Iterator[str]:
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy().astype(str)
    # Preprocess project text
    text_lists = [df_[col].to_list() for col in columns]
    # Create project documents
    docs = [preprocessor(html.unescape(text)) for text in create_documents(text_lists)]
    return docs


def create_documents(lists_of_texts: Iterator[str]) -> Iterator[str]:
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']
    Parameters
    ----------
        lists_of_texts:
            Contains lists of texts to be joined up and processed to create
            the "documents"; i-th element of each list corresponds to the
            i-th entity/document
    Yields
    ------
        Iterator[str]
            Document generator
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            ". ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def setup_spacy_model(model_parameters=DEF_LANGUAGE_MODEL) -> spacy.Language:
    """
    Load and set up a spacy language model
    Args:
        model_parameters (dict): Dictionary containing language model parameters.
        The dictionary is expected to have the following structure:
            {
                "model":
                    Spacy language model name; for example "en_core_web_sm",
                "disable":
                    Pipeline components to disable for faster processing
                    (e.g. disable=["ner"] to disable named entity recognition).
                    If not required, set it to None.
            }
    Returns:
        (spacy.Language): A spacy language model
    """
    nlp = spacy.load(model_parameters["model"])
    if model_parameters["disable"] is not None:
        nlp.select_pipes(disable=model_parameters["disable"])
    return nlp


def remove_newline(text):
    """Removes new lines from documents"""
    return re.sub("\n", " ", text.lower())


def process_spacy_doc_to_tokens(doc: spacy.tokens.doc.Doc) -> list:
    """Process a spacy document
    Args:
        doc: spacy tokenised document
    Returns:
        A list of tokens after processing
    """

    no_stops = [
        x
        for x in doc
        if (x.is_stop is False)
        & (x.is_punct is False)
        & (x.is_digit is False)
        & (x.like_url is False)
        & (x.is_space is False)
    ]

    drop_ents = [x.text for x in doc.ents if x.label_ in set(DROP_NERS)]

    no_ents = [
        x.lemma_ for x in no_stops if all(x.text not in ent for ent in drop_ents)
    ]

    return no_ents


def make_ngram(
    tokenised_corpus: list, n_gram: int = 2, threshold: float = 0.35, min_count: int = 5
) -> list:
    """Extract bigrams from tokenised corpus
    Args:
        tokenised_corpus: List of tokenised corpus
        n_gram: maximum length of n-grams. Defaults to 2 (bigrams)
        threshold:
        min_count: minimum number of token occurrences
    Returns:
        ngrammed_corpus
    """
    tokenised = tokenised_corpus.copy()
    t = 1
    # Loops while the ngram length less / equal than our target
    while t < n_gram:
        phrases = models.Phrases(
            tokenised, min_count=min_count, threshold=threshold, scoring="npmi"
        )
        ngram_phraser = models.phrases.Phraser(phrases)
        tokenised = ngram_phraser[tokenised]
        t += 1
    return list(tokenised), ngram_phraser


def ngrammer(text: str, ngram_phraser, nlp: spacy.Language = None):
    if nlp is None:
        nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)
    return ngram_phraser[process_text(nlp(remove_newline(text)))]


def process_and_tokenise_corpus(text_corpus: Iterator[str], nlp=None, **kwargs) -> list:
    """Pre-process a corpus of text
    Args:
        text_corpus: list of documents
    Returns:
        Tokenised and ngrammed list
    """
    if nlp is None:
        nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)

    corpus = [remove_newline(x) for x in text_corpus]

    logging.info("Tokenising")
    tokenised = [process_spacy_doc_to_tokens(doc) for doc in nlp.pipe(corpus)]

    logging.info("Ngramming")
    tok_ngram, ngram_phraser = make_ngram(tokenised, **kwargs)

    return tok_ngram, ngram_phraser


def process_corpus(text_corpus: Iterator[str], nlp=None, verbose: bool = True) -> list:
    """
    Same as process_and_tokenise_corpus but without creating phrases
    """
    if nlp is None:
        nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)
    corpus = [remove_newline(x) for x in text_corpus]
    if verbose:
        logging.info("Tokenising")
    tokenised = [process_spacy_doc_to_tokens(doc) for doc in nlp.pipe(corpus)]
    return tokenised


def process_text_disc(doc: spacy.tokens.doc.Doc) -> list:
    """Adapted from process_spacy_doc_to_tokens for discourse analysis"""

    no_stops = [
        x
        for x in doc
        if (x.is_stop is False)
        & (x.is_punct is False)
        & (x.is_digit is False)
        & (x.like_url is False)
        & (x.is_space is False)
    ]

    drop_ents = [x.text for x in doc.ents if x.label_ in set(DROP_NERS_MIN)]

    no_ents = [x for x in no_stops if all(x.text not in ent for ent in drop_ents)]

    return no_ents


def simple_tokenizer(text: str) -> Iterator[str]:
    return [token.strip() for token in text.split(" ") if len(token) > 0]
