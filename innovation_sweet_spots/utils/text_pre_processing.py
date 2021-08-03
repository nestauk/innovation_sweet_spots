# Functions to pre-process text

import logging
import re

import spacy
from gensim import models

from innovation_sweet_spots.analysis.text_analysis import setup_spacy_model

DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["tok2vec"]}
DROP_NERS = ["ORG", "DATE", "QUANTITY", "PERSON", "CARDINAL", "ORDINAL", "GPE", "LOC"]
DROP_NERS_MIN = []



def remove_newline(text):
    """Removes new lines from documents"""
    return re.sub("\n", " ", text.lower())


def process_text(doc: spacy.tokens.doc.Doc) -> list:
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


def ngrammer(text, ngram_phraser, nlp=None):
    if nlp is None:
        nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)
    return ngram_phraser[process_text(nlp(remove_newline(text)))]


def pre_process_corpus(text_corpus: list, nlp=None, **kwargs) -> list:
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
    tokenised = [process_text(doc) for doc in nlp.pipe(corpus)]

    logging.info("Ngramming")
    tok_ngram, ngram_phraser = make_ngram(tokenised, **kwargs)

    return tok_ngram, ngram_phraser

def process_text_disc(doc: spacy.tokens.doc.Doc) -> list:
    """Adapted from process_text for discourse analysis
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

    drop_ents = [x.text for x in doc.ents if x.label_ in set(DROP_NERS_MIN)]

    no_ents = [
        x.lemma_ for x in no_stops if all(x.text not in ent for ent in drop_ents)
    ]

    return no_ents