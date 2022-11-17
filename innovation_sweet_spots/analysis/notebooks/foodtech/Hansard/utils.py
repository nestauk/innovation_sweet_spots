import pandas as pd
from typing import Iterable
import innovation_sweet_spots.utils.text_processing_utils as tpu

nlp = tpu.setup_spacy_model()
import nltk.data

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def remove_space_after_comma(text):
    """util function to process search terms with comma"""
    return ",".join([s.strip() for s in text.split(",")])


def process_foodtech_terms(foodtech_terms: pd.DataFrame, nlp=nlp) -> Iterable[str]:
    terms_df = foodtech_terms.query("use == '1'").reset_index(drop=True)

    terms = [s.split(",") for s in terms_df.Terms.to_list()]
    terms_processed = []
    for term in terms:
        terms_processed.append(
            [" ".join(t) for t in tpu.process_corpus(term, nlp=nlp, verbose=False)]
        )
    assert len(terms_processed) == len(terms_df)
    terms_df["terms_processed"] = terms_processed
    return terms_df


def select_terms_by_label(
    text: str, terms_df: pd.DataFrame, terms: Iterable[str], column: str = "Tech area"
) -> Iterable:
    term_number = terms_df[terms_df[column] == text].index
    return [terms[i] for i in term_number]


def compile_term_dict(terms_df, column="Tech area") -> dict:
    tech_area_terms = {}
    terms_processed = terms_df.terms_processed.to_list()
    for tech_area in terms_df[column].unique():
        tech_area_terms[tech_area] = select_terms_by_label(
            text=tech_area, terms_df=terms_df, terms=terms_processed, column=column
        )
    return tech_area_terms


def find_sentences_with_terms(text, terms, all_terms: bool = True):
    """util function that finds terms in sentences"""
    # split text into sentences
    sentences = tokenizer.tokenize(text)
    # keep sentences with terms
    sentences_with_terms = []
    # number of terms in the query
    n_terms = len(terms)
    for i, sentence in enumerate(sentences):
        terms_detected = 0
        # check all terms
        for term in terms:
            if term in sentence.lower():
                terms_detected += 1
        # check if all terms were found
        if all_terms and (terms_detected == n_terms):
            sentences_with_terms.append(sentence)
        # check if at least one term was found
        elif (all_terms is False) and (terms_deteced > 0):
            sentences_with_terms.append(sentence)
        else:
            pass
    return sentences_with_terms


def check_articles_for_comma_terms(text: str, terms: str):
    terms = [term.strip() for term in terms.split(",")]
    sentences_with_terms = find_sentences_with_terms(text, terms, all_terms=True)
    if len(sentences_with_terms) >= 1:
        return True
    else:
        return False
