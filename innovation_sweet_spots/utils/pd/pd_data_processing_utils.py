#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:37:14 2021

@author: jdjumalieva
"""
import os
from collections import defaultdict
import pandas as pd
import re
from typing import Iterator

from innovation_sweet_spots.utils import text_cleaning_utils as tcu
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH


DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["tok2vec"]}


det_article_replacement = {
    # old patterns: replacement pattern
    "^the\s": "",
    "^a\s": "",
    "^an\s": "",
}

compiled_det_art_patterns = [re.compile(a) for a in det_article_replacement.keys()]
det_art_replacement = list(det_article_replacement.values())


def clean_articles(articles):
    """Performs minimal clearning of article text.

    Minimum cleaning includes: split camel-case, convert to lower case, clean
    punctuation, remove extra spaces.

    Args:
        articles: A list of articles.

    Returns:
        clean_article_text (list): clean text of articles.

    Raises:
        TypeError: If incorrect data type was passed in args.
    """
    clean_article_text = [tcu.clean_text_minimal(article) for article in articles]
    clean_article_text = [
        elem for elem in clean_article_text if elem != "sentence is blank"
    ]
    return clean_article_text


def generate_sentence_corpus(clean_article_text, nlp_model):
    """Generates a collection of sentences and a spacy corpus.

    Cleans article text, processes the articles with spacy and break them up into
    sentences.

    Args:
        articles: A list of articles.
        nlp: A spacy language model.

    Returns:
       A list of sentences.
       A list of processed spacy docs.
    """
    spacy_docs = [nlp_model(article) for article in clean_article_text]
    article_sentences = [
        [sent.text for sent in article.sents] for article in spacy_docs
    ]
    return article_sentences, spacy_docs


def generate_sentence_corpus_by_year(
    article_text_df, nlp_model, year_field="year", text_field="text", id_field="id"
):
    """Generates spacy processed docs and sentence records for each year in the dataset.

    Several outputs are produced at once to avoid repeating pre-processing with
    spacy.

    Args:
        article_text_df: A pandas dataframe with article text, id and year.
        nlp_model: A spacy language model used.

    Returns:
        A dict with spacy docs for each year.
        A list of tuples with sentence, id, year.
    """
    sentence_records = []
    processed_articles_by_year = defaultdict(dict)
    for year, group in article_text_df.groupby(year_field):
        clean_article_text = clean_articles(group[text_field].values)
        sentences, processed_articles = generate_sentence_corpus(
            clean_article_text, nlp_model
        )
        ids = group[id_field]
        for sentence_bunch in zip(sentences, ids):
            article_id = sentence_bunch[1]
            for sentence in sentence_bunch[0]:
                sentence_records.append((sentence, article_id, year))
        processed_articles_by_year[str(year)] = processed_articles
    return (processed_articles_by_year, sentence_records)


def get_noun_chunks(spacy_corpus, remove_det_articles=False):
    """Extracts noun phrases from articles using spacy's inbuilt methods.

    Args:
        spacy_corpus: A dict mapping years to a list of spacy processed documents.
        remove_det_articles: A boolean option to remove determiner articles (a, an, the).
            The default is False.

    Returns:
        A deduplicated list of strings.
    """
    noun_chunks = []
    for article in spacy_corpus:
        for chunk in article.noun_chunks:
            noun_chunks.append(chunk)  # could change to chunk.lemma_ if needed

    # convert spacy tokens to string
    noun_chunks_str = [str(elem) for elem in noun_chunks]
    if remove_det_articles:
        noun_chunks_str = [remove_determiner_articles(elem) for elem in noun_chunks_str]
    dedup_noun_chunks = list(set(noun_chunks_str))
    return dedup_noun_chunks


def remove_determiner_articles(text):
    """Removes determiner articles 'a', 'an', 'the' at the start of the string.

    This function is used to clean up noun phrases.

    Args:
        text: A string with some text.

    Returns:
        A string with determiner articles removed.
    """
    for a, pattern in enumerate(compiled_det_art_patterns):
        text = pattern.sub(det_art_replacement[a], text)
    return text


def get_flat_sentence_mentions(search_terms, sentence_collection):
    """Retrieves sentences that mention a given search_term.

    Identifies sentences that contain search_term using regex.

    Args:
        search_term: A term of interest.
        sentence_collection: A list of tuples with sentence, id, year.

    Returns:
        A dict mapping years to sentences containing the search_term.
    """
    base = r"{}"
    expr = "(?:\s|^){}(?:,?\s|\.|$)"
    for term in search_terms:
        combined_expressions = [
            base.format("".join(expr.format(term))) for term in search_terms
        ]
    joined_expressions = "|".join(combined_expressions)

    year_flat_sentences = dict()
    sentence_collection_df = pd.DataFrame(sentence_collection)
    sentence_collection_df.columns = ["sentence", "id", "year"]
    for year, sentences in sentence_collection_df.groupby("year"):
        sentences_with_term = sentences[
            sentences["sentence"].str.contains(joined_expressions)
        ]
        year_flat_sentences[str(year)] = sentences_with_term
    return year_flat_sentences


###########
# Utility functions for quickly checking collocations
def check_collocations(sentence_collection_df, collocated_term, groupby_field="year"):
    """Retrieves sentences with collocations.

    Args:
        sentence_collection_df: A pandas dataframe with sentences.
        collocated_term: A string referring to the term of interest.
        groupby_field: A string referring to the field on which sentence dataframe
            will be grouped.
    Returns:
        A pandas groupby object.
    """
    base = r"{}"
    expr = "(?:\s|^){}(?:,?\s|\.|$)"
    combined_expr = base.format("".join(expr.format(collocated_term)))
    collocation_df = sentence_collection_df[
        sentence_collection_df["sentence"].str.contains(combined_expr, regex=True)
    ]
    grouped_by_year = collocation_df.groupby(groupby_field)
    return grouped_by_year


def collocation_summary(grouped_sentences):
    """Prints a quick summary of collocations.

    Args:
        grouped_sentences: A pandas groupby object.

    Returns:
        None.
    """
    num_years = len(grouped_sentences)
    num_sentences = sum([len(group) for name, group in grouped_sentences])
    print(
        f"The terms were mentioned together in {num_sentences} sentences across {num_years} years."
    )


def view_collocations(
    grouped_sentences,
    metadata_dict,
    sentence_record_dict,
    url_field="webUrl",
    title_field="webTitle",
    output_to_file=True,
    output_path=OUTPUT_DATA_PATH,
):
    """Prints sentences and corresponding article metadata grouped by year.

    Args:
        grouped_sentences: A pandas groupby object.
        metadata_dict: A dict mapping article IDs to original article metadata.
        sentence_record_dict: A dict mapping sentences to article IDs.

    Returns:
        None.
    """
    results = []
    for year, group in grouped_sentences:
        print(year)
        for ix, row in group.iterrows():
            sentence = row["sentence"]
            sent_id = sentence_record_dict[sentence]
            web_url = metadata_dict[sent_id][url_field]
            article_title = metadata_dict[sent_id][title_field]
            print(article_title)
            print(sentence, end="\n\n")
            print(web_url, end="\n\n")
            print("----------")
            results.append([year, article_title, sentence, web_url])
    if output_to_file:
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(output_path / "sentences_w_collocations.csv", index=False)


def view_collocations_given_year(
    year_sentences,
    metadata_dict,
    sentence_record_dict,
    url_field="webUrl",
    title_field="webTitle",
):
    """Prints sentences and corresponding article metadata for a given year.

    Args:
        year_sentences: A pandas dataframe.
        metadata_dict: A dict mapping article IDs to original article metadata.
        sentence_record_dict: A dict mapping sentences to article IDs.

    Returns:
        None.
    """
    for ix, row in year_sentences.iterrows():
        sentence = row["sentence"]
        sent_id = sentence_record_dict[sentence]
        web_url = metadata_dict[sent_id][url_field]
        article_title = metadata_dict[sent_id][title_field]
        print(article_title)
        print(sentence, end="\n\n")
        print(web_url, end="\n\n")
        print("----------")


# Utility functions for quickly checking mentions (adapted from above)
def check_mentions(sentence_collection_df, term, groupby_field="year"):
    """Retrieves sentences with mentions.

    Args:
        sentence_collection_df: A pandas dataframe with sentences.
        collocated_term: A string referring to the term of interest.
        groupby_field: A string referring to the field on which sentence dataframe
            will be grouped.
    Returns:
        A pandas groupby object.
    """
    base = r"{}"
    expr = "(?:\s|^){}(?:,?\s|\.|$)"
    combined_expr = base.format("".join(expr.format(term)))
    mentions_df = sentence_collection_df[
        sentence_collection_df["sentence"].str.contains(combined_expr, regex=True)
    ]
    grouped_by_year = mentions_df.groupby(groupby_field)
    return grouped_by_year


def mentions_summary(grouped_sentences):
    """Prints a quick summary of mentions.

    Args:
        grouped_sentences: A pandas groupby object.

    Returns:
        None.
    """
    num_years = len(grouped_sentences)
    num_sentences = sum([len(group) for name, group in grouped_sentences])
    print(
        f"The term was mentioned in {num_sentences} sentences across {num_years} years."
    )


def view_mentions(
    grouped_sentences,
    metadata_dict,
    sentence_record_dict,
    url_field="webUrl",
    title_field="webTitle",
    output_to_file=True,
    output_path=OUTPUT_DATA_PATH,
):
    """Prints sentences and corresponding article metadata grouped by year.

    Args:
        grouped_sentences: A pandas groupby object.
        metadata_dict: A dict mapping article IDs to original article metadata.
        sentence_record_dict: A dict mapping sentences to article IDs.

    Returns:
        None.
    """
    results = []
    for year, group in grouped_sentences:
        print(year)
        for ix, row in group.iterrows():
            sentence = row["sentence"]
            sent_id = sentence_record_dict[sentence]
            web_url = metadata_dict[sent_id][url_field]
            article_title = metadata_dict[sent_id][title_field]
            print(article_title)
            print(sentence, end="\n\n")
            print(web_url, end="\n\n")
            print("----------")
            results.append([year, article_title, sentence, web_url])
    if output_to_file:
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(output_path / "sentences_w_mentions.csv", index=False)
