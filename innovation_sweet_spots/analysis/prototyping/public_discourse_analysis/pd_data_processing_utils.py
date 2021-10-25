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
from innovation_sweet_spots.utils import text_pre_processing as tpu
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH


DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["tok2vec"]}


det_article_replacement = {
    # old patterns: replacement pattern
    "^the\s": "",
    "^a\s": "",
    "^an\s": ""
    }

compiled_det_art_patterns = [re.compile(a) for a in det_article_replacement.keys()]
det_art_replacement = list(det_article_replacement.values())


def clean_articles(articles: Iterator[str]):
    """
    Clean article text.
    Parameters
    ----------
    articles (list): text of articles.

    Returns
    -------
    clean_article_text (list): clean text of articles.
    """
    # Minimum cleaning includes: split camel-case, convert to lower case, 
    # clean punctuation, remove extra spaces
    clean_article_text = [tcu.clean_text_minimal(article) for article in articles]
    clean_article_text = [elem for elem in clean_article_text if elem != 'sentence is blank']
    return clean_article_text
    
    
def generate_sentence_corpus (clean_article_text: Iterator[str], nlp_model):
    """
    Clean article text, process with spacy and break up into sentences.
    
    Parameters
    ----------
    articles (list): text of articles.
    nlp (spacy.lang.en.English, optional): spacy model used. 
    
    Returns
    -------
    article_sentences (list): list of sentences.
    spacy_docs (list[]): list of processed spacy docs (spacy.tokens.doc.Doc).
    """
    spacy_docs = [nlp_model(article) for article in clean_article_text]
    article_sentences = [[sent.text for sent in article.sents] for article in spacy_docs]
    return article_sentences, spacy_docs


def generate_sentence_corpus_by_year(article_text_df, nlp_model, year_field = 'year', 
                                     text_field = 'text', id_field = 'id'):
    """
    Generate corpus of sentences, spacy processed docs and sentence records
    for each year in the dataset.
    Several outputs are produced at once to avoid repeating pre-processing with
    spacy.
    
    Parameters
    ----------
    article_text_df (pandas.core.frame.DataFrame): dataframe with article text, id and year.
    nlp_model (spacy.lang.en.English): spacy model used.
    
    Returns
    -------
    processed_articles_by_year (dict): spacy docs for each year.
    sentence_records (list): list of tuples with sentence, id, year.
    """
    sentence_records = []
    processed_articles_by_year = defaultdict(dict)
    for year, group in article_text_df.groupby(year_field):
        clean_article_text = clean_articles(group[text_field])
        sentences, processed_articles = generate_sentence_corpus\
            (clean_article_text, nlp_model)
        ids = group[id_field]
        for sentence_bunch in zip(sentences, ids):
            article_id = sentence_bunch[1]
            for sentence in sentence_bunch[0]:
                sentence_records.append((sentence, article_id, year))
        processed_articles_by_year[str(year)] = processed_articles
    return(processed_articles_by_year, sentence_records)


# Addressing determiner articles, but not stopwords
def get_noun_chunks(spacy_corpus, remove_det_articles = False):
    """
    Extract noun phrases from articles using spacy's inbuilt methods.
    
    Parameters
    ----------
    spacy_corpus (): spacy processed documents.
    remove_det_articles (boolean): option to remove determiner articles (a, an, the).
    the default is False.
    
    Returns
    -------
    dedup_noun_chunks (list): deduplicated list of noun chunks as strings.
    """
    noun_chunks = []
    for article in spacy_corpus:
        for chunk in article.noun_chunks:
            noun_chunks.append(chunk) #could change to chunk.lemma_ if needed 

    #convert spacy tokens to string
    noun_chunks_str = [str(elem) for elem in noun_chunks]
    if remove_det_articles:
        noun_chunks_str = [remove_determiner_articles(elem) for elem in noun_chunks_str]
    dedup_noun_chunks = list(set(noun_chunks_str))
    return dedup_noun_chunks


def remove_determiner_articles(text):
    """
    Remove determiner articles 'a', 'an', 'the' at the start of the string. 
    Used to clean up noun phrases.
    
    Parameters
    ----------
    text (str): some text.
    
    Returns
    -------
    text (str): text with determiner articles removed.
    """
    for a, pattern in enumerate(compiled_det_art_patterns):
        text = pattern.sub(det_art_replacement[a], text)
    return text


def get_flat_sentence_mentions(search_term, sentence_collection):
    """
    Identify sentences that contain search_term using regex.
    Convert list of lists (i.e. lists of sentences in a given article) into
    a flat list of sentences.
    
    Parameters
    ----------
    search_term (str): term of interest.
    sentence_collection (list): list of tuples with sentence, id, year.
    
    Returns
    -------
    year_flat_sentences (dict): sentences with term in each year.
    """
    base = r'{}'
    expr = '(?:\s|^){}(?:,?\s|\.|$)'
    combined_expr = base.format(''.join(expr.format(search_term)))
    year_flat_sentences = dict()
    sentence_collection_df = pd.DataFrame(sentence_collection)
    sentence_collection_df.columns = ['sentence', 'id', 'year']
    for year, sentences in sentence_collection_df.groupby('year'):
        sentences_with_term = sentences[sentences['sentence'].\
                                        str.contains(combined_expr, regex = True)]
        year_flat_sentences[str(year)] = sentences_with_term
    return year_flat_sentences


def combine_term_sentences(term_sentence_dict, search_terms):
    """
    Bring together all sentences that mention any term from the search term list.
    
    Parameters
    ----------
    term_sentence_dict (dict): dictionary with sentences for each year and each term.
    search_terms (list): list of terms.
    
    Returns
    -------
    combined_sentences (dict): dictionary with years as keys and sentence dataframe as values.
    """
    combined_sentences = defaultdict(dict)
    all_keys = [term_sentence_dict[term].keys() for term in search_terms]
    all_years = sorted(list(set([year for sublist in all_keys for year in sublist])))
    for year in all_years:
        year_sents = []
        for term in search_terms:
            year_term_sentences = term_sentence_dict[term].\
            get(year, pd.DataFrame({'sentence': [], 'id':[], 'year':[]}))
            year_sents.append(year_term_sentences)
        year_corpus = pd.concat(year_sents)
        combined_sentences[str(year)] = year_corpus.drop_duplicates()
    return combined_sentences

###########
# Utility functions for quickly checking collocations
def check_collocations(sentence_collection_df, collocated_term, groupby_field = 'year'):
    """
    Retrieve sentences where the collocated_term was mentioned together with one
    of the search terms.
    Parameters
    ----------
    sentence_collection_df (pandas.core.frame.DataFrame): dataframe with sentences
    collocated_term (str): term of interest.
    groupby_field (str): this is the field on which sentence dataframe will be grouped.
    Returns
    -------
    grouped_by_year : pandas groupby object.
    """
    base = r'{}'
    expr = '(?:\s|^){}(?:,?\s|$)'
    combined_expr = base.format(''.join(expr.format(collocated_term)))    
    collocation_df = sentence_collection_df[sentence_collection_df['sentence'].\
                                                str.contains(combined_expr, regex = True)]
    grouped_by_year = collocation_df.groupby(groupby_field)
    return grouped_by_year


def collocation_summary(grouped_sentences):
    """
    Print quick summary of collocations.
    Parameters
    ----------
    grouped_sentences : pandas groupby object.
    Returns
    -------
    None.
    """
    num_years = len(grouped_sentences)
    num_sentences = sum([len(group) for name, group in grouped_sentences])
    print(f"The terms were mentioned together in {num_sentences} sentences across {num_years} years.")


def view_collocations(grouped_sentences, metadata_dict, sentence_record_dict, 
                      url_field = 'webUrl', title_field = 'webTitle'):
    """
    Print sentences grouped by year.
    Parameters
    ----------
    grouped_sentences : pandas groupby object.
    metadata_dict: dict with sentence IDs and urls
    sentence_record_dict: dict with sentences and corresponding IDs and year
    sentence_year (int): optional year to subset sentences
    
    Returns
    -------
    None.
    """
    for year, group in grouped_sentences:
        print(year)
        for ix, row in group.iterrows():
            sentence = row['sentence']
            sent_id = sentence_record_dict[sentence]
            web_url = metadata_dict[sent_id][url_field]
            article_title = metadata_dict[sent_id][title_field]
            print(article_title)
            print(sentence, end = "\n\n")
            print(web_url, end = "\n\n")
            print('----------')