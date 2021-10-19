#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:36:41 2021

@author: jdjumalieva
"""
import os
import re
from bs4 import BeautifulSoup
from itertools import groupby
from collections import defaultdict
import pandas as pd
from typing import Iterator



def combine_articles(articles):
    """
    Generate a flat list of articles from a nested list where each element
    corresponds to result from a Guardian API for a given search term.

    Parameters
    ----------
    articles (list): a nested list where each element.
    corresponds to JSON response from a Guardian API for a given search term.

    Returns
    -------
    aggregated articles (list)

    """
    assert isinstance(articles, list), 'parameter articles={} not of <class "list">'.\
        format(articles)
    aggregated_articles = [article for sublist in articles for article in sublist]
    return aggregated_articles


def filter_by_category(aggregated_articles, category_list, field='sectionName'):
    """
    From a list of articles only retain those that fall under specifide category.

    Parameters
    ----------
    aggregated_articles (list): list of articles (each a JSON nested dict).
    field (str): specific name of a field that refers to article category. 
                default value is 'sectionName'.
    category_list (list): list of category names (str).

    Returns
    -------
    filtered_articles (list)

    """
    assert isinstance(aggregated_articles, list), 'parameter aggregated_articles={} not of <class "list">'.\
        format(aggregated_articles)
    assert isinstance(field, str), 'parameter field={} not of <class "str">'.\
        format(field)
    assert isinstance(category_list, list), 'parameter category_list={} not of <class "list">'.\
        format(category_list)
    filtered_articles = [a for a in aggregated_articles if a[field] in category_list]
    return filtered_articles
    

def sort_by_year(filtered_articles, date_field = 'webPublicationDate'):
    """
    Sort articles and group by year.

    Parameters
    ----------
    filtered_articles (list): list of articles (each a JSON nested dict).
    date_field (str): field that corresponds to article date. default is 
                        'webPublicationDate'.

    Returns
    -------
    articles_by_year (dict): year as key, list of articles as values.
                            articles grouped by year using itertools.

    """
    assert isinstance(filtered_articles, list), 'parameter filtered_articles={} not of <class "list">'.\
        format(filtered_articles)
    assert isinstance(date_field, str), 'parameter date_field={} not of <class "str">'.\
        format(date_field)   
    sorted_articles = sorted(filtered_articles, key = lambda x: x[date_field][:4])
    articles_by_year = defaultdict(list)
    for k,v in groupby(sorted_articles,key=lambda x:x[date_field][:4]):
        articles_by_year[k] = list(v)
    return articles_by_year
        

def extract_text_from_html(html, tags: Iterator[str]) -> Iterator[str]:
    """
    Extract text from specified html tags.

    Parameters
    ----------
    html (str): content of an article in html format.
    tags (list): list of tags to extract content from.

    Returns
    -------
    no_html (list): list of chunks (e.g. paragraphs) of text extracted from the tags.

    """
    segments = BeautifulSoup(html, "html.parser").find_all(tags)
    no_html = [seg.get_text() for seg in segments]
    return no_html


def get_text_segments(articles: Iterator[dict], tags: Iterator[str]) -> Iterator[str]:
    """
    Extract segments of article text from Guardian articles in html format.

    Parameters
    ----------
    articles (list): list of dicts containing result of the Guardian
    search_content function.
    tags (list): list of tags to extract content from.

    Returns
    -------
    clean_segments (list): nested list of article segments.

    """
    article_html = [article["fields"]["body"] for article in articles]
    clean_segments = [extract_text_from_html(article, tags) for article in article_html]
    return clean_segments


def get_article_text(articles: Iterator[dict], tags: Iterator[str]):
    """
    Extract full text of an article from Guardian articles in html format.

    Parameters
    ----------
    articles (list): list of dicts containing result of the Guardian
    search_content function.
    tags (list): list of tags to extract content from.

    Returns
    -------
    article_text (list): list of article text.

    """
    article_segments = get_text_segments(articles, tags)
    article_text = [" ".join(segment) for segment in article_segments]
    return article_text


def get_article_metadata(grouped_articles, fields_to_extract=["id"]):
    """
    Extract useful article fields from raw data returned by the Guardian API.

    Parameters
    ----------
    grouped_articles (dict): articles grouped by year using itertools.
    fields_to_extract (list): list of fields. the default is "id", other useful
                                fields are "webUrl" and "webTitle"

    Returns
    -------
    year_article_df_combined (pandas.core.frame.DataFrame): dataframe with
    content of article fields.

    """
    year_article_dfs = []
    for year, articles in grouped_articles.items():
        year_data = dict()
        for field in fields_to_extract:
            article_field = [article[field] for article in articles]
            year_data[field] = article_field
            year_data["year"] = year
        year_article_df = pd.DataFrame(year_data)
        year_article_dfs.append(year_article_df)
    year_article_df_combined = pd.concat(year_article_dfs)
    year_article_df_combined = year_article_df_combined.drop_duplicates()
    return year_article_df_combined


def get_article_text_df(grouped_articles, tags):
    """
    Extract article text from articles grouped by year. Deduplicate articles.

    Parameters
    ----------
    grouped_articles (dict): articles grouped by year using itertools.
    tags (list): specified html tags.

    Returns
    -------
    year_articles (pandas.core.frame.DataFrame): dataframe with article text, id and year.

    """
    year_articles_across_years = []
    for year, articles in grouped_articles.items():
        article_text = get_article_text(articles, tags)
        article_id = [article["id"] for article in articles]
        year_article_df = pd.DataFrame({"id": article_id, "text": article_text})
        year_articles_across_years.append(year_article_df)
    year_articles = pd.concat(year_articles_across_years)
    year_articles = year_articles.drop_duplicates()
    return year_articles


def subset_articles(article_text, refine_terms, required_terms, text_field = 'text'):
    """
    This function is used to extract a subset of articles that contain specific terms.
    It is useful for removing irrelevant articles in instances when a broad general 
    search term is used during data collection, e.g. `hydrogen`.
    

    Parameters
    ----------
    article_text (pandas.core.frame.DataFrame): dataframe that contains article text and id.
    refine_terms (list): list of terms that we use to disambiguate the general term.
    required_terms (list): an article must mention at least one of these terms.
    text_field (str): name of the column with text, the default is 'text'.

    Returns
    -------
    deduplicated_df (pandas.core.frame.DataFrame): dataframe with a subset of articles.

    """
    # we first generate a 'thematic' subset of articles that are relevant
    # e.g. applications of hydrogen to heating
    base = r'{}'
    expr = '(?:\s|^){}(?:,?\s|$)'
    subsets = []
    for term in refine_terms:
        combined_expr = base.format(''.join(expr.format(term))) 
        print(term)
        subset = article_text[article_text[text_field].str.contains(combined_expr)]
        subsets.append(subset)
        print(len(subset))
    subset_df = pd.concat(subsets) #duplicates v likely as article may contain more than one refine_term
    deduplicated_df = subset_df.drop_duplicates()  
    # required_terms tend to be geographic areas that we are studying
    # we select only articles that mention those areas from a 'thematic' set
    # defined above        
    if len(required_terms) > 0:
        filtered_subsets = []
        for term in required_terms:
            print(term)
            filtered_subset = deduplicated_df[deduplicated_df[text_field].str.contains(term)]
            filtered_subsets.append(filtered_subset)
            print(len(filtered_subset))
        filtered_subset_df = pd.concat(filtered_subsets)
        deduplicated_df = filtered_subset_df.drop_duplicates()
    return deduplicated_df 


def remove_articles(article_text, term: str, text_field = 'text'):
    """
    A wrapper around pandas subsetting. This function is used to explicitely
    remove any articles that mention a particular term ('e.g. hydrogen peroxide).

    Parameters
    ----------
    article_text (pandas.core.frame.DataFrame): dataframe that contains article text and id.
    term (str): term that we don't want to be mentioned in an article
    text_field (str): name of the column with text, the default is 'text'.

    Returns
    -------
    article_text (pandas.core.frame.DataFrame): dataframe with a subset of articles.

    """
    article_text = article_text[~article_text[text_field].str.contains(term)]
    return article_text  