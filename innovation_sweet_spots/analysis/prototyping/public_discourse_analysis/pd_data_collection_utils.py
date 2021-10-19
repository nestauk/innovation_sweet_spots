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


def filter_by_category(aggregated_articles, category_list, field='sectionName'):
    """Returns a subset of articles that fall under specified category.
    
    Args:
        aggregated_articles: A list of articles (each a JSON nested dict).
        field: A string that specifies the field containing information on article 
            category. Default value is 'sectionName'.
        category_list: A list of category names.

    Returns:
        A list of filtered articles.
    
    Raises:
        TypeError: If aggregated_articles and category_list are not of type list.
            And field is not of type str.
    """
    if not isinstance(aggregated_articles, list):
        raise TypeError, 'parameter aggregated_articles={} not of <class "list">'
    if not isinstance(category_list, list):
        raise TypeError, 'parameter category_list={} not of <class "list">'
    if not isinstance(field, str):
        raise TypeError, 'parameter field={} not of <class "str">'   
        
    filtered_articles = [a for a in aggregated_articles if a[field] in category_list]
    return filtered_articles
    

def sort_by_year(filtered_articles, date_field='webPublicationDate'):
    """Sorts articles and groups them by year.

    Args:
        filtered_articles: A list of articles (each a JSON nested dict).
        date_field: A string that corresponds to the field containing information
            about article date. Default is 'webPublicationDate'.

    Returns:
        A dict mapping years to the corresponding lists of articles.
    
    Raises:
        TypeError: If filtered_articles is not of type list and date_field is not
            of type str.
    """
    if not isinstance(filtered_articles, list):
        raise TypeError, 'parameter filtered_articles={} not of <class "list">'
    if not isinstance(date_field, str):
        raise TypeError, 'parameter date_field={} not of <class "str">'
        
    sorted_articles = sorted(filtered_articles, key = lambda x: x[date_field][:4])
    articles_by_year = defaultdict(list)
    for k,v in groupby(sorted_articles,key=lambda x:x[date_field][:4]):
        articles_by_year[k] = list(v)
    return articles_by_year
        

def extract_text_from_html(html, tags):
    """Extracts text from specified html tags.

    Args:
        html: A content of an article in html format.
        tags: A list of tags to extract content from.

    Returns:
        A list of chunks (e.g. paragraphs) of text extracted from the tags.
        
    Raises:
        TypeError: If html is not of type str and tags is not of type list.
    """
    if not isinstance(html, str):
        raise TypeError, 'parameter html={} not of <class "str">'
    if not isinstance(tags, list):
        raise TypeError, 'parameter tags={} not of <class "list">'
        
    segments = BeautifulSoup(html, "html.parser").find_all(tags)
    no_html = [seg.get_text() for seg in segments]
    return no_html


def get_text_segments(articles, tags):
    """Extracts segments of article text from Guardian articles in html format.

    Args:
        articles: A list of dicts containing article content.
        tags: A list of html tags to extract content from.

    Returns:
        A nested list of article segments.
    """
    article_html = [article["fields"]["body"] for article in articles]
    clean_segments = [extract_text_from_html(article, tags) for article in article_html]
    return clean_segments


def get_article_text(articles, tags):
    """Extracts full text of an article from Guardian articles in html format.

    Args:
        articles: A list of dicts containing article content.
        tags: A list of tags to extract content from.

    Returns:
        A list that contains full text of each article.
    """
    article_segments = get_text_segments(articles, tags)
    article_text = [" ".join(segment) for segment in article_segments]
    return article_text


def get_article_metadata(grouped_articles, fields_to_extract=["id"]):
    """Extracts useful article fields from raw data returned by the Guardian API.

    Args:
        grouped_articles: Articles grouped by year using itertools.
        fields_to_extract: A list of fields to extract data from. The default 
            is "id", other useful fields are "webUrl" and "webTitle"

    Returns:
        A pandas dataframe with content of article fields.
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
    """Extracts article text, deduplicates articles and organises them by year.

    Args:
        grouped_articles: A dict with articles mapped to years using itertools.
        tags: A list of html tags to extract content from.

    Returns:
        A pandas dataframe with article text, id and year.
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


def subset_articles(article_text, refine_terms, required_terms, text_field='text'):
    """Extracts a subset of articles that contain specific terms.
    
    It is useful for removing irrelevant articles in instances when a broad general 
    search term is used during data collection, e.g. `hydrogen`.
    
    Args:
        article_text: A pandas dataframe that contains article text and id.
        refine_terms: A list of terms that we use to disambiguate the general term.
        required_terms: A list of terms. An article must mention at least one of these terms.
        text_field: A string referring to the name of the column with text, the default is 'text'.

    Returns:
        A pandas dataframe with a subset of articles.
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


def remove_articles(article_text, term, text_field='text'):
    """Explicitely removes any articles that mention a particular term.

    Args:
        article_text: A dataframe that contains article text and id.
        term: A string referring to the term that we don't want to be mentioned in an article
        text_field: A string referring to the name of the column with text, the default is 'text'.

    Returns:
        A pandas dataframe with a subset of articles.

    """
    article_text = article_text[~article_text[text_field].str.contains(term)]
    return article_text  