#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:36:41 2021

@author: jdjumalieva
"""
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd


def filter_by_category(aggregated_articles, category_list, field="sectionName"):
    """Returns a subset of articles that fall under specified category.

    Args:
        aggregated_articles: A list of articles (each a JSON nested dict).
        field: A string that specifies the field containing information on article
            category. Default value is 'sectionName'.
        category_list: A list of category names.

    Returns:
        A list of filtered articles.

    Raises:
        TypeError: If incorrect data type was passed in args.
    """
    if not isinstance(aggregated_articles, list):
        raise TypeError("parameter aggregated_articles should be a list")
    if not isinstance(category_list, list):
        raise TypeError("parameter category_list should be a list")
    if not isinstance(field, str):
        raise TypeError("parameter field should be a string")

    if len(category_list) > 0:
        return [a for a in aggregated_articles if a[field] in category_list]
    else:
        return aggregated_articles


def extract_text_from_html(html, tags):
    """Extracts text from specified html tags.

    Args:
        html: A content of an article in html format.
        tags: A list of tags to extract content from.

    Returns:
        A list of chunks (e.g. paragraphs) of text extracted from the tags.

    Raises:
        TypeError: If incorrect data type was passed in args.
    """
    if not isinstance(html, str):
        raise TypeError("parameter html should be a string")
    if not isinstance(tags, list):
        raise TypeError("parameter tags should be a list")

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


def get_article_metadata(filtered_articles, fields_to_extract, id_field="id"):
    """Extracts useful article fields from raw data returned by the Guardian API.

    Args:
        filtered_articles: A list of raw articles.
        fields_to_extract: A list of fields to extract data from. The useful fields
        are "webUrl", "webTitle" and "webPublicationDate"
        id_field: A string that referrs to field containing article ID.

    Returns:
        A dict mapping article IDs to other useful fields.
    """
    metadata_dict = defaultdict(dict)
    for article in filtered_articles:
        article_id = article[id_field]
        for field in fields_to_extract:
            article_field = article[field]
            metadata_dict[article_id][field] = article_field
    return metadata_dict


def get_article_text_df(filtered_articles, tags):
    """Extracts article text, deduplicates articles and organises them by year.

    Args:
        grouped_articles: A list of raw articles.
        tags: A list of html tags to extract content from.

    Returns:
        A pandas dataframe with article text, id and year.
    """
    article_text = get_article_text(filtered_articles, tags)
    article_id = [article["id"] for article in filtered_articles]
    article_df = pd.DataFrame(
        {"id": article_id, "text": article_text}
    ).drop_duplicates()
    return article_df


def subset_articles(article_text, filter_1, filter_2, text_field="text"):
    """Extracts a subset of articles that contain specific terms.

    It is useful for removing irrelevant articles in instances when a broad general
    search term is used during data collection, e.g. `hydrogen`. In this example,
    the filter_1 terms were used to generate a subset of articles related to heating.
    The filter_2 terms were then used to select articles relevant to the UK from this subset.

    Args:
        article_text: A pandas dataframe that contains article text and id.
        filter_1: A list of terms that we use for the first round of filtering.
        filter_2: A list of terms that we use for subsequent filtering.
        text_field: A string referring to the name of the column with text, the default is 'text'.

    Returns:
        A pandas dataframe with a subset of articles.
    """
    # we first generate a 'thematic' subset of articles that are relevant
    # e.g. applications of hydrogen to heating
    base = r"{}"
    expr = "(?:\s|^){}(?:,?\s|\.|$)"
    combined_expressions = [
        base.format("".join(expr.format(term))) for term in filter_1
    ]
    joined_expressions = "|".join(combined_expressions)
    subset_df = article_text[article_text[text_field].str.contains(joined_expressions)]
    deduplicated_df = subset_df.drop_duplicates()

    # required_terms tend to be geographic areas that we are studying
    # we select only articles that mention those areas from a 'thematic' set
    # defined above
    if len(filter_2) > 0:
        for term in filter_2:
            combined_terms = "|".join(term)
        filtered_subset_df = deduplicated_df[
            deduplicated_df[text_field].str.contains(combined_terms)
        ]
        deduplicated_df = filtered_subset_df.drop_duplicates()
    return deduplicated_df


def remove_articles(article_text, term, text_field="text"):
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
