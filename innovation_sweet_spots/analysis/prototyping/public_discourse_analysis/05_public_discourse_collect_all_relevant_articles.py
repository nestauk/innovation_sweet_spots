#!/usr/bin/env python3
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

# %% [markdown]
# # Collect and preprocess all relevant Guardian articles
#
# - Identify relevant tags
# - Fetch articles using Guardian API
# - Filter out articles that don't refer to the UK or any UK nation; deduplicate
# - Extract metadata and text from html
# - Save for further analysis

# %% [markdown]
# ## 1. Import dependencies

# %%
import spacy
import collections
import pickle
import os
from itertools import groupby
import pandas as pd
import csv
from urllib.parse import urlencode, quote
import requests
import json
import time

# %%
# Change first element to location of project folder.
os.chdir(os.path.join('/Users/jdjumalieva/Documents/Analysis/', 'innovation_sweet_spots'))

# %%
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.analysis import analysis_utils as iss
from innovation_sweet_spots.analysis import discourse_utils as disc
from innovation_sweet_spots.utils import text_cleaning_utils as tcu
from innovation_sweet_spots.utils import text_pre_processing as tpu
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ['p', 'h2']
CATEGORIES = ['Environment', 'Guardian Sustainable Business', 'Technology', 'Science', 'Business', 
               'Money', 'Cities', 'Politics', 'Opinion', 'Global Cleantech 100', 'The big energy debate',
              'UK news', 'Life and style']

# %%
nlp = spacy.load("en_core_web_sm")

# %%
BASE_URL = "http://content.guardianapis.com/"
API_KEY = open(os.environ["GUARDIAN_API_KEY"], "r").read()


def create_url_tag(
    search_term: str, api_key: str = API_KEY, adjusted_parameters: dict = {}, 
):
    """NB: Use double quotes for the search term"""
    parameters = config["guardian_api"].copy()
    parameters["api-key"] = api_key
    for key in adjusted_parameters:
        parameters[key] = adjusted_parameters[key]
    search_query = f'tag={search_term}&'
    url = f"{BASE_URL}search?" + search_query + urlencode(parameters)
    return url


# %%
def get_request(url):
    """Return response"""
    r = requests.get(url)
    time.sleep(0.25)
    return r


# %%
def search_tags(
    search_term: str,
    api_key: str = API_KEY,
    use_cached: bool = False,
    save_to_cache: bool = False,
#    fpath=API_RESULTS_DIR,
    only_first_page: bool = False,
):
    # Check if we have already made such a search
    print(search_term)
    if use_cached:
        results_list = get_content_from_cache(search_term, fpath)
        if results_list is not False:
            return results_list

    # Do a new search
    url = create_url_tag(search_term, api_key)
    r = get_request(url)
    if r.status_code != 200:
        return r
    else:
        response = r.json()["response"]
        n_total_results = response["total"]
        if n_total_results > 0:
            n_pages_total = response["pages"]
            current_page = response["currentPage"]  # should always = 1
            results_list = [response["results"]]
            # Get results from all pages
            if (n_pages_total > 1) and (not only_first_page):
                while current_page < n_pages_total:
                    # Update url and call again
                    current_page += 1
                    url = create_url_tag(search_term, api_key, {"page": current_page})
                    r = get_request(url)
                    if r.status_code == 200:
                        results_list.append(r.json()["response"]["results"])
            results_list = [r for page_results in results_list for r in page_results]
            percentage_collected = round(len(results_list) / n_total_results * 100)
            # Save the results
            logging.info(f"Collected {len(results_list)} ({percentage_collected}%) results")
            if save_to_cache:
                save_content_to_cache(search_term, results_list, fpath)
            return results_list
        else:
            logging.info(f"Search for {search_term} returned no results")
            return []



# %%
#config["guardian_api"].copy()

# %% [markdown]
# ## 2. Fetch relevant Guardian articles

# %%
search_terms =  ['hydrogen boiler', 'hydrogen boilers', 'hydrogen-ready boiler', 'hydrogen-ready boilers', 
                 'hydrogen ready boiler', 'hydrogen ready boilers',
                 'hydrogen heating', 'hydrogen heat', 'hydrogen systems','hydrogen']

# %%
curated_list = ['environment/environment',
                'environment/energy',
                'environment/climate-crisis',
                'environment/carbon-emissions',
                'environment/renewableenergy',
                'environment/ethical-living',
                'environment/green-politics',
                'environment/energyefficiency',
                'environment/energy-storage',
                'environment/greenbuilding',
                'environment/green-economy',
                'environment/carbonfootprints',
                'big-energy-debate/big-energy-debate',
                'sustainable-business/cleantech',
                'technology/energy'
]

# %%
# Retrieve all articles with a given tag
tagged_articles = [search_tags(tag) for tag in curated_list]

# %%
aggregated_tag_articles = [article for sublist in tagged_articles for article in sublist]

# %%
filtered_t_articles = [a for a in aggregated_tag_articles if a['sectionName'] in CATEGORIES]

# %%
len(aggregated_tag_articles) #total number of articles

# %%
# Group returned articles by year.
sorted_t_articles = sorted(filtered_t_articles, key = lambda x: x['webPublicationDate'][:4])

articles_by_year_t = collections.defaultdict(list)
for k,v in groupby(sorted_t_articles,key=lambda x:x['webPublicationDate'][:4]):
    articles_by_year_t[k] = list(v)

# %%
len(sorted_t_articles) #articles within defined categories

# %%
metadata_t = disc.get_article_metadata(articles_by_year_t, 'year', fields_to_extract=['id', 'webUrl', 'webTitle', 'sectionName'])

# %%
metadata_t.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_metadata_all_environment.csv'), index = False)


# %%
len(metadata_t) #after initial deduplication

# %%
article_text_t = disc.get_article_text_df(articles_by_year_t, TAGS, metadata_t)

# %%
location_filter = ['UK', 'Britain', 'Scotland', 'Wales', 'England', 'Northern Rreland', 'Britons', 'London']
article_text_t = disc.subset_articles(article_text_t, location_filter, [])
article_text_t = article_text_t[~article_text_t['text'].str.contains('Australia')]

# %%
article_text_t.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_all_environment.csv'), index = False, quoting = csv.QUOTE_NONNUMERIC)

# %%
len(article_text_t)

# %%
# Number of articles per year
for year, group in article_text_t.groupby('year'):
    print(len(group))

# %%
env_articles = article_text_t.groupby('year')['text'].count()

# %%
env_articles.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'mentions_all_env.csv'))
