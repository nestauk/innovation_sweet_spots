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
# # Collect Guardian articles, subset relevant documents and extract text
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html

# %% [markdown]
# ## 1. Import dependencies

# %%
import os
import pandas as pd
import csv
import json
from collections import defaultdict
import pickle
import itertools

# %%

# %%
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import (
    pd_data_collection_utils as dcu,
)
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ["p", "h2"]
CATEGORIES = [
    "Environment",
    "Guardian Sustainable Business",
    "Technology",
    "Science",
    "Business",
    "Money",
    "Cities",
    "Politics",
    "Opinion",
    "Global Cleantech 100",
    "The big energy debate",
    "UK news",
    "Life and style",
]

# %% [markdown]
# ## 2. Fetch relevant Guardian articles

# %%
search_terms = ["heat pump", "heat pumps"]

# %%
# For each search term download corresponding articles.
articles = [
    guardian.search_content(search_term, use_cached=True)
    for search_term in search_terms
]

# %%
# # Alternatively, load articles collected previously
# with open (os.path.join(DISC_OUTPUTS_DIR, 'articles_hp.json'), "r") as f:
#     articles = json.load(f)

# %%
# Combine results across set of search terms
aggregated_articles = list(itertools.chain(*articles))

# %%
# Only keep articles from specified sections
filtered_articles = dcu.filter_by_category(aggregated_articles, CATEGORIES)

# %% [markdown]
# ## 3. Extract metadata and text

# %%
# Extract article metadata
# This dict is later used for quick retrieval of original articles when reviewing results
metadata = dcu.get_article_metadata(
    filtered_articles, fields_to_extract=["webUrl", "webTitle", "webPublicationDate"]
)

# %%
# Extract article text
article_text = dcu.get_article_text_df(filtered_articles, TAGS)

# %%
# Save raw data from the Guardian API
# with open (os.path.join(DISC_OUTPUTS_DIR, 'articles_hp.json'), "w") as outfile:
#    json.dump(articles, outfile, indent = 4)

# %% [markdown]
# ### 3.1. Filtering out irrelevant articles

# %% [markdown]
# Sometimes we initially have to use a broad search term, such as hydrogen, to catch all the relevant documents. But we then end up with lots of results that we want to filter out. These are irrelevant either because they mention the search term in a context that we are not interested in (e.g. hydrogen in articles about space) or that relate to a geography that's outside of our scope (e.g. Australia).
#
# Below is an example of how to generate a subset of articles:
# * hydrogen (focus on energy and heating + UK specific)
# * heat pumps (UK specific)

# %%
# disambiguation_terms = ['home', 'homes', 'heating, cooling', 'hot water', 'electricity', 'boiler', 'boilers',
#                        'house', 'houses', 'building', 'buildings', 'radiators', 'low carbon', 'carbon emissions',
#                        'domestic', 'heating and cooling', 'heating fuel', 'heating systems', 'heating schemes',
#                        'hydrogen for heating', 'electric heating', 'electrification of heating', 'electrifying heating',
#                        'heating and lighting', 'district heating', 'clean burning hydrogen', 'hydrogen gas heating',
#                        'gas fired hydrogen', 'gas grid', 'climate targets', 'climate goals', 'households',
#                        'energy grid', 'energy grids', 'central heating', 'heating homes','net zero', 'net-zero',
#                        'appliances', 'hobs']
# required_terms = ['UK', 'Britain', 'Scotland', 'Wales', 'England', 'Northern Ireland', 'Britons', 'London']
# article_text = dcu.subset_articles(article_text, disambiguation_terms, required_terms)
# Exclude articles that mention hydrogen peroxide
# article_text = dcu.remove_articles(article_text, 'peroxide')

# %%
# Ensuring heat pump articles focus on UK
required_terms = [
    "UK",
    "Britain",
    "Scotland",
    "Wales",
    "England",
    "Northern Ireland",
    "Britons",
    "London",
]
article_text = dcu.subset_articles(article_text, required_terms, [])
article_text = dcu.remove_articles(article_text, "Australia")
# %%
# Persist processed outputs to disk

# Metadata for all articles
with open(os.path.join(DISC_OUTPUTS_DIR, "metadata_dict_hp.pkl"), "wb") as outfile:
    pickle.dump(metadata, outfile)

# Article text
article_text.to_csv(
    os.path.join(DISC_OUTPUTS_DIR, "article_text_hp.csv"),
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
