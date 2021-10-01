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
# # Collect and preprocess Guardian articles
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html

# %% [markdown]
# ## 1. Import dependencies

# %%
import os
import pandas as pd
import csv

# %%
# Change first element to location of project folder.
os.chdir(os.path.join('/Users/jdjumalieva/Documents/Analysis/', 'innovation_sweet_spots'))

# %%
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import pd_data_collection_utils as dcu
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ['p', 'h2']
CATEGORIES = ['Environment', 'Guardian Sustainable Business', 'Technology', 'Science', 'Business', 
               'Money', 'Cities', 'Politics', 'Opinion', 'Global Cleantech 100', 'The big energy debate',
              'UK news', 'Life and style']

# %% [markdown]
# ## 2. Fetch relevant Guardian articles

# %%
search_terms = ['heat pump', 'heat pumps']

# %%
# For each search term download corresponding articles.
articles = [guardian.search_content(search_term, use_cached = True) for search_term in search_terms]

# %%
# Combine results across set of search terms
aggregated_articles = dcu.combine_articles(articles)

# %%
# Only keep articles from specified sections
filtered_articles = dcu.filter_by_category(aggregated_articles, CATEGORIES)

# %%
articles_by_year = dcu.sort_by_year(filtered_articles)

# %% [markdown]
# ## 3. Extract metadata and text

# %%
# Extract article metadata
metadata = dcu.get_article_metadata(articles_by_year, fields_to_extract=['id', 'webUrl', 'webTitle', 
                                                                                  'webPublicationDate'])

# %%
# Extract article text
article_text = dcu.get_article_text_df(articles_by_year, TAGS)

# %%
article_text.head()

# %%
# Persist processed outputs to disk

# Metadata for all articles
metadata.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_metadata_hp.csv'), index = False)

# Article text
article_text.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_hp.csv'), 
                    index = False, 
                    quoting = csv.QUOTE_NONNUMERIC)


# %%
