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
# # Collect and preprocess Guardian articles
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html
# - Generate sentence corpus
# - Analyse mentions over time

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

# %% [markdown]
# ## 2. Fetch relevant Guardian articles

# %%
search_terms = ['heat pump', 'heat pumps']
#search_terms =  ['hydrogen boiler', 'hydrogen boilers', 'hydrogen-ready boiler', 'hydrogen-ready boilers', 
#                'hydrogen ready boiler', 'hydrogen ready boilers',
#                 'hydrogen heating', 'hydrogen heat', 'hydrogen systems','hydrogen']

# %%
# For each search term download corresponding articles.
articles = [guardian.search_content(search_term, use_cached = True) for search_term in search_terms]

# %%
aggregated_articles = [article for sublist in articles for article in sublist]

# %%
filtered_articles = [a for a in aggregated_articles if a['sectionName'] in CATEGORIES]

# %%
# Group returned articles by year.
sorted_articles = sorted(filtered_articles, key = lambda x: x['webPublicationDate'][:4])

articles_by_year = collections.defaultdict(list)
for k,v in groupby(sorted_articles,key=lambda x:x['webPublicationDate'][:4]):
    articles_by_year[k] = list(v)
# %%
with open(os.path.join(DISC_OUTPUTS_DIR, 'articles_by_year_hp.pkl'), "wb") as outfile:
        pickle.dump(articles_by_year, outfile)

# %%
# Read in previously generated inputs
#with open(os.path.join(DISC_OUTPUTS_DIR, 'articles_by_year_hp.pkl'), "rb") as infile:
#        articles_by_year = pickle.load(infile)

# %% [markdown]
# ## 3. Preprocess articles

# %%
# Extract article metadata
metadata = disc.get_article_metadata(articles_by_year, 'year', fields_to_extract=['id', 'webUrl', 'webTitle'])

# %%
# Extract article text
article_text = disc.get_article_text_df(articles_by_year, TAGS, metadata)

# %%
required_terms = ['UK', 'Britain', 'Scotland', 'Wales', 'England', 'Northern Rreland', 'Britons', 'London']
article_text = disc.subset_articles(article_text, required_terms, [])
article_text = article_text[~article_text['text'].str.contains('Australia')]

# %%
# Hydrogen energy: we can further refine article selection when initial search includes broad terms
#disambiguation_terms = ['home', 'homes', 'heating, cooling', 'hot water', 'electricity', 'boiler', 'boilers', 
#                        'house', 'houses', 'building', 'buildings', 'radiators', 'low carbon', 'carbon emissions',
#                        'domestic', 'heating and cooling', 'heating fuel', 'heating systems', 'heating schemes',
#                        'hydrogen for heating', 'electric heating', 'electrification of heating', 'electrifying heating', 
#                        'heating and lighting', 'district heating', 'clean burning hydrogen', 'hydrogen gas heating',
#                        'gas fired hydrogen', 'gas grid', 'climate targets', 'climate goals', 'households',
#                        'energy grid', 'energy grids', 'central heating', 'heating homes','net zero', 'net-zero',
#                        'appliances', 'hobs']
#required_terms = ['UK', 'Britain', 'Scotland', 'Wales', 'England', 'Northern Rreland', 'Britons', 'London']
#article_text = disc.subset_articles(article_text, disambiguation_terms, required_terms)
# Exclude articles that mention hydrogen peroxide
#article_text = article_text[~article_text['text'].str.contains('peroxide')]
#article_text = article_text[~article_text['text'].str.contains('hydrogen bomb')]
#article_text = article_text[~article_text['text'].str.contains('Australia')]
#article_text = article_text[~article_text['text'].str.contains('Australia')]

# %%
# Identify and filter out articles related to hydrogen applications in transport
#transport_terms = ['car', 'cars', 'cab', 'cabs', 'taxi', 'vehicle', 'vehicles', 'bus', 'buses', 'train', 'trains',
#                   'fleet', 'truck', 'trucks', 'van', 'vans', 'aircraft', 'flight', 'flights', 'plane', 'planes', 
#                   'boat', 'boats', 'ship', 'transport system']
#transport_related = disc.subset_articles(article_text, transport_terms, [])

# %%
#article_text = article_text[~article_text.index.isin(transport_related.index)]

# %%
# Generate sentence corpus, spacy corpus of articles and sentence records (include original article id).
# Cleaning is minimal (normalising, but keeping punctuation, retaining stopwords, basic lemmatisation).
# Current performance is ~ 2.5 min per 1K articles. Results on rroad topics may contain tens of thousands of articles.
# In such cases first use a sample of articles.

sentences_by_year, processed_articles_by_year, sentence_records = disc.get_sentence_corpus(article_text, 
                                                                                           nlp,
                                                                                          'year',
                                                                                          'text',
                                                                                          'id')

# %%
sentence_record_dict = {elem[0]: elem[1] for elem in sentence_records}

# %%
metadata_dict = collections.defaultdict(dict)
for ix, row in article_text.iterrows():
    metadata_dict[row['id']]['url'] = row['webUrl'] 
    metadata_dict[row['id']]['title'] = row['webTitle']

# %%
# Use spacy functionality to identify noun phrases in all sentences in the corpus.
# These often provide a useful starting point for analysing language used around a given technology.
noun_chunks_all_years = {str(year): disc.get_noun_chunks(processed_articles, remove_det_articles = True) for\
                        year, processed_articles in processed_articles_by_year.items()}

# %%
# Persist processed outputs to disk

# Metadata for all articles
#metadata.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_metadata_hydrogen.csv'), index = False)

article_text.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_hp.csv'), index = False, quoting = csv.QUOTE_NONNUMERIC)


with open(os.path.join(DISC_OUTPUTS_DIR, 'sentences_by_year_hp.pkl'), "wb") as outfile:
        pickle.dump(sentences_by_year, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'processed_articles_by_year_hp.pkl'), "wb") as outfile:
    pickle.dump(processed_articles_by_year, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_records_hp.pkl'), "wb") as outfile:
        pickle.dump(sentence_records, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_record_dict_hp.pkl'), "wb") as outfile:
        pickle.dump(sentence_record_dict, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'metadata_dict_hp.pkl'), "wb") as outfile:
        pickle.dump(metadata_dict, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'noun_chunks_hp.pkl'), "wb") as outfile:
        pickle.dump(noun_chunks_all_years, outfile)

# %% [markdown]
# ## 4. Analyse frequency of mentions over time

# %%
# Dataframe with sentences that contain search terms.
term_sentences = disc.combine_flat_sentence_mentions(search_terms, sentence_records)

# %%
# Number of mentions (calculated as number of sentences with mentions).
term_mentions = disc.collate_mentions(search_terms, term_sentences)
term_mentions.append(disc.total_docs(article_text, 'year')) #replace with sample_articles is using a sample

# %%
# Combined data frame with number of mentions across all terms and total number of articles.
mentions_all = pd.DataFrame.from_records(term_mentions)
mentions_all = mentions_all.T
mentions_all.columns = search_terms + ['total_documents']
mentions_all.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'mentions_df_hp.csv'))

# %%
combined_term_sentences = disc.combine_term_sentences(term_sentences, search_terms)

# %%
with open(os.path.join(DISC_OUTPUTS_DIR, 'term_sentences_hp.pkl'), "wb") as outfile:
        pickle.dump(term_sentences, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'combined_sentences_hp.pkl'), "wb") as outfile:
        pickle.dump(combined_term_sentences, outfile)

# %%
mentions_all

# %%
