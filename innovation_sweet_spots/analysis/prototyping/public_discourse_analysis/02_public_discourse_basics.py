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
# # Basic functionality for public discourse data analysis
#
# - Fetching Guardian articles with certain keywords
# - Reviewing article categories
# - Cleaning and preprocessing article text
# - Calculating sentence sentiment
# - Identifying other key terms and associated sentiment

# %% [markdown]
# ## 1. Import dependencies

# %%
# cd '/Users/jdjumalieva/Documents/Analysis/innovation_sweet_spots/'

# %%
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.analysis import analysis_utils as iss
from innovation_sweet_spots.analysis import discourse_utils as disc
from innovation_sweet_spots.utils import text_cleaning_utils as tcu
from innovation_sweet_spots.utils import text_pre_processing as tpu
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import collections
import pickle
import os
from itertools import groupby
# %%
TAGS = ['p', 'h2']
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ## 2. Fetching relevant Guardian articles

# %%
search_terms = ['heat pumps']
search_term = search_terms[0]

articles = guardian.search_content(search_term)

# %%
# group returned articles by year
sorted_articles = sorted(articles, key = lambda x: x['webPublicationDate'][:4])

articles_by_year = collections.defaultdict(list)
for k,v in groupby(sorted_articles,key=lambda x:x['webPublicationDate'][:4]):
    articles_by_year[k] = list(v)

# %% [markdown]
# ### 2.1 Mentions in the news

# %%
disc.show_num_articles(disc.get_num_articles(articles_by_year))

# %% [markdown]
# ### 2.2 Review most common article categories for each year

# %%
top10_across_years = disc.get_top_n_categories(articles_by_year)
print(f"The top 10 categories of articles in 2019 were:\n{top10_across_years['2019']}")


# %% [markdown]
# ## 3. Cleaning and pre-processing article text

# %%
# 3.1 Extracting content from specified html tags to avoid irrelevant in-text links 
# to other articles

article_segments = disc.get_text_segments(articles, TAGS)
article_text = [' '.join(segment) for segment in article_segments]

# %%
# 3.2 Cleaning content

# This involves minimal cleaning (keeping punctuation and stopwords, no lemmatisation)
clean_article_text = [tcu.clean_text_minimal(article) for article in article_text]

# %%
# 3.3 Extracting sentences
# nlp = spacy.load("en_core_web_sm")

processed_articles = [nlp(article) for article in clean_article_text]

# %%
# To do: persist article sentences, as they form the corpus for further analysis
article_sentences = [[sent.text for sent in article.sents] for article in processed_articles]

with open(os.path.join(DISC_OUTPUTS_DIR, 'article_sentences.pkl'), "wb") as outfile:
        pickle.dump(article_sentences, outfile)

flat_article_sentences = [item for sublist in article_sentences for item in sublist]
flat_article_sentences = list(set(flat_article_sentences))
# %%
# Relevant articles
sentence_mentions = [[sent for sent in art if search_term in sent] for art in article_sentences]
flat_sentence_mentions = [item for sublist in sentence_mentions for item in sublist]
# %%
# 3.4 Extract noun chunks related to search term
noun_chunks = []
for article in processed_articles:
    for chunk in article.noun_chunks:
        noun_chunks.append(chunk)
        # print(chunk)

noun_chunks_str = [str(elem) for elem in noun_chunks]
dedup_noun_chunks = list(set(noun_chunks_str))

# %%
# 3.5 Tokenize and extract ngrams
# Filter out stopwords, punctuation, etc. and certain entities
# Use sentence as a unit of analysis
tokenised = [tpu.process_text_disc(doc) for doc in nlp.pipe(flat_article_sentences)]

# %%
# Extract ngrams
count_model = CountVectorizer(tokenizer = disc.identity_tokenizer,
                              lowercase = False, 
                              ngram_range=(1,3),
                              min_df = 5) # default unigram model
X = count_model.fit_transform(tokenised)
Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
Xc.setdiag(0)

# %%
vocab = count_model.vocabulary_
names = count_model.get_feature_names()
count_list = X.toarray().sum(axis=0)
count_dict = dict(zip(names,count_list))

# %%
# Calculate PPMI
# ngrams = np.sum(Xc.todense())/2
search_index = names.index(search_term)

# %%
# Revisited calculation of PMI defining context as a sentence
pmis = {}
for ix, name in enumerate(names):
    association = disc.pmi(np.sum(X[:, search_index]),
                  np.sum(X[:, ix]),
                  Xc[search_index, ix],
                  len(flat_article_sentences),
                  len(flat_article_sentences))
    pmis[name] = association
    
pruned_pmis = {k:v for k,v in pmis.items() if v >0}

# %%
# 3.6 Identiy noun chunk associations
# Remove determiners in noun chunks to increase number of matches
dedup_noun_chunks = [elem.replace(' a ', ' ').replace(' the ',' ') for elem in dedup_noun_chunks]
dedup_noun_chunks = [elem.strip() for elem in dedup_noun_chunks]


pruned_noun_chunks = [elem for elem in dedup_noun_chunks if elem in names]
pruned_noun_chunks = [elem for elem in pruned_noun_chunks if count_dict.get(elem, 0) > 3]


chunk_pmi = {chunk: pruned_pmis.get(chunk, 0) for chunk in pruned_noun_chunks}
chunk_pmis = {k:v for k,v in chunk_pmi.items() if v >0}

# Below need to remove noun chunks with any terms from the search term 
# contains_search_term = []
# for noun_chunk in pruned_noun_chunks:
#     for substring in search_term.split():
#         if substring in noun_chunk:
#             contains_search_term.append(noun_chunk)

# %%
# Study sentiment around noun chunks in sentences
sentence_sentiment = iss.get_sentence_sentiment(flat_article_sentences)

# %%
# Link noun chunks with high association to sentence sentiment

noun_chunk_sentiments = collections.defaultdict(list)
for ix, row in sentence_sentiment.iterrows():
    for noun_chunk in chunk_pmis:
        # print(noun_chunk)
        if noun_chunk in row['sentences']:
            noun_chunk_sentiments[noun_chunk].append(row['compound'])

# %%
# Calculate average sentiment for noun chunks
noun_chunk_agg_sent = {k: np.mean(v) for k,v in noun_chunk_sentiments.items()}

sorted_sent = sorted(noun_chunk_agg_sent.items(), 
                     key = lambda x: x[1], 
                     reverse = True)

# %%
prop_sent = {k: disc.prop_pos(v) for k,v in \
             noun_chunk_sentiments.items()}

