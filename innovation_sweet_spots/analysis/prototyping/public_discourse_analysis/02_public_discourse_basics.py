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
import collections
import pickle
import os
from itertools import groupby
import pandas as pd
# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ['p', 'h2']

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
# %%
# generate mapping of articles to IDs and original urls
id_map = collections.defaultdict(dict)
for year, articles in articles_by_year.items():
    for ix, article in enumerate(articles):
        record_id = (ix, article['id'], article['webUrl'])
        id_map[year][ix] = record_id

# %% [markdown]
# #### 2.1 Mentions in the news

# %%
disc.show_num_articles(disc.get_num_articles(articles_by_year))

# %% [markdown]
# #### 2.2 Review most common article categories for each year

# %%
top10_across_years = disc.get_top_n_categories(articles_by_year)
print(f"The top 10 categories of articles in 2019 were:\n{top10_across_years['2019']}")


# %% [markdown]
# ## 3. Preparing articles for further analysis

# %%
# Extract text from html

article_text_by_year = {y: disc.get_article_text(v, TAGS) for y,v in articles_by_year.items()}

combined = collections.defaultdict(dict)
for year, articles in article_text_by_year.items():
    combined_record = list(zip(article_text_by_year[year], id_map[year].values()))
    combined[year] = combined_record


# %%
# Generate clean corpus of sentences.
# nlp = spacy.load("en_core_web_sm")


# Cleaning is minimal (keeping punctuation and stopwords, no lemmatisation)
sentences_by_year = collections.defaultdict(dict)
processed_articles_by_year = collections.defaultdict(dict) # spacy corpus of articles
for year, articles in combined.items():
    sentences, processed_articles = disc.generate_sentence_corpus\
        ([art[0] for art in combined[year]], nlp)
    sentences_with_record = list(zip(sentences, [art[1] for art in articles]))
    sentences_by_year[year] = sentences_with_record
    processed_articles_by_year[year] = processed_articles

# Persist sentence corpus to disk
with open(os.path.join(DISC_OUTPUTS_DIR, 'sentences_by_year.pkl'), "wb") as outfile:
        pickle.dump(sentences_by_year, outfile)

# %% [markdown]
# ## 4. Evaluating sentiment around search terms for a given year
# %%
year_flat_sentences = collections.defaultdict(list)
year_sentence_records = collections.defaultdict(list)
for year in articles_by_year:
    year_articles = [elem[0] for elem in sentences_by_year[year]]
    year_article_records = {elem[1][0]: (elem[1][1], elem[1][2]) for elem in sentences_by_year[year]}
    
    sentences_with_term = [[sent for sent in art if search_term in sent] for art in year_articles]
    
    flat_sentences_with_ix = []
    for ix, sentences in enumerate(sentences_with_term):
        for sentence in sentences:
            flat_sentences_with_ix.append((ix, sentence))
        
    
    flat_sentences = [item[1] for item in flat_sentences_with_ix]
    sentence_records = [year_article_records[elem[0]] for elem in flat_sentences_with_ix]
    
    year_flat_sentences[year] = flat_sentences
    year_sentence_records[year] = sentence_records
# %%
# Sentiment for different types of context

# Whole sentence
sentiments_all_years = []
for year in year_flat_sentences:
    # Sentence
    sentiments = disc.calculate_sentiment(year_flat_sentences[year],
                                                   search_term, 
                                                   context = 'sentence')
    
    # # Window of n words to the left and right of the search term
    # sentiments = disc.calculate_sentiment(year_flat_sentences[year],
    #                                            search_term, 
    #                                            context = 'n_words', 
    #                                            n_words = 5)
    
    # # Phrase containing search term (using dependency subtrees)
    # sentiments = disc.calculate_sentiment(year_flat_sentences[year],
    #                                            search_term, 
    #                                            context = 'phrase', 
    #                                            nlp)
    
    sentiments['url'] = [elem[1] for elem in year_sentence_records[year]] # can add guardian ID too
    sentiments['year'] = year
    sentiments_all_years.append(sentiments)

sentiments_across_years_df = pd.concat(sentiments_all_years)

# %%
sentiments_across_years_df.head()

# %%
# Produce aggregate sentiment stats over years
agg_sentiments = sentiments_across_years_df.groupby('year').agg({'neg': 'mean', 
                                                                 'neu': 'mean', 
                                                                 'pos': 'mean', 
                                                                 'compound': 'mean'})

# %%
agg_sentiments

# %% [markdown]
# ## 5. Identifying most relevant terms
# %%
# Identifying terms that often co-occur with search term using PMI

# First identify noun chunks
year_noun_chunks = collections.defaultdict(list)
for year in articles_by_year:
    year_articles = [elem[0] for elem in sentences_by_year[year]]
    year_sentences = [sent for art in year_articles for sent in art]
    noun_chunks = disc.get_noun_chunks(processed_articles_by_year[year])
    year_noun_chunks[year] = noun_chunks

# %%
# Persist noun chunks to disk
with open(os.path.join(DISC_OUTPUTS_DIR, 'year_noun_chunks.pkl'), "wb") as outfile:
        pickle.dump(year_noun_chunks, outfile)

# %%
related_terms = collections.defaultdict(list)
normalised_ranks = collections.defaultdict(list)


for year in articles_by_year:
    year_articles = [elem[0] for elem in sentences_by_year[year]]
    year_sentences = [sent for art in year_articles for sent in art]
    noun_chunks = year_noun_chunks[year]
    
    # Then get terms with positive PMI
    tokenised_sentences = disc.get_spacy_tokens(year_sentences, nlp)

    cooccurrence_matrix, doc_term_matrix, token_names, token_counts = disc.get_ngrams(\
                                                    tokenised_sentences,
                                                    token_range = (1,3), 
                                                    min_mentions = 3)

    # Calculate PMI defining sentence as a context
    pmis = disc.calculate_positive_pmi(cooccurrence_matrix, 
                                              doc_term_matrix, 
                                              token_names, 
                                              token_counts,
                                              search_term)

    # Identify most relevant noun phrases
    key_related_terms = disc.get_related_terms(noun_chunks, 
                                        pmis, 
                                        token_names, 
                                        token_counts, 
                                        min_mentions = 3)

    # Add normalised cooccurrence rank
    normalised_rank = disc.get_normalised_rank(cooccurrence_matrix, token_names, 
                                                 token_counts, search_term, 
                                                 threshold =3)

    # sorted(normalised_ranks.items(), key = lambda x: x[1])[:20]
    # Add comparison of standard deviation of normalised ranks over time
    
    related_terms[year] = list(key_related_terms.items())
    normalised_ranks[year] = list(normalised_rank.items())

# %%
# Write to disk
with open(os.path.join(DISC_OUTPUTS_DIR, 'related_terms.pkl'), "wb") as outfile:
        pickle.dump(related_terms, outfile)
        
with open(os.path.join(DISC_OUTPUTS_DIR, 'normalised_ranks.pkl'), "wb") as outfile:
        pickle.dump(normalised_ranks, outfile)     

# %%
# View top 50 terms with the highest pointwise mutual information in the most recent year
sorted(key_related_terms.items(), key = lambda x: x[1], reverse = True)[:50]

# %%
# View top 50 terms with the highest normalised rank in the most recent year
list(normalised_rank.items())[:50]

# %% [markdown]
# ## Appendix

# %%
# Illustration of subtree phrase extraction
sample_sentence = "switching heating from gas to electricity would mean relying on heat pumps."
print(disc.get_phrase(sample_sentence, 'heat pumps', nlp_model = None))

# %%
sample_sentence = "heating remains a major source of carbon and the government plans to open a grant scheme from april 2022 to help households and small businesses invest in heat pumps and biomass boilers, backed by 100m of exchequer funding."
print(disc.get_phrase(sample_sentence, 'heat pumps', nlp_model = None))

# %%
from spacy import displacy

# %%
nlp = spacy.load("en_core_web_sm")

# %%
displacy.serve(nlp(sample_sentence), style="dep")

