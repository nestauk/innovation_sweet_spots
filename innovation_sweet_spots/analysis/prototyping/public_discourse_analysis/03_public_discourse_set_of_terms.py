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
# # Extending functionality to perform analysis on a set of terms
#
# - Perform analysis on individual terms
# - Aggregate results

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
import csv
# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ['p', 'h2']

# %%
nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# ## 2. Fetching relevant Guardian articles

# %%
search_terms = ['heat pump', 'heat pumps']
#["heat", "thermal", "temperature", "waste heat", "storage", "heating", "electricity", "cooling"]

# %%
articles = [guardian.search_content(search_term) for search_term in search_terms]

# %%
aggregated_articles = [article for sublist in articles for article in sublist]

# %%
# group returned articles by year
sorted_articles = sorted(aggregated_articles, key = lambda x: x['webPublicationDate'][:4])

articles_by_year = collections.defaultdict(list)
for k,v in groupby(sorted_articles,key=lambda x:x['webPublicationDate'][:4]):
    articles_by_year[k] = list(v)
# %%
with open(os.path.join(DISC_OUTPUTS_DIR, 'articles_by_year_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(articles_by_year, outfile)

# %% [markdown]
# #### Review most common article categories for each year

# %%
top10_across_years = disc.get_top_n_categories(articles_by_year)
print(f"The top 10 categories of articles in 2020 were:\n {top10_across_years['2020']}")


# %% [markdown]
# ## 3. Preprocess articles

# %%
# Extract article metadata
metadata = disc.get_article_metadata(articles_by_year, fields_to_extract=['id', 'webUrl'])

# %%
# Extract article text
article_text = disc.get_article_text_df(articles_by_year, TAGS, metadata)

# %%
# Read in outputs generated earlier
#with open(os.path.join(DISC_OUTPUTS_DIR, 'articles_by_year_heat_pumps.pkl'), "rb") as infile:
#        articles_by_year = pickle.load(infile)

# %%
#article_text = pd.read_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text.csv'))

# %%
#sample_articles = article_text.sample(n = 10000, random_state = 808) #if working with a large dataset

# %%
#Extract sentences, spacy corpus of articles and sentence records (include original article id)
# Cleaning is minimal (keeping punctuation and stopwords, basic lemmatisation)
# Current performance is about 2.5 min per 1K articles, so using a sample for illustrative purposes (for broad topics). 
# There are 95K articles in total for broad heating topic.

sentences_by_year, processed_articles_by_year, sentence_records = disc.get_sentence_corpus(article_text, nlp)

# %%
# Persist processed outputs to disk

# Metadata for all articles
metadata.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_metadata_heat_pumps.csv'), index = False)

article_text.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_heat_pumps.csv'), index = False, quoting = csv.QUOTE_NONNUMERIC)


with open(os.path.join(DISC_OUTPUTS_DIR, 'sentences_by_year_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(sentences_by_year, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'processed_articles_by_year_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(processed_articles_by_year, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_records_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(sentence_records, outfile)        

# %%
# Read in outputs
#with open(os.path.join(DISC_OUTPUTS_DIR, 'sentences_by_year_heat_pumps.pkl'), "rb") as infile:
#        sentences_by_year = pickle.load(infile)

#with open(os.path.join(DISC_OUTPUTS_DIR, 'processed_articles_by_year_heat_pumps.pkl'), "rb") as infile:
#        processed_articles_by_year = pickle.load(infile)

#with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_records_heat_pumps.pkl'), "rb") as infile:
#        sentence_records = pickle.load(infile) 

# %% [markdown]
# ## 4. Analyse mentions of term in the news

# %%
# Dataframe with sentences that contain search terms
term_sentences = disc.combine_flat_sentence_mentions(search_terms, sentence_records)

# %%
# Number of mentions (calculated as number of sentences with mentions)
term_mentions = disc.collate_mentions(search_terms, term_sentences)
term_mentions.append(disc.total_docs(article_text)) #replace with sample_articles is using a sample

# %%
# Combined data frame with number of mentions across all terms and total number of articles
mentions_all = pd.DataFrame.from_records(term_mentions)
mentions_all = mentions_all.T
mentions_all.columns = search_terms + ['total_articles']

# %%
mentions_all

# %%
combined_term_sentences = disc.combine_term_sentences(term_sentences, search_terms)

# %% [markdown]
# ## 5. Identifying most relevant terms
# %%
noun_chunks_all_years = {str(year): disc.get_noun_chunks(processed_articles, remove_det_articles = True) for\
                        year, processed_articles in processed_articles_by_year.items()}

# %%
# Persist noun chunks to disk
#with open(os.path.join(DISC_OUTPUTS_DIR, 'noun_chunks_all_years_hp.pkl'), "wb") as outfile:
#        pickle.dump(noun_chunks_all_years, outfile)

# %%
# Read in previously identified noun chunks
#with open(os.path.join(DISC_OUTPUTS_DIR, 'noun_chunks_all_years_hp.pkl'), "rb") as infile:
#        noun_chunks_all_years = pickle.load(infile)

# %%
related_terms = collections.defaultdict(dict)
normalised_ranks = collections.defaultdict(dict)

for year in sentences_by_year:
    print(year)
    year_articles = [elem for elem in sentences_by_year[year]] # nested list of sentences within each article
    year_sentences = [sent for art in year_articles for sent in art]
    noun_chunks = noun_chunks_all_years[str(year)]
    for term in search_terms:
        #print(term)
        key_terms, normalised_rank = disc.get_key_terms(term, year_sentences, nlp, noun_chunks,
                                                            mentions_threshold = 3, token_range = (1,3))

        related_terms[year][term] = list(key_terms.items())
        normalised_ranks[year][term] = list(normalised_rank.items())

# %%
# Write to disk
with open(os.path.join(DISC_OUTPUTS_DIR, 'related_terms_hp.pkl'), "wb") as outfile:
        pickle.dump(related_terms, outfile)
        
with open(os.path.join(DISC_OUTPUTS_DIR, 'normalised_ranks_hp.pkl'), "wb") as outfile:
        pickle.dump(normalised_ranks, outfile)     

# %% [markdown]
# ### Combine related terms and normalised ranks

# %%
combined_pmi = disc.combine_pmi(related_terms, search_terms)

# %%
combined_ranks = disc.combine_ranks(normalised_ranks, search_terms)

# %%
combined_pmi_dict = collections.defaultdict(dict)
for year in combined_pmi:
    for term in combined_pmi[year]:
        combined_pmi_dict[year][term[0]] = term[1]

# %%
# Dictionary: for each year return frequency of mentions, normalised rank and pmi for a given term.
pmi_inters_ranks = collections.defaultdict(dict)
for year in combined_ranks:
    for term in combined_ranks[year]:
        if term[0] in combined_pmi_dict[year]:
            pmi_inters_ranks[year][term[0]] = (term[1], combined_ranks[year][term], combined_pmi_dict[year][term[0]])

# %%
agg_pmi = disc.agg_combined_pmi_rank(pmi_inters_ranks)

# %%
# Aggregate pmi
agg_pmi.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'combined_pmi_rank_hp.csv'), index = False)

# %% [markdown]
# ### Quick exploration of collocations

# %%
flat_sentences = pd.concat([combined_term_sentences[y] for y in combined_term_sentences])

# %%
grouped_sentences = disc.check_collocations(flat_sentences, 'hydrogen')
disc.collocation_summary(grouped_sentences)

# %%
disc.view_collocations(grouped_sentences)

# %% [markdown]
# ## 6. Analysis of language used to describe search terms

# %%
term_phrases = disc.noun_chunks_w_term(noun_chunks_all_years, search_terms)

# %%
#term_phrases

# %%
period = ['2007', '2008', '2009', '2010']
adjectives = []
for year in period:
    adjectives.append(disc.find_pattern(combined_term_sentences[year]['sentence'], nlp, adj_phrase))

# %%
collections.Counter([elem for sublist in adjectives for elem in sublist])

# %%
period = ['2018', '2019', '2020', '2021']
nouns = []
for year in period:
    nouns.append(disc.find_pattern(combined_term_sentences[year]['sentence'], nlp, noun_phrase))

# %%
collections.Counter([elem for sublist in nouns for elem in sublist])

# %%
period = ['2011', '2012', '2013']
hp_are = []
for year in period:
    hp_are.append(disc.find_pattern(combined_term_sentences[year]['sentence'], nlp, term_is))

# %%
collections.Counter([elem for sublist in hp_are for elem in sublist])

# %%
period = ['2018', '2019', '2020', '2021']
verb_s = []
for year in period:
    verb_s.append(disc.find_pattern(combined_term_sentences[year]['sentence'], nlp, verb_subj))

# %%
collections.Counter([elem for sublist in verb_s for elem in sublist])

# %%
period = ['2011', '2012', '2013']
verb_o = []
for year in period:
    verb_o.append(disc.find_pattern(combined_term_sentences[year]['sentence'], nlp, verb_obj))

# %%
collections.Counter([elem for sublist in verb_o for elem in sublist])

# %%
subject_phrase_dict = collections.defaultdict(list)
object_phrase_dict = collections.defaultdict(list)
given_term = 'heat pump'
for year in term_sentences[given_term]:
    given_term_sentences = term_sentences[given_term][year]['sentence']
    subj_triples, obj_tripels = disc.get_svo_triples(given_term_sentences, given_term, nlp)
    subject_phrases, object_phrases = disc.get_svo_phrases(subj_triples, obj_tripels)
    subject_phrase_dict[year] = subject_phrases
    object_phrase_dict[year] = object_phrases

# %%
period = ['2018', '2019', '2020', '2021']
verb_o = []
for year in period:
    print(object_phrase_dict[year])

# %% [markdown]
# ## Appendix

# %%
noun_phrase = [{"POS": "ADJ", "OP": "*"}, 
               {'POS': 'NOUN'},
               {'POS': 'NOUN', 'OP': '?'},
               {'TEXT': 'heat'},
               {"TEXT": {'IN': ['pump', 'pumps']}}, 
              ]

adj_phrase = [{"POS": "ADV", "OP": "*"}, 
            {'POS': 'ADJ'},
            {"POS": "ADJ", "OP": "*"}, 
            {'POS': 'NOUN', 'OP': '?'},
            {'TEXT': 'heat'},
            {"TEXT": {'IN': ['pump', 'pumps']}}, 
            ]


term_is = [{'TEXT': 'heat'},
           {"TEXT": {'IN': ['pump', 'pumps']}}, 
           {"LEMMA": "be"}, 
           {"DEP": "neg", "OP": "?"}, 
           {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}]


verb_obj = [{'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
            {'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
            {'POS': 'VERB'},
            {'OP': '?'},            
            {'TEXT': 'heat'},
            {"TEXT": {'IN': ['pump', 'pumps']}}, 
            ]

verb_subj = [{'TEXT': 'heat'},
             {"TEXT": {'IN': ['pump', 'pumps']}}, 
             {'POS': 'VERB'},
             {'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
             {'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
             ]   


# %% [markdown]
# ### Evaluating sentiment around search terms for a given year
# %%
aggregated_sentiment_all_terms = dict()
sentence_sentiment_all_terms = dict()
for term in search_terms:
    aggregated_sentiment, sentence_sentiment = disc.agg_term_sentiments(term, term_sentences)
    aggregated_sentiment_all_terms[term] = aggregated_sentiment
    sentence_sentiment_all_terms[term] = sentence_sentiment

# %%
with open(os.path.join(DISC_OUTPUTS_DIR, 'aggregated_sentiment_all_terms_hp.pkl'), "wb") as outfile:
        pickle.dump(aggregated_sentiment_all_terms, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_sentiment_all_terms_hp.pkl'), "wb") as outfile:
        pickle.dump(sentence_sentiment_all_terms, outfile)   
