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
# # Clean text, build sentence corpus and analyse mentions over time
#
# - Basic cleaning of articles
# - Extracting sentences (default context) and noun chunks for subsequent analysis
# - Analyse mentions of search terms over time
# - Appendix: alternative options for defining context around search terms

# %% [markdown]
# ## 1. Import dependencies

# %%
import os
import pandas as pd
import csv
import spacy
import pickle

# %%
# Change first element to location of project folder.
os.chdir(os.path.join('/Users/jdjumalieva/Documents/Analysis/', 'innovation_sweet_spots'))

# %%
from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import pd_data_processing_utils as dpu
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
search_terms = ['heat pump', 'heat pumps']

# %%
nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# ## 2. Clean text, build sentence corpus

# %%
# Read in article_text dataframe
article_text = pd.read_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_hp.csv'))

# %%
with open(os.path.join(DISC_OUTPUTS_DIR, 'metadata_dict_hp.pkl'), "rb") as infile:
        metadata_dict = pickle.load(infile)

# %%
article_text['year'] = article_text['id'].apply(lambda x: metadata_dict[x]['date'][:4])

# %% [markdown]
# ### 2.1. Extract sentences

# %%
# Generate corpus of articles and sentences (flat list including ids for linking to metadata).
# Spacy processed corpus of articles is used to extract noun chunks from text.
# Cleaning is minimal (split camel-case, convert to lower case, normalise, but keep punctuation, 
# remove extra spaces, retain stopwords, no lemmatisation).
# Current performance is ~ 2.5 min per 1K articles. Results on broad topics may contain tens of thousands of articles.
# In such case first use a sample of articles.

processed_articles_by_year, sentence_records = dpu.generate_sentence_corpus_by_year(article_text, 
                                                                        nlp,
                                                                        'year',
                                                                        'text',
                                                                        'id')

# %%
# Link sentences to article IDs
sentence_record_dict = {elem[0]: elem[1] for elem in sentence_records}

# %% [markdown]
# ### 2.2. Extract noun chunks

# %%
# Use spacy functionality to identify noun phrases in all sentences in the corpus.
# These often provide a useful starting point for analysing language used around a given technology.
noun_chunks_all_years = {str(year): dpu.get_noun_chunks(processed_articles, remove_det_articles = True) for\
                        year, processed_articles in processed_articles_by_year.items()}

# %% [markdown]
# ## 3. Analyse mentions over time

# %% [markdown]
# ### 3.1. Extract sentences that mention search terms

# %%
# Dict with year, search term as keys and corpus of sentences as values.
term_sentences = {term: dpu.get_flat_sentence_mentions(term, sentence_records) for term in search_terms}

# %%
# More streamlined version of the above: sentence corpus aggregated across the set of search terms
combined_term_sentences = dpu.combine_term_sentences(term_sentences, search_terms)

# %% [markdown]
# ### 3.2. Count mentions across set of terms over time

# %%
mentions_s = []
# Count number of sentences that mentioned each individual search term
for term in search_terms:
    term_mentions = pd.Series({y: len(s) for y,s in term_sentences[term].items() })
    mentions_s.append(term_mentions)
# Count total number of articles that mentioned any of the search terms
all_documents = article_text.groupby('year')['id'].count()
mentions_s.append(all_documents)
# Collect into a single dataframe and specify column names
mentions_df = pd.concat(mentions_s, axis =1)
mentions_df.columns = search_terms + ['total_documents']


# %%
mentions_df
