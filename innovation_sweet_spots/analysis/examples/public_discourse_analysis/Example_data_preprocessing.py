# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Clean text, build sentence corpus and analyse mentions over time
#
# - Basic cleaning of articles
# - Extracting sentences (default context) and noun chunks for subsequent analysis
# - Analyse mentions of search terms over time
# - Appendix: quick exploration of co-locations

# %% [markdown]
# ## 1. Import dependencies

# %%
import os
import pandas as pd
import csv
import spacy
import pickle

# %%
from innovation_sweet_spots.utils.pd import (
    pd_data_processing_utils as dpu,
)
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
search_terms = ["heat pump", "heat pumps"]

# %%
nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# ## 2. Clean text, build sentence corpus

# %%
# Read in article_text dataframe
article_text = pd.read_csv(DISC_OUTPUTS_DIR / "article_text_hp.csv")

# %%
with open(os.path.join(DISC_OUTPUTS_DIR, "metadata_dict_hp.pkl"), "rb") as infile:
    metadata_dict = pickle.load(infile)

# %%
article_text["year"] = article_text["id"].apply(
    lambda x: metadata_dict[x]["webPublicationDate"][:4]
)

# %% [markdown]
# ### 2.1. Extract sentences

# %%
# Generate corpus of articles and sentences (flat list including ids for linking to metadata).
# Spacy processed corpus of articles is used to extract noun chunks from text.
# Cleaning is minimal (split camel-case, convert to lower case, normalise, but keep punctuation,
# remove extra spaces, retain stopwords, no lemmatisation).
# Current performance is ~ 2.5 min per 1K articles. Results on broad topics may contain tens of thousands of articles.
# In such case first use a sample of articles.

processed_articles_by_year, sentence_records = dpu.generate_sentence_corpus_by_year(
    article_text, nlp, "year", "text", "id"
)

# %%
# Link sentences to article IDs
sentence_record_dict = {elem[0]: elem[1] for elem in sentence_records}

# %% [markdown]
# ### 2.2. Extract noun chunks

# %%
# Use spacy functionality to identify noun phrases in all sentences in the corpus.
# These often provide a useful starting point for analysing language used around a given technology.
noun_chunks_all_years = {
    str(year): dpu.get_noun_chunks(processed_articles, remove_det_articles=True)
    for year, processed_articles in processed_articles_by_year.items()
}

# %% [markdown]
# ## 3. Analyse mentions over time

# %% [markdown]
# ### 3.1. Extract sentences that mention search terms

# %%
# Dict with year, search term as keys and corpus of sentences as values.
term_sentences = {
    term: dpu.get_flat_sentence_mentions([term], sentence_records)
    for term in search_terms
}

# %%
# Sentence corpus aggregated across the set of search terms
combined_term_sentences = dpu.get_flat_sentence_mentions(search_terms, sentence_records)

# %% [markdown]
# ### 3.2. Count mentions across set of terms over time

# %%
mentions_s = []
# Count number of sentences that mentioned each individual search term
for term in search_terms:
    term_mentions = pd.Series({y: len(s) for y, s in term_sentences[term].items()})
    mentions_s.append(term_mentions)
# Count total number of articles that mentioned any of the search terms
all_documents = article_text.groupby("year")["id"].count()
mentions_s.append(all_documents)
# Collect into a single dataframe and specify column names
mentions_df = pd.concat(mentions_s, axis=1)
mentions_df.columns = search_terms + ["total_documents"]


# %%
mentions_df

# %%
# Save outputs
with open(
    os.path.join(DISC_OUTPUTS_DIR, "processed_articles_by_year_hp.pkl"), "wb"
) as outfile:
    pickle.dump(processed_articles_by_year, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, "sentence_records_hp.pkl"), "wb") as outfile:
    pickle.dump(sentence_records, outfile)

with open(
    os.path.join(DISC_OUTPUTS_DIR, "sentence_record_dict_hp.pkl"), "wb"
) as outfile:
    pickle.dump(sentence_record_dict, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, "noun_chunks_hp.pkl"), "wb") as outfile:
    pickle.dump(noun_chunks_all_years, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, "term_sentences_hp.pkl"), "wb") as outfile:
    pickle.dump(term_sentences, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, "combined_sentences_hp.pkl"), "wb") as outfile:
    pickle.dump(combined_term_sentences, outfile)

mentions_df.to_csv(os.path.join(DISC_OUTPUTS_DIR, "mentions_df_hp.csv"))

# %% [markdown]
# ## Appendix

# %% [markdown]
# ### Quick exploration of co-locations

# %%
flat_sentences = pd.concat(
    [combined_term_sentences[y] for y in combined_term_sentences]
)

# %%
# Retrieve sentences where a given term was used together with any of the search terms
grouped_sentences = dpu.check_collocations(flat_sentences, "retrofit")
dpu.collocation_summary(grouped_sentences)

# %%
dpu.view_collocations(grouped_sentences, metadata_dict, sentence_record_dict)

# %% [markdown]
# ### Quick search across the corpus

# %%
sentence_collection_df = pd.DataFrame(sentence_records)
sentence_collection_df.columns = ["sentence", "id", "year"]
# sentences_by_year = {y: v for y, v in sentence_collection_df.groupby('year')}

# %%
# Retrieve sentences with a given term
grouped_sentences = dpu.check_mentions(sentence_collection_df, "retrofit")
dpu.mentions_summary(grouped_sentences)

# %%
dpu.view_mentions(grouped_sentences, metadata_dict, sentence_record_dict)

# %%
