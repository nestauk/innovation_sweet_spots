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

# %%
from innovation_sweet_spots.getters.hansard import get_hansard_data
import innovation_sweet_spots.analysis.analysis_utils as iss

# %% [markdown]
# # Hansard

# %%
import importlib

importlib.reload(iss)

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# %%
hans = get_hansard_data(nrows=200000)

# %%
hans_docs = iss.create_documents_from_dataframe(hans, columns=["speech"])

# %%
search_term = "fuel cell"

# %%
is_term_present = iss.is_term_present(search_term, hans_docs)
docs_with_term = [doc for i, doc in enumerate(hans_docs) if is_term_present[i]]

# %%
speeches = iss.search_via_docs(search_term, hans_docs, hans)

# %%
speeches.info()

# %%
iss.get_hansard_mentions_per_year(speeches)

# %%
iss.show_time_series(iss.get_hansard_mentions_per_year(speeches), y="counts")

# %%
importlib.reload(iss)

# %%
sentences = iss.get_sentences_with_term(search_term, docs_with_term)
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
for s in sentiment_df[sentiment_df.compound > 0].sentences.to_list():
    print(s)
    print("")

# %%
