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
from innovation_sweet_spots.getters.guardian import search_content
import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
search_term = "heat pumps"
articles = search_content(search_term)

# %%
sentences = iss.get_guardian_sentences_with_term(search_term, articles, field="body")
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
sentiment_df

# %%
