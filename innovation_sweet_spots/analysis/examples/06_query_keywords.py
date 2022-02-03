# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selecting GtR projects by keywords and key phrases

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
from innovation_sweet_spots.getters.preprocessed import get_tokenised_gtr_corpus
from innovation_sweet_spots.analysis.query_terms import QueryTerms
import pandas as pd

GTR = wu.GtrWrangler()

# %%
Query = QueryTerms(corpus=get_tokenised_gtr_corpus())

# %%
SEARCH_TERMS = [["heat pump"], ["hydrogen", "heat"]]

# %%
query_df = Query.find_matches(SEARCH_TERMS, return_only_matches=True)
query_df

# %%
GTR.add_project_data(query_df, id_column="id", columns=["title"])
