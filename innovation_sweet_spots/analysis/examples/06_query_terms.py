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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selecting projects and companies by keywords and key phrases

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
from innovation_sweet_spots.getters.preprocessed import (
    get_pilot_gtr_corpus,
    get_pilot_crunchbase_corpus,
)
from innovation_sweet_spots.analysis.query_terms import QueryTerms
import pandas as pd

GTR = wu.GtrWrangler()
CB = wu.CrunchbaseWrangler()

# %%
SEARCH_TERMS = [
    ["food"],
    ["food reformulation"],
    ["food", "reformulation"],
    ["food", "reform"],
    ["food", "formula"],
    ["novel food"],
]

# %%
SEARCH_TERMS = [
    ["heat pump"],
]

# %% [markdown]
# # Check research projects

# %%
Query = QueryTerms(corpus=get_pilot_gtr_corpus())

# %%
query_df = Query.find_matches(SEARCH_TERMS, return_only_matches=True)
query_df

# %%
GTR.add_project_data(query_df, id_column="id", columns=["title", "start"])

# %% [markdown]
# # Check companies

# %%
QueryCB = QueryTerms(corpus=get_pilot_crunchbase_corpus())

# %%
query_df = QueryCB.find_matches(SEARCH_TERMS, return_only_matches=True)
query_df

# %%
CB.add_company_data(query_df, id_column="id", columns=["name"])

# %%
