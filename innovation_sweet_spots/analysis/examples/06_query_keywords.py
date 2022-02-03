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

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots.analysis.query_terms import (
    find_documents_with_set_of_terms,
    token_list_to_string,
)
import pandas as pd

# %%
DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/tokenised_data/"
CORPUS_FILEPATH = DIR / "gtr_docs_tokenised_full.p"
SEARCH_TERMS = [["heat pump"], ["hydrogen", "heat"]]

# %%
# Load the tokenised corpus file (a dict id: tokenised text)
tokenised_corpus = load_pickle(CORPUS_FILEPATH)

# %%
# Prepare text corpus (takes about 15 sec for all projects)
text_corpus = [token_list_to_string(s) for s in tokenised_corpus.values()]

# %%
# Find matches and prepare output
matches = find_documents_with_set_of_terms(SEARCH_TERMS, text_corpus, verbose=True)
matches["id"] = list(tokenised_corpus.keys())
df = pd.DataFrame(matches)

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib

importlib.reload(wu)
GTR = wu.GtrWrangler()

# %%
GTR.add_project_data(df.query("has_any_terms==True"), id_column="id", columns=["title"])

# %%
