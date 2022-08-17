# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
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
# # Keyword search in research funding data
#
# - GtR, NIHR data
# - Fetch search terms from Google Sheet
# - Preprocess search terms before querying
# - Aggregate results > Indicate which search terms were present in the documents

# %%
from innovation_sweet_spots.getters.preprocessed import (
    get_nihr_corpus,
    get_gtr_corpus,
)
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

# %% [markdown]
# # Fetching input data

# %%
gtr_corpus = get_gtr_corpus()
nihr_corpus = get_nihr_corpus()


# %%
foodtech_terms = get_foodtech_search_terms()

# %% [markdown]
# # Preprocessing search terms

# %%
import innovation_sweet_spots.utils.text_processing_utils as tpu
import pandas as pd
from typing import Iterable

import importlib

importlib.reload(tpu)


# %%
nlp = tpu.setup_spacy_model()

# %%
terms_df = foodtech_terms.query("use == 1").reset_index(drop=True)
len(terms_df)

# %%
terms = [s.split(",") for s in terms_df.Terms.to_list()]
terms_processed = []
for term in terms:
    terms_processed.append(
        [" ".join(t) for t in tpu.process_corpus(term, nlp=nlp, verbose=False)]
    )
assert len(terms_processed) == len(terms_df)

# %%
terms_df.head(5)

# %%
terms_df.tail(5)


# %%
def select_terms_by_label(
    text: str, terms_df: pd.DataFrame, terms: Iterable[str], column: str = "Tech area"
) -> Iterable:
    term_number = terms_df[terms_df[column] == text].index
    return [terms[i] for i in term_number]


# %%
tech_area_terms = {}
for tech_area in terms_df["Tech area"].unique():
    tech_area_terms[tech_area] = select_terms_by_label(
        text=tech_area, terms_df=terms_df, terms=terms_processed, column="Tech area"
    )

# %% [markdown]
# # Search terms

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms

# %%
Query_nihr = QueryTerms(corpus=nihr_corpus)
Query_gtr = QueryTerms(corpus=gtr_corpus)

# %%
# for tech_area in tech_area_terms:
# query_df = Query_nihr.find_matches(tech_area_terms[tech_area], return_only_matches=True)

# %%
query_df = Query_gtr.find_matches(tech_area_terms["Lab meat"], return_only_matches=True)


# %%
query_df_ = query_df.copy()
query_df_["search_terms_list"] = [
    [ast.literal_eval(x) for x in list(row[row].index)]
    for i, row in query_df.drop(["has_any_terms", "id"], axis=1).iterrows()
]

# %%
query_df_[["id", "search_terms_list"]]

# %%
# query_df = Query_nihr.find_matches(tech_area_terms['Biomedical'], return_only_matches=True)
# query_df

# %%
# query_df = Query_gtr.find_matches(tech_area_terms['Lab meat'], return_only_matches=True)
# query_df

# %%

# %%
