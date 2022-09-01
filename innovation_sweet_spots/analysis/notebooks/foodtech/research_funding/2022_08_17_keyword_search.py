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
foodtech_terms = get_foodtech_search_terms(from_local=False)

# %%
foodtech_terms

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
def process_foodtech_terms(foodtech_terms: pd.DataFrame, nlp) -> Iterable[str]:
    terms_df = foodtech_terms.query("use == '1'").reset_index(drop=True)

    terms = [s.split(",") for s in terms_df.Terms.to_list()]
    terms_processed = []
    for term in terms:
        terms_processed.append(
            [" ".join(t) for t in tpu.process_corpus(term, nlp=nlp, verbose=False)]
        )
    assert len(terms_processed) == len(terms_df)
    terms_df["terms_processed"] = terms_processed
    return terms_df


def select_terms_by_label(
    text: str, terms_df: pd.DataFrame, terms: Iterable[str], column: str = "Tech area"
) -> Iterable:
    term_number = terms_df[terms_df[column] == text].index
    return [terms[i] for i in term_number]


def compile_term_dict(terms_df, column="Tech area") -> dict:
    tech_area_terms = {}
    terms_processed = terms_df.terms_processed.to_list()
    for tech_area in terms_df[column].unique():
        tech_area_terms[tech_area] = select_terms_by_label(
            text=tech_area, terms_df=terms_df, terms=terms_processed, column=column
        )
    return tech_area_terms


# %%
# Remove the following terms
dont_use_terms = ["no-kill meat"]
terms_df = process_foodtech_terms(
    foodtech_terms.query("Terms not in @dont_use_terms"), nlp
)
tech_area_terms = compile_term_dict(terms_df, "Tech area")

# %%
tech_area_terms

# %% [markdown]
# # Use search terms

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms

# %%
import ast

# %%
Query_nihr = QueryTerms(corpus=nihr_corpus)
Query_gtr = QueryTerms(corpus=gtr_corpus)

# %%
tech_areas_to_check = list(tech_area_terms.keys())[:-3]


# %%
def get_hit_terms(df: pd.DataFrame, column_name: str = "found_terms") -> pd.DataFrame:
    """Collates all found terms into one string and adds a new column 'found terms'"""
    df = df.copy()
    hit_terms = []
    for i, row in df.drop("has_any_terms", axis=1).iterrows():
        hit_terms.append(f'[{", ".join(row[row==True].index)}]')
    df[column_name] = hit_terms
    return df


# %%
def get_document_hits(
    Query_instance, tech_area_terms, tech_areas_to_check, filter_hits
):
    all_hits = {}
    query_results = pd.DataFrame()
    for tech_area in tech_areas_to_check:
        query_df = Query_instance.find_matches(
            tech_area_terms[tech_area], return_only_matches=True
        )
        hits = set(query_df.id.to_list()).intersection(set(filter_hits.id.to_list()))
        query_df = (
            get_hit_terms(query_df.query("id in @hits"))
            .assign(tech_area=tech_area)
            .sort_values("found_terms")
        )[["id", "found_terms", "tech_area"]]
        assert len(query_df) == len(hits)
        query_results = pd.concat([query_results, query_df], ignore_index=True)
        all_hits[tech_area] = hits
    return query_results, all_hits


# %% [markdown]
# ### Gateway to Research

# %%
food_hits = Query_gtr.find_matches(
    tech_area_terms["Food terms"], return_only_matches=True
)

# %%
gtr_query_results, gtr_all_hits = get_document_hits(
    Query_gtr, tech_area_terms, tech_areas_to_check, food_hits
)

# %%
pd.DataFrame(
    data={
        "tech_area": tech_areas_to_check,
        "hits": [len(gtr_all_hits[hits]) for hits in gtr_all_hits],
    }
)

# %%
gtr_query_results

# %%
gtr_query_results_, gtr_all_hits_ = get_document_hits(
    Query_gtr, tech_area_terms, ["Food technology terms"], food_hits
)

# %%
gtr_query_results_new = gtr_query_results_.query(
    "id not in @gtr_query_results.id.to_list()"
)
len(gtr_query_results_new)

# %% [markdown]
# ### Export for review

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters import gtr_2022 as gtr
import pandas as pd

gtr_df = gtr.get_gtr_projects()

# %%
from innovation_sweet_spots.utils.io import save_json, load_json

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/foodtech/interim/research_funding/"


# %%
def prep_export_table(df_to_export, gtr_df):
    gtr_columns_to_export = [
        "id",
        "title",
        "abstractText",
        "techAbstractText",
        "grantCategory",
        "leadFunder",
        "leadOrganisationDepartment",
        "found_terms",
        "tech_area",
    ]

    gtr_query_details = df_to_export.merge(gtr_df, on="id")
    return gtr_query_details[gtr_columns_to_export]


# %%
gtr_query_details = prep_export_table(gtr_query_results, gtr_df)

# %%
gtr_query_details.to_csv(OUTPUTS_DIR / "gtr_projects_v2022_08_22.csv", index=False)

# %%
gtr_all_hits_export = gtr_all_hits.copy()
for key in gtr_all_hits_export:
    gtr_all_hits_export[key] = list(gtr_all_hits_export[key])
gtr_all_hits_export["Food terms"] = food_hits.id.to_list()

# %%
save_json(gtr_all_hits_export, OUTPUTS_DIR / "gtr_projects_v2022_08_22.json")

# %% [markdown]
# ### New exports

# %%
new_table = prep_export_table(gtr_query_results_new, gtr_df)
new_table.to_csv(OUTPUTS_DIR / "gtr_projects_v2022_08_31_foodtech.csv", index=False)

# %% [markdown]
# ## NIHR

# %%
nihr_food_hits = Query_nihr.find_matches(
    tech_area_terms["Food terms"], return_only_matches=True
)

# %%
nihr_query_results, nihr_all_hits = get_document_hits(
    Query_nihr, tech_area_terms, tech_areas_to_check, nihr_food_hits
)

# %%
pd.DataFrame(
    data={
        "tech_area": tech_areas_to_check,
        "hits": [len(nihr_all_hits[hits]) for hits in nihr_all_hits],
    }
)

# %% [markdown]
# ### Export for review

# %%
NIHR_DIR = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
nihr_df = pd.read_csv(NIHR_DIR)

# %%
nihr_df.info()

# %%
nihr_query_details = nihr_query_results.merge(
    nihr_df.rename(columns={"recordid": "id"}), on="id"
)

# %%
# nihr_query_details

# %%
nihr_columns_to_export = [
    "id",
    "project_title",
    "scientific_abstract",
    "plain_english_abstract",
    "programme",
    "organisation_type",
    "award_amount_m",
    "found_terms",
    "tech_area",
]

# %%
nihr_query_details[nihr_columns_to_export].to_csv(
    OUTPUTS_DIR / "nihr_projects_v2022_08_22.csv", index=False
)

# %%
nihr_all_hits_export = nihr_all_hits.copy()
for key in nihr_all_hits_export:
    nihr_all_hits_export[key] = list(nihr_all_hits_export[key])
nihr_all_hits_export["Food terms"] = nihr_food_hits.id.to_list()

# %%
save_json(nihr_all_hits_export, OUTPUTS_DIR / "nihr_projects_v2022_08_22.json")

# %%
