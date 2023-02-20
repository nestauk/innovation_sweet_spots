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
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Food tech: Hansard parliamentary debate trends
#

# %%
from innovation_sweet_spots.getters import hansard
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_hansard_corpus
import utils
import innovation_sweet_spots.analysis.query_terms as query_terms
from innovation_sweet_spots import PROJECT_DIR
from ast import literal_eval
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
import re

# %% [markdown]
# ## Prepare search terms

# %%
# Noisy terms (based on checking Guardian articles)
terms_to_remove = [
    "supply chain",
    "delivery platform",
    "meal box",
    "smart kitchen",
    "novel food",
    "kitchen",
]

# %%
# Load search terms table
df_search_terms_table = get_foodtech_search_terms(from_local=False)

# %%
# Process search terms (lemmatise etc) for querying the Hansard corpus
df_search_terms = (
    df_search_terms_table.query("Terms not in @terms_to_remove")
    .assign(Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma))
    .pipe(utils.process_foodtech_terms)
)

# %%
# Search term dictionary
tech_area_terms = utils.compile_term_dict(df_search_terms)

# %% [markdown]
# ## Load Hansard speeches data

# %%
# Dataframe with parliamentary debates
df_debates = hansard.get_debates().drop_duplicates("id", keep="first")
assert len(df_debates.id.unique()) == len(df_debates)
len(df_debates)

# %% [markdown]
# # Keyword search

# %%
# Remove the last three tech areas (food terms, innovation terms and food technology terms)
tech_areas_to_check = list(tech_area_terms.keys())[:-3]

# %%
# Double check the aras
tech_areas_to_check

# %%
# Preprocessed Hansard corpus for keyword search (same data as in the table, but lemmatised etc)
hansard_corpus = get_hansard_corpus()
# Use query util for keyword search
Query_hansard = QueryTerms(corpus=hansard_corpus)

# %%
## Get speeches with food terms
food_hits = Query_hansard.find_matches(
    tech_area_terms["Food terms"], return_only_matches=True
)

# %%
## Get speeches with innovation terms
innovation_hits = Query_hansard.find_matches(
    tech_area_terms["Innovation terms"], return_only_matches=True
)

# %%
import importlib
importlib.reload(query_terms)

## Get speeches with technology terms
gtr_query_results__, gtr_all_hits__ = query_terms.get_document_hits(
    Query_hansard, tech_area_terms, tech_areas_to_check, None
)

# %%
## Get speeches with technology terms
gtr_query_results, gtr_all_hits = query_terms.get_document_hits(
    Query_hansard, tech_area_terms, tech_areas_to_check, food_hits
)

# %%
# Check how many hits we found
len(gtr_query_results__)

# %%
2793695-(1393761)

# %%
# Check how many hits we found
len(gtr_query_results)

# %%
1-(61380/1399934)

# %% [markdown]
# ### Additional filtering: Check that terms are in the same sentence
#
# A check for cases where the search term is a combiation of multiple terms, eg "food, reformulation".
# This will require that both "food" and "reformulation" are mentioned in the same sentnece.

# %%
# Add debate speech texts to the query results
hansard_query_results = gtr_query_results.merge(
    df_debates[["id", "speech", "speakername", "year"]], on="id"
)

# Create a column with found terms
hansard_query_results["found_terms_list"] = hansard_query_results.found_terms.apply(
    literal_eval
)
# Add non-processed search terms to the table (we'll use those to search in the full debate text)
hansard_query_results_ = (
    # Create a row for each search term
    hansard_query_results.explode("found_terms_list").astype({"found_terms_list": str})
    # Add the non-processed version of the term to the table
    .merge(
        df_search_terms[["terms_processed", "Terms"]].astype({"terms_processed": str}),
        left_on="found_terms_list",
        right_on="terms_processed",
    )
)
# Check that the non-processed terms are in the same sentence
has_terms_in_same_sentence = [
    utils.check_articles_for_comma_terms(row.speech, row.Terms)
    for i, row in hansard_query_results_.iterrows()
]

# %%
# Filter out the hits where terms are not in the same sentence
hansard_query_results_filtered = (
    hansard_query_results_[has_terms_in_same_sentence]
    .copy()
    .merge(
        df_search_terms[["Tech area", "Sub Category", "Category"]].drop_duplicates(
            "Tech area"
        ),
        how="left",
        left_on="tech_area",
        right_on="Tech area",
    )
    .drop_duplicates(["id", "Sub Category"])
)

# %%
# Save filtered query results to be used in next step analysis
PD_INTERIM_DIR = PROJECT_DIR / "outputs/foodtech/interim/public_discourse/"
PD_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

hansard_query_results_filtered.to_csv(
    PD_INTERIM_DIR / "hansard_hits_v2022_11_22.csv", index=False
)


# %% [markdown]
# ## Check sentences with specific terms

# %%
def extract_mention_sentence(text, search_term, any_term=False):
    text = re.sub("hon\.", "hon", text)
    text = re.sub("Hon\.", "Hon", text)
    sents = text.lower().split(".")
    sents_with_terms = []
    for sent in sents:
        if any_term:
            has_mention = False
            for s in search_term:
                has_mention = has_mention or (s in sent)
        else:
            has_mention = True
            for s in search_term:
                has_mention = has_mention and (s in sent)
        if has_mention == True:
            sents_with_terms.append(sent)
    return sents_with_terms


# %%
# Query hansard for food environment
search_term = ["food environment"]
hits = Query_hansard.find_matches([search_term], return_only_matches=True)
hits = hits.merge(df_debates[["id", "speech", "speakername", "year"]], on="id")

# %%
# See food environment hits per year
hits.groupby("year").agg(counts=("id", "count"))

# %%
# Process sentences
hits["sents"] = hits.speech.apply(
    lambda x: extract_mention_sentence(x, search_term, any_term=True)
)

# %%
# Show speeches with sentences that contain the search term
hits[hits.sents.astype(bool)]

# %% [markdown]
# ## Check specific terms

# %%
# Query hansard for deliveroo matches
deliveroo = Query_hansard.find_matches([["deliveroo"]], return_only_matches=True)

# %%
# Add additional info from debates data
deliveroo = deliveroo.merge(
    df_debates[["id", "speech", "speakername", "year"]], on="id"
)

# %%
# View deliveroo matches
deliveroo

# %%
