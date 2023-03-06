# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linking our taxonomy to other datasets

# %%
from innovation_sweet_spots.utils import google_sheets as gs
import utils

import importlib

importlib.reload(utils)
import json

# %%
# Load taxonomy
taxonomy_df = gs.download_google_sheet(utils.AFS_GOOGLE_SHEET_ID, "taxonomy")

# %%
# Load existing data
initial_list_df = gs.download_google_sheet(utils.AFS_GOOGLE_SHEET_ID, "initial_list")

# %% [markdown]
# ## Our taxonomy

# %%
taxonomy_df.head(2)

# %%
utils.get_taxonomy_dict(taxonomy_df, "theme", "subtheme")

# %% [markdown]
# ## Existing labels

# %%
initial_categories = utils.get_taxonomy_dict(initial_list_df, "source", "category")

# Print pretty json
print(json.dumps(initial_categories, indent=4))

# %%
initial_categories_df = (
    initial_list_df[["source", "category"]]
    .drop_duplicates()
    .sort_values(["source", "category"])
)
initial_categories_df.head(2)
# gs.upload_to_google_sheet(initial_categories_df, utils.AFS_GOOGLE_SHEET_ID, 'taxonomy_alignment')

# %% [markdown]
# ## Crunchbase categories
#
# - Load Crunchbase wrangler and get industries, embed, and find similar ones to our list

# %%
from innovation_sweet_spots.analysis import wrangling_utils as wu
from innovation_sweet_spots.utils.embeddings_utils import QueryEmbeddings
import sentence_transformers

CB = wu.CrunchbaseWrangler()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)

industry_vectors = embedding_model.encode(CB.industries)
group_vectors = embedding_model.encode(CB.industry_groups)

q_industries = QueryEmbeddings(industry_vectors, CB.industries, embedding_model)
q_groups = QueryEmbeddings(group_vectors, CB.industry_groups, embedding_model)


# %%
## Iterate through taxonomy keywords, fetch top 25 most similar industries and count them up
import pandas as pd

# import defaultdict
from collections import defaultdict

# Take each column of theme, subtheme and keywords, and combine into one list
taxonomy_keywords = (
    taxonomy_df.theme.to_list()
    + taxonomy_df.subtheme.to_list()
    + taxonomy_df.keywords.to_list()
)
taxonomy_keywords = (
    pd.DataFrame(data={"keywords": taxonomy_keywords})
    .keywords.str.lower()
    .drop_duplicates()
    .sort_values()
    .to_list()
)

# Iterate through keywords, find most similar industries and count them up

industry_counts = defaultdict(int)
for keyword in taxonomy_keywords:
    similar_industries = q_industries.find_most_similar(keyword)[0:25].text.to_list()
    for industry in similar_industries:
        industry_counts[industry] += 1

# %%
(
    pd.DataFrame(
        data={"industry": industry_counts.keys(), "counts": industry_counts.values()}
    ).sort_values("counts", ascending=False)
).iloc[51:100]

# %%
q_industries.find_most_similar("children").head(25).text.to_list()

# %%
industry_vectors = eu.Vectors()

# %%
# Set up a querying class
query = eu.QueryEmbeddings(
    vectors=concept_embeddings.vectors,
    texts=concept_embeddings.vector_ids,
    model=concept_embeddings.model,
)


def find_most_similar_concepts(
    query: eu.QueryEmbeddings, search_term: str
) -> pd.DataFrame:
    """Find most similar concepts and display their information"""
    return (
        query.find_most_similar(term)
        .rename(columns={"text": "openalex_id"})
        .merge(concepts, how="left")
    )
