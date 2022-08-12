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
# # Finding OpenAlex concepts related to our food tech categories

# %% [markdown]
# ## Setup

# %%
from innovation_sweet_spots.getters.openalex import (
    get_openalex_concept_list,
    get_openalex_concept_embeddings,
)
import innovation_sweet_spots.utils.embeddings_utils as eu
import pandas as pd
import json
from innovation_sweet_spots.getters.processed_foodtech import FOODTECH_PATH

# Load OpenAlex concepts and vectorised embeddings
concepts = get_openalex_concept_list()
concept_embeddings = get_openalex_concept_embeddings()
# Load foodtech taxonomy table (latest version)
foodtech_taxonomy = pd.read_csv(FOODTECH_PATH / "interim/taxonomy_v2022_07_27.csv")
foodtech_taxonomy_dict = json.load(
    open(FOODTECH_PATH / "interim/taxonomy_v2022_07_27.json", "r")
)

# %%
concepts.head(5)

# %%
foodtech_taxonomy.sample(3)

# %% [markdown]
# The taxonomy table contains names of Major and Minor categories of innovations. The column "label" is a more granular keyword or key phrase related to the Minor innovation category.
#
# Note that this taxonomy is presently based on the VC investments data, as is not fully aligned with the Minor, sub-categories presented at the workshop. For example, it is missing "reformulation" Minor category.
#
# Nonetheless, it would be useful to explore the OpenAlex concepts and find concepts that are closely related to the keywords and phrases in the "Major", "Minor" and "label" columns. This would help, in turn, to find research publications in OpenAlex related to the food tech categories.

# %% [markdown]
# ## Querying related concepets
#
# I guess one strategy could be to find the top 20 or so most similar concepts for each of the labels, and then go through with them together to consider which ones could be useful...

# %%
# Set up a querying class
query = eu.QueryEmbeddings(
    vectors=concept_embeddings.vectors,
    texts=concept_embeddings.vector_ids,
    model=concept_embeddings.model,
)


# %%
# Helper function
def find_most_similar_concepts(
    query: eu.QueryEmbeddings, search_term: str
) -> pd.DataFrame:
    """Find most similar concepts and display their information"""
    return (
        query.find_most_similar(term)
        .rename(columns={"text": "openalex_id"})
        .merge(concepts, how="left")
    )


# %%
term = "Innovative food"

# %%
find_most_similar_concepts(query, term).head(20)

# %%
