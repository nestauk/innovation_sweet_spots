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
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: innovation_sweet_spots
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
from innovation_sweet_spots import PROJECT_DIR

# Load OpenAlex concepts and vectorised embeddings
concepts = get_openalex_concept_list()
concept_embeddings = get_openalex_concept_embeddings()
# Load foodtech taxonomy table (latest version)
foodtech_taxonomy = pd.read_csv(
    PROJECT_DIR / "inputs/data/misc/foodtech/foodtech_search_terms.csv"
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
# ## Querying related concepts
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


# %% [markdown]
# Checking top 20 similar concepts for a single term

# %%
term = "Innovative food"

# %%
find_most_similar_concepts(query, term).head(20)

# %%
foodtech_taxonomy

# %% [markdown]
# Grab top 5 similar concepts for each term

# %%
pred_df = []
for term in foodtech_taxonomy["Terms"].unique():

    top_5_similar = find_most_similar_concepts(query, term).head(10)
    top_5_similar.insert(0, "taxonomy_term", "")
    top_5_similar["taxonomy_term"] = term

    pred_df.append(top_5_similar)

# %%
sheet = pd.concat(pred_df).reset_index(drop=True)

# %%
user_directory = ""
sheet.to_csv(
    f"{user_directory}/innovation_sweet_spots/inputs/data/taxonomy_terms_and_openalex.csv"
)
