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
# # Example of optimising clusters for Gateway to Research project abstracts

# %% [markdown]
# ## Library imports

# %%
import nltk

nltk.download("stopwords")
from innovation_sweet_spots.getters.gtr_2022 import get_gtr_file
from innovation_sweet_spots.utils.embeddings_utils import Vectors
from innovation_sweet_spots.utils.cluster_analysis_utils import (
    param_grid_search,
    highest_silhouette_model_params,
)
import innovation_sweet_spots.utils.cluster_analysis_utils as cau
import innovation_sweet_spots.utils.embeddings_utils as eu
from innovation_sweet_spots import PROJECT_DIR

# %% [markdown]
# ## Load data

# %%
# Load GtR project abstracts
gtr_project_abstracts = get_gtr_file(filename="gtr_projects-projects.json")[
    ["id", "abstractText"]
].rename(columns={"abstractText": "abstract"})

# %%
# View GtR project abtracts
gtr_project_abstracts.head(5)

# %%
# For testing, reduce data by sampling 10,000 records
gtr_project_abstracts = gtr_project_abstracts.sample(n=10_000, random_state=1)

# %% [markdown]
# ## Embed data

# %%
# Define constants
EMBEDDINGS_DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "gtr_projects"

# %%
# Instansiate Vectors class
gtr_vectors = Vectors(
    model_name="all-MiniLM-L6-v2",
    vector_ids=None,
    filename=FILENAME,
    folder=EMBEDDINGS_DIR,
)

# %%
# Make vectors
gtr_vectors.generate_new_vectors(
    new_document_ids=gtr_project_abstracts.id.values,
    texts=gtr_project_abstracts.abstract.values,
)

# %%
# Save vectors
gtr_vectors.save_vectors()

# %% [markdown]
# ## Clustering
# Here we will use parameter search to find parameters with the highest mean silhouette score.
# <br><br>
# Note that the search parameters need to be in the dictionary format below, with keys being the parameter and the value being a list of values to search through. For example for `hdbscan_search_params`, when performing grid search values `10` and `100` will be input for `min_cluster_size`.
# <br><br>
# `param_grid_search` will search through all combinations of the parameters in the dictionary.

# %%
# Define HDBSCAN search parameters
hdbscan_search_params = {
    "min_cluster_size": [10, 100],
    "min_samples": [1, 25],
    "cluster_selection_method": ["leaf"],
    "prediction_data": [True],
}

# Define K-Means search parameters
kmeans_search_params = {"n_clusters": [8, 20, 30], "init": ["k-means++"]}

# %%
# %%time
# Parameter grid search using HDBSCAN
hdbscan_search_results = param_grid_search(
    vectors=gtr_vectors.vectors,
    search_params=hdbscan_search_params,
    cluster_with_hdbscan=True,
)

# %%
# View results
hdbscan_search_results

# %%
# Find HDBSCAN model params with highest silhouette score
optimal_hdbscan_params = highest_silhouette_model_params(hdbscan_search_results)
optimal_hdbscan_params

# %%
# %%time
# Parameter grid search using K-Means
kmeans_search_results = param_grid_search(
    vectors=gtr_vectors.vectors,
    search_params=kmeans_search_params,
    cluster_with_hdbscan=False,
)

# %%
# View results
kmeans_search_results

# %%
# Find K-Means model params with highest silhouette score
optimal_kmeans_params = highest_silhouette_model_params(kmeans_search_results)
optimal_kmeans_params

# %% [markdown]
# ## Visualising the optimal clustering result

# %%
importlib.reload(cau);

# %% [markdown]
# ### k-means result

# %%
optimal_labels = cau.kmeans_clustering(gtr_vectors.vectors, optimal_kmeans_params)

# %%
vectors_2d, fig = cau.cluster_visualisation(
    gtr_vectors.vectors,
    optimal_labels,
    # Add short abstracts to the visualisation 
    extra_data=(
        gtr_project_abstracts[['id', 'abstract']]
        .assign(abstract=lambda df: df.abstract.apply(lambda x: str(x)[0:300] + '...'))
    ),
)

# %% [markdown]
# ### hdbscan result

# %%
optimal_labels = cau.hdbscan_clustering(gtr_vectors.vectors, optimal_hdbscan_params)

# %%
vectors_2d, fig = cau.cluster_visualisation(
    vectors_2d,
    optimal_labels['labels'],
    # Add short abstracts to the visualisation 
    extra_data=(
        gtr_project_abstracts[['id', 'abstract']]
        .assign(abstract=lambda df: df.abstract.apply(lambda x: str(x)[0:300] + '...'))
    ),
)

# %%
fig

# %%
