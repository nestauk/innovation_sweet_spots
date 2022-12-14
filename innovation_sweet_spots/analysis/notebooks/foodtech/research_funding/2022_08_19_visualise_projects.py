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

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms

# %%
from innovation_sweet_spots.getters.preprocessed import (
    get_nihr_corpus,
    get_gtr_corpus,
)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
import umap

# %% [markdown]
# # Calculate embeddings: GtR

# %%
gtr_corpus = get_gtr_corpus()

# %%
Query_gtr = QueryTerms(corpus=gtr_corpus)

# %%
tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words="english", max_features=10000)
tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(Query_gtr.text_corpus)

# %%
tfidf_word_doc_matrix.shape

# %%
tfidf_embedding = umap.UMAP(metric="manhattan").fit(tfidf_word_doc_matrix)

# %%
umap_embeddings = tfidf_embedding.transform(tfidf_word_doc_matrix)

# %%
umap_embeddings[0, :]

# %% [markdown]
# ## Visualise GtR embeddings

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters import gtr_2022 as gtr
import pandas as pd

gtr_df = gtr.get_gtr_projects()
OUTPUTS_DIR = PROJECT_DIR / "outputs/foodtech/interim/research_funding/"

# %%
keyword_hits_df = pd.read_csv(OUTPUTS_DIR / "gtr_projects_v2022_08_22.csv")

# %%
keyword_hits_df.info()

# %%
import altair as alt

alt.data_transformers.disable_max_rows()

# %%
# gtr_df[['x', 'y']] = umap_embeddings

# %%
# alt.Chart(gtr_df, width=500, height=500).mark_circle(size=20).encode(
#     x='x',
#     y='y',
#     color='leadFunder',
#     tooltip=['title', 'abstractText']
# ).interactive()

# %%

# %% [markdown]
# # Calculate embeddings: NIHR

# %%
