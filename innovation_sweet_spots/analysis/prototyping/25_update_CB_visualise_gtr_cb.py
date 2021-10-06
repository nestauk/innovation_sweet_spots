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

# %% [markdown]
# # Update CB and visualise all GTR and CB projects
#
# - Import top2vec vectors
# - Generate UMAP embeddings
# - Highlight projects identified via semi-automated approach
#   - Check for potential *areas* of false negatives
# - Highlight projects identified via topic analysis
#   - Show overlaps of areas like energy management and sensors

# %%
import umap
from innovation_sweet_spots.analysis.topic_analysis import (
    get_top2vec_model,
    umap_document_vectors,
    get_clustering,
)
import altair as alt

# %%
TOP2VEC_RUN = "August2021_gtr_cb_stopwords_e100"
UMAP_ARGS = {
    "n_neighbors": 50,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 112,
}

# %%
# Import the top2vec vectors and clusters
top2vec_model = get_top2vec_model(TOP2VEC_RUN)

# %%
top2vec_clusters = get_clustering(TOP2VEC_RUN)
top2vec_clusters.head(1)

# %%
# Generate a 2D visualisation
xy, umap_model = umap_document_vectors(top2vec_model)

# %%
xy.shape

# %%
df_viz = top2vec_clusters.merge(topic_docs, how="left")
df_viz["x"] = xy[:, 0]
df_viz["y"] = xy[:, 1]

# %%
GTR_DOCS_ALL = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_GTR.csv"
)

# %%
df_viz = df_viz.merge(GTR_DOCS_ALL[["doc_id", "tech_category"]], how="left")

# %%
alt.data_transformers.disable_max_rows()

# %%
df_viz.head(2)

# %%
len(df_viz)

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %%
fig = (
    alt.Chart(
        df_viz[df_viz.tech_category == "Building insulation"],
        #         df_viz.loc[0:110000],
        width=750,
        height=750,
    )
    .mark_circle(size=25, opacity=0.9)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        #         size='size',
        #         color='cluster',
        shape="source",
        color=alt.Color("tech_category", scale=alt.Scale(scheme="category20")),
        tooltip=["title", "cluster_keywords"],
    )
    .interactive()
)

# fig_categories= (
#     alt.Chart(
#         df_viz.iloc[0:107165],
#         width=750,
#         height=750,
#     )
#     .mark_circle(size=25, opacity=0.2)
#     .encode(
#         x=alt.X("x", axis=alt.Axis(grid=False)),
#         y=alt.Y("y", axis=alt.Axis(grid=False)),
#         #         size='size',
#         #         color='cluster',
#         shape='source',
#         color=alt.Color('tech_category', scale=alt.Scale(scheme="category20")),
#         tooltip=['title', 'cluster_keywords'],
#     )
#     .interactive()
# )

# %%
fig

# %%
# alt_save.save_altair(fig, f"viz_all_docs", driver)

# %% [markdown]
# # Add the new CB companies to the mix
# - Import the new CB companies
# - Tokenise
# - Infer top2vec vectors & create master top2vec vector matrix (npy)
# - Infer closest top2vec clusters (optional?)
# - Infer topic model probabilities & create master topic prob matrix (npy)

# %%
from innovation_sweet_spots.getters.crunchbase import CB_PATH
from innovation_sweet_spots.utils.io import load_pickle, save_pickle
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils import text_pre_processing
import innovation_sweet_spots.analysis.analysis_utils as iss

import pandas as pd
import os
import sys

# %%
CB_NEW = pd.read_csv(CB_PATH.parent / "cb_2021/new_cb_uk_orgs.csv")

# %%
### Tokenise

# Create documents texts
cb_org_texts = iss.create_documents_from_dataframe(
    CB_NEW,
    columns=["short_description", "long_description"],
    preprocessor=(lambda x: x),
)
# Ngrammer
ngram_phraser = load_pickle(PROJECT_DIR / f"outputs/models/ngram_phraser_gtr_cb_full.p")
# Language model
nlp = text_pre_processing.setup_spacy_model(text_pre_processing.DEF_LANGUAGE_MODEL)
tokenised_corpus = [
    text_pre_processing.ngrammer(s, ngram_phraser, nlp) for s in cb_org_texts
]

# %%
save_pickle(
    tokenised_corpus, PROJECT_DIR / "outputs/data/cb/new_cb_uk_docs_tokenised.p"
)

# %%
### Infer top2vec vectors and topics
top2vec_model.add_documents(tokenised_corpus, tokenizer="preprocessed")

# %%
# top2vec_model.model.infer_vector

# %%
len(top2vec_clusters)

# %%
top2vec_cb_new = CB_NEW[["id"]].rename(columns={"id": "doc_id"}).copy()
top2vec_cb_new["cluster_id"] = top2vec_model.doc_top[len(top2vec_clusters) :]
top2vec_cb_new["source"] = "cb"
top2vec_cb_new["version"] = "new"
top2vec_cb_new = top2vec_cb_new.merge(
    top2vec_clusters[["cluster_id", "cluster_keywords"]].drop_duplicates("cluster_id"),
    how="left",
)

# %%
len(top2vec_cb_new)

# %%
top2vec_clusters_all = top2vec_clusters.copy()
top2vec_clusters_all["version"] = "base"
top2vec_clusters_all = top2vec_clusters_all.append(top2vec_cb_new, ignore_index=True)

# %%
top2vec_clusters_all

# %%
top2vec_model.save(
    PROJECT_DIR
    / "outputs/models/top2vec_clusters_August2021_gtr_cb_stopwords_e100_new.p"
)

# %%
top2vec_clusters_all.to_csv(
    PROJECT_DIR
    / "outputs/data/top2vec_clusters_August2021_gtr_cb_stopwords_e100_new.csv",
    index=False,
)

# %%
### Tomotopy
import tomotopy as tp
import numpy as np
from tqdm.notebook import tqdm

# %%
mdl_new = tp.LDAModel.load(
    str(PROJECT_DIR / "outputs/models/topic_models/narrow_corpus_lda_model.bin")
)
doc_topic_dists = np.load(
    PROJECT_DIR / "outputs/data/results_august/narrow_topics_full_corpus.npy"
)

# %%
topic_docs = pd.read_csv(
    PROJECT_DIR / "outputs/models/topic_models/topic_model_docs.csv"
)
topic_docs["version"] = "base"

df = top2vec_clusters_all[top2vec_clusters_all.version == "new"].copy()
df["top2vec_index"] = df.index.to_list()
df = df.merge(
    CB_NEW[["id", "name"]].rename(columns={"id": "doc_id", "name": "title"}), how="left"
)

topic_docs = topic_docs.append(
    df[["doc_id", "title", "source", "top2vec_index", "version"]], ignore_index=True
)

# %%
topic_docs.to_csv(
    PROJECT_DIR / "outputs/models/topic_models/topic_model_all_docs.csv", index=False
)

# %%
len(topic_docs)

# %%
doc_topic_dists.shape

# %%
# Infer new topics
infer_doc_topic_dists = []
for doc in tqdm(tokenised_corpus, total=len(tokenised_corpus)):
    doc_instance = mdl_new.make_doc(doc)
    infer_doc_topic_dist, _ = mdl_new.infer(doc_instance)
    infer_doc_topic_dists.append(infer_doc_topic_dist)
infer_doc_topic_dists = np.array(infer_doc_topic_dists)

# %%
len(tokenised_corpus)

# %%
infer_doc_topic_dists.shape

# %%
doc_topic_dists_all = np.vstack((doc_topic_dists, infer_doc_topic_dists))

# %%
np.save(
    PROJECT_DIR / "outputs/models/topic_models/narrow_topic_probabilities_all_docs.npy",
    doc_topic_dists_all,
)

# %% [markdown]
# ### Select for review the new cb documents

# %%
import innovation_sweet_spots.analysis.search_terms_utils as search_terms_utils
import innovation_sweet_spots.analysis.search_terms as search_terms

# %%
import importlib

importlib.reload(search_terms_utils)
importlib.reload(search_terms)

# %%
corpus_document_texts = []
for tokens_list in tokenised_corpus:
    s_list = [s for s_list in [tok.split("_") for tok in tokens_list] for s in s_list]
    corpus_document_texts.append(" " + " ".join(s_list) + " ")

# %%
search_terms_utils.find_docs_with_all_terms(
    ["heat pump"], corpus_document_texts, CB_NEW
)

# %%
# narrow_set_categories = pd.read_excel(PROJECT_DIR / 'outputs/data/results_august/narrow_set_top2vec_cluster_counts_checked.xlsx')
# narrow_set_categories.head(20)

# %%
# search_terms_utils.get_categories_to_tomotopy_narrow_topics()

# %%
cb_new = top2vec_clusters_all[top2vec_clusters_all.version == "new"]

# %%
doc_df = topic_docs[["doc_id", "title"]]

# %%
dfs = dict()
for key in search_terms.categories_keyphrases:
    dfs[key] = search_terms_utils.add_info(
        search_terms_utils.get_docs_with_keyphrases(
            search_terms.categories_keyphrases[key], corpus_document_texts, cb_new
        ),
        top2vec_clusters_all,
        doc_df,
        search_terms.categories_to_lda_topics[key],
        doc_topic_dists_all,
    ).reset_index(drop=True)

# %%
[len(dfs[key]) for key in dfs]

# %%
# df = CB_NEW[-CB_NEW.long_description.isnull()]
# df[df.long_description.str.lower().str.contains('heat stor')]

# %%
dfs["Heat storage"].merge(
    CB_NEW[["id", "name", "short_description", "long_description"]],
    left_on="doc_id",
    right_on="id",
)

# %%
with pd.ExcelWriter(
    PROJECT_DIR
    / "outputs/data/results_august/ISS_technologies_to_review_August_27.xlsx"
) as writer:
    for key in dfs:
        dfs[key].merge(
            CB_NEW[["id", "name", "short_description", "long_description"]],
            left_on="doc_id",
            right_on="id",
        ).to_excel(writer, sheet_name=key)

# %%
### TO-DO: ADD THE OTHER SELECTION CRITERIA
import innovation_sweet_spots.getters.crunchbase as crunchbase

crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"

# %%
cb_cats = crunchbase.get_crunchbase_organizations_categories()

# %%
cb_new.head(1)

# %%
cb_cats.head(1)

# %%
cb_new_cats = cb_new.merge(cb_cats, left_on="doc_id", right_on="organization_id")

# %%
cb_new_cats.head(1)

# %%
key = "Solar"
search_terms.categories_to_cb_categories[key]

# %%
importlib.reload(search_terms)

# %%
c

# %%
dfs = dict()
for key in search_terms.reference_category_keyphrases:
    if search_terms.categories_to_cb_categories[key] is not None:
        found_docs_category = cb_new_cats[
            cb_new_cats.category_name.isin(
                search_terms.categories_to_cb_categories[key]
            )
        ]
        found_docs_category = found_docs_category[
            ["doc_id", "cluster_id", "source", "cluster_keywords", "version"]
        ]
    else:
        found_docs_category = pd.DataFrame()

    found_docs = pd.concat(
        [
            found_docs_category,
            search_terms_utils.get_docs_with_keyphrases(
                search_terms.reference_category_keyphrases[key],
                corpus_document_texts,
                cb_new,
            ),
        ]
    ).drop_duplicates("doc_id")

    dfs[key] = search_terms_utils.add_info(
        found_docs,
        top2vec_clusters_all,
        doc_df,
        search_terms.reference_categories_to_lda_topics[key],
        doc_topic_dists_all,
    ).reset_index(drop=True)

# %%
[len(dfs[key]) for key in dfs]

# %%

# %%
with pd.ExcelWriter(
    PROJECT_DIR
    / "outputs/data/results_august/ISS_technologies_to_review_August_27_reference.xlsx"
) as writer:
    for key in dfs:
        dfs[key].merge(
            CB_NEW[["id", "name", "short_description", "long_description"]],
            left_on="doc_id",
            right_on="id",
        ).to_excel(writer, sheet_name=key)

# %%
