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
# # Inspect top2vec clusters
#
# - Check average cluster size
# - Try reducing the number of clusters using top2vec API
# - Map businesses to these clusters; which number of epochs is better for this purpose?

# %%
from innovation_sweet_spots.utils.io import load_pickle, save_pickle
from innovation_sweet_spots.analysis import top2vec
import innovation_sweet_spots.analysis.analysis_utils as iss
import pandas as pd
import numpy as np
from innovation_sweet_spots import logging, PROJECT_DIR
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
from innovation_sweet_spots.getters.green_docs import (
    get_green_gtr_docs,
    get_green_cb_docs,
)
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.green_document_utils import (
    find_green_gtr_projects,
    find_green_cb_companies,
)
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import adjusted_mutual_info_score as ami_score
import altair as alt

alt.data_transformers.disable_max_rows()
from tqdm.notebook import tqdm

# %%
import importlib

# %%
runs = ["July2021"]
runs = ["July2021_projects_orgs"]
clusterings = [iss_topics.get_clustering(run) for run in runs]
clusterings[0].head(1)

# %% [markdown]
# ## Characterise basic stats

# %%
i = 0
counts = iss_topics.get_cluster_counts(clusterings[i])
iss_topics.plot_histogram(counts)

# %%
df = clusterings[0]

# %%
# df.groupby(['cluster_id', 'source']).count().head(20)

# %%
df

# %% [markdown]
# ## Prepare

# %%
importlib.reload(iss_topics)

# %%
# Import gtr projecs and green project texts
gtr_projects = gtr.get_gtr_projects()
green_docs = get_green_gtr_docs()
documents = list(green_docs.values())

# %%

# %%
# Import top2vec model
run = runs[0]
top2vec_model = iss_topics.get_top2vec_model(run)
# Get topic labels
wiki_labels = iss_topics.get_wiki_topic_labels(run)

# %%
# Generate visualisation embeddings
xy, umap_model = iss_topics.umap_document_vectors(top2vec_model)

# %%
# Create a dataframe with each project
clustering = iss_topics.get_clustering(run).merge(
    gtr_projects[["project_id", "title"]], left_on="doc_id", right_on="project_id"
)

# Dataframe with the unique topics
cluster_topics = (
    clustering.sort_values("cluster_id")
    .drop_duplicates("cluster_id")[["cluster_id", "cluster_keywords"]]
    .reset_index(drop=True)
)
# Add wiki labels
cluster_topics["wiki_labels"] = cluster_topics["cluster_id"].apply(
    lambda x: wiki_labels[x]
)
# Add wiki labels to the main dataframe
clustering = clustering.merge(cluster_topics[["cluster_id", "wiki_labels"]], how="left")

# Generate visualisation embeddings
clustering["x"] = xy[:, 0]
clustering["y"] = xy[:, 1]

# %%
cluster_topics.head(1)

# %%
clustering.head(1)

# %%
len(np.unique(top2vec_model.doc_top))


# %% [markdown]
# ## Create reduced hierarchy

# %%
# # Check silhouette score for reduced topics
# n_reduced_topics = list(reversed((range(2, top2vec_model.get_num_topics(), 1))))
# silhouette_coeffs = []
# silhouette_coeffs.append(
#     silhouette_score(
#         X=top2vec_model._get_document_vectors(norm=False),
#         labels=cluster_labels,
#         metric="cosine",
#     )
# )
# for n in tqdm(n_reduced_topics, total=len(n_reduced_topics)):
#     # Reduce the number of topics
#     topic_hierarchy = top2vec_model.hierarchical_topic_reduction(n)
#     # Create new labels
#     cluster_labels_reduced = dict(zip(range(len(topic_hierarchy)), topic_hierarchy))
#     base_labels_to_reduced = {
#         c: key for key in cluster_labels_reduced for c in cluster_labels_reduced[key]
#     }
#     clustering_reduced = clustering.copy()
#     clustering_reduced["reduced_cluster_id"] = clustering_reduced.cluster_id.apply(
#         lambda x: base_labels_to_reduced[x]
#     )
#     # Assess 'quality' of clustering
#     silhouette_coeff = silhouette_score(
# #         top2vec_model._get_document_vectors(norm=False),
#         top2vec_model.umap_model.embedding_,
#         clustering_reduced.reduced_cluster_id.to_list(),
#         metric="cosine",
#     )
#     silhouette_coeffs.append(silhouette_coeff)
# n_reduced_topics = [top2vec_model.get_num_topics()] + n_reduced_topics

# %%
# df_silhouette = pd.DataFrame(
#     data={"n_reduced_topics": n_reduced_topics, "silhouette_coeff": silhouette_coeffs}
# )

# iss.show_time_series_fancier(
#     df_silhouette, y="silhouette_coeff", x="n_reduced_topics", show_trend=False
# )

# %%
def topic_keywords(documents, clusters, topic_words, n=10):
    # Create large "cluster documents" for finding best topic words based on tf-idf scores
    document_cluster_memberships = clusters
    cluster_ids = sorted(np.unique(document_cluster_memberships))
    cluster_docs = {i: [] for i in cluster_ids}
    for i, clust in enumerate(document_cluster_memberships):
        cluster_docs[clust] += documents[i]

    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(list(cluster_docs.values()))

    id_to_token = dict(
        zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
    )

    clust_words = []
    for i in range(X.shape[0]):
        x = X[i, :].todense()
        topic_word_counts = [
            X[i, vectorizer.vocabulary_[token]] for token in topic_words[i]
        ]
        best_i = np.flip(np.argsort(topic_word_counts))
        top_n = best_i[0:n]
        words = [topic_words[i][t] for t in top_n]
        clust_words.append(words)
    logging.info(f"Generated keywords for {len(cluster_ids)} topics")
    return clust_words


# %%
def reduce_topics(n, top2vec_model, clustering):
    # Reduce the number of topics
    topic_hierarchy = top2vec_model.hierarchical_topic_reduction(n)
    # Create new labels
    cluster_labels_reduced = dict(zip(range(len(topic_hierarchy)), topic_hierarchy))
    base_labels_to_reduced = {
        c: key for key in cluster_labels_reduced for c in cluster_labels_reduced[key]
    }
    clustering_reduced = clustering.copy()
    clustering_reduced["reduced_cluster_id"] = clustering_reduced.cluster_id.apply(
        lambda x: base_labels_to_reduced[x]
    )
    topic_words, _, _ = top2vec_model.get_topics(reduced=True)
    clust_words = topic_keywords(
        documents, clustering_reduced["reduced_cluster_id"].to_list(), topic_words, n=10
    )
    clust_words = {i: keywords for i, keywords in enumerate(clust_words)}
    clustering_reduced["reduced_cluster_keywords"] = clustering_reduced[
        "reduced_cluster_id"
    ].apply(lambda x: clust_words[x])
    return clustering_reduced


# %%
def reduce_clustering(clustering, reductions):
    clustering_reduced = clustering.copy()
    # Reduce clustering and generate labels
    for reduction in reductions:
        level = reduction["level"]
        n_clusters = reduction["n_clusters"]
        clustering_reduced = reduce_topics(
            n_clusters, top2vec_model, clustering_reduced
        ).rename(
            columns={
                "reduced_cluster_id": f"cluster_id_level_{level}",
                "reduced_cluster_keywords": f"keywords_level_{level}",
            }
        )
    # Rename the original labels
    clustering_reduced = clustering_reduced.rename(
        columns={
            "cluster_id": f"cluster_id_level_{level+1}",
            "cluster_keywords": f"keywords_level_{level+1}",
            "wiki_labels": f"wiki_labels_level_{level+1}",
        }
    )
    return clustering_reduced


# %%
reductions = [{"level": 1, "n_clusters": 10}, {"level": 2, "n_clusters": 50}]

clustering_reduced = reduce_clustering(clustering, reductions)

# %%
clustering_reduced.head(1)

# %%
# Dataframe with the unique topics
cluster_topics = (
    clustering.sort_values("cluster_id")
    .drop_duplicates("cluster_id")[["cluster_id", "cluster_keywords"]]
    .reset_index(drop=True)
)

# %%
importlib.reload(iss_topics)

# %%
counts = iss_topics.get_cluster_counts(clustering_reduced, "cluster_id_level_1")
iss_topics.plot_histogram(counts)


# %% [markdown]
# ### Visualise clusters

# %%

# %% [markdown]
# ### Characterising clusters

# %%
def get_cluster_funding_level(clust, level, clustering, funded_projects):
    clust_proj_ids = clustering[
        clustering[f"cluster_id_level_{level}"] == clust
    ].doc_id.to_list()
    # Rerank based on clustering
    #     clust_proj_ids = clust_proj_ids[
    #         np.flip(np.argsort(model.cluster.probabilities_[clust_proj_ids]))
    #     ]
    df = funded_projects[funded_projects.project_id.isin(clust_proj_ids)].copy()
    cluster_funding = iss.gtr_funding_per_year(df, min_year=2010)
    return cluster_funding


# %%
green_projects = find_green_gtr_projects()

# %%
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
link_gtr_topics = gtr.get_link_table("gtr_topic")

gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(green_projects, gtr_project_funds)
del link_gtr_funds

# %%
project_to_org = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
green_projects.iloc[999]

# %%
n_total = len(cluster_topics)


# %%
def cluster_col_name(level):
    return f"cluster_id_level_{level}"


def keywords_col_name(level):
    return f"keywords_level_{level}"


# %%
def describe_clusters(clustering_reduced, funded_projects, level=3):
    # Columns we will be creating
    data_cols = [
        "funding_2016_2020",
        "funding_growth",
        "n_projects_2016_2020",
        "n_projects_growth",
    ]
    # Dataframe with the unique topics
    cluster_topics = (
        clustering_reduced.sort_values(cluster_col_name(level))
        .drop_duplicates(cluster_col_name(level))[
            [cluster_col_name(level), keywords_col_name(level)]
        ]
        .reset_index(drop=True)
    )
    for col in data_cols:
        cluster_topics[col] = 0

    logging.info(f"Assessing {len(cluster_topics)} level {level} clusters")
    for i, c in enumerate(cluster_topics[cluster_col_name(level)].to_list()):
        cluster_funding = get_cluster_funding_level(
            c, level, clustering_reduced, funded_projects
        )
        cluster_topics.loc[i, "funding_growth"] = iss.estimate_growth_level(
            cluster_funding, growth_rate=True
        )
        cluster_topics.loc[i, "n_projects_growth"] = iss.estimate_growth_level(
            cluster_funding, column="no_of_projects", growth_rate=True
        )
        cluster_topics.loc[i, "n_projects_2016_2020"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].no_of_projects.sum()
        cluster_topics.loc[i, "funding_2016_2020"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].amount_total.sum()

    return cluster_topics


# %%
clusters_level_1 = describe_clusters(clustering_reduced, funded_projects, level=1)
clusters_level_2 = describe_clusters(clustering_reduced, funded_projects, level=2)
clusters_level_3 = describe_clusters(clustering_reduced, funded_projects, level=3)

# %%
clusters_level_3.sort_values("funding_growth")

# %% [markdown]
# ### Produce time series graphs

# %% [markdown]
# ## Mapping businesses to clusters

# %%
import innovation_sweet_spots.getters.green_docs as iss_get_green

importlib.reload(iss_get_green)

# %%
green_cb_corpus_UK, green_cb_orgs_UK = iss_get_green.get_green_cb_docs_by_country()

# %%
importlib.reload(iss_topics)

# %%
top2vec_model_cb = iss_topics.get_top2vec_model("July2021")

# %%
top2vec_model_cb.add_documents(green_cb_corpus_UK, tokenizer="preprocessed")

# %%
dv = top2vec_model_cb._get_document_vectors(norm=False)
cb_vecs = dv[len(green_projects) :, :]
cb_vecs.shape

# %%
import innovation_sweet_spots.analysis.embeddings_utils as iss_emb

# %%
cb_clusts = []
for i in range(len(cb_vecs)):
    closest_docs = iss_emb.find_most_similar_vect(
        cb_vecs[i, :], dv[0 : len(green_projects)], "cosine"
    )
    best_cluster = int(
        clustering_reduced.iloc[closest_docs[0:15]]
        .groupby("cluster_id_level_3")
        .agg(counts=("doc_id", "count"))
        .sort_values("counts")
        .tail(1)
        .index[0]
    )
    cb_clusts.append(best_cluster)

# %%
cb_viz = green_cb_orgs_UK.copy()
cb_viz["cluster_id_level_3"] = cb_clusts
cb_viz = cb_viz.merge(
    clusters_level_3[["cluster_id_level_3", "keywords_level_3"]], how="left"
)

# %%
df = cb_viz[["name", "short_description", "cluster_id_level_3", "keywords_level_3"]]

# %%
pd.set_option("max_colwidth", 200)
df[df.cluster_id_level_3 == 11]

# %%
