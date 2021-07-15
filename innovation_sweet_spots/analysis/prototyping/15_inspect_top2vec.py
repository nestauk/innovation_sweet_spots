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

# %%
from innovation_sweet_spots.getters.green_docs import get_green_gtr_docs

# %%
import importlib

importlib.reload(iss_topics)

# %%
from sklearn.metrics import adjusted_mutual_info_score as ami_score
import altair as alt

# %%
runs = [
    "epochs50",
    "epochs50_v2",
    "epochs50_leaf",
    "epochs50_leaf_v2",
    "epochs50_leaf_v2_repeat",
]
runs = ["test_1", "test_2"]
clusterings = [iss_topics.get_clustering(run) for run in runs]
clusterings[0].head(1)

# %%
print(ami_score(clusterings[0].cluster_id, clusterings[1].cluster_id))
# print(ami_score(clusterings[2].cluster_id, clusterings[3].cluster_id))
# print(ami_score(clusterings[1].cluster_id, clusterings[3].cluster_id))
# print(ami_score(clusterings[2].cluster_id, clusterings[4].cluster_id))

# %%
i = 2
counts = iss_topics.get_cluster_counts(clusterings[i])
iss_topics.plot_histogram(counts)

# %% [markdown]
# ## Visualise

# %%
from innovation_sweet_spots.getters import gtr

alt.data_transformers.disable_max_rows()

# %%
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

# %%
importlib.reload(iss_topics)

# %%
gtr_projects = gtr.get_gtr_projects()

# %%
green_docs = get_green_gtr_docs()
documents = list(green_docs.values())

# %%
run = runs[2]
top2vec_model = iss_topics.get_top2vec_model(run)

# %%
xy, umap_model = iss_topics.umap_document_vectors(top2vec_model)

# %%
clustering = get_clustering(run).merge(
    gtr_projects[["project_id", "title"]], left_on="doc_id", right_on="project_id"
)
clustering["x"] = xy[:, 0]
clustering["y"] = xy[:, 1]

# %%
cluster_topics = (
    clustering.sort_values("cluster_id")
    .drop_duplicates("cluster_id")[["cluster_id", "cluster_keywords"]]
    .reset_index(drop=True)
)

# %%
clustering.head(1)

# %%
# iss_topics.plot_clustering(clustering)

# %%
len(np.unique(top2vec_model.doc_top))

# %%
cluster_labels = clustering.cluster_id.to_list()
# silhouette_umap = silhouette_score(top2vec_model.umap_model.embedding_,cluster_labels)
silhouette_coeff = silhouette_score(
    top2vec_model._get_document_vectors(norm=False), cluster_labels, metric="cosine"
)

# %%
top2vec_model.get_num_topics()

# %%
from tqdm.notebook import tqdm

# %%
n_reduced_topics = list(reversed((range(10, top2vec_model.get_num_topics(), 10)))) + [5]
silhouette_coeffs = []
silhouette_coeffs.append(
    silhouette_score(
        X=top2vec_model._get_document_vectors(norm=False),
        labels=cluster_labels,
        metric="cosine",
    )
)
for n in tqdm(n_reduced_topics, total=len(n_reduced_topics)):
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
    # Assess 'quality' of clustering
    silhouette_coeff = silhouette_score(
        top2vec_model._get_document_vectors(norm=False),
        clustering_reduced.reduced_cluster_id.to_list(),
        metric="cosine",
    )
    silhouette_coeffs.append(silhouette_coeff)
n_reduced_topics = [top2vec_model.get_num_topics()] + n_reduced_topics

# %%
df_silhouette = pd.DataFrame(
    data={"n_reduced_topics": n_reduced_topics, "silhouette_coeff": silhouette_coeffs}
)

# %%
iss.show_time_series_fancier(
    df_silhouette, y="silhouette_coeff", x="n_reduced_topics", show_trend=False
)


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
df = reduce_topics(10, top2vec_model, clustering)
df

# %% [markdown]
# ### Mapping businesses to clusters

# %%
uk_green_corpus = [corpus_cb[i] for i in uk_green_cb.index]
uk_green_company_texts = [green_company_texts[i] for i in uk_green_cb.index]

# %%
model.add_documents(uk_green_company_texts, tokenizer=bigrammer)

# %%
dv = model._get_document_vectors(norm=False)
