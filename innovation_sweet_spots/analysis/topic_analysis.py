## topic analysis_utils
import pandas as pd
import numpy as np
import altair as alt
from innovation_sweet_spots import logging, PROJECT_DIR
from innovation_sweet_spots.analysis import top2vec
import umap
import json
from innovation_sweet_spots.getters import gtr, crunchbase

umap_args_plotting = {
    "n_neighbors": 15,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 112,
}


def get_doc_details(clustering):
    gtr_projects = gtr.get_gtr_projects()[
        ["project_id", "title", "abstractText"]
    ].rename(
        columns={
            "project_id": "doc_id",
            "title": "title",
            "abstractText": "description",
        }
    )
    cb_orgs = (
        crunchbase.get_crunchbase_orgs_full()[["id", "name", "short_description"]]
        .drop_duplicates("id")
        .rename(
            columns={
                "id": "doc_id",
                "name": "title",
                "short_description": "description",
            }
        )
    )
    if "source" not in clustering.columns:
        green_country_orgs["source"] = "cb"
        gtr_projects["source"] = "gtr"
    combined_df = pd.concat([gtr_projects, cb_orgs], axis=0, ignore_index=True)
    clustering_details = clustering.merge(combined_df, on="doc_id", how="left")
    del combined_df
    return clustering_details


def get_wiki_topic_labels(run):
    clust_labels = json.load(
        open(
            PROJECT_DIR
            / f"outputs/gtr_green_project_cluster_words_{run}_wiki_labels.json",
            "r",
        )
    )
    clust_labels = {d["id"]: d["labels"] for d in clust_labels}
    return clust_labels


def get_clustering(run):
    return pd.read_csv(PROJECT_DIR / f"outputs/data/gtr/top2vec_clusters_{run}.csv")


def get_top2vec_model(run):
    return top2vec.Top2Vec.load(
        PROJECT_DIR / f"outputs/models/top2vec_green_projects_{run}.p"
    )


def get_cluster_counts(clusterings, cluster_col="cluster_id"):
    counts = (
        clusterings.groupby(cluster_col)
        .agg(counts=("doc_id", "count"))
        .reset_index()
        # .merge(
        #     clusterings[["cluster_id", "cluster_keywords"]].drop_duplicates(
        #         "cluster_id"
        #     ),
        #     how="left",
        # )
    )
    return counts


def plot_histogram(df, x="counts", bin_step=10):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(f"{x}:Q", bin=alt.Bin(step=bin_step)),
            y="count()",
        )
    )


def plot_clustering(
    clustering, colour_col="cluster_keywords", tooltip=None, shape="source"
):
    if tooltip is None:
        tooltip = ["title", colour_col]
    return (
        alt.Chart(
            clustering,
            width=750,
            height=750,
        )
        .mark_circle(size=25)
        .encode(
            x=alt.X("x", axis=alt.Axis(grid=False)),
            y=alt.Y("y", axis=alt.Axis(grid=False)),
            #         size='size',
            #         color='cluster',
            shape=shape,
            color=alt.Color(colour_col, scale=alt.Scale(scheme="category20")),
            tooltip=tooltip,
        )
        .interactive()
    )


def umap_document_vectors(top2vec_model):
    umap_model = umap.UMAP(**umap_args_plotting).fit(
        top2vec_model._get_document_vectors(norm=False)
    )
    xy = umap_model.transform(top2vec_model._get_document_vectors(norm=False))
    return xy, umap_model


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
