## topic analysis_utils
import pandas as pd
import numpy as np
import altair as alt
from innovation_sweet_spots import logging, PROJECT_DIR
from innovation_sweet_spots.analysis import top2vec
import umap

umap_args_plotting = {
    "n_neighbors": 15,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 111,
}


def get_clustering(run):
    return pd.read_csv(PROJECT_DIR / f"outputs/data/gtr/top2vec_clusters_{run}.csv")


def get_top2vec_model(run):
    return top2vec.Top2Vec.load(
        PROJECT_DIR / f"outputs/models/top2vec_green_projects_{run}.p"
    )


def get_cluster_counts(clusterings):
    counts = (
        clusterings.groupby("cluster_id")
        .agg(counts=("doc_id", "count"))
        .reset_index()
        .merge(
            clusterings[["cluster_id", "cluster_keywords"]].drop_duplicates(
                "cluster_id"
            ),
            how="left",
        )
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


def plot_clustering(clustering, colour_col="cluster_keywords"):
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
            color=alt.Color(colour_col, scale=alt.Scale(scheme="category20")),
            tooltip=["title", "cluster_id", "cluster_keywords"],
        )
        .interactive()
    )


def umap_document_vectors(top2vec_model):
    umap_model = umap.UMAP(**umap_args_plotting).fit(
        top2vec_model._get_document_vectors(norm=False)
    )
    xy = umap_model.transform(top2vec_model._get_document_vectors(norm=False))
    return xy, umap_model
