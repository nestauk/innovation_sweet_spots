"""
Various utils for drawing plots
"""
import umap
import altair as alt
import numpy as np
import pandas as pd

# Distinct palette of colors
COLOUR_PAL = [
    "#000075",
    "#e6194b",
    "#3cb44b",
    "#f58231",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#DCDCDC",
    "#9a6324",
    "#800000",
    "#808000",
    "#ffe119",
    "#46f0f0",
    "#4363d8",
    "#911eb4",
    "#aaffc3",
    "#000000",
    "#ffd8b1",
    "#808000",
    "#000075",
    "#DCDCDC",
]

RANDOM_STATE = 111
N_NEIGHBOURS = 20
MIN_DIST = 0.01
N_DIM = 2


def default_umap_viz(vectors: np.ndarray):
    reducer = umap.UMAP(
        random_state=RANDOM_STATE,
        n_neighbors=N_NEIGHBOURS,
        min_dist=MIN_DIST,
        n_components=N_DIM,
    )
    low_dim_embeddings = reducer.fit_transform(vectors)
    return low_dim_embeddings


def viz_dataframe(df: pd.DataFrame, vectors: np.ndarray, xy=None):
    if xy is None:
        xy = default_umap_viz(vectors)
    df_viz = df.copy()
    df_viz["x"] = xy[:, 0]
    df_viz["y"] = xy[:, 1]
    return df_viz
