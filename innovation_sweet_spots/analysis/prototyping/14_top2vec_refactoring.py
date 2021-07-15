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

# %%
from innovation_sweet_spots.getters.green_docs import get_green_gtr_docs
from innovation_sweet_spots.analysis import top2vec
from innovation_sweet_spots import PROJECT_DIR, logging

# %%
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import silhouette_score

# %%
from innovation_sweet_spots.utils.io import load_pickle

# %%
import importlib

# %%
corpus_gtr = get_green_gtr_docs()

# %%
importlib.reload(top2vec)

# %%
UMAP_ARGS = {"n_neighbors": 15, "n_components": 5, "metric": "cosine"}

HDBSCAN_ARGS = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

DEFAULT_DOC2VEC_ARGS = {
    "vector_size": 300,
    "min_count": 10,
    "window": 15,
    "sample": 1e-5,
    "negative": 0,
    "hs": 1,
    "epochs": 1,
    "dm": 0,
    "dbow_words": 1,
    "workers": 1,
    "corpus_file": None,
}

RANDOM_STATE = 111

# %%
RANDOM_STATES = [284, 270, 418, 21, 566, 712, 37, 972, 742, 698]

# %%
top2vec_model = top2vec.Top2Vec(
    documents=docs_to_train,
    speed="test-learn",
    tokenizer="preprocessed",
    doc2vec_args=DOC2VEC_ARGS,
    umap_args=UMAP_ARGS,
    hdbscan_args=HDBSCAN_ARGS,
    random_state=RANDOM_STATE,
)

# %%
d = pd.DataFrame(data={"id": [1, 2], "terms": [["a", "b"], ["c", "d"]]}).T.to_dict()
[d[key] for key in d]

# %%
top2vec_model.doc

# %%
top2vec_model.save("test.p")

# %%
mod = top2

# %%
mod = top2vec.Top2Vec.load(
    "/Users/karliskanders/Documents/innovation_sweet_spots/outputs/models/top2vec_green_projects.p"
)

# %%
len(mod.doc_top)

# %%
vectors = top2vec_model._get_document_vectors(norm=False)

# %%
silhouette_score(vectors, top2vec_model.doc_top, random_state=random_state)

# %%
silhouette_score(
    top2vec_model.umap_model.embedding_,
    top2vec_model.doc_top,
    random_state=random_state,
)

# %%
from datetime import date

today = date.today()
today.strftime("%Y_%m_%d")

# %%
silhouette_avg = silhouette_score(X, cluster_labels)

# %%
X = top2vec_model.umap_model.embedding_
cluster_labels = top2vec_model.doc_top

# %%
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print(
    "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
)

# %%
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        (
            "Silhouette analysis for KMeans clustering on sample data "
            "with n_clusters = %d" % n_clusters
        ),
        fontsize=14,
        fontweight="bold",
    )

plt.show()

# %% [markdown]
# # Evaluate

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# %%
RESULTS_DIR = PROJECT_DIR / "outputs/evaluations"
SESSION_SUFFIX = ""
results = load_pickle(RESULTS_DIR / f"top2vec_parameter_test_epochs{SESSION_SUFFIX}.p")

# %%
results["epoch_ami"] = results["epoch_ami"].reshape(-1, 1)

# %%
results["silhouette_umap"].shape

# %%
# results.keys()

# %%
# results['silhouette_umap']

# %%
# results_matrix[0,:]

# %%
param = "test_epochs"
eval_metric = "silhouette_doc2vec"


# %%
def create_evaluation_table(results, param, eval_metric):
    results_matrix = results[eval_metric]
    cols = [str(p) for p in results[param]]
    results_df = (
        pd.DataFrame(
            results_matrix.T, columns=cols, index=range(results_matrix.shape[1])
        )
        .melt(value_vars=cols)
        .rename(columns={"variable": param, "value": eval_metric})
    )
    return results_df


def plot_evaluation(results_df, save=False):
    param = results_df.columns[0]
    eval_metric = results_df.columns[1]
    plt.figure(figsize=(8, 4))
    if len(results_df[param].unique()) != len(results_df):
        sns.violinplot(y=eval_metric, x=param, data=results_df, color="gray")
    sns.swarmplot(y=eval_metric, x=param, data=results_df, color="k", edgecolor="white")
    if save:
        fpath = RESULTS_DIR / f"top2vec_parameter_{param}_{eval_metric}.png"
        plt.savefig(fpath, format="png", dpi=150)
        logging.info(fpath)


# %%
results.keys()

# %%
results["epoch_ami"]

# %%
results["eval_metric"]

# %%
param = "epochs"
eval_metric = "epoch_ami"
create_evaluation_table(results, param, eval_metric)

# %%
param = "epochs"
# eval_metric = 'silhouette_doc2vec'
# eval_metric = 'silhouette_umap'
# eval_metric = 'n_clusters'
eval_metric = "cluster_ami"
plot_evaluation(create_evaluation_table(results, param, eval_metric), save=True)

# %%

# %%
