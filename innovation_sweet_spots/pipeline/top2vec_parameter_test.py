from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.getters.green_docs import get_green_gtr_docs
from innovation_sweet_spots.analysis import top2vec
from innovation_sweet_spots.utils.io import save_pickle

from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd
import sys
import seaborn as sns
from matplotlib import pyplot as plt

# Baseline parameter sets for testing
UMAP_ARGS = {"n_neighbors": 15, "n_components": 5, "metric": "cosine"}

HDBSCAN_ARGS = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

DOC2VEC_ARGS = {
    "vector_size": 300,
    "min_count": 10,
    "window": 15,
    "sample": 1e-5,
    "negative": 0,
    "hs": 1,
    "epochs": 50,
    "dm": 0,
    "dbow_words": 1,
    "workers": 1,
    "corpus_file": None,
}

# Testing parameters
SEED = 999
N = 10
TEST_EPOCHS = [25, 50]
TEST_NEIGHBOURS = [10, 15, 20, 30, 50]
TEST_MIN_CLUSTER_SIZE = [10, 15, 20, 30, 50]

RESULTS_DIR = PROJECT_DIR / "outputs/evaluations/top2vec_parameter_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_SUFFIX = ""


def train_evaluate_model(
    documents, doc2vec_args, umap_args, hdbscan_args, random_state
):
    # Train the model
    top2vec_model = top2vec.Top2Vec(
        documents=documents,
        speed="None",
        tokenizer="preprocessed",
        doc2vec_args=doc2vec_args,
        umap_args=umap_args,
        hdbscan_args=hdbscan_args,
        random_state=random_state,
    )
    # Evaluate
    cluster_labels, silhouette_doc2vec, silhouette_umap = evaluate_model(top2vec_model)
    return cluster_labels, silhouette_doc2vec, silhouette_umap


def evaluate_model(top2vec_model):
    # Get the cluster labels
    cluster_labels = top2vec_model.doc_top
    # Calculate silhouette coefficients
    if len(np.unique(cluster_labels)) > 1:
        silhouette_doc2vec = silhouette_score(
            top2vec_model._get_document_vectors(norm=False), cluster_labels
        )
        silhouette_umap = silhouette_score(
            top2vec_model.umap_model.embedding_, cluster_labels
        )
    else:
        silhouette_doc2vec = np.nan
        silhouette_umap = np.nan
    return cluster_labels, silhouette_doc2vec, silhouette_umap


def setup_result_arrays(n_params, n_runs):
    # Arrays for storing outputs
    silhouette_doc2vec = np.zeros((n_params, n_runs))
    silhouette_umap = np.zeros((n_params, n_runs))
    cluster_ami = []
    epoch_ami = []
    n_clusters = np.zeros((n_params, n_runs))
    return silhouette_doc2vec, silhouette_umap, cluster_ami, epoch_ami, n_clusters


def test_number_of_epochs(documents, test_epochs=TEST_EPOCHS, n=N):
    """
    Performs n runs of top2vec for each value of epochs, and
    checks the silhouette score for each run, and adjusted mutual information
    across runs
    """
    logging.info(f"Testing model with respect to different number of epochs")
    np.random.seed(SEED)
    random_states = [np.random.randint(1e3) for i in range(n)]

    # Arrays for storing outputs
    (
        silhouette_doc2vec,
        silhouette_umap,
        cluster_ami,
        epoch_ami,
        n_clusters,
    ) = setup_result_arrays(len(test_epochs), n)

    # Run loops across epochs and random states
    for j, epochs in enumerate(test_epochs):
        doc2vec_args = DOC2VEC_ARGS.copy()
        doc2vec_args["epochs"] = epochs
        logging.info(f"Building {n} models with {epochs} epochs")
        cluster_labels_all_params = []
        for i, random_state in enumerate(random_states):
            cluster_labels, s_doc2vec, s_umap = train_evaluate_model(
                documents, doc2vec_args, UMAP_ARGS, HDBSCAN_ARGS, random_state
            )
            cluster_labels_all_params.append(cluster_labels)
            n_clusters[j, i] = n_clust = len(np.unique(cluster_labels))
            silhouette_doc2vec[j, i] = s_doc2vec
            silhouette_umap[j, i] = s_umap
            n_clusters[j, i] = len(np.unique(cluster_labels))
        ami_avg, ami_values = ensemble_AMI(
            cluster_labels_all_params, return_matrix=False
        )
        cluster_ami.append(ami_values)
        epoch_ami.append(ami_avg)

    results = {
        "test_param": "epochs",
        "epochs": test_epochs,
        "random_states": random_states,
        "silhouette_doc2vec": silhouette_doc2vec,
        "silhouette_umap": silhouette_umap,
        "cluster_ami": np.array(cluster_ami),
        "epoch_ami": np.array(epoch_ami).reshape(-1, 1),
        "n_clusters": n_clusters,
        "session_suffix": SESSION_SUFFIX,
    }
    save_pickle(
        results, RESULTS_DIR / f"top2vec_parameter_test_epochs{SESSION_SUFFIX}.p"
    )
    batch_plot_evaluation(results)
    return results


def test_umap_neighbours(documents, test_params=TEST_NEIGHBOURS, n=N):
    """
    For a given set of doc2vec vectors, performs n runs of umap model training
    for each value of n_neighbors, and checks the silhouette score for each run,
    and adjusted mutual information across runs
    """
    logging.info(
        f"Testing model with respect to different number of UMAP nearest neighbours"
    )
    np.random.seed(SEED)
    random_states = [np.random.randint(1e3) for i in range(n)]
    # Arrays for storing outputs
    (
        silhouette_doc2vec,
        silhouette_umap,
        cluster_ami,
        epoch_ami,
        n_clusters,
    ) = setup_result_arrays(len(test_params), n)
    # Initial model
    top2vec_model = top2vec.Top2Vec(
        documents=documents,
        speed="None",
        tokenizer="preprocessed",
        doc2vec_args=DOC2VEC_ARGS,
        umap_args=UMAP_ARGS,
        hdbscan_args=HDBSCAN_ARGS,
        random_state=random_states[0],
    )
    # Run loops across epochs and random states
    for j, n_neighbors in enumerate(test_params):
        top2vec_model.umap_args["n_neighbors"] = n_neighbors
        logging.info(f"Building {n} models with {n_neighbors} nearest neighbors")

        cluster_labels_all_params = []
        for i, random_state in enumerate(random_states):
            top2vec_model.umap_seed = random_state
            top2vec_model.generate_umap_model()
            top2vec_model.cluster_docs()
            top2vec_model.process_topics()
            cluster_labels, s_doc2vec, s_umap = evaluate_model(top2vec_model)
            cluster_labels_all_params.append(cluster_labels)
            n_clusters[j, i] = n_clust = len(np.unique(cluster_labels))
            silhouette_doc2vec[j, i] = s_doc2vec
            silhouette_umap[j, i] = s_umap
            n_clusters[j, i] = len(np.unique(cluster_labels))
        ami_avg, ami_values = ensemble_AMI(
            cluster_labels_all_params, return_matrix=False
        )
        cluster_ami.append(ami_values)
        epoch_ami.append(ami_avg)

    results = {
        "test_param": "n_neighbors",
        "n_neighbors": test_params,
        "random_states": random_states,
        "silhouette_doc2vec": silhouette_doc2vec,
        "silhouette_umap": silhouette_umap,
        "cluster_ami": np.array(cluster_ami),
        "epoch_ami": np.array(epoch_ami).reshape(-1, 1),
        "n_clusters": n_clusters,
        "session_suffix": SESSION_SUFFIX,
    }
    save_pickle(
        results,
        RESULTS_DIR
        / f'top2vec_parameter_test_{results["test_param"]}{SESSION_SUFFIX}.p',
    )
    batch_plot_evaluation(results)
    return results


def test_min_cluster_size(documents, test_params=TEST_MIN_CLUSTER_SIZE, n=N):
    """
    For a given set of doc2vec vectors and umap embeddings, performs n runs of
    clusterings
    """
    logging.info(
        f"Testing model with respect to different number of UMAP nearest neighbours"
    )
    np.random.seed(SEED)
    random_states = [np.random.randint(1e3) for i in range(n)]
    # Arrays for storing outputs
    (
        silhouette_doc2vec,
        silhouette_umap,
        cluster_ami,
        epoch_ami,
        n_clusters,
    ) = setup_result_arrays(len(test_params), n)
    # Initial model
    top2vec_model = top2vec.Top2Vec(
        documents=documents,
        speed="None",
        tokenizer="preprocessed",
        doc2vec_args=DOC2VEC_ARGS,
        umap_args=UMAP_ARGS,
        hdbscan_args=HDBSCAN_ARGS,
        random_state=random_states[0],
    )
    # Run loops across epochs and random states
    for j, min_cluster_size in enumerate(test_params):
        top2vec_model.hdbscan_args["min_cluster_size"] = min_cluster_size
        logging.info(
            f"Building {n} models with {min_cluster_size} minimal cluster size"
        )

        cluster_labels_all_params = []
        for i, random_state in enumerate(random_states):
            top2vec_model.hdbscan_seed = random_state
            top2vec_model.cluster_docs()
            top2vec_model.process_topics()
            cluster_labels, s_doc2vec, s_umap = evaluate_model(top2vec_model)
            cluster_labels_all_params.append(cluster_labels)
            n_clusters[j, i] = n_clust = len(np.unique(cluster_labels))
            silhouette_doc2vec[j, i] = s_doc2vec
            silhouette_umap[j, i] = s_umap
            n_clusters[j, i] = len(np.unique(cluster_labels))
        ami_avg, ami_values = ensemble_AMI(
            cluster_labels_all_params, return_matrix=False
        )
        cluster_ami.append(ami_values)
        epoch_ami.append(ami_avg)

    results = {
        "test_param": "min_cluster_size",
        "min_cluster_size": test_params,
        "random_states": random_states,
        "silhouette_doc2vec": silhouette_doc2vec,
        "silhouette_umap": silhouette_umap,
        "cluster_ami": np.array(cluster_ami),
        "epoch_ami": np.array(epoch_ami).reshape(-1, 1),
        "n_clusters": n_clusters,
        "session_suffix": SESSION_SUFFIX,
    }
    save_pickle(
        results,
        RESULTS_DIR
        / f'top2vec_parameter_test_{results["test_param"]}{SESSION_SUFFIX}.p',
    )
    batch_plot_evaluation(results)


def ensemble_AMI(P, verbose: bool = True, return_matrix=True):
    """
    Calculates pairwise adjusted mutual information (AMI) scores across
    the clustering ensemble.
    Parameters
    ----------
    P (list of lists of int):
        Clustering ensemble, i.e., a list of clustering results, where each
        clustering result is a list of integers. These integers correspond
        to cluster labels.
    verbose (boolean):
        Determines whether information about the results is printed.
    Returns
    -------
    ami_avg (float):
        Average adjusted mutual information across the ensemble
    ami_matrix (numpy.ndarray):
        The complete matrix with adjusted mutual information scores between
        all pairs of clustering results
    """
    ami_matrix = np.zeros((len(P), len(P)))
    for i in range(0, len(P)):
        for j in range(i, len(P)):
            ami_matrix[i][j] = ami_score(P[i], P[j], average_method="arithmetic")

    ami_matrix += np.triu(ami_matrix).T
    np.fill_diagonal(ami_matrix, 1)
    ami_avg = np.mean(ami_matrix[np.triu_indices_from(ami_matrix, k=1)])

    if not return_matrix:
        ami_matrix = ami_matrix[np.triu_indices_from(ami_matrix, k=1)]

    if verbose:
        logging.info(
            f"Average pairwise AMI across {len(P)} partitions is {np.round(ami_avg,4)}"
        )
    return ami_avg, ami_matrix


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
        plt.savefig(fpath, format="png", dpi=300)
        logging.info(fpath)


def batch_plot_evaluation(results):
    param = results["test_param"]
    eval_metrics = [
        "silhouette_doc2vec",
        "silhouette_umap",
        "cluster_ami",
        "epoch_ami",
        "n_clusters",
    ]
    for eval_metric in eval_metrics:
        plot_evaluation(create_evaluation_table(results, param, eval_metric), save=True)


if __name__ == "__main__":
    # Check if session name has been provided
    if len(sys.argv) > 1:
        SESSION_SUFFIX = f"_{sys.argv[1]}"
    documents = list(get_green_gtr_docs().values())
    test_number_of_epochs(documents)
    test_umap_neighbours(documents)
    test_min_cluster_size(documents)
