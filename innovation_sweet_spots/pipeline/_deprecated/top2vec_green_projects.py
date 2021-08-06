from innovation_sweet_spots import PROJECT_DIR, logging, config
from innovation_sweet_spots.getters.green_docs import get_green_gtr_docs
from innovation_sweet_spots.analysis import top2vec
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import json
from innovation_sweet_spots.analysis.green_document_utils import find_green_gtr_projects

# Ensure reproducibility
import os
import sys

hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

SEED = 11111
SESSION_SUFFIX = "July2021"

RESULTS_DIR = PROJECT_DIR / "outputs/models"

UMAP_ARGS = {"n_neighbors": 25, "n_components": 5, "metric": "cosine"}

HDBSCAN_ARGS = {
    "min_cluster_size": 20,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
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

TOP_N_WORDS = 10


def topic_keywords(documents, top2vec_model, n=TOP_N_WORDS):
    # Create large "cluster documents" for finding best topic words based on tf-idf scores
    document_cluster_memberships = top2vec_model.doc_top
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

    topic_words, word_scores, topic_nums = top2vec_model.get_topics(
        top2vec_model.get_num_topics()
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


if __name__ == "__main__":
    SESSION_SUFFIX = "_" + SESSION_SUFFIX
    green_docs = get_green_gtr_docs()
    documents = list(green_docs.values())
    logging.info(f"Training top2vec model with {len(documents)} documents")

    # Train the topic model
    top2vec_model = top2vec.Top2Vec(
        documents=documents,
        speed="None",
        tokenizer="preprocessed",
        doc2vec_args=DOC2VEC_ARGS,
        umap_args=UMAP_ARGS,
        hdbscan_args=HDBSCAN_ARGS,
        random_state=SEED,
    )

    # Save the topic model
    fpath = RESULTS_DIR / f"top2vec_green_projects{SESSION_SUFFIX}.p"
    top2vec_model.save(fpath)
    logging.info(f"Saved the top2vec model to {fpath}")

    # Generate cluster topic keywords
    clust_words = topic_keywords(documents, top2vec_model)
    cluster_keywords = pd.DataFrame(
        data={
            "cluster_keywords": clust_words,
            "cluster_id": list(range(len(clust_words))),
        }
    )

    # Generate and save table with documents and keywords
    result_table = pd.DataFrame(
        data={
            "doc_id": list(green_docs.keys()),
            "cluster_id": top2vec_model.doc_top,
        }
    ).merge(cluster_keywords, how="left")
    fpath = PROJECT_DIR / f"outputs/data/gtr/top2vec_clusters{SESSION_SUFFIX}.csv"
    result_table.to_csv(fpath, index=False)
    logging.info(f"Saved a table with document clusters in {fpath}")

    # Save file for generating wiki labels
    keyword_dict = (
        cluster_keywords.rename(
            columns={"cluster_id": "id", "cluster_keywords": "terms"}
        )
    ).T.to_dict()
    keyword_list = [keyword_dict[key] for key in keyword_dict]
    json.dump(
        keyword_list,
        open(
            PROJECT_DIR
            / f"outputs/gtr_green_project_cluster_words{SESSION_SUFFIX}.json",
            "w",
        ),
        indent=4,
    )
