from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import get_lookup, save_lookup
from wiki_topic_labels import suggest_labels
from typing import Iterator
import sys
from pathlib import Path

DEF_WIKI_ARGS = {
    "topn": 4,
    "bootstrap_size": 3,
    "boost_with_categories": False,
    "contextual_anchors": [],
}

DEF_ANCHORS = []


def get_wiki_topic_labels(
    lists_of_terms: Iterator[Iterator[str]], wiki_args=DEF_WIKI_ARGS
):
    labels = []
    for i, terms in enumerate(lists_of_terms):
        suggested_labels = suggest_labels(terms, **DEF_WIKI_ARGS)
        labels.append(suggested_labels)
        logging.info(f"({i}) {suggested_labels}")
    return labels


def get_terms(fpath=PROJECT_DIR / "outputs/gtr_green_project_cluster_words"):
    return get_lookup(fpath)


def get_labels(fpath, save=True, outfile=None):

    logging.info(f"Loading terms from {fpath}")

    cluster_terms = get_terms(fpath)

    logging.info(f"Suggesting labels for {len(cluster_terms)} sets of terms")

    term_lists = [d["terms"] for d in cluster_terms]
    labels = get_wiki_topic_labels(term_lists)

    for i, label_set in enumerate(labels):
        cluster_terms[i]["labels"] = label_set

    if save:
        if outfile is None:
            outfile = f"{fpath.parent / fpath.stem}_wiki_labels.json"
        save_lookup(cluster_terms, outfile)
        logging.info(f"Saved labels in {outfile}")

    return cluster_terms


if __name__ == "__main__":
    # Check if path has been provided
    if len(sys.argv) > 1:
        fpath = Path(sys.argv[1])
    else:
        raise FileNotFoundError("Provide a path to the input file")
    # Suggest labels
    get_labels(fpath)
