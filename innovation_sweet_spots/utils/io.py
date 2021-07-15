# Generic read / save functions

import json

from innovation_sweet_spots import PROJECT_DIR, logging
from typing import Iterator
import numpy as np
import pickle


def save_lookup(name, path_name):
    with open(path_name, "w") as outfile:
        json.dump(name, outfile, indent=4)


def get_lookup(path_name):
    with open(path_name, "r") as infile:
        return json.load(infile)


def save_list_of_terms(list_of_terms: Iterator[str], fpath):
    """Writes a text file with comma-separated terms"""
    with open(fpath, "w") as outfile:
        outfile.write(", ".join(sorted(list_of_terms)))
    logging.info(f"Saved {len(list_of_terms)} terms in {fpath}")


def read_list_of_terms(fpath):
    with open(fpath, "r") as infile:
        txt = infile.read()
    return sorted(txt.split(", "))


def save_array(embeddings: np.ndarray, fpath):
    np.save(embeddings, fpath)
    logging.info(f"Saved array of shape {embeddings.shape} in {fpath}")


def load_array(fpath):
    embeddings = np.load(fpath)
    logging.info(f"Loaded in array of shape {embeddings.shape}")
    return embeddings


def save_pickle(data, fpath):
    with open(fpath, "wb") as outfile:
        pickle.dump(data, outfile)
    logging.info(f"Saved a pickle file in {fpath}")


def load_pickle(fpath):
    with open(fpath, "rb") as infile:
        data = pickle.load(infile)
    logging.info(f"Loaded data from a pickle file in {fpath}")
    return data
