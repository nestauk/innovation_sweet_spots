# Generic read / save functions

import json

from innovation_sweet_spots import PROJECT_DIR, logging
from typing import Iterator
import numpy as np


def save_lookup(name, path_name):
    with open(f"{PROJECT_DIR}/{path_name}.json", "w") as outfile:
        json.dump(name, outfile)


def get_lookup(path_name):
    with open(f"{PROJECT_DIR}/{path_name}.json", "r") as infile:
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
