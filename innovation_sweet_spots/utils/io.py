"""
Various input output utilities
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from yaml import safe_load
from typing import TypeVar, Iterator
import pathlib
import pickle
import json
import os
import pandas as pd

PathLike = TypeVar("PathLike", str, pathlib.Path, None)

DEF_CONFIG_PATH = PROJECT_DIR / "innovation_sweet_spots/config"


def import_config(filename: str, path=DEF_CONFIG_PATH) -> dict:
    """Load a config .yaml file"""
    with open(path / filename, "r", encoding="utf-8") as yaml_file:
        config_dict = safe_load(yaml_file)
    return config_dict


def save_pickle(data, filepath: PathLike):
    """Pickles data at the provided filepath"""
    with open(filepath, "wb") as outfile:
        pickle.dump(data, outfile)
    logging.info(f"Saved a pickle file in {filepath}")


def load_pickle(filepath: PathLike):
    """Loads a pickle file"""
    with open(filepath, "rb") as infile:
        data = pickle.load(infile)
    logging.info(f"Loaded data from a pickle file in {filepath}")
    return data


def save_text_items(list_of_terms: Iterator[str], filepath: PathLike):
    """Writes a text file with comma-separated text terms"""
    with open(filepath, "w") as outfile:
        outfile.write(", ".join(list_of_terms))
    logging.info(f"Saved {len(list_of_terms)} terms in {filepath}")


def read_text_items(filepath: PathLike) -> Iterator[str]:
    """Reads in a text file with comma-separated text terms"""
    with open(filepath, "r") as infile:
        txt = infile.read()
    list_of_terms = txt.split(", ")
    logging.info(f"Loaded {len(list_of_terms)} text items from {filepath}")
    return list_of_terms


def save_json(data, filepath: PathLike):
    """Saves a dictionary as a json file"""
    with open(filepath, "w") as outfile:
        json.dump(data, outfile, indent=4)


def load_json(filepath: PathLike):
    """Loads a json file as a dictionary"""
    with open(filepath, "r") as infile:
        return json.load(infile)


def get_filenames_in_folder(folder: PathLike, extension: str = None) -> Iterator[str]:
    """Get names of all files in the folder and filter by their extension"""
    return [
        file
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file))
        and ((extension is None) or file.endswith(f".{extension}"))
    ]


def combine_tables_from_folder(
    folder: PathLike, deduplicate_by: Iterator[str] = None
) -> pd.DataFrame:
    """Finds all csv files in a given folder, loads and combines them into one dataframe and deduplicates rows"""
    csv_files = get_filenames_in_folder(folder, "csv")
    return pd.concat(
        [pd.read_csv(str(folder / file)) for file in csv_files]
    ).drop_duplicates(deduplicate_by)
