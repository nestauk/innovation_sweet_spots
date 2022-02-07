"""
Various input output utilities
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from yaml import safe_load
from typing import TypeVar
import pathlib
import pickle

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
