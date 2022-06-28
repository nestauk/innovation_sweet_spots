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
from zipfile import ZipFile

PathLike = TypeVar("PathLike", str, pathlib.Path, None)

DEF_CONFIG_PATH = PROJECT_DIR / "innovation_sweet_spots/config"
VERBOSE = True


def import_config(filename: str, path=DEF_CONFIG_PATH) -> dict:
    """Load a config .yaml file"""
    with open(path / filename, "r", encoding="utf-8") as yaml_file:
        config_dict = safe_load(yaml_file)
    return config_dict


def save_pickle(data, filepath: PathLike):
    """Pickles data at the provided filepath"""
    with open(filepath, "wb") as outfile:
        pickle.dump(data, outfile)
    if VERBOSE:
        logging.info(f"Saved a pickle file in {filepath}")


def load_pickle(filepath: PathLike):
    """Loads a pickle file"""
    with open(filepath, "rb") as infile:
        data = pickle.load(infile)
    if VERBOSE:
        logging.info(f"Loaded data from a pickle file in {filepath}")
    return data


def save_text_items(list_of_terms: Iterator[str], filepath: PathLike):
    """Writes a text file with comma-separated text terms"""
    with open(filepath, "w") as outfile:
        outfile.write(", ".join(list_of_terms))
    if VERBOSE:
        logging.info(f"Saved {len(list_of_terms)} terms in {filepath}")


def read_text_items(filepath: PathLike) -> Iterator[str]:
    """Reads in a text file with comma-separated text terms"""
    with open(filepath, "r") as infile:
        txt = infile.read()
    list_of_terms = txt.split(", ")
    if VERBOSE:
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
    """
    Finds all csv files in a given folder, loads and combines them
    into one dataframe and deduplicates rows. NB: It is assumed that
    all tables have the same column names

    Args:
        folder: Location of the csv files
        deduplicate_by: List of column names to use for deduplication

    Returns:
        Dataframe with the combined table
    """
    csv_files = get_filenames_in_folder(folder, "csv")
    return pd.concat(
        [pd.read_csv(str(folder / file)) for file in csv_files]
    ).drop_duplicates(deduplicate_by)


def download_file(url: str, fpath: PathLike):
    """Downloads a file from url and unzips (if archived) in the local fpath"""
    filename = url.split("/")[-1]
    with BytesIO() as fileobj:
        stream_to_file(url, fileobj)
        extract_to_disk = (
            extract_from_zip
            if filename.endswith(".xxx")
            else (lambda obj, path: bytesio_to_file(obj, path / filename))
        )
        extract_to_disk(fileobj, fpath)


def unzip_files(
    path_to_zip_archive: PathLike, extract_path: PathLike, delete: bool = False
):
    """Extracts the zip file and optionally deletes it"""
    logging.info(f"Extracting the archive {path_to_zip_archive}")
    with ZipFile(path_to_zip_archive, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    if delete:
        os.remove(path_to_zip_archive)
