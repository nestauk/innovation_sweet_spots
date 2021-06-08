"""
Modules for collecting data analysis inputs.

To fetch all inputs, simply run "python inputs.py"
"""
from innovation_sweet_spots import PROJECT_DIR, logging, db_config_path
from pathlib import Path
from data_getters.gtr import build_projects
from data_getters.core import get_engine
from pandas import read_sql_table, read_csv, concat
from yaml import safe_load
import datetime
import os
import json
import zipfile
import urllib

INPUTS_PATH = PROJECT_DIR / "inputs/data/"
GTR_PATH = INPUTS_PATH / "gtr"
CB_PATH = INPUTS_PATH / "cb"
CB_DATA_SPEC_PATH = PROJECT_DIR / "innovation_sweet_spots/config/cb_data_spec.yaml"
HANSARD_PATH = INPUTS_PATH / "hansard"
ZENODO_BASE = "https://zenodo.org/record/4843485/files/{}?download=1"
ZENODO_FILES = ["hansard-speeches-v310.csv.zip", "parliamentary_posts.json"]
ZENODO_URLS = map(ZENODO_BASE.format, ZENODO_FILES)


def get_gtr_projects(
    fpath=GTR_PATH / "gtr_projects.json", fields=["id"], use_cached=True
):
    """
    Downloads GTR projects from Nesta database and stores them locally.
    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_gtr_projects; get_gtr_projects(use_cached=False);"

    Note that in the present implementation, the default value for non-serializable
    objects is datetime.datetime.isoformat, which means that pandas._libs.tslibs.nattype.NaTType objects
    will become "0001-01-01T00:00:00"

    The other non-serializable type in the GTR data is pandas._libs.tslibs.timestamps.Timestamp,
    which should have no issue with being formated in isoformat

    Parameters
    ----------
    fpath : str
        Location on disk for saving the projects table
    fields : list of str
        Use the default value; for additional functionality see 'data_getters' documentation
    use_cached: bool
        If use_cached=True, the function will load in the local version of the dataset, if possible;
        set use_cached=False to download (and overwrite the existing) data.


    Returns
    -------
    projects : list of dict
        List of dictionaries with project data
    """
    use_cached = use_cached and os.path.exists(fpath)
    if not use_cached:
        logging.info(f"Collection of GTR projects in progress")
        projects = build_projects(
            config_path=db_config_path,
            chunksize=5000,
            table_wildcards=["gtr_projects"],
            desired_fields=fields,
        )
        with open(fpath, "w") as f:
            json.dump(projects, f, default=datetime.datetime.isoformat)
        logging.info(
            f"Collected {len(projects)} GTR projects and stored them in {fpath}"
        )
    else:
        projects = json.load(open(fpath, "r"))
        logging.info(f"Loaded in the file {fpath}")
    return projects


def get_cb_data(fpath=CB_PATH, cb_data_spec_path=CB_DATA_SPEC_PATH, use_cached=True):
    """
    Downloads Crunchbase data from Nesta database and stores it locally.
    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_cb_data; get_cb_data(use_cached=False);"

    Parameters
    ----------
    fpath : str
        Location on disk for saving the projects table
    cb_data_spec_path : str
       Path to the config file that specifies which tables and columns to load in
    use_cached: bool
        If use_cached=True, the function will load in the local version of the dataset, if possible;
        set use_cached=False to download (and overwrite the existing) data

    Returns
    -------
    dict of str: pandas.DataFrame:
        Dictionary with dataframes with keys corresponding to table names
    """
    # Import specification of which tables and columns to download
    with open(cb_data_spec_path, "r", encoding="utf-8") as yaml_file:
        cb_tables = safe_load(yaml_file)
    tables = {}

    logging.info(f"Collection of business organisation data in progress")
    con = get_engine(db_config_path)
    # Download (or load in from the local storage) the specified tables one by one
    for table_name, columns in cb_tables.items():
        savepath = f"{fpath}/{table_name}.csv"
        use_cached_table = use_cached and os.path.exists(savepath)
        if not use_cached_table:
            chunks = read_sql_table(table_name, con, columns=columns, chunksize=1000)
            # Combine all chunks
            df = concat(chunks, axis=0).reset_index()
            df.to_csv(savepath, index=False)
            logging.info(
                f"Downloaded {table_name} ({len(df)} rows) and stored in {savepath}"
            )
        else:
            df = read_csv(savepath)
            logging.info(f"Loaded in the file {savepath}")
        tables[table_name] = df
    return tables


def download_hansard_data(fpath=HANSARD_PATH):
    """
    Downloads version 3.1.0 of the Hansard Speeches dataset.
    Find more information about the dataset here: https://zenodo.org/record/4843485

    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import download_hansard_data; download_hansard_data();"

    Parameters
    ----------
    fpath : str
        Location on disk for saving the data
    """
    # Download from Zenodo
    logging.info(f"Collection and storing of Hansard data in progress")
    zip_files = []
    for filename, url in zip(ZENODO_FILES, ZENODO_URLS):
        logging.info(f"Collecting {filename}")
        filepath = fpath / filename
        urllib.request.urlretrieve(url, fpath / filename)
        if filepath.suffix == ".zip":
            zip_files.append(filepath)
    # Extract the zip files and delete them
    for zip_file in zip_files:
        unzip_files(zip_file, fpath, delete=True)
    logging.info(f"Data successfully stored in {fpath}")


def unzip_files(path_to_zip_archive, extract_path, delete=False):
    """Extracts the zip file and optionally deletes it"""
    logging.info(f"Extracting the archive {path_to_zip_archive}")
    with zipfile.ZipFile(path_to_zip_archive, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    if delete:
        os.remove(path_to_zip_archive)


if __name__ == "__main__":
    """Downloads all input files"""
    GTR_PATH.mkdir(parents=True, exist_ok=True)
    get_gtr_projects()
    CB_PATH.mkdir(parents=True, exist_ok=True)
    get_cb_data()
    HANSARD_PATH.mkdir(parents=True, exist_ok=True)
    download_hansard_data()
    # Add other getter functions here
