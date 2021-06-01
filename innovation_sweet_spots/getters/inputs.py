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

GTR_PATH = f"{PROJECT_DIR}/inputs/data/gtr_projects.json"
INPUTS_PATH = f"{PROJECT_DIR}/inputs/data"
CB_PATH = f"{INPUTS_PATH}/cb"
CB_DATA_SPEC_PATH = f"{PROJECT_DIR}/innovation_sweet_spots/config/cb_data_spec.yaml"


def get_gtr_projects(fpath=GTR_PATH, fields=["id"], use_cached=True):
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
        If use_cached=True, the function will load in the local version of the dataset;
        set use_cached=False to download (and overwrite the existing) data


    Returns
    -------
    projects : list of dict
        List of dictionaries with project data
    """
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
        try:
            projects = json.load(open(fpath, "r"))
            logging.info(f"Loaded in the file {fpath}")
        except FileNotFoundError:
            projects = []
            logging.error(
                f"File {fpath} does not exist! Set use_cached=False to download the data."
            )
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
        If use_cached=True, the function will load in the local version of the dataset;
        set use_cached=False to download (and overwrite the existing) data

    Returns
    -------
    list of dict:
        Dictionaries with the requested tables and paths where they have been saved.
        The dictionaries follow the structure:
            {"name": name of the table (corresponds to the table names from cb_data_spec_path),
             "path": path where the table is stored,
             "data": pandas dataframe with the data}
    """
    # Import specification of which tables and columns to download
    with open(cb_data_spec_path, "r", encoding="utf-8") as yaml_file:
        cb_tables = safe_load(yaml_file)
        cb_table_names = list(cb_tables.keys())

    tables = []
    if not use_cached:
        logging.info(f"Collection of business organisation data in progress")
        con = get_engine(db_config_path)
        # Download the specified tables one by one
        for table in cb_table_names:
            chunks = read_sql_table(
                table, con, columns=cb_tables[table], chunksize=1000
            )
            # Combine all chunks
            df = concat([c for c in chunks], axis=0).reset_index()
            savepath = f"{fpath}/{table}.csv"
            df.to_csv(savepath, index=False)
            tables.append({"name": table, "path": savepath, "data": df})
            logging.info(f"Collected {table} ({len(df)} rows) and stored in {savepath}")
    else:
        for table in cb_table_names:
            loadpath = f"{fpath}/{table}.csv"
            try:
                df = read_csv(loadpath)
                logging.info(f"Loaded in the file {loadpath}")
            except FileNotFoundError:
                df = []
                logging.error(
                    f"File {loadpath} does not exist! Set use_cached=False to download the data."
                )
            tables.append({"name": table, "path": loadpath, "data": df})
    return tables


if __name__ == "__main__":
    """Downloads all input files"""
    data_folder = f"{PROJECT_DIR}/inputs/data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    get_gtr_projects(use_cached=False)
    if not os.path.exists(CB_PATH):
        os.makedirs(CB_PATH)
    get_cb_data(use_cached=False)
    # Add other getter functions here
