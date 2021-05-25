"""
Modules for collecting data analysis inputs.

To fetch all inputs, simply run "python inputs.py"
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from pathlib import Path
from data_getters.gtr import build_projects
from data_getters.core import get_engine
import pandas as pd
from pandas import read_sql_table
import yaml
import urllib.request

INPUTS_PATH = f"{PROJECT_DIR}/inputs/data/"
GTR_PATH = f"{INPUTS_PATH}gtr_projects.csv"
CB_DATA_SPEC_PATH = f"{PROJECT_DIR}/innovation_sweet_spots/config/cb_data_spec.yaml"


def get_gtr_projects(fpath=GTR_PATH, fields=["id"]):
    """
    Downloads GTR projects from Nesta database and stores them locally.
    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_gtr_projects; get_gtr_projects();"

    Parameters
    ----------
    fpath : str
        Location on disk for saving the projects table
    fields : list of str
        Use the default value; for additional functionality see 'data_getters' documentation

    Returns
    -------
    df : pandas.DataFrame
        Table with projects
    """
    logging.info(f"Collection of GTR projects in progress")
    projects = build_projects(
        config_path=config["database_config_path"],
        chunksize=5000,
        table_wildcards=["gtr_projects"],
        desired_fields=fields,
    )
    df = pd.DataFrame(projects)
    df.to_csv(fpath, index=False)
    n_projects = len(df)
    logging.info(f"Collected {n_projects} GTR projects and stored them in {fpath}")
    return df


def get_cb_data(fpath=INPUTS_PATH, cb_data_spec_path=CB_DATA_SPEC_PATH):
    """
    Downloads Crunchbase data from Nesta database and stores it locally.
    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_cb_data; get_cb_data();"

    Parameters
    ----------
    fpath : str
       Location on disk for saving the tables
    cb_data_spec_path : str
       Path to the config file that specifies which tables and columns to load in

    Returns
    -------
    dict:
        Dictionary with the requested tables and paths where they have been saved
    """
    logging.info(f"Collection of business organisation data in progress")
    con = get_engine(config["database_config_path"])

    # Import specification of which tables and columns to download
    with open(cb_data_spec_path, "r", encoding="utf-8") as yaml_file:
        cb_tables = yaml.safe_load(yaml_file)

    # Download the specified tables one by one
    tables = {}
    for table in list(cb_tables.keys()):
        chunks = read_sql_table(table, con, columns=cb_tables[table], chunksize=1000)
        # Combine all chunks
        df = pd.concat([c for c in chunks], axis=0).reset_index()
        savepath = f"{fpath}{table}.csv"
        df.to_csv(savepath, index=False)
        tables[table] = {"path": savepath, "data": df}
        logging.info(f"Collected {table} ({len(df)} rows) and stored in {savepath}")
    return tables


def get_hansard_data(fpath=f"{INPUTS_PATH}hansard/"):
    """
    Downloads version 3.0.1 of the Hansard Speeches dataset.
    Find more information here: https://zenodo.org/record/4066772#.YK0WXpPYpTY

    Parameters
    ----------
    fpath : str
        Location on disk for saving the projects table
    """
    urls = [
        "https://zenodo.org/record/4066772/files/government_posts.json?download=1",
        "https://zenodo.org/record/4066772/files/opposition_posts.json?download=1",
        "https://zenodo.org/record/4066772/files/parliamentary_posts.json?download=1",
        "https://zenodo.org/record/4066772/files/hansard-speeches-v301.csv?download=1",
    ]
    logging.info(f"Collection and storing of Hansard data in progress")
    for url in urls:
        filename = url.split("/")[-1].split("?")[0]
        logging.info(f"Collecting {filename}")
        urllib.request.urlretrieve(url, f"{fpath}{filename}")
    logging.info(f"Data successfully stored in {fpath}")


if __name__ == "__main__":
    get_gtr_projects()
    get_cb_data()
    get_hansard_data()
    # Add other getter functions here
