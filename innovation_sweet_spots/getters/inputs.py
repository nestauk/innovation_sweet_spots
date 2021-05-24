"""
Modules for collecting data analysis inputs.

To fetch all inputs, simply run "python inputs.py"
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from pathlib import Path
from data_getters.gtr import build_projects
from data_getters.core import get_engine
import pandas as pd

INPUTS_PATH = f"{PROJECT_DIR}/inputs/data"
gtr_path = f"{INPUTS_PATH}/gtr_projects.csv"
cb_path = f"{INPUTS_PATH}/cb_data.csv"


def get_gtr_projects(fpath=gtr_path, fields=["id"]):
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


def get_cb_data(fpath=cb_path):
    """
    Downloads Crunchbase data from Nesta database and stores them locally.
    Function can be used from command line as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_cb_data; get_cb_data();"
    """
    con = get_engine(config["database_config_path"])
    chunks = pd.read_sql_table(
        "crunchbase_organizations", con, columns=["company_name"], chunksize=1000
    )
    for i, df in enumerate(chunks):
        print(df.head())
        if i == 3:
            break


if __name__ == "__main__":
    get_gtr_projects()
    # Add other getter functions here
