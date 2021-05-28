"""
Modules for collecting data analysis inputs.

To fetch all inputs, simply run "python inputs.py"
"""
from innovation_sweet_spots import PROJECT_DIR, logging, db_config_path
from pathlib import Path
from data_getters.gtr import build_projects
import datetime
import os
import json

GTR_PATH = f"{PROJECT_DIR}/inputs/data/gtr_projects.json"


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


if __name__ == "__main__":
    """Downloads all input files"""
    data_folder = f"{PROJECT_DIR}/inputs/data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    get_gtr_projects(use_cached=False)
    # Add other getter functions here
