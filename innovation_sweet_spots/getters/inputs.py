"""
Modules for collecting data analysis inputs.

To fetch all inputs, simply run "python inputs.py"
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from pathlib import Path
from data_getters.gtr import build_projects
import pandas as pd

GTR_PATH = f"{PROJECT_DIR}/inputs/data/gtr_projects.csv"


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
    if not (use_cached or os.path.exists(fpath)):
        projects = build_projects(
            config_path=config["database_config_path"],
            chunksize=5000,
            table_wildcards=["gtr_projects"],
            desired_fields=fields,
        )
        with open(fpath, "w") as f:
            json.dump(projects, f)
        logging.info(f"Collected {len(projects)} GTR projects and stored them in {fpath}")
    df = pd.read_json(fpath, orient='records')
    return df
    return projects


if __name__ == "__main__":
    get_gtr_projects()
    # Add other getter functions here
