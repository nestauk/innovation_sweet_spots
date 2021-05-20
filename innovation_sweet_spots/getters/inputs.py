"""
Modules for collecting data analysis inputs
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from pathlib import Path
from data_getters.gtr import build_projects
import pandas as pd


def get_gtr_projects(
    fpath=f"{PROJECT_DIR}/inputs/data/gtr_projects.csv", fields=["id"]
):
    """
    Downloads GTR projects from Nesta database and stores them locally.
    Function can be used from command line simply as follows:
        python -c "from innovation_sweet_spots.getters.inputs import get_gtr_projects; get_gtr_projects()"
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
    logging.info(f"Collected {len(df)} GTR projects and stored them in {fpath}")
