"""
innovation_sweet_spots.getters.nihr

Module for easy access to downloaded NIHR data
"""
# import numpy as np
# import pandas as pd
# import altair as alt
# import os
import logging
import pandas as pd
from innovation_sweet_spots.utils.read_from_s3 import load_csv_from_s3
from innovation_sweet_spots import PROJECT_DIR

NIHR_FOLDER = PROJECT_DIR / "inputs/data/nihr"


def get_nihr_summary_data(from_local: bool = True) -> pd.DataFrame:
    """Load NIHR .csv file from S3"""
    if from_local:
        return pd.read_csv(NIHR_FOLDER / "nihr_summary_data.csv")
    else:
        return load_csv_from_s3("inputs/data/nihr/", "nihr_summary_data")


def save_nihr_to_local():
    """Save NIHR summary data locally"""
    data = get_nihr_summary_data()
    # Create the folder if it doesn't exist
    NIHR_FOLDER.mkdir(parents=True, exist_ok=True)
    path = NIHR_FOLDER / "nihr_summary_data.csv"
    logging.info(f"Saving data on {len(data)} NIHR projects in {path}")
    data.to_csv(path, sep=",", index=False)


if __name__ == "__main__":
    # Load NIHR data and save locally
    save_nihr_to_local()
