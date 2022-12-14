"""
innovation_sweet_spots.getters.gtr_2022

Module for easy access to downloaded GtR data (updated for summer 2022)

"""
import pandas as pd
from innovation_sweet_spots.utils.io import load_json
from innovation_sweet_spots.getters.path_utils import GTR_2022_PATH


def get_gtr_file(filename: str) -> pd.DataFrame:
    """Load Gateway to Research csv or json file as a dataframe

    Args:
        filename: For example 'gtr_projects-funds.json'

    Returns:
        Dataframe of provided filename
    """
    filepath = GTR_2022_PATH / filename
    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    if filepath.suffix == ".json":
        return pd.DataFrame(load_json(filepath))
