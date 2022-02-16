"""
innovation_sweet_spots.getters.dealroom

Module for easy access to downloaded Dealroom data

"""
import pandas as pd
from innovation_sweet_spots.getters.path_utils import DEALROOM_PATH


def get_foodtech_companies() -> pd.DataFrame:
    """Dataset used in the food tech themed Innovation Sweet Spots"""
    return pd.read_csv(DEALROOM_PATH / "dealroom_foodtech.csv")
