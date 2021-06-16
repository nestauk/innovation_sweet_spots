import pandas as pd
from innovation_sweet_spots import logging
from innovation_sweet_spots.getters.path_utils import HANSARD_PATH
from typing import Iterable


def get_hansard_data(n_skiprows=1750000):
    return pd.read_csv(
        HANSARD_PATH / "hansard-speeches-v310.csv", skiprows=range(1, 2000000)
    )
