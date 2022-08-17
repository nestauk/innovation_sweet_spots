"""
innovation_sweet_spots.getters.gtr_2022

Module for easy access to downloaded GtR data (updated for summer 2022)

"""
import pandas as pd
from innovation_sweet_spots import logging
from innovation_sweet_spots.utils.io import load_json
from innovation_sweet_spots.getters.path_utils import GTR_2022_PATH, PILOT_OUTPUTS
from typing import Iterable

PATH = GTR_2022_PATH


def get_gtr_projects() -> pd.DataFrame:
    """Main GtR projects table"""
    return pd.DataFrame(load_json(PATH / "gtr_projects-projects.json"))
