import pandas as pd
from innovation_sweet_spots.getters.path_utils import INPUTS_PATH


def get_tech_navigator():
    return pd.read_csv(INPUTS_PATH / "misc/Tech_Nav-Grid view.csv")
