import pandas as pd
from innovation_sweet_spots.getters.path_utils import INPUTS_PATH
from innovation_sweet_spots.utils.io import read_list_of_terms

MISC_PATH = INPUTS_PATH / "misc"


def get_tech_navigator():
    """Heating tech, compiled by Kyle Usher"""
    return pd.read_csv(MISC_PATH / "Tech_Nav-Grid view.csv")


# def get_ik_keywords():
#     """Green keywords, derived from EGSS/LCREE by India Kerle"""
#     return read_list_of_terms(MISC_PATH / "green_keywords_IK.txt")
