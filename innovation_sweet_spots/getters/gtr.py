"""
Module for easy access to downloaded GTR data

"""
import pandas as pd
from innovation_sweet_spots.getters.inputs import GTR_PATH


def get_gtr_projects():
    return pd.read_csv(f"{GTR_PATH}/gtr_projects.csv")


def get_link_table():
    """NB: Large table with 34M+ rows"""
    return pd.read_csv(f"{GTR_PATH}/gtr_link_table.csv")


def get_gtr_funds():
    return pd.read_csv(f"{GTR_PATH}/gtr_funds.csv")


def get_gtr_topics():
    return pd.read_csv(f"{GTR_PATH}/gtr_topics.csv")


def get_gtr_organisations():
    return pd.read_csv(f"{GTR_PATH}/gtr_organisations.csv")
