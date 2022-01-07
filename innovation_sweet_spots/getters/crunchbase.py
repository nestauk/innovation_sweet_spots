"""
innovation_sweet_spots.getters.crunchbase

Module for easy access to downloaded CB data

"""
import pandas as pd
from innovation_sweet_spots.getters.path_utils import CB_PATH


def get_crunchbase_category_groups():
    return pd.read_csv(f"{CB_PATH}/crunchbase_category_groups.csv")


def get_crunchbase_orgs():
    """Loads and deduplicates Crunchbase organisations table"""
    return pd.read_csv(f"{CB_PATH}/crunchbase_organizations.csv").drop_duplicates()


def get_crunchbase_organizations_categories():
    return pd.read_csv(f"{CB_PATH}/crunchbase_organizations_categories.csv")


def get_crunchbase_funding_rounds():
    return pd.read_csv(f"{CB_PATH}/crunchbase_funding_rounds.csv")


def get_crunchbase_investments():
    return pd.read_csv(f"{CB_PATH}/crunchbase_investments.csv")


def get_crunchbase_investors():
    return pd.read_csv(f"{CB_PATH}/crunchbase_investors.csv")
