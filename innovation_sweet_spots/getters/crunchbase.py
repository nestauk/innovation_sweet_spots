"""
innovation_sweet_spots.getters.crunchbase

Module for easy access to downloaded CB data

"""
import pandas as pd
from innovation_sweet_spots.getters.path_utils import CB_PATH


def get_crunchbase_category_groups():
    """Table with company categories (also called 'industries') and broader 'category groups'"""
    return pd.read_csv(CB_PATH / "crunchbase_category_groups.csv")


def get_crunchbase_orgs():
    """
    Loads and deduplicates the main Crunchbase organisations table;
    dtype = object, to avoid warnings about mixed data types
    """
    return pd.read_csv(
        CB_PATH / "crunchbase_organizations.csv", dtype=object
    ).drop_duplicates()


def get_crunchbase_organizations_categories():
    """Table with companies and their categories"""
    return pd.read_csv(CB_PATH / "crunchbase_organizations_categories.csv")


def get_crunchbase_funding_rounds():
    """Table with investment rounds (NB: one round can have several investments)"""
    return pd.read_csv(CB_PATH / "crunchbase_funding_rounds.csv")


def get_crunchbase_investments():
    """Table with investments"""
    return pd.read_csv(CB_PATH / "crunchbase_investments.csv")


def get_crunchbase_investors():
    """Table with investors"""
    return pd.read_csv(CB_PATH / "crunchbase_investors.csv")
