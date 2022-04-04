"""
innovation_sweet_spots.getters.crunchbase

Module for easy access to downloaded CB data

"""
import pandas as pd
from innovation_sweet_spots.getters.path_utils import CB_PATH, CB_GTR_LINK_PATH


def get_crunchbase_category_groups() -> pd.DataFrame:
    """Table with company categories (also called 'industries') and broader 'category groups'"""
    return pd.read_csv(CB_PATH / "crunchbase_category_groups.csv")


def get_crunchbase_orgs(nrows: int = None) -> pd.DataFrame:
    """
    Loads and deduplicates the main Crunchbase organisations table;
    dtype = object, to avoid warnings about mixed data types
    """
    return pd.read_csv(
        CB_PATH / "crunchbase_organizations.csv", dtype=object, nrows=nrows
    ).drop_duplicates()


def get_crunchbase_organizations_categories() -> pd.DataFrame:
    """Table with companies and their categories"""
    return pd.read_csv(CB_PATH / "crunchbase_organizations_categories.csv")


def get_crunchbase_funding_rounds() -> pd.DataFrame:
    """Table with investment rounds (NB: one round can have several investments)"""
    return pd.read_csv(CB_PATH / "crunchbase_funding_rounds.csv")


def get_crunchbase_investments() -> pd.DataFrame:
    """Table with investments"""
    return pd.read_csv(CB_PATH / "crunchbase_investments.csv")


def get_crunchbase_investors() -> pd.DataFrame:
    """Table with investors"""
    return pd.read_csv(CB_PATH / "crunchbase_investors.csv")


def get_crunchbase_people() -> pd.DataFrame:
    """Table with people"""
    return pd.read_csv(CB_PATH / "crunchbase_people.csv")


def get_crunchbase_degrees() -> pd.DataFrame:
    """Table with university degrees"""
    return pd.read_csv(CB_PATH / "crunchbase_degrees.csv")


def get_crunchbase_ipos() -> pd.DataFrame:
    """Table with crunchbase ipos"""
    return pd.read_csv(CB_PATH / "crunchbase_ipos.csv")


def get_crunchbase_acquisitions() -> pd.DataFrame:
    """Table with crunchbase acquisitions"""
    return pd.read_csv(CB_PATH / "crunchbase_acquisitions.csv")


def get_crunchbase_gtr_lookup() -> pd.DataFrame:
    """Table with UK Crunchbase organisation ids with fuzzy
    matched Gateway to Research organisation ids"""
    return pd.read_csv(CB_GTR_LINK_PATH / "cb_gtr_id_lookup.csv", index_col=0)
