"""
Module for easy access to downloaded CB data

"""
import pandas as pd
import sys
from innovation_sweet_spots.getters.path_utils import CB_PATH, OUTPUT_CB_PATH


def get_crunchbase_category_groups():
    return pd.read_csv(f"{CB_PATH}/crunchbase_category_groups.csv")


def get_crunchbase_orgs():
    return pd.read_csv(f"{CB_PATH}/crunchbase_organisations.csv").drop_duplicates()


def get_crunchbase_orgs_full():
    return pd.read_csv(f"{CB_PATH}/crunchbase_organizations.csv").drop_duplicates()


def get_crunchbase_organizations_categories():
    return pd.read_csv(f"{CB_PATH}/crunchbase_organizations_categories.csv")


def get_crunchbase_funding_rounds():
    return pd.read_csv(f"{CB_PATH}/crunchbase_funding_rounds.csv")


def get_crunchbase_investments():
    return pd.read_csv(f"{CB_PATH}/crunchbase_investments.csv")


def get_crunchbase_investors():
    return pd.read_csv(f"{CB_PATH}/crunchbase_investors.csv")


def get_cleaned_green_crunchbase_texts():
    return pd.read_csv(f"{OUTPUT_CB_PATH}/cb_green_org_clean_text.csv")
