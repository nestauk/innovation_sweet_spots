import logging
from pathlib import Path
from typing import Dict
import pandas as pd

from daps1_utils import fetch_daps_table, save_daps_table

CB_PATH = Path(__file__).parents[3] / "inputs/data/crunchbase"

CB_TABLES = [
    "crunchbase_organizations",
    "crunchbase_funding_rounds",
    "crunchbase_organizations_categories",
    "crunchbase_category_groups",
    "crunchbase_people",
    "crunchbase_degrees",
    "crunchbase_investment_partners",
    "crunchbase_investments",
    "crunchbase_investors",
    "crunchbase_ipos",
    "crunchbase_acquisitions",
]


def fetch_save_table(name: str):
    """Fetch and save a Crunchbase table"""
    df = pd.concat(fetch_daps_table(name))
    save_daps_table(df, name, CB_PATH)


def fetch_save_crunchbase():
    """Fetch and save Crunchbase data"""
    for table in CB_TABLES:
        fetch_save_table(table)


def get_cb_names(con) -> Dict[str, str]:
    """Fetch non-null `{id: name}` pairs from `crunchbase_organizations`."""

    query = """
    SELECT id, name
    FROM crunchbase_organizations
    """
    return pd.read_sql_query(query, con).set_index("id").name.dropna().to_dict()
