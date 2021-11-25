import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from daps1_utils import fetch_daps_table, save_daps_table

CB_PATH = Path(__file__).parents[3] / "inputs/data/cb"
logger = logging.getLogger(__name__)


def filter_uk(table: pd.DataFrame, ids: set, var_name: str = "org_id"):
    """Gets UK companies from crunchbase
    Args:
        table: crunchbase table
        ids: UK company ids
        var_name: name of org id variable
    Returns:
        filtered table
    """
    return table.loc[table[var_name].isin(ids)].reset_index(drop=True)


def fetch_save_crunchbase():
    """Fetch and save crunchbase data"""
    cb_orgs = pd.concat(fetch_daps_table("crunchbase_organizations", fields="all"))
    save_daps_table(cb_orgs, "crunchbase_organizations", CB_PATH)

    # Commented out filtering by country
    # cb_uk = cb_orgs.loc[cb_orgs["country"] == "United Kingdom"].drop_duplicates(
    #     subset=["id"]
    # )
    # logging.info(len(cb_uk))
    # save_daps_table(cb_uk, "crunchbase_organisations", CB_PATH)

    # cb_uk_ids = set(cb_uk["id"])

    cb_funding_rounds = pd.concat(
        fetch_daps_table("crunchbase_funding_rounds", fields="all")
    )
    # cb_funding_rounds_uk = filter_uk(cb_funding_rounds, cb_uk_ids)

    cb_orgs_cats = pd.concat(
        fetch_daps_table("crunchbase_organizations_categories", fields="all")
    )
    # cb_org_cats_uk = filter_uk(cb_orgs_cats, cb_uk_ids, "organization_id")

    category_group = pd.concat(
        fetch_daps_table("crunchbase_category_groups", fields="all")
    )

    save_daps_table(cb_funding_rounds, "crunchbase_funding_rounds", CB_PATH)
    save_daps_table(cb_orgs_cats, "crunchbase_organizations_categories", CB_PATH)
    save_daps_table(category_group, "crunchbase_category_groups", CB_PATH)


def get_uk_names(con) -> Dict[str, str]:
    """Fetch non-null `{id: name}` pairs from `crunchbase_organizations` in the UK."""

    query = """
    SELECT id, name
    FROM crunchbase_organizations
    WHERE country = 'United Kingdom'
    """
    return pd.read_sql_query(query, con).set_index("id").name.dropna().to_dict()
