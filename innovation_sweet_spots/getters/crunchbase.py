"""
innovation_sweet_spots.getters.crunchbase

Module for easy access to downloaded CB data

"""
import pandas as pd
from innovation_sweet_spots.getters.path_utils import CB_PATH


def restore_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to ensure that utils functions written to process
    the older Crunchbase data snapshots also work with the newer
    snapshots (where columns storing unique identifiers have been renamed).
    """
    return df.rename(
        columns={
            "uuid": "id",
            "org_uuid": "org_id",
            "lead_investor_uuids": "lead_investor_ids",
            "funding_round_uuid": "funding_round_id",
            "investor_uuid": "investor_id",
            "featured_job_organization_uuid": "featured_job_organization_id",
            "person_uuid": "person_id",
            "institution_uuid": "institution_id",
            "acquiree_uuid": "acquiree_id",
            "acquirer_uuid": "acquirer_id",
        }
    )


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
    return pd.read_csv(CB_PATH / "crunchbase_funding_rounds.csv").pipe(
        restore_column_names
    )


def get_crunchbase_investments() -> pd.DataFrame:
    """Table with investments"""
    return pd.read_csv(CB_PATH / "crunchbase_investments.csv").pipe(
        restore_column_names
    )


def get_crunchbase_investors() -> pd.DataFrame:
    """Table with investors"""
    return pd.read_csv(CB_PATH / "crunchbase_investors.csv").pipe(restore_column_names)


def get_crunchbase_people() -> pd.DataFrame:
    """Table with people"""
    return pd.read_csv(CB_PATH / "crunchbase_people.csv").pipe(restore_column_names)


def get_crunchbase_degrees() -> pd.DataFrame:
    """Table with university degrees"""
    return pd.read_csv(CB_PATH / "crunchbase_degrees.csv").pipe(restore_column_names)


def get_crunchbase_ipos() -> pd.DataFrame:
    """Table with crunchbase ipos"""
    return pd.read_csv(CB_PATH / "crunchbase_ipos.csv").pipe(restore_column_names)


def get_crunchbase_acquisitions() -> pd.DataFrame:
    """Table with crunchbase acquisitions"""
    return pd.read_csv(CB_PATH / "crunchbase_acquisitions.csv").pipe(
        restore_column_names
    )
