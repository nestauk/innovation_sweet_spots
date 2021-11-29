"""
Module for easy access to downloaded GTR data

"""
import pandas as pd
from innovation_sweet_spots import logging
from innovation_sweet_spots.getters.path_utils import GTR_PATH, OUTPUT_GTR_PATH
from typing import Iterable
from ast import literal_eval


def get_gtr_projects():
    return pd.read_csv(f"{GTR_PATH}/gtr_projects.csv")


def get_gtr_funds():
    return pd.read_csv(f"{GTR_PATH}/gtr_funds.csv")


def get_gtr_funds_api() -> pd.DataFrame:
    """
    GtR project funding data that has been fetched directly using GtR API.
    This was done, as the database's funding data was ambiguous.
    Contains three columns ['index_col', 'project_id', 'amount'], with amount in GBP.
    """
    df = (
        pd.read_csv(
            f"{GTR_PATH}/gtr_funds_api.csv", names=["index_col", "project_id", "amount"]
        )
        # Remove the index column, which is redundant
        .drop("index_col", axis=1)
    )
    return df


def get_gtr_topics() -> pd.DataFrame:
    return pd.read_csv(f"{GTR_PATH}/gtr_topics.csv")


def get_gtr_organisations() -> pd.DataFrame:
    return pd.read_csv(f"{GTR_PATH}/gtr_organisations.csv")


def get_link_table(table: str = None):
    if table is None:
        # NB: Large table with 34M+ rows
        return pd.read_csv(f"{GTR_PATH}/gtr_link_table.csv")
    else:
        fpath = get_path_to_specific_link_table(table)
        return pd.read_csv(fpath)


def get_path_to_specific_link_table(table: str):
    return GTR_PATH / f"links/link_{table}.csv"


def pullout_gtr_links(tables: Iterable[str]):
    """
    Pulls out all rows related to the links between projects and
    items in the specified tables, and saves them in a csv file.
    """
    # Get all of the links and query the table of interest
    link_table = get_link_table()
    for table in tables:
        specific_link_table = link_table.query(f"table_name=='{table}'")
        # Save the links in a separate csv file
        fpath = get_path_to_specific_link_table(table)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        specific_link_table.to_csv(fpath, index=False)
        logging.info(
            f"Links between GTR projects and items in {table} saved in {fpath}"
        )


def get_cleaned_project_texts() -> pd.DataFrame:
    df = pd.read_csv(f"{OUTPUT_GTR_PATH}/gtr_project_clean_text.csv")
    df["project_text"] = df["project_text"].apply(lambda x: literal_eval(x))
    return df


def add_funding_data(
    gtr_docs: pd.DataFrame, gtr_funds: pd.DataFrame, project_id_column="project_id"
) -> pd.DataFrame:
    """
    Update research project funding column with funds
    """
    gtr_docs = gtr_docs.copy()
    # If amount column already exists, keep it as amount_old
    if "amount" in gtr_docs.columns:
        gtr_docs["amount_old"] = gtr_docs["amount"].copy()
        gtr_docs.drop("amount", axis=1)
    # Add the provided funding data
    gtr_docs = gtr_docs.merge(
        gtr_funds[[project_id_column, "amount"]], on=project_id_column, how="left"
    )
    # Check if there are projects with missing funding data (i.e. no corresponding
    # project in gtr_funds dataframe) and impute with 0
    missing_amount = gtr_docs.amount.isnull()
    n_missing = missing_amount.sum()
    logging.info(f"{n_missing} documents without funding info")
    if n_missing > 0:
        gtr_docs.loc[missing_amount, "funding_category"] = "NO_FUND_INFO"
        gtr_docs["amount"] = gtr_docs["amount"].fillna(0)
    return gtr_docs
