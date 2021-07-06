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


def get_gtr_topics():
    return pd.read_csv(f"{GTR_PATH}/gtr_topics.csv")


def get_gtr_organisations():
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


def get_cleaned_project_texts():
    df = pd.read_csv(f"{OUTPUT_GTR_PATH}/gtr_project_clean_text.csv")
    df["project_text"] = df["project_text"].apply(lambda x: literal_eval(x))
    return df
