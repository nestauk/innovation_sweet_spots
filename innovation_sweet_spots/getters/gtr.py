"""
innovation_sweet_spots.getters.gtr

Module for easy access to downloaded GtR data

"""
import pandas as pd
from innovation_sweet_spots import logging
from innovation_sweet_spots.getters.path_utils import GTR_PATH
from typing import Iterable

# Path to the tables linking projects to other data
GTR_LINKS_PATH = GTR_PATH / "links"


def get_gtr_projects() -> pd.DataFrame:
    """Main GtR projects table"""
    return pd.read_csv(GTR_PATH / "gtr_projects.csv")


def get_gtr_funds() -> pd.DataFrame:
    """Links between project ids and funding ids"""
    return pd.read_csv(GTR_PATH / "gtr_funds.csv")


def get_gtr_funds_api() -> pd.DataFrame:
    """Links between project ids and funding ids, retreived using API calls"""
    return pd.read_csv(GTR_PATH / "gtr_funds_api.csv")


def get_gtr_topics() -> pd.DataFrame:
    """GtR project research topics"""
    return pd.read_csv(GTR_PATH / "gtr_topic.csv")


def get_gtr_organisations() -> pd.DataFrame:
    """GtR research organisations"""
    return pd.read_csv(GTR_PATH / "gtr_organisations.csv")


def get_gtr_organisations_locations() -> pd.DataFrame:
    """GtR research organisations"""
    return pd.read_csv(GTR_PATH / "gtr_organisations_locations.csv")


def get_link_table(table: str = None) -> pd.DataFrame:
    """
    Returns table specifying links between projects and other data

    Args:
        table: String specifying the link table to fetch;
            if table=None, this will fetch the full links table
            (NB: Large table with 34M+ rows)
            Useful values for 'table' include:
              - gtr_funds (returns links between projects and funding data)
              - gtr_organisations (projects and organisations)
              - gtr_persons (projects and persons)
              - gtr_topic (projects and research topics/labels)

    """
    if table is None:
        # Get the full links table
        return pd.read_csv(GTR_PATH / "gtr_link_table.csv")
    else:
        # Get the specific links defined by table variable
        fpath = get_path_to_specific_link_table(table)
        try:
            return pd.read_csv(fpath)
        except FileNotFoundError:
            # Generate the table if it doesn't exist
            logging.info(
                f"Link table {fpath} not found. Creating the table now (might take a while)"
            )
            pullout_gtr_links(tables=[table])
            return pd.read_csv(fpath)


def get_path_to_specific_link_table(table: str):
    """Default path to the pulled out links tables"""
    return GTR_LINKS_PATH / f"link_{table}.csv"


def pullout_gtr_links(tables: Iterable[str]):
    """
    Pulls out all rows related to the links between projects and
    items in the specified tables, and saves them in a csv file.
    """
    # Get all the links and query the table of interest
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
