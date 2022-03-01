"""
innovation_sweet_spots.utils.airtable_utils

Module for processing Airtable tables
Find more info on using Airtable API here: https://pyairtable.readthedocs.io/en/latest/api.html
"""
from innovation_sweet_spots import logging
import pyairtable
from pyairtable.api.table import Table

importlib.reload(airtable)
import pandas as pd
from os import PathLike
from typing import Iterator, Dict, Union


def get_ids(table: Union[Dict, Table]) -> Iterator[str]:
    """Produces a list of airtable record ids"""
    return [record["id"] for record in records]


def get_fields(table: Union[Dict, Table]) -> Iterator[Dict]:
    """Produces a list of airtable record fields"""
    return [record["fields"] for record in records]


def get_created_time(table: Union[Dict, Table]) -> Iterator:
    """Produces a list of airtable record creation times"""
    return [record["createdTime"] for record in records]


def table_to_dataframe(table: Union[Dict, Table]) -> pd.DataFrame:
    """Converts an airtable object to a dataframe"""
    records = table.all() if type(table) is not list else table
    return (
        pd.DataFrame(get_fields(records))
        .assign(createdTime=get_created_time(records))
        .assign(airtable_id=get_ids(records))
        .astype({"createdTime": "datetime64[ns]"})
    )


def dict_from_dataframe(
    dataframe: pd.DataFrame, columns: Iterator[str]
) -> Iterator[Dict]:
    """Generates a dictionary detailing the columns (fields)"""
    records = df[columns].to_dict(orient="records")
    ids = df.airtable_id.to_list()
    return [{"id": ids[i], "fields": record} for i, record in enumerate(records)]


def table_filename(table: Table):
    """Generate table filename"""
    return f"tableId_{t.base_id}_tableName_{t.table_name}.csv"


def save_table_locally(table: Union[Dict, Table], folder: PathLike):
    """Saves a local dataframe copy of the airtable table"""
    df = table_to_dataframe(table)
    filepath = folder / table_filename(table)
    df.to_csv(filepath, index=False)
    logging.info(f"Table saved at {filepath}")
