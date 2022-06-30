"""
innovation_sweet_spots.utils.airtable_utils

Module for processing Airtable tables
Find more info on using Airtable API here: https://pyairtable.readthedocs.io/en/latest/api.html
"""
from innovation_sweet_spots import logging
import pyairtable
from pyairtable.api.table import Table

import pandas as pd
from os import PathLike
from typing import Iterator, Dict, Union

RELEVANT_CB_COLUMNS = [
    "id",
    "cb_url",
    "email",
    "city",
    "short_description",
    "long_description",
    "homepage_url",
    "twitter_url",
    "airtable_id",
]


def get_ids(records: Union[Dict, Table]) -> Iterator[str]:
    """Produces a list of airtable record ids"""
    return [record["id"] for record in records]


def get_fields(records: Union[Dict, Table]) -> Iterator[Dict]:
    """Produces a list of airtable record fields"""
    return [record["fields"] for record in records]


def get_created_time(records: Union[Dict, Table]) -> Iterator:
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
    records = dataframe[columns].to_dict(orient="records")
    ids = dataframe.airtable_id.to_list()
    return [{"id": ids[i], "fields": record} for i, record in enumerate(records)]


def table_filename(table: Table):
    """Generate table filename"""
    return f"tableId_{table.base_id}_tableName_{table.table_name}.csv"


def save_table_locally(table: Union[Dict, Table], folder: PathLike):
    """Saves a local dataframe copy of the airtable table"""
    df = table_to_dataframe(table)
    filepath = folder / table_filename(table)
    df.to_csv(filepath, index=False)
    logging.info(f"Table saved at {filepath}")


def check_if_has_investment(amount: float):
    """Returns Yes if investment is above 0, otherwise N/A"""
    if amount > 0:
        return "Yes"
    else:
        return "N/A"


def investment_amount_to_range(amount: float, return_na: bool = True) -> str:
    """Convert amounts to range in millions"""
    amount /= 1e3
    if (amount >= 0.001) and (amount < 1):
        return "0-1"
    elif (amount >= 1) and (amount < 4):
        return "1-4"
    elif (amount >= 4) and (amount < 15):
        return "4-15"
    elif (amount >= 15) and (amount < 40):
        return "15-40"
    elif (amount >= 40) and (amount < 100):
        return "40-100"
    elif (amount >= 100) and (amount < 250):
        return "100-250"
    elif amount >= 250:
        return "250+"
    else:
        return "n/a"


def get_lowest_range(investment_range: str):
    if investment_range == "n/a":
        return -1
    elif investment_range == "250+":
        return 251
    else:
        return int(investment_range.split("-")[0])


def update_investment_range_field():
    pass


# def replace_nulls(df_original: pd.DataFrame, df_updated: pd.DataFrame, column: str):
#     """Replace nulls of the updated dataframe with data from the original dataframe"""
#     df_new = df_updated.copy()
#     df_new.loc[df_new[column].isnull(), column] = df_original.loc[
#         df_new[column].isnull(), column
#     ]
#     return df_new


def crunchbase_description_column(company_data_to_update: pd.DataFrame) -> list:
    """Update description"""
    cb_description = []
    for i, row in company_data_to_update.iterrows():
        descr = (
            row.short_description
            if type(row.long_description) is not str
            else row.long_description
        )
        cb_description.append(descr)
    return cb_description


def get_company_data_to_update(
    company_data: pd.DataFrame, company_funds: pd.DataFrame
) -> pd.DataFrame:
    """Prepare data to be sent to Airtable"""
    return (
        company_data[RELEVANT_CB_COLUMNS]
        .copy()
        .merge(
            company_funds[["org_id", "raised_amount_gbp"]],
            left_on="id",
            right_on="org_id",
            how="left",
        )
        .assign(
            has_investment=lambda df: df.raised_amount_gbp.apply(
                check_if_has_investment
            ),
            raised_investment_range=lambda df: df.raised_amount_gbp.apply(
                investment_amount_to_range
            ),
            crunchbase_description=lambda df: crunchbase_description_column(df),
        )
        .rename(
            columns={
                "id": "crunchbase_id",
                "cb_url": "crunchbase_url",
            }
        )
        .fillna("n/a")
    )
