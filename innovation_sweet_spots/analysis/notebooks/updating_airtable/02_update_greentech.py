# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.getters import airtable
from innovation_sweet_spots.utils import airtable_utils as au

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

# %%
from innovation_sweet_spots import logging
import pandas as pd
from os import PathLike
from typing import Iterator, Dict, Union

# %%
import importlib

importlib.reload(airtable)
importlib.reload(au)

# %% [markdown]
# # Green tech table
#
# - Load Airtable table
# - Load the local list (eg, ISS pilot Crunchbase companies)
# - Compare company lists
#   - Which companies are on Airtable but not in this list
#   - Suggest missing Crunchbase ids
#   - Populate empty fields with CB data iif possible
#   - Check if investment data can be updated
#   - Update records on Airtable
#   - Update local table, with a new name

# %%
import re

# %%
CB = CrunchbaseWrangler()

# %%

# %%
import innovation_sweet_spots.utils.airtable_utils as au

importlib.reload(airtable)
importlib.reload(au)

# %%
# Fetch from Airtable and update the local copy
# t = airtable.get_greentech_table()
# records = t.all()
au.save_table_locally(t, airtable.AIRTABLE_PATH)

# %%

# %%
# Create a table
df = au.table_to_dataframe(records)

# %%

# %%

# %%
cb_urls = df.crunchbase_url.to_list()

# %%
list(CB.cb_organisations.columns)

# %%
cb_selected = CB.cb_organisations.query("cb_url in @cb_urls")

# %%
import numpy as np


def replace_nulls(df_original: pd.DataFrame, df_updated: pd.DataFrame, column: str):
    """Replace nulls of the updated dataframe with data from the original dataframe"""
    df_new = df_updated.copy()
    df_new.loc[df_new[column].isnull(), column] = df_original.loc[
        df_new[column].isnull(), column
    ]
    return df_new


def amount_to_range(amount: float, return_na: bool = True) -> str:
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


# %%
# len(cb_selected)

# %%
# dict_from_dataframe(df, columns)

# %%
# df_.iloc[24]

# %%
columns = [
    "cb_url",
    "id",
    "address",
    "email",
    "city",
    "twitter_url",
    "employee_count",
    "total_funding_usd",
    "short_description",
]
df_ = df.drop("twitter_url", axis=1).merge(
    cb_selected[columns], left_on="crunchbase_url", right_on="cb_url", how="left"
)

# %%
df_ = replace_nulls(df, df_, "twitter_url")

# %%
investment_amounts = (
    CB.get_funding_rounds(df_.rename(columns={"company_title": "name"}))
    .groupby("org_id")
    .agg(total_raised_amount_gbp=("raised_amount_gbp", "sum"))
)


# %%
df__ = df_.merge(
    investment_amounts.reset_index(), how="left", left_on="id", right_on="org_id"
)

# %%
df__["raised_investment_range"] = df__["total_raised_amount_gbp"].apply(amount_to_range)

# %%
df__[["total_raised_amount_gbp", "raised_investment_range"]].sort_values(
    "total_raised_amount_gbp"
)

# %%
df__final = (
    df__[
        [
            "airtable_id",
            "id",
            "city",
            "email",
            "twitter_url",
            "raised_investment_range",
            "short_description",
        ]
    ]
    .rename(columns={"id": "crunchbase_id"})
    .copy()
)
df__final = df__final.fillna("n/a")
df__final.twitter_url = df__final.twitter_url.apply(
    lambda x: re.sub("\?lang=en", "", x)
)

# %%
df__final.twitter_url = df__final.twitter_url.apply(
    lambda x: "n/a" if "twitter.com/search" in x else x
)

# %%
dict_from_dataframe

# %%
# updates = dict_from_dataframe(df__final, ['crunchbase_id', 'city', 'email', 'twitter_url', 'raised_investment_range'])
updates = dict_from_dataframe(df__final, ["short_description"])

# %%
import time
from tqdm.notebook import tqdm

# %%
updates[0:3]

# %%
for update in tqdm(updates, total=len(updates)):
    t.update(update["id"], update["fields"])
    time.sleep(0.25)

# %%
CB.cb_organisations.query('name == "Eon UK PLC"')[["email"]]
