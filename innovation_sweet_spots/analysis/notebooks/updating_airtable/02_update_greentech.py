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

# %% [markdown]
# # Fetching and updating Airtable data
#

# %% [markdown]
# ## Import dependencies

# %%
from innovation_sweet_spots.getters import airtable
from innovation_sweet_spots.utils import airtable_utils as au

# %%
import time
from tqdm.notebook import tqdm

# %%
from innovation_sweet_spots import logging

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

# %%
CB = CrunchbaseWrangler()

# %%
import innovation_sweet_spots.utils.airtable_utils as au

importlib.reload(airtable)
importlib.reload(au)

# %%
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

FIELDS_TO_UPDATE = [
    "crunchbase_id",
    "crunchbase_url",
    "email",
    "city",
    "crunchbase_description",
    "homepage_url",
    "twitter_url",
    "raised_investment_range",
]

# %% [markdown]
# ## Process 1: Update info on new companies

# %%
# Fetch from Airtable and update the local copy
t = airtable.get_greentech_table()
records = t.all()
au.save_table_locally(t, airtable.AIRTABLE_PATH)

# %% [markdown]
# ### Update missing info from Airtable data

# %%
# Create a table
airtable_df = au.table_to_dataframe(records)

# Check for companies that don't have Crunchbase ID
companies_without_crunchbase_id = airtable_df[
    ((airtable_df.crunchbase_id == "n/a") | (airtable_df.crunchbase_id.isnull()))
]

logging.info(
    f"There are {len(companies_without_crunchbase_id)} potentially new companies"
)


# %%
# Among the new companies check those that have a Crunchbase URL
companies_with_urls = companies_without_crunchbase_id[
    (
        (companies_without_crunchbase_id.crunchbase_url != "n/a")
        & (companies_without_crunchbase_id.crunchbase_url.isnull() == False)
    )
]

# Fetch company data for the new companies
company_data = CB.cb_organisations.query(
    "cb_url in @companies_with_urls.crunchbase_url.to_list()"
).merge(
    airtable_df[["crunchbase_url", "airtable_id"]],
    left_on="cb_url",
    right_on="crunchbase_url",
    how="left",
)

logging.info(f"Updating data for {len(company_data)} companies")

# %%
company_funds = (
    CB.get_funding_rounds(company_data)
    .groupby("org_id", as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
)

# %%
importlib.reload(au)

# %%
company_data_to_update = au.get_company_data_to_update(company_data, company_funds)

# %%
updates = au.dict_from_dataframe(company_data_to_update, FIELDS_TO_UPDATE)

# %% [markdown]
# ### Update airtable records

# %%
# updates = au.dict_from_dataframe(company_data_to_update, FIELDS_TO_UPDATE)
# updates[0:3]

# %%
for update in tqdm(updates, total=len(updates)):
    t.update(update["id"], update["fields"])
    time.sleep(0.25)

# %% [markdown]
# ## Process 2: Get the updated table and update locally

# %%
from datetime import datetime
from innovation_sweet_spots import PROJECT_DIR

# %%
# Fetch from Airtable and update the local copy
t = airtable.get_greentech_table()
records = t.all()
au.save_table_locally(t, airtable.AIRTABLE_PATH)

# %%
# Create a table
airtable_df = au.table_to_dataframe(records)

# %% [markdown]
# ### Get latest table of ids, names and tech categories

# %%
columns = ["crunchbase_id", "company_title", "crunchbase_url", "tech_category"]
filename = f"greentech_Crunchbase_companies_{datetime.today().strftime('%Y-%m-%d')}.csv"
(
    airtable_df[columns]
    .explode("tech_category")
    .to_csv(PROJECT_DIR / f"outputs/finals/pilot_outputs/{filename}", index=False)
)

# %% [markdown]
# ### Process 3: Update investment amounts

# %%
# Fetch from Airtable and update the local copy
t = airtable.get_greentech_table()
records = t.all()
au.save_table_locally(t, airtable.AIRTABLE_PATH)

# %%
# Check for companies that don't have Crunchbase ID
companies_with_crunchbase_id = airtable_df[
    (
        (airtable_df.crunchbase_id != "n/a")
        & (airtable_df.crunchbase_id.isnull() == False)
    )
]

# %%
new_investment_ranges = (
    CB.get_funding_rounds(
        (
            companies_with_crunchbase_id
            # Harmonise column names
            .rename(
                columns={
                    "crunchbase_id": "id",
                    "company_title": "name",
                }
            )
        )[["id", "name"]]
    )
    .groupby("org_id", as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(
        raised_investment_range=lambda df: df.raised_amount_gbp.apply(
            au.investment_amount_to_range
        )
    )
    .rename(columns={"org_id": "crunchbase_id"})
)

# %%
compare_investments = new_investment_ranges.merge(
    companies_with_crunchbase_id[
        [
            "airtable_id",
            "crunchbase_id",
            "tech_category",
            "company_title",
            "raised_investment_range",
            "crunchbase_url",
        ]
    ].rename(columns={"raised_investment_range": "raised_investment_range_old"}),
    how="left",
)

# %%
df_changes = compare_investments[
    compare_investments["raised_investment_range"]
    != compare_investments["raised_investment_range_old"]
]
df_changes.to_csv(
    PROJECT_DIR
    / f"outputs/finals/pilot_outputs/airtable_updates/update_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.csv"
)


# %%
df_changes

# %%
importlib.reload(au)


# %%
### Double check that the amount has increased

# %%
df_changes_to_update = df_changes[
    df_changes.raised_investment_range.apply(au.get_lowest_range)
    > df_changes.raised_investment_range_old.apply(au.get_lowest_range)
]

# %%
df_changes_to_update

# %%
updates = au.dict_from_dataframe(df_changes_to_update, ["raised_investment_range"])

# %%
len(updates)

# %%
categories_to_report = ["Energy management", "Low carbon heating"]
updated_companies = (
    df_changes_to_update.explode("tech_category")
    .query("tech_category in @categories_to_report")
    .drop_duplicates("crunchbase_id")
    .company_title.to_list()
)
updated_companies

# %%
len(updated_companies)

# %%
for update in tqdm(updates, total=len(updates)):
    t.update(update["id"], update["fields"])
    time.sleep(0.25)

# %%
