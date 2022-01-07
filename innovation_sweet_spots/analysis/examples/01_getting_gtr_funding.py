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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting GTR funding data

# %%
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler

# Initiate an instance of GtrWrangler and get the data
wrangler = GtrWrangler()

# %% [markdown]
# ## Fetching funding data
#
# Example of getting project funding amounts, and the start and end dates of funding

# %%
# Import projects table
gtr_projects = gtr.get_gtr_projects()

# %%
# Choose a smaller set of projects
df_proj = gtr_projects.head(3)

# %%
# Add data on funding amounts and the start and end dates of funding
# (the first run might take longer as it needs to load in funding data)
df_proj_with_funds = wrangler.get_funding_data(df_proj)


# %%
# NB: The project 'start' and funding start 'fund_start' dates are most of the time identical, but that's not always the case
df_proj_with_funds

# %% [markdown]
# ## Underlying data tables
#
# Examples of the underlying funding data, which is processed by the `GtrWrangler`

# %%
# Funding data table
gtr_funds = gtr.get_gtr_funds()

# %%
gtr_funds.head(3)

# %%
gtr_funds.groupby("category").agg(counts=("id", "count"))

# %%
gtr_funds.groupby("currencyCode").agg(counts=("id", "count"))

# %%
# Alternative funding data table, that has been generated by retrieving data directly via API (as opposed to fetching from Nesta's database)
# This is used, because the database funding amount data is somewhat ambiguous
gtr.get_gtr_funds_api().head(3)

# %%
# Import funding to projects links table
gtr_funds_link_table = gtr.get_link_table("gtr_funds")

# %%
gtr_funds_link_table.head(3)

# %%
# Note multiple funding ids for the same project_id
gtr_funds_link_table.project_id.duplicated().sum()

# %%
