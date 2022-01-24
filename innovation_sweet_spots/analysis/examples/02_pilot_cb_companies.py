# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
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
# # Companies detected in the pilot project
#
# Check and prepare a compact company table

# %%
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR
import re

OUTPUTS_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs"
FILEPATH = OUTPUTS_DIR / "ISS_pilot_Crunchbase_companies.csv"


# %%
def list_to_comma_sep_string(input_list):
    """
    Converts ['a', 'b'] -> 'a, b'
    """
    return re.sub("'", "", str(input_list)[1:-1])


# %%
# Import the company table
cb_df = pd.read_csv(FILEPATH)
# Collapse the tech categories
company_categories = (
    cb_df.groupby("doc_id")
    .agg({"tech_category": lambda x: list_to_comma_sep_string(x.tolist())})
    .reset_index()
)
# Create a new table
cb_df = (
    cb_df.drop_duplicates("doc_id")
    .drop("tech_category", axis=1)
    .merge(company_categories, on="doc_id", how="left")
)[["title", "tech_category", "homepage_url", "cb_url"]]

# %%
cb_df

# %%
# Export
cb_df.to_csv(OUTPUTS_DIR / "misc/ISS_pilot_Crunchbase_companies_dedup.csv", index=False)

# %%
