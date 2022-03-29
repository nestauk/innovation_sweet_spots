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
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib

importlib.reload(wu)

# %%
DR = wu.DealroomWrangler()

# %%
df = DR.company_data

# %%
DR.company_subindustries

# %%
list(df.columns)
