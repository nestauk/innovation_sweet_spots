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
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting GTR research project participant data
#
# Organisations and people

# %%
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR

# Initiate an instance of GtrWrangler and get the data
GtR = GtrWrangler()

# %% [markdown]
# ## Fetching organisation data
#

# %%
# Import projects table
gtr_projects = gtr.get_gtr_projects().head(3)

# %%
GtR.get_organisations_and_locations(gtr_projects)

# %% [markdown]
# ## Finding participants

# %%
GtR.get_persons(gtr_projects)
