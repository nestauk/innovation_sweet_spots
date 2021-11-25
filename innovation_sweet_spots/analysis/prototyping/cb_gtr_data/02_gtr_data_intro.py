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
# # GtR data exploration
# - Load data
#
# Prerequisite: Fetching GtR data by running `make fetch-daps1` from the main repo directory

# %%
from innovation_sweet_spots.getters import gtr
import innovation_sweet_spots.analysis.analysis_utils as iss
import altair as alt
