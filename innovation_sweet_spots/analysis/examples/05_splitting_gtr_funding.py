# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Splitting GtR funding across the duration of the project

# %%
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler
import innovation_sweet_spots.analysis.analysis_utils as au

# %%
# Initiate an instance of GtrWrangler
gtr_wrangler = GtrWrangler()

# %%
# Get pilot GtR projects
gtr_docs = gtr.get_pilot_GtR_projects()

# %%
# Get a sample of 5 records
gtr_docs_sample = gtr_docs.head(5)
gtr_docs_sample

# %%
# Add funding information to sample
funding_data_sample = gtr_wrangler.get_funding_data(gtr_docs_sample)
funding_data_sample

# %%
# See how the data looks after splitting by year
funding_data_split = gtr_wrangler.split_funding_data(
    funding_data_sample, time_period="year"
)
funding_data_split

# %%
# See how the data looks after splitting by month
funding_data_split = gtr_wrangler.split_funding_data(
    funding_data_sample, time_period="month"
)
funding_data_split
