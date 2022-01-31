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
from innovation_sweet_spots.utils.io import import_config
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR

# %% [markdown]
# ### See how to split data across the duration of the project

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

# %% [markdown]
# ### Check no data is lost when being split
# You will have needed to have run `innovation_sweet_spots/pipeline/pilot/timeseries_gtr.py` with `SPLIT = True` and `SPLIT = False` for these next checks to work.

# %%
PARAMS = import_config("iss_pilot.yaml")
categories = PARAMS["technology_categories"]
TIME_SERIES = PROJECT_DIR / "outputs/finals/pilot_outputs/time_series/"

# %%
# Print sums for each category for not split and split time series
for category in categories:
    print(category)
    print(
        pd.read_csv(TIME_SERIES / f"Time_series_GtR_{category}.csv").amount_total.sum()
    )
    print(
        pd.read_csv(
            TIME_SERIES / f"Time_series_GtR_split_{category}.csv"
        ).amount_total.sum()
    )
    print("*" * 10)
