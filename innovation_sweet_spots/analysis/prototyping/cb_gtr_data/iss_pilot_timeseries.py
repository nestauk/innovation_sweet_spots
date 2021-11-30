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
# # Time series of green technologies
#
# Time series of research funding and business investment related to specific technologies, generated as part of the ISS pilot project.

# %%
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.analysis.analysis_utils as iss
import altair as alt
import pandas as pd

# %% [markdown]
# ## Time series

# %%
# Fetch the time series generated as part of the ISS pilot
outputs_dir = PROJECT_DIR / "outputs/finals/pilot_outputs"
iss_time_series = pd.read_csv(outputs_dir / "ISS_pilot_Time_series.csv")

# %%
iss_time_series.head(1)

# %% [markdown]
# The columns hold the following data:
# `year` = year (between 2007 and 2021)
# `no_of_projects` = number of research projects started in a specific `year` (GtR)
# `amount_total` = research funding amount awareded in a specific `year` (GtR)
# `raised_amount_total_gbp` = raised investment amount (GBP) a specific `year` (Crunchbase)
# `no_of_orgs_founded` = number of new companies started in the `year` (Crunchbase)
# `articles` = number of articles in the Guardian with the technology keywords and phrases
# `speeches` = number of speeches in Hansard with the technology keywords and phrases

# %%
# Technology categories
print(iss_time_series.tech_category.unique())

# %%
iss_time_series

# %% [markdown]
# - It would be interesting to check potential relationships, e.g. correlations and lags, between GtR and Crunchbase time series, i.e. `amount_total` or `no_of_projects` versus `raised_amount_gbp_total` or `no_of_rounds`.
# - Makes sense to check the following, broader tech categories: 'Low carbon heating', 'EEM', 'Solar' 'Wind & offshore' 'Hydrogen & fuel cells', 'Batteries', 'Bioenergy', 'Carbon capture & storage' (the other, more granular low carbon heating categories will have very small number of investment deals)
# - If the time series are too noisy, you could use try using a moving average

# %%
# Rolling moving average
iss.get_moving_average(
    iss_time_series[iss_time_series.tech_category == "EEM"], rename_cols=True
)

# %% [markdown]
# ## Underlying data

# %%
# GtR projects
gtr_projects = pd.read_csv(outputs_dir / "ISS_pilot_GtR_projects.csv")
gtr_projects.head()

# %%
# Crunchbase companies (note: to get the deals/rounds, you'll have to add investments data)
cb_companies = pd.read_csv(outputs_dir / "ISS_pilot_Crunchbase_companies.csv")
cb_companies.head()

# %%
# To add investment deals data
from innovation_sweet_spots.getters import crunchbase

CB_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %%
# Investment deals for companies related to solar technologies
cb_df = cb_companies[cb_companies.tech_category == "Solar"]
iss.get_cb_org_funding_rounds(cb_df, CB_funding_rounds)

# %%
