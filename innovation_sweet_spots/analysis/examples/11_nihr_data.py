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
# # Checking NIHR data

# %%
from innovation_sweet_spots.getters.nihr import get_nihr_summary_data
from innovation_sweet_spots import PROJECT_DIR

NIHR_FOLDER = PROJECT_DIR / "inputs/data/nihr"

# Fetch data from s3
nihr_df = get_nihr_summary_data()

# %%
len(nihr_df)

# %%
sorted(list(nihr_df.columns))

# %%
nihr_df.iloc[0]

# %%
nihr_df.sample(5)

# %% [markdown]
# # Panda profiler

# %%
from pandas_profiling import ProfileReport

# %%
profile = ProfileReport(nihr_df, title="NIHR summary data report", explorative=False)

# %%
profile.to_file(NIHR_FOLDER / "nihr_summary_data_report.html")

# %%
