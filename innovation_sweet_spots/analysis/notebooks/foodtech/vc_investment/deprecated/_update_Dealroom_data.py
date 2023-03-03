# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Updating Dealroom company data with the latest snapshot

# %%
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR

# %% [markdown]
# ## Load in the updated snapshot

# %%
FOLDER = "inputs/data/dealroom/"
df_old = pd.read_csv(
    PROJECT_DIR / f"{FOLDER}/dealroom_foodtech.csv"
).astype({"id": str})

# %%
# Updated snapshot (accessed November 24th, 2022)
FOLDER_UPDATED = "inputs/data/dealroom/raw_exports/foodtech_2022_11"
df_updated = pd.read_excel(
    PROJECT_DIR / f"{FOLDER_UPDATED}/df8381fc-6e7d-4eab-9ed5-59ecf66074dd.xlsx"
).astype({"id": str})

# %% [markdown]
# ### Check differences between old and new version
#
# NB: The new snapshot only includes the final companies included in our trend analysis.
#
# There are differences in columns, but none of the different columns are used in our analysis.

# %%
len(df_old), len(df_updated)

# %%
# New columns in the new dataset
df_old.columns.difference(df_updated.columns)

# %%
# Missing columns from the old dataset
df_updated.columns.difference(df_old.columns)

# %% [markdown]
# ## Prepare the new dataset

# %%
df_updated.to_csv(
    PROJECT_DIR / f"{FOLDER}/dealroom_foodtech_2022_11_24.csv",
    index=False,
)
