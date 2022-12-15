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

# %%
from innovation_sweet_spots.getters.google_sheets import get_foodtech_reviewed_gtr
import numpy as np
from innovation_sweet_spots.utils.io import save_pickle
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    combine_labels,
    add_binarise_labels,
    rename_columns,
)

# %% [markdown]
# ## Downloading data from google sheets and converting into a dataframe
#
# The following process is for datasets that potentially have multiple rows with the same id but with different labels.
# The functions used in the next stage, in the 01 notebook expect that the dataframes are in the same format as produced in this notebook i.e. id | text | label_1 | label_2 | .... | label_n | where the label columns contain values of 0s and 1s.

# %%
# Load and process foodtech GtR abstracts from googlesheets that have been partially reviewed
ft_gtr = (
    get_foodtech_reviewed_gtr(from_local=False, save_locally=False)
    .replace("", np.NaN)
    .fillna(value={"reviewed": "0"})
    .astype({"reviewed": "int8"})
    .dropna(subset=["abstractText"])[
        ["id", "title", "abstractText", "tech_area", "tech_area_checked", "reviewed"]
    ]
    .pipe(
        combine_labels,
        groupby_cols=["id", "abstractText", "reviewed"],
        label_column="tech_area_checked",
    )
    .pipe(add_binarise_labels, label_column="tech_area_checked", not_valid_label="-")
    .sample(frac=1, random_state=1)
    .pipe(rename_columns, text_column="abstractText")
    .drop(columns=["tech_area_checked"])
)

# %%
# Calculate 80% training split
split = int(len(ft_gtr) * 0.8)

# %%
# Set path to save dataframes
SAVE_DF_PATH = PROJECT_DIR / "inputs/data/review_labelling/dataframes/foodtech_gtr/"
SAVE_DF_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Make and save training dataframe using records that have been human reviewed
train_df = (
    ft_gtr.query("reviewed == 1")[:split]
    .drop(columns="reviewed")
    .reset_index(drop=True)
)
save_pickle(train_df, SAVE_DF_PATH / "train_df.pickle")
train_df.head(2)

# %%
# Make and save validation dataframe using records that have been human reviewed
valid_df = (
    ft_gtr.query("reviewed == 1")[split:]
    .drop(columns="reviewed")
    .reset_index(drop=True)
)
save_pickle(valid_df, SAVE_DF_PATH / "valid_df.pickle")
valid_df.head(2)

# %%
# Make and save set of records to be classified using the model that have not been human reviewed
to_review_df = (
    ft_gtr.query("reviewed == 0").drop(columns="reviewed").reset_index(drop=True)
)
save_pickle(to_review_df, SAVE_DF_PATH / "to_review_df.pickle")
to_review_df.head(2)
