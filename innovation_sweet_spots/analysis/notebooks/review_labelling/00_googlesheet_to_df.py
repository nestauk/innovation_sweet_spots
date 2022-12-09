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
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from innovation_sweet_spots.utils.io import save_pickle
from innovation_sweet_spots import PROJECT_DIR

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
)

# %%
# View the data
ft_gtr

# %%
# Check to see if any were reviewed but the reviewed column was not updated
reviewed_not_recorded = ft_gtr.query("reviewed == 0").query(
    "tech_area != tech_area_checked"
)
reviewed_not_recorded

# %%
# Currently, in cases where the project belongs to multiple tech areas,
# There are multiple rows with the same id but different tech_area values
ft_gtr.id.value_counts()

# %%
# Make the data have 1 row per id and a list in the colum tech_area_checked
ft_gtr_grouped = (
    ft_gtr.groupby(["id", "title", "abstractText", "reviewed"])["tech_area_checked"]
    .apply(list)
    .reset_index()
)

# %%
# View the new format
ft_gtr_grouped

# %%
# Records that have not been reviewed
not_reviewed = ft_gtr_grouped.query("reviewed == 0")
not_reviewed

# %%
# All the not reviewed projects were found by key word search to be in the Biomedical field
# Once our model is trained, we can use it to make predictions for these records
not_reviewed.tech_area_checked.value_counts()

# %%
# Make dummy cols
mlb = MultiLabelBinarizer()

dummy_cols = pd.DataFrame(
    mlb.fit_transform(ft_gtr_grouped["tech_area_checked"]),
    columns=mlb.classes_,
    index=ft_gtr_grouped.index,
)

# %%
# View dummy cols
dummy_cols

# %%
# When '-' is recorded it means it is not part of the other fields
# Therefore we need to make sure when the '-' column has a 1,
# the other columns must be 0.
# We can then remove the '-' col.
cols_excl_dash = [col for col in dummy_cols.columns if col != "-"]
dummy_cols = dummy_cols[cols_excl_dash].mask(dummy_cols["-"] == 1, 0)

# %%
# Combine and shuffle data
ft_gtr_w_dummy = pd.concat([ft_gtr_grouped, dummy_cols], axis=1).sample(
    frac=1, random_state=1
)


# %%
def keep_relevant_cols(ft_gtr_df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not needed"""
    return ft_gtr_df.drop(columns=["title", "reviewed", "tech_area_checked"])


def rename_columns(ft_gtr_df: pd.DataFrame) -> pd.DataFrame:
    """For column names:
        Replace space and - with _
        Make lowercase
        Change 'abstracttext' -> 'text'
    """
    ft_gtr_df.columns = ft_gtr_df.columns.str.replace("\s|-", "_", regex=True)
    return ft_gtr_df.rename(columns=str.lower).rename(columns={"abstracttext": "text"})


def process_columns(ft_gtr_df: pd.DataFrame) -> pd.DataFrame:
    """Pipe keep_relevant_cols and rename_columns functions"""
    return ft_gtr_df.pipe(keep_relevant_cols).pipe(rename_columns)


# %%
# Calculate 80% training split
split = int(len(ft_gtr_w_dummy) * 0.8)

# %%
# Set path to save dataframes
SAVE_DF_PATH = PROJECT_DIR / "inputs/data/review_labelling/dataframes/foodtech_gtr/"
SAVE_DF_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Make and save training dataframe
train_df = ft_gtr_w_dummy.query("reviewed == 1")[:split].pipe(process_columns)
save_pickle(train_df, SAVE_DF_PATH / "train_df.pickle")
train_df.head(3)

# %%
# Make and save validation dataframe
valid_df = ft_gtr_w_dummy.query("reviewed == 1")[split:].pipe(process_columns)
save_pickle(valid_df, SAVE_DF_PATH / "valid_df.pickle")
valid_df.head(3)

# %%
# Make and save set of records to be reviewed using the model
to_review_df = ft_gtr_w_dummy.query("reviewed == 0").pipe(process_columns)
save_pickle(to_review_df, SAVE_DF_PATH / "to_review_df.pickle")
to_review_df.head(3)

# %%
