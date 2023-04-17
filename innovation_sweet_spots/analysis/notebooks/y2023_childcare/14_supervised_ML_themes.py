# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Evaluating relevant childcare companies
#
# - Fetch childcare companies from the google sheets
# - Prepare additional set of negative samples
# - Prepare dataset for training a model
# - Train a model to predict "relevant" childcare companies
#
#

# +
from sklearn.model_selection import train_test_split
import numpy as np
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.getters.google_sheets as gs
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)

from innovation_sweet_spots.utils.io import save_pickle, load_pickle
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import df_to_hf_ds
import utils

# -

from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    combine_labels,
    add_binarise_labels,
    rename_columns,
)

# ## Inputs
#

# Get longlist of companies
longlist_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)
# Load a table with processed company descriptions
processed_texts = get_preprocessed_crunchbase_descriptions()

# Get processed labels
import pandas as pd

id_to_subtheme_df = (
    pd.read_csv(
        PROJECT_DIR
        / "outputs/2023_childcare/interim/openai/subtheme_labels_v2023_04_06.csv"
    )
    # replace <Workforce: Training>, <Workforce: Recruitment> and <Workforce: Optimisation> labels with <Workforce>
    .assign(
        subthemes_list=lambda df: df.subthemes_list.str.replace(
            "Workforce: Training", "Workforce"
        )
    )
    .assign(
        subthemes_list=lambda df: df.subthemes_list.str.replace(
            "Workforce: Recruitment", "Workforce"
        )
    )
    .assign(
        subthemes_list=lambda df: df.subthemes_list.str.replace(
            "Workforce: Optimisation", "Workforce"
        )
    )
    # repl
)

id_to_subtheme_df.subthemes_list.value_counts()


# ## Prep datasets

LABEL_COLUMN = "subtheme"

longlist_df.to_csv(
    PROJECT_DIR
    / "outputs/2023_childcare/model/dataset_v2023_04_12/longlist_v2023_04_12.csv",
    index=False,
)

# +
# Process data for supervised ML
data = (
    longlist_df[["cb_id", "relevant", "cluster_relevance"]]
    .copy()
    .merge(id_to_subtheme_df[["cb_id", "subthemes_list"]], on="cb_id", how="left")
    .rename(columns={"subthemes_list": "subtheme"})
)

# Add processed company descriptions
data = (
    data.rename({"cb_id": "id"}, axis=1)
    .merge(processed_texts, on="id")
    .drop(["name", "cluster_relevance"], axis=1)
    .rename({"description": "text"}, axis=1)
    # if relevant = 1, but subtheme is <Not relevant>, then set subtheme to nan (to avoid training wrong labels)
    .assign(
        subtheme=lambda df: np.where(
            (df.relevant == "1") & (df.subtheme == "<Not relevant>"),
            np.nan,
            df.subtheme,
        )
    )
)[["id", "text", "relevant", "subtheme"]]

# -

data

data_model = (
    data
    # add a dummy row for the null label
    .append(
        {
            "id": "null",
            "text": "null",
            "relevant": 0,
            LABEL_COLUMN: "-",
        },
        ignore_index=True,
    )
    # assign reviewed=1 if subtheme not null
    .assign(reviewed=lambda df: df.subtheme.notnull().astype(int))
    # assign empty string to null subthemes
    .assign(subtheme=lambda df: df.subtheme.fillna(""))
    # .astype({"reviewed": "int8"})
    .pipe(
        combine_labels,
        groupby_cols=["id", "text", "relevant", "reviewed"],
        label_column=LABEL_COLUMN,
    )
    .pipe(add_binarise_labels, label_column=LABEL_COLUMN, not_valid_label="-")
    .sample(frac=1, random_state=1)
    .pipe(rename_columns, text_column="text")
    .drop(columns=[LABEL_COLUMN, "relevant", ""])
)

data_model.columns

data_model.head(2)

# +
train_df, test_df = train_test_split(
    data_model.query("reviewed == 1").drop(columns=["reviewed"]),
    test_size=0.2,
    random_state=42,
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Unlabelled data, to be labelled by the model
to_review_df = (
    data_model.query("reviewed != 1").drop(columns=["reviewed"])
).reset_index(drop=True)
# -

SAVE_DS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/dataset_v2023_04_12"
SAVE_DS_PATH.mkdir(exist_ok=True, parents=True)

# Make and save training dataframe using records that have been human reviewed
save_pickle(train_df, SAVE_DS_PATH / "train_df.pickle")
save_pickle(test_df, SAVE_DS_PATH / "test_df.pickle")
save_pickle(to_review_df, SAVE_DS_PATH / "to_review_df.pickle")

# ## Create hugging face datasets

# Make datasets
train_ds = df_to_hf_ds(train_df)
test_ds = df_to_hf_ds(test_df)
to_review_ds = df_to_hf_ds(to_review_df)

# Save datasets
save_pickle(train_ds, SAVE_DS_PATH / "train_ds.pickle")
save_pickle(test_ds, SAVE_DS_PATH / "test_ds.pickle")
save_pickle(to_review_ds, SAVE_DS_PATH / "to_review_ds.pickle")
