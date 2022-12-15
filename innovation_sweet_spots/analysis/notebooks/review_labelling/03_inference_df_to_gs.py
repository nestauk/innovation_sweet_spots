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
import pandas as pd
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_pickle
import numpy as np
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    dummies_to_labels,
    binarise_predictions,
    load_tokenizer,
    load_training_args,
    load_model,
    load_trainer,
    add_reviewer_and_predicted_labels,
)
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots.utils.google_sheets import upload_to_google_sheet

# %% [markdown]
# ## Use model to make predictions and reupload data to google sheets

# %%
# Load and process foodtech GtR abstracts from googlesheets that have been partially reviewed
ft_gtr = (
    get_foodtech_reviewed_gtr(from_local=False, save_locally=False)
    .replace("", np.NaN)
    .fillna(value={"reviewed": "0"})
    .astype({"reviewed": "int8"})
    .dropna(subset=["abstractText"])
)

# %%
# Load dataframes
LOAD_DF_PATH = PROJECT_DIR / "inputs/data/review_labelling/dataframes/foodtech_gtr/"
train_df = load_pickle(LOAD_DF_PATH / "train_df.pickle")
valid_df = load_pickle(LOAD_DF_PATH / "valid_df.pickle")
to_review_df = load_pickle(LOAD_DF_PATH / "to_review_df.pickle")

# %%
# Load datasets
LOAD_DS_PATH = PROJECT_DIR / "inputs/data/review_labelling/datasets/foodtech_gtr/"
train_ds = load_pickle(LOAD_DS_PATH / "train_ds.pickle")
valid_ds = load_pickle(LOAD_DS_PATH / "valid_ds.pickle")
to_review_ds = load_pickle(LOAD_DS_PATH / "to_review_ds.pickle")

# %%
# Load model trainer
FOOD_TECH_RESULTS_PATH = (
    PROJECT_DIR / "outputs/data/review_labelling/models/foodtech_gtr/results/"
)
MODEL_PATH = FOOD_TECH_RESULTS_PATH / "checkpoint-1500"
NUM_LABELS = len(train_ds[0]["labels"])

model = load_model(num_labels=NUM_LABELS, model_path=MODEL_PATH)
tokenizer = load_tokenizer()
training_args = load_training_args(output_dir=FOOD_TECH_RESULTS_PATH)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds
)

# %%
# Make predictions and add labels columns for each dataset then combine
train = add_reviewer_and_predicted_labels(
    df=train_df, ds=train_ds, trainer=trainer, model_dataset_name="train"
)
valid = add_reviewer_and_predicted_labels(
    df=valid_df, ds=valid_ds, trainer=trainer, model_dataset_name="validation"
)
to_review = add_reviewer_and_predicted_labels(
    df=valid_df,
    ds=valid_ds,
    trainer=trainer,
    model_dataset_name="to_review (reviewer_labels not reviewed)",
)
all_datasets = pd.concat([train, valid, to_review])

# %%
# Add data to google sheets data
merged = ft_gtr.merge(all_datasets, how="left", on="id").drop(columns="text").fillna("")

# %%
# Reupload to google sheets
upload_to_google_sheet(
    df=merged,
    google_sheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
    wks_name="ukri_updated",
)
