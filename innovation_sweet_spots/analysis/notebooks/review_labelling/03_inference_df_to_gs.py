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
)
from innovation_sweet_spots.utils.io import load_pickle

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
# Set label columns
valid_label_columns = valid_df.drop(columns=["id", "text"]).columns

# %%
# Load datasets
LOAD_DS_PATH = PROJECT_DIR / "inputs/data/review_labelling/datasets/foodtech_gtr/"
train_ds = load_pickle(LOAD_DS_PATH / "train_ds.pickle")
valid_ds = load_pickle(LOAD_DS_PATH / "valid_ds.pickle")
to_review_ds = load_pickle(LOAD_DS_PATH / "to_review_ds.pickle")

# %%
# Load model trainer
NUM_LABELS = len(train_ds[0]["labels"])
MODEL_PATH = (
    PROJECT_DIR
    / "outputs/data/review_labelling/models/foodtech_gtr/results/checkpoint-1500"
)
model = load_model(num_labels=NUM_LABELS, model_path=MODEL_PATH)
tokenizer = load_tokenizer()
training_args = load_training_args(
    output_dir=PROJECT_DIR
    / "outputs/data/review_labelling/models/foodtech_gtr/results/"
)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds
)

# %%
# Add reviewer labels to validation set
valid_df["reviewer_labels"] = dummies_to_labels(
    dummy_cols=valid_df.drop(columns=["id", "text"])
)
valid_df = valid_df[["id", "text", "reviewer_labels"]]
valid_df

# %%
# Create dummy col binarised model predictions for validation set
trainer_predictions = trainer.predict(valid_ds)
binarised_preds = binarise_predictions(
    predictions=trainer_predictions.predictions, threshold=0.5
)
binarised_preds = pd.DataFrame(binarised_preds)
binarised_preds.columns = valid_label_columns

# %%
# Convert dummy cols to set of predicted labels for validation set
# And add to valid_df
valid_df["model_labels"] = dummies_to_labels(binarised_preds)

# %%
valid_df

# %%
# Add column to indicate whether reviewer and model labels match
valid_df["reviewer_model_match"] = np.where(
    valid_df.reviewer_labels == valid_df.model_labels, 1, 0
)

# %%

# %%
# Add column to indicate was dataset the record is from
valid_df["model_dataset"] = "validation"

# %%
# Join predictions and related info to googlesheet data
to_join = valid_df.drop(columns="text")
merged = ft_gtr.merge(to_join, how="left", on="id")

# %%
merged

# %%
# Reupload to google sheets
from innovation_sweet_spots.utils.google_sheets import upload_to_google_sheet

upload_to_google_sheet(
    df=merged,
    google_sheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
    wks_name="ukri_updated",
)

# %%
