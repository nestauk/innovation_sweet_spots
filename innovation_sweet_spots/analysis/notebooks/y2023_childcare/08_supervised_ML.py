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
#     display_name: base
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

# ## Inputs
#

# Get longlist of companies
longlist_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)
# Load a table with processed company descriptions
processed_texts = get_preprocessed_crunchbase_descriptions()

# ## Prep datasets

# +
# Process data for supervised ML
data = longlist_df[["cb_id", "relevant", "cluster_relevance"]].copy()

# Select a random 50% sample of 'noise' clusters and set their relevance to 0
# This is somewhat ad-hoc, and we could also not use the noise clusters at all here
companies_in_noise_clusters = (
    data.query('cluster_relevance == "noise"')
    .query('relevant != "1"')
    .sample(frac=0.5, random_state=42)
)
data.loc[companies_in_noise_clusters.index, "relevant"] = "0"

# Add processed company descriptions
data = (
    data.rename({"cb_id": "id"}, axis=1)
    .merge(processed_texts, on="id")
    .drop(["name", "cluster_relevance"], axis=1)
    .rename({"description": "text"}, axis=1)
)[["id", "text", "relevant"]]

# Create a training set and a test set out from the labelled data,
# with 80% of the relevant and not-relevant labels in the training set
data_labelled = data.query('relevant != "not evaluated"').astype({"relevant": "int"})

# Unlabelled data, to be labelled by the model
data_unlabelled = (
    data.query('relevant == "not evaluated"')
    # make all values in relevant null
    .assign(relevant=0)
)

train_df, test_df = train_test_split(
    data_labelled, test_size=0.2, stratify=data_labelled["relevant"], random_state=42
)

# Split column relevant into relevant and not_relevant
train_df = train_df.assign(not_relevant=lambda x: 1 - x["relevant"])
test_df = test_df.assign(not_relevant=lambda x: 1 - x["relevant"])
data_unlabelled = data_unlabelled.assign(not_relevant=0)

# -

# ## Create hugging face datasets

# Make datasets
train_ds = df_to_hf_ds(train_df)
test_ds = df_to_hf_ds(test_df)
to_review_ds = df_to_hf_ds(data_unlabelled)

# Save datasets
SAVE_DS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/dataset_v2023_03_03"
save_pickle(train_ds, SAVE_DS_PATH / "train_ds.pickle")
save_pickle(test_ds, SAVE_DS_PATH / "test_ds.pickle")
save_pickle(to_review_ds, SAVE_DS_PATH / "to_review_ds.pickle")

# ## Training a model

from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    load_training_args,
    load_model,
    load_trainer,
    compute_metrics,
)

train_ds = load_pickle(SAVE_DS_PATH / "train_ds.pickle")
test_ds = load_pickle(SAVE_DS_PATH / "test_ds.pickle")
to_review_ds = load_pickle(SAVE_DS_PATH / "to_review_ds.pickle")

# Set number of labels
NUM_LABELS = len(train_ds[0]["labels"])
NUM_LABELS

# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/"
SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load model
model = load_model(num_labels=NUM_LABELS)

# Train model with early stopping
training_args = load_training_args(output_dir=SAVE_TRAINING_RESULTS_PATH)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds
)
trainer.train()

# Evaluate model
trainer.evaluate()

# View f1, roc and accuracy of predictions on validation set
predictions = trainer.predict(test_ds)
compute_metrics(predictions)
