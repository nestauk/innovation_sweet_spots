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

# +
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_pickle
import utils

from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    load_tokenizer,
    load_training_args,
    load_model,
    load_trainer,
    add_reviewer_and_predicted_labels,
    label_counts,
    plot_label_counts,
)

import pandas as pd

# -

# Load dataframes
train_df = load_pickle(utils.SAVE_DS_PATH / "train_df.pickle")
valid_df = load_pickle(utils.SAVE_DS_PATH / "test_df.pickle")
to_review_df = load_pickle(utils.SAVE_DS_PATH / "to_review_df.pickle")

# Load datasets
train_ds = load_pickle(utils.SAVE_DS_PATH / "train_ds.pickle")
valid_ds = load_pickle(utils.SAVE_DS_PATH / "test_ds.pickle")
to_review_ds = load_pickle(utils.SAVE_DS_PATH / "to_review_ds.pickle")

# +
# Load model trainer
MODEL_PATH = utils.SAVE_DS_PATH.parent / "checkpoint-500_v1"
NUM_LABELS = len(train_ds[0]["labels"])

model = load_model(num_labels=NUM_LABELS, model_path=MODEL_PATH)
tokenizer = load_tokenizer()
training_args = load_training_args(output_dir=utils.SAVE_DS_PATH.parent)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds
)
# -

# Make predictions and add labels columns for each dataset then combine
train = add_reviewer_and_predicted_labels(
    df=train_df, ds=train_ds, trainer=trainer, model_dataset_name="train"
)
valid = add_reviewer_and_predicted_labels(
    df=valid_df, ds=valid_ds, trainer=trainer, model_dataset_name="validation"
)
to_review = add_reviewer_and_predicted_labels(
    df=to_review_df,
    ds=to_review_ds,
    trainer=trainer,
    model_dataset_name="to_review",
)

predictions_train = trainer.predict(train_ds)
predictions_valid = trainer.predict(valid_ds)

predictions_review = trainer.predict(to_review_ds)

# +
import torch

sigmoid = torch.nn.Sigmoid()


def get_relevance_score(predictions, df: pd.DataFrame) -> pd.DataFrame:
    # get probabilities
    probs = sigmoid(torch.Tensor(predictions.predictions))
    # dataframe
    df_probs = (
        df.copy()
        .assign(model_relevant_prob=probs[:, 0])
        .assign(model_not_relevant_prob=probs[:, 1])
    )
    df_probs["model_relevant"] = df_probs["model_relevant_prob"] / (
        df_probs["model_relevant_prob"] + df_probs["model_not_relevant_prob"]
    )
    return df_probs


def get_all_relevance_scores(list_of_predictions, list_of_dfs) -> pd.DataFrame:
    df_probs = []
    for i, predictions in enumerate(list_of_predictions):
        df_probs.append(get_relevance_score(predictions, list_of_dfs[i]))
    return pd.concat(df_probs, ignore_index=True)


# -

df = get_all_relevance_scores(
    [predictions_train, predictions_valid, predictions_review],
    [train_df, valid_df, to_review_df],
)

# remove altair row limit
alt.data_transformers.disable_max_rows()
# plot scatter plot of model_relevant_prob vs model_not_relevant_prob
import altair as alt

alt.Chart(df).mark_circle().encode(
    x="model_relevant_prob",
    y="model_not_relevant_prob",
)


df = df.drop_duplicates("id")
len(df)

# Get longlist of companies
longlist_df = utils.gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)

len(longlist_df)

longlist_df.columns

# Add data to google sheets data
merged = (
    longlist_df[["company_name", "homepage_url", "cb_id", "relevant"]].merge(
        df[["id", "model_relevant"]].rename(columns={"id": "cb_id"}),
        how="left",
        on="cb_id",
    )
    # .drop(columns="text")
    .fillna("")
)

# +
# pd.set_option("max_colwidth", 1000)
# merged

# +
# # Plot model predicted label counts
# model_label_counts = label_counts(df=merged, column_to_count="model_relevant")
# plot_label_counts(
#     model_label_counts,
#     excl_labels=[],
#     title="Model predicted label counts",
# )
# -

merged

utils.gs.upload_to_google_sheet

# Reupload to google sheets
import innovation_sweet_spots.utils.google_sheets as gs

gs.upload_to_google_sheet(
    df=merged,
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2_model_relevance",
)

# +
# Add data to google sheets data
# merged = ft_gtr.merge(all_datasets, how="left", on="id").drop(columns="text").fillna("")
# -

merged
