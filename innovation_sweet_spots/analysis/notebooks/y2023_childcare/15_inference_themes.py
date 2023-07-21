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

# # Code to do inference on themes

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
    df_to_hf_ds,
    label_counts,
    plot_label_counts,
)

# Reupload to google sheets
import innovation_sweet_spots.utils.google_sheets as gs

import pandas as pd
import random
import numpy as np
import torch

# +

SAVE_DS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/dataset_v2023_04_12"
# -

# ## Import datasets

# Get longlist of companies
longlist_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)

# +
# Load dataframes
train_df = load_pickle(SAVE_DS_PATH / "train_df.pickle")
valid_df = load_pickle(SAVE_DS_PATH / "test_df.pickle")
to_review_df = load_pickle(SAVE_DS_PATH / "to_review_df.pickle")

# iterate through columns and replace some but not all 0s randomly with 1s in to_review_df dataframe
# otherwise trainer complains about lack of classes
for col in to_review_df.columns:
    if col in ["id", "text"]:
        continue
    else:
        # replace column with random 1s and 0s
        to_review_df[col] = np.random.choice(
            [0, 1], size=len(to_review_df[col]), p=[0.5, 0.5]
        )
# -

# Load datasets
train_ds = load_pickle(SAVE_DS_PATH / "train_ds.pickle")
valid_ds = load_pickle(SAVE_DS_PATH / "test_ds.pickle")
to_review_ds = df_to_hf_ds(to_review_df)

# ## Inference

# +
# Load model trainer
MODEL_PATH = (
    PROJECT_DIR
    / "outputs/2023_childcare/model_themes_v2023_04_12_colab/checkpoint-2000"
)
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
# all_datasets = pd.concat([train, valid, to_review])

# +
# reviewed_datasets = ['train', 'validation']
# -

# - In cases, where no model prediction has been made, assess the probabilities and choose the highest one
# - For train and validation, overwrite the model predictions with "reviewer labels" (or combine them?)

# ### Get the probabilities

# +
from innovation_sweet_spots import logging

sigmoid = torch.nn.Sigmoid()

# label array
label_array = list(train_df.columns[2:])
label_array

# relevant ids
relevant_ids = longlist_df.query("relevant == '1'").cb_id.to_list()


# -


def process_labels(
    predictions_df: pd.DataFrame,
    predictions,
    replace_model_with_reviewer: bool,
    label_array: list,
):
    """
    Process the labels from the model predictions and add them to the predictions dataframe.

    Args:
        predictions (TrainerPrediction): The predictions from the model.
        predictions_df (pd.DataFrame): The dataframe containing the predictions.
        replace_model_with_reviewer (bool): Whether to replace the model labels with the reviewer labels.

    Returns:
        pd.DataFrame: The predictions dataframe with the model labels added.
    """
    predictions_df = predictions_df.copy().reset_index(drop=True)
    logging.info(f"------")
    logging.info(f"Processing labels for {len(predictions_df)} rows")
    # Get the probabilities
    probs_array = sigmoid(torch.Tensor(predictions.predictions))
    # find best prediction for each row
    preds = np.argmax(probs_array, axis=1)
    # Find the label for each row
    labels = [label_array[i] for i in preds]
    # Check the ids that had no label predictions
    rows_to_add_labels = predictions_df.query("model_labels == ''").index.to_list()

    # Add the new labels
    logging.info(f"Mising model labels: Adding {len(rows_to_add_labels)} model labels")
    predictions_df.loc[rows_to_add_labels, "model_labels"] = [
        {labels[i]} for i in rows_to_add_labels
    ]

    # Check disagreements between reviewer and model
    n = len(predictions_df.query("reviewer_labels != model_labels"))
    logging.info(
        f"Disagreements: Found {n} disagreements between reviewer and model labels"
    )
    # Replace model labels with reviewer labels if required
    if replace_model_with_reviewer:
        logging.info(f"Replacing {n} model labels with reviewer labels")
        predictions_df["model_labels"] = predictions_df["reviewer_labels"]
    return predictions_df


train_processed = process_labels(
    train, predictions_train, replace_model_with_reviewer=True, label_array=label_array
)
valid_processed = process_labels(
    valid, predictions_valid, replace_model_with_reviewer=True, label_array=label_array
)
to_review_processed = process_labels(
    to_review,
    predictions_review,
    replace_model_with_reviewer=False,
    label_array=label_array,
)

all_datasets = pd.concat([train_processed, valid_processed, to_review_processed])

# number of labels per row
all_datasets.model_labels.apply(len).value_counts()

# ### Manual corrections

# LOAD IN MANUALLY LABELED DATA
updated_ids_df = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/interim/model_themes_v2023_04_06_updates.csv"
)

# +
from ast import literal_eval

updated_ids_df = (
    updated_ids_df.assign(model_labels=lambda df: df.subthemes_y.apply(literal_eval))
    .assign(model_labels=lambda df: df.model_labels.apply(set))
    .drop(["subthemes_x", "subthemes_y"], axis=1)
    .rename(columns={"cb_id": "id"})
)

updated_ids_df

# -

# UPDATE model_labels values for rows that have the same cb_id in all_datasets and updated_ids_df
all_datasets = all_datasets.merge(
    updated_ids_df[["id", "model_labels"]], on="id", how="left"
).assign(model_labels=lambda df: df.model_labels_y.combine_first(df.model_labels_x))

# ## Inference, all prob values

predictions_train[-1]

predictions_valid[-1]

# ### Check if there are ambiguous cases
#
# Or any cases with multiple classifciations at all

sigmoid = torch.nn.Sigmoid()
probs_array = sigmoid(torch.Tensor(predictions_review.predictions))
# get the second largest value in each row
x = probs_array.sort(axis=1)[0][:, -2]
# histogram with bins of 0.1
# pd.cut(x, bins=np.arange(0, 1, 0.1)).value_counts()
# probs_array.sum(axis=1).max()

# calculate entropy of each row
entropy = -torch.sum(probs_array * torch.log(probs_array), dim=1)

# use non-scientific notation
max(probs_array[0])

entropy.min()

# ## Finalise the inference
# Keep all companies that are:
# - Evaluated manually as 'relevant'
# - Not evaluated manually, but model relevance = 1, and ChatGPT is not <not_relevant>
#
# NB: Will need to manually review cases, where evaluated manually as 'relevant' but ChatGPT is <not_relevant>

# +
# # Get longlist of companies
# longlist_df = utils.gs.download_google_sheet(
#     google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
#     wks_name="list_v2",
# )
# -

all_datasets.info()

updated_ids_not_relevant = updated_ids_df[
    updated_ids_df["model_labels"].apply(str) == "{'<not_relevant>'}"
].id.to_list()
updated_ids_relevant = updated_ids_df[
    updated_ids_df["model_labels"].apply(str) != "{'<not_relevant>'}"
].id.to_list()

# +
merged_df = (
    longlist_df.merge(
        all_datasets.rename(columns={"id": "cb_id"})[
            ["cb_id", "model_labels", "pred_probs"]
        ],
        how="left",
        on="cb_id",
    )
    # convert model_labels column to list
    .assign(
        model_labels=lambda df: df.model_labels.apply(
            lambda x: x if x is None else list(x)
        )
    )
    # round pred_probs to 2 decimal places
    .assign(
        pred_probs=lambda df: df.pred_probs.apply(
            lambda x: x if x is None else [np.round(prob, 2) for prob in x]
        )
    )
    # rename columns
    .rename(
        columns={
            "model_labels": "model_subthemes",
            "pred_probs": "model_subtheme_probs",
        }
    )
    # explode the subthemes and subtheme_probs columns
    .explode(["model_subthemes"])
    # create a column named keep
    .assign(keep=False)
    # assign keep to True if relevant = 1 or relevant = not evaluated and subthemes is not '<not_relevant>'
    .assign(
        keep=lambda df: (df.relevant == "1")
        | ((df.relevant == "not evaluated") & (df.model_subthemes != "<not_relevant>"))
    )
    .fillna("")
    .reset_index(drop=True)
)
# set column 'relevant' to 1 if column cb_id values are in updated_ids_relevant; keep other rows the same
merged_df.loc[merged_df.cb_id.isin(updated_ids_relevant), "relevant"] = 1
# set assign column 'relevant' to 1 if column cb_id values are in updated_ids_relevant; keep other rows the same
merged_df.loc[merged_df.cb_id.isin(updated_ids_not_relevant), "relevant"] = 0

# merged_df = merged_df[merged_df.keep == True]

# -

merged_df.model_subthemes.value_counts()

merged_df.relevant

# # Merge, upload and review

# +
# updated_df = gs.download_google_sheet(
#     google_sheet_id="141iLNJ5e4NHlsxf73L0GX3LMxmZk-VxIDxoYug5Aglg",
#     wks_name="list_v3",
# )
# -

gs.upload_to_google_sheet(
    df=merged_df,
    google_sheet_id="141iLNJ5e4NHlsxf73L0GX3LMxmZk-VxIDxoYug5Aglg",
    wks_name="list_v3",
)

sorted(list(merged_df.model_subthemes.unique()))

# # Evaluate the results

# - How many companies relevant in the end?
# - How many companies in each theme?
# - Visualise the space of companies... use streamlit or flourish?
# - What are the investment trends (use the new data)

sorted(merged_df.model_subthemes.unique())

# ### Basic stats

final_df = merged_df[merged_df.keep == True]
len(final_df)

final_df.model_theme.value_counts()

# ### Visualisation


# ### Investments stats


# +


# sigmoid = torch.nn.Sigmoid()


# def get_relevance_score(predictions, df: pd.DataFrame) -> pd.DataFrame:
#     # get probabilities
#     probs = sigmoid(torch.Tensor(predictions.predictions))
#     # dataframe
#     df_probs = (
#         df.copy()
#         .assign(model_relevant_prob=probs[:, 0])
#         .assign(model_not_relevant_prob=probs[:, 1])
#     )
#     df_probs["model_relevant"] = df_probs["model_relevant_prob"] / (
#         df_probs["model_relevant_prob"] + df_probs["model_not_relevant_prob"]
#     )
#     return df_probs


# def get_all_relevance_scores(list_of_predictions, list_of_dfs) -> pd.DataFrame:
#     df_probs = []
#     for i, predictions in enumerate(list_of_predictions):
#         df_probs.append(get_relevance_score(predictions, list_of_dfs[i]))
#     return pd.concat(df_probs, ignore_index=True)

# +
# df = get_all_relevance_scores(
#     [predictions_train, predictions_valid, predictions_review],
#     [train_df, valid_df, to_review_df],
# )
