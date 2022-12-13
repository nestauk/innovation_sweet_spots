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
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from innovation_sweet_spots.utils.io import save_pickle, load_pickle
from innovation_sweet_spots import PROJECT_DIR
import transformers
import datasets
import torch

# %%
# Load dataframes
LOAD_DF_PATH = PROJECT_DIR / "inputs/data/review_labelling/dataframes/foodtech_gtr/"
train_df = load_pickle(LOAD_DF_PATH / "train_df.pickle")
valid_df = load_pickle(LOAD_DF_PATH / "valid_df.pickle")
to_review_df = load_pickle(LOAD_DF_PATH / "to_review_df.pickle")


# %%
def create_labels(dataset: Dataset, cols_to_skip: list) -> Dataset:
    cols = dataset.column_names
    return dataset.map(
        lambda row: {
            "labels": torch.FloatTensor(
                [(row[col]) for col in cols if col not in cols_to_skip]
            )
        }
    )


def tokenize_dataset(dataset: Dataset, text_column: str) -> Dataset:
    remove_cols = dataset.column_names
    remove_cols.remove("labels")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", problem_type="multi_label_classification"
    )
    return dataset.map(
        lambda row: tokenizer(row[text_column], truncation=True),
        batched=True,
        remove_columns=remove_cols,
    )


def df_to_hf_ds(
    df: pd.DataFrame, non_label_cols: list = ["text", "id"], text_column: str = "text"
) -> Dataset:
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = create_labels(dataset, cols_to_skip=non_label_cols)
    return tokenize_dataset(dataset, text_column="text")


# %%
# Make datasets
train_ds = df_to_hf_ds(train_df)
valid_ds = df_to_hf_ds(valid_df)
to_review_ds = df_to_hf_ds(to_review_df)

# %%
# Set path to save datasets
SAVE_DS_PATH = PROJECT_DIR / "inputs/data/review_labelling/datasets/foodtech_gtr/"
SAVE_DS_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Save datasets
save_pickle(train_ds, SAVE_DS_PATH / "train_ds.pickle")
save_pickle(valid_ds, SAVE_DS_PATH / "valid_ds.pickle")
save_pickle(to_review_ds, SAVE_DS_PATH / "to_review_ds.pickle")
