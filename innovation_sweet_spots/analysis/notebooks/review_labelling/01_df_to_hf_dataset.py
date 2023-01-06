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
from innovation_sweet_spots.utils.io import save_pickle, load_pickle
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import df_to_hf_ds

# %% [markdown]
# ## Converting dataframe into huggingface dataset

# %%
# Load dataframes
LOAD_DF_PATH = PROJECT_DIR / "inputs/data/review_labelling/dataframes/foodtech_gtr/"
train_df = load_pickle(LOAD_DF_PATH / "train_df.pickle")
valid_df = load_pickle(LOAD_DF_PATH / "valid_df.pickle")
to_review_df = load_pickle(LOAD_DF_PATH / "to_review_df.pickle")

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
