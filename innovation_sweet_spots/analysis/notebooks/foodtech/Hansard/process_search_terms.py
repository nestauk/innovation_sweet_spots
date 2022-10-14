# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Script to process search terms for keyword queriesm

# %%
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
import utils as utils

# %%
import importlib

importlib.reload(utils)

# %%
import innovation_sweet_spots.utils.text_processing_utils as tpu

nlp = tpu.setup_spacy_model()

# %%
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

# %%
# Import search terms
foodtech_terms = get_foodtech_search_terms(from_local=False)
# Import precision check results
df_precision_path = (
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/Food_terms_precision.csv"
)

df_search_results = pd.read_csv(df_search_results_path)
df_precision = pd.read_csv(df_precision_path)

# %%
df_search_terms = foodtech_terms.copy()
df_search_terms["Terms"] = df_search_terms["Terms"].apply(
    utils.remove_space_after_comma
)

# %%
imprecise_terms = df_precision.query("proportion_correct < 0.5").terms.to_list()

# %%
# Remove the following terms
dont_use_terms = ["no-kill meat", "supply chain", "kitchen"]

# %%
terms_to_remove = imprecise_terms + dont_use_terms

# %%
len(terms_to_remove)

# %%
# Check number of search terms
df_search_terms.groupby("Category", as_index=False).agg(
    counts=("Terms", "count")
).sort_values("Category")


# %%
# Check number of search terms, after removing imprecise ones
df_search_terms.query("Terms not in @terms_to_remove").groupby(
    "Category", as_index=False
).agg(counts=("Terms", "count")).sort_values("Category")


# %%
# Check number of search terms, after removing imprecise ones
df_search_terms.query("Terms not in @terms_to_remove").groupby(
    "Sub Category", as_index=False
).agg(counts=("Terms", "count")).sort_values("Sub Category")


# %%
len(df_search_terms), len(df_search_terms.query("Terms not in @terms_to_remove"))

# %%
terms_df = utils.process_foodtech_terms(
    df_search_terms.query("Terms not in @terms_to_remove"), nlp
)
tech_area_terms = utils.compile_term_dict(terms_df, "Tech area")

# %%
from innovation_sweet_spots.utils.io import save_pickle

save_pickle(
    tech_area_terms,
    PROJECT_DIR / "outputs/foodtech/interim/foodtech_search_terms_v2.pickle",
)

# %%
tech_area_terms

# %%

# %%
