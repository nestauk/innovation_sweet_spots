# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
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
# # Collect Guardian articles, subset relevant documents and extract text
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html

# %% [markdown]
# ## 1. Import dependencies

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as au
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH
import importlib
import pandas as pd

# %%
CATEGORIES = [
    # "Environment",
    # "Guardian Sustainable Business",
    # "Technology",
    # "Science",
    # "Business",
    # "Money",
    # "Cities",
    # "Politics",
    # "Opinion",
    # "Global Cleantech 100",
    # "The big energy debate",
    # "UK news",
    # "Life and style",
]

REQUIRED_TERMS = [
    "UK",
    "Britain",
    "Scotland",
    "Wales",
    "England",
    "Northern Ireland",
    "Britons",
    "London",
]


# %% [markdown]
# # Trying out search terms

# %%
table = pd.read_csv("Food_terms.csv")
table = table[table["use"] == 1]

search_terms = [x.replace(", ", ",") for x in table["Terms"].values]

importlib.reload(au)

# %%
query_id = "AutoSearch"

banned_terms = []

# %%
au.get_guardian_articles

# %%
search_terms[96]

# %%
for i in enumerate(search_terms):
    articles, metadata = au.get_guardian_articles(
        search_terms=[i[1]],
        use_cached=True,
        allowed_categories=CATEGORIES,
        query_identifier=query_id + str(i[0]),
        save_outputs=True,
    )


# %%
files = [
    OUTPUT_DATA_PATH
    / (
        "discourse_analysis_outputs/"
        + query_id
        + str(i)
        + "/"
        + "document_text_"
        + query_id
        + str(i)
        + ".csv"
    )
    for i in range(len(search_terms))
]
combined_csv = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates()


combined_csv_copy = combined_csv.copy()
for i in enumerate(files):
    # When merging table with an OUTER, rows which are NOT NA are the only rows in common
    row_in_common = (
        pd.merge(combined_csv, pd.read_csv(i[1]), how="outer", on=["id"])["text_y"]
    ).notna() * 1
    combined_csv_copy[search_terms[i[0]]] = row_in_common.values

combined_csv = combined_csv_copy.drop("text", axis=1)
combined_csv_copy["URL"] = combined_csv["id"].apply(
    lambda x: "https://www.theguardian.com/" + str(x)
)
# If you don't truncate the text you get some strange issues whith very long values
combined_csv_copy["text"] = combined_csv_copy["text"].apply(lambda x: x[:400])
combined_csv_copy["Headline"] = combined_csv_copy["id"].apply(
    lambda x: x.split("/")[-1]
)


# %%
combined_csv_copy.to_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/foodtech_all.csv"
)
