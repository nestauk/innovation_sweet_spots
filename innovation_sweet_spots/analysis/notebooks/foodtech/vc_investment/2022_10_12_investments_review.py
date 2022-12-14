# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Investment

# +
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu
import utils
import innovation_sweet_spots.utils.text_cleaning_utils as tcu

import altair as alt
import pandas as pd

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES
# -

from innovation_sweet_spots import PROJECT_DIR

import numpy as np

# +
# Initialise a Dealroom wrangler instance
import importlib

importlib.reload(wu)
DR = wu.DealroomWrangler()

# Number of companies
len(DR.company_data)

# +
# Reviewed companies
from innovation_sweet_spots.getters.google_sheets import get_foodtech_reviewed_vc

reviewed_df = get_foodtech_reviewed_vc(from_local=False)
# -

# ## Reviewed taxonomy
# - Check all major categories
# - Check all sub-categories

import ast
import re


def process_reviewed_text(text: str) -> str:
    text = re.sub("‘", "'", text)
    text = re.sub("’", "'", text)
    return text


process_reviewed_text(
    "{'agritech': ['agritech (all other)'], 'food waste': ['food waste (all other)’]}"
)

for i, row in reviewed_df.iterrows():
    try:
        ast.literal_eval(process_reviewed_text(row.taxonomy_checked))
    except:
        print(row.NAME, row.taxonomy_checked)

taxonomy_assigments = reviewed_df.taxonomy_checked.apply(
    lambda x: ast.literal_eval(process_reviewed_text(x))
)

from collections import defaultdict

all_keys = set()
all_values = set()
taxonomy = defaultdict(set)
for t in taxonomy_assigments:
    all_keys = all_keys.union(set(list(t.keys())))
    all_values = all_values.union(set([t for tt in list(t.values()) for t in tt]))
    for key in t:
        taxonomy[key] = taxonomy[key].union(set(t[key]))


for key in taxonomy:
    taxonomy[key] = list(taxonomy[key])
taxonomy = dict(taxonomy)

taxonomy

df = reviewed_df[reviewed_df.taxonomy_checked != "{}"]

assert len(df[df.duplicated("id", keep=False)]) == 0

all_ids = df.id.to_list()
tax_assignments = list(
    df.taxonomy_checked.apply(lambda x: ast.literal_eval(process_reviewed_text(x)))
)

company_to_taxonomy_dict = dict(zip(all_ids, tax_assignments))

# #### create taxonomy dataframe

# +
# enabling_tech.keys()
# -

rejected_tags = [
    "pet food",
    "pet care",
    "pet",
    "veterinary",
]


def create_taxonomy_dataframe(
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a taxonomy dataframe from a dictionary
    """
    taxonomy_df = []
    for major in taxonomy.keys():
        for minor in taxonomy[major]:
            # for label in taxonomy[major][minor]:
            taxonomy_df.append([major, minor])
    taxonomy_df = pd.DataFrame(taxonomy_df, columns=["Major", "Minor"])

    # if DR is not None:
    #     # Number of companies for each label (NB: also accounts for multiple labels of different types with the same name)
    #     category_counts = (
    #         DR.company_labels.groupby(["Category", "label_type"], as_index=False)
    #         .agg(counts=("id", "count"))
    #         .sort_values("counts", ascending=False)
    #     )
    #     taxonomy_df = taxonomy_df.merge(
    #         category_counts[["Category", "label_type", "counts"]]
    #     )
    return taxonomy_df


taxonomy_df = create_taxonomy_dataframe(taxonomy)

taxonomy_df.iloc[0:20]

# ## Refining taxonomy assignments

# - [1] Major category update:
#     - Create a dataframe with company > category links from the dictionary
#     - Check companies in multiple categories
#     - Check if sub-industries can help separate (keep track of those in multiple sub industries)
#     - Update the dataframe accordingly
#     - Convert the dataframe back into dictionary
# - [2] Review companies that are in multiple major categories
#     - Prioritise by investment size, maybe check only those that have more than 1M
# - [3] Minor category update:
#     - Create a dataframe
#     - Check companies in multiple minor categories
#     - Check which minor categories are in the allowed major (check dataframe; make all else False)
# - [4] Any special rules (eg within major groups)
#

from tqdm.notebook import tqdm

df = taxonomy_df.drop_duplicates(["Minor", "Major"])
minor_to_major = dict(zip(df.Minor, df.Major))

company_to_taxonomy_labels = []
for company_id in all_ids:
    for cat in company_to_taxonomy_dict[company_id]:
        company_to_taxonomy_labels.append([company_id, cat, "Major"])
        for minor_cat in company_to_taxonomy_dict[company_id][cat]:
            company_to_taxonomy_labels.append([company_id, minor_cat, "Minor"])

company_to_taxonomy_df = pd.DataFrame(
    company_to_taxonomy_labels, columns=["id", "Category", "level"]
)

len(company_to_taxonomy_df.id.unique())

company_to_taxonomy_df.head(5)

# Remove categories
drop_categories = ["taste", "vegan", "algae", "oleaginous"]
company_to_taxonomy_df = company_to_taxonomy_df[
    company_to_taxonomy_df.Category.isin(drop_categories) == False
].reset_index(drop=True)

# +
# Check category "innovative food (all other")
category_to_check = "innovative food (all other)"
ids = list(company_to_taxonomy_df.query("Category == @category_to_check").id.unique())
print(len(ids))

innovative_food_terms = [
    "reformulation",
    "plant-based",
    "fermentation",
    "alt protein",
    "lab meat",
    "insects",
]

indices = []
for company_id in ids:
    df = company_to_taxonomy_df.query("id == @company_id and level == 'Minor'")
    if df.Category.isin(innovative_food_terms).sum() > 0:
        indices.append(df.query("Category == @category_to_check").index[0])


print(len(indices))

company_to_taxonomy_df = company_to_taxonomy_df.drop(indices).reset_index(drop=True)

# +
# Check category "innovative food (all other")
category_to_check = "alt protein"
ids = list(company_to_taxonomy_df.query("Category == @category_to_check").id.unique())
print(len(ids))

alt_protein_terms = [
    "plant-based",
    "fermentation",
    "lab meat",
    "insects",
]

indices = []
for company_id in ids:
    df = company_to_taxonomy_df.query("id == @company_id and level == 'Minor'")
    if df.Category.isin(alt_protein_terms).sum() > 0:
        indices.append(df.query("Category == @category_to_check").index[0])

print(len(indices))
company_to_taxonomy_df = company_to_taxonomy_df.drop(indices).reset_index(drop=True)

# +
# Check category "innovative food (all other")
category_to_check = "diet"
ids = list(company_to_taxonomy_df.query("Category == @category_to_check").id.unique())
print(len(ids))

alt_terms = ["personalised nutrition"]

indices = []
for company_id in ids:
    df = company_to_taxonomy_df.query("id == @company_id and level == 'Minor'")
    if df.Category.isin(alt_terms).sum() > 0:
        indices.append(df.query("Category == @category_to_check").index[0])

print(len(indices))

company_to_taxonomy_df = company_to_taxonomy_df.drop(indices).reset_index(drop=True)
# -

company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'health tech'").index, "Category"
] = "health (all other)"


# +
# company_to_taxonomy_df[company_to_taxonomy_df.duplicated(['id','Category'])]

# +
# Check category "innovative food (all other")
category_to_check = "health (all other)"
ids = list(company_to_taxonomy_df.query("Category == @category_to_check").id.unique())
print(len(ids))

alt_terms = [
    "personalised nutrition",
    "diet",
    "biomedical",
]

indices = []
for company_id in ids:
    df = company_to_taxonomy_df.query("id == @company_id and level == 'Minor'")
    if df.Category.isin(alt_terms).sum() > 0:
        indices.append(df.query("Category == @category_to_check").index[0])

print(len(indices))

company_to_taxonomy_df = company_to_taxonomy_df.drop(indices).reset_index(drop=True)

# +
company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'food waste (all other)'").index,
    "Category",
] = "waste reduction"

company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'innovative food (all other)'").index,
    "Category",
] = "innovative food (other)"

company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'health (all other)'").index, "Category"
] = "health (other)"

company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'alt protein'").index, "Category"
] = "alt protein (other)"
# -

taxonomy = {
    "agritech": [
        "agritech (all other)",
        "precision agriculture",
        "crop protection",
        "vertical farming",
    ],
    "food waste": ["packaging", "waste reduction"],
    "health": [
        "health (other)",
        "diet",
        "biomedical",
        "dietary supplements",
        "personalised nutrition",
    ],
    "innovative food": [
        "plant-based",
        "fermentation",
        "lab meat",
        "insects",
        "alt protein (other)",
        "reformulation",
        "innovative food (other)",
    ],
    "logistics": ["supply chain", "delivery", "meal kits"],
    "retail and restaurants": ["retail", "restaurants"],
    "cooking and kitchen": ["kitchen tech", "dark kitchen"],
}

taxonomy_df = create_taxonomy_dataframe(taxonomy)

df = taxonomy_df.drop_duplicates(["Minor", "Major"])
minor_to_major = dict(zip(df.Minor, df.Major))

company_to_taxonomy_df.groupby(["level", "Category"]).count()

# ### Export the outputs of processing reviewed data

taxonomy_df.Major = taxonomy_df.Major.str.capitalize()
taxonomy_df.Minor = taxonomy_df.Minor.str.capitalize()
taxonomy_df = taxonomy_df.rename(
    columns={
        "Major": "Category",
        "Minor": "Sub Category",
    }
)

df = taxonomy_df.drop_duplicates(["Category", "Sub Category"])
minor_to_major = dict(zip(df["Sub Category"], df["Category"]))

company_to_taxonomy_df.Category = company_to_taxonomy_df.Category.str.capitalize()
company_to_taxonomy_df.loc[
    company_to_taxonomy_df.level == "Major", "level"
] = "Category"
company_to_taxonomy_df.loc[
    company_to_taxonomy_df.level == "Minor", "level"
] = "Sub Category"
company_to_taxonomy_df

output_folder = PROJECT_DIR / "outputs/foodtech/venture_capital"
from innovation_sweet_spots.utils.io import save_json, load_json

taxonomy_df.to_csv(output_folder / "vc_tech_taxonomy.csv", index=False)

save_json(minor_to_major, output_folder / "vc_tech_taxonomy_minor_to_major.json")

company_to_taxonomy_df.to_csv(output_folder / "vc_company_to_taxonomy.csv", index=False)





