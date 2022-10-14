# -*- coding: utf-8 -*-
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

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu
import utils
import innovation_sweet_spots.utils.text_cleaning_utils as tcu

importlib.reload(wu)
import altair as alt
import pandas as pd

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES

# %%
importlib.reload(au)
importlib.reload(wu)

# %%
from innovation_sweet_spots import PROJECT_DIR

# %%
import numpy as np

# %%
# Functionality for saving charts
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
# Initialise a Dealroom wrangler instance
import importlib

importlib.reload(wu)
DR = wu.DealroomWrangler()

# Number of companies
len(DR.company_data)

# %%
# Reviewed companies
from innovation_sweet_spots.getters.google_sheets import get_foodtech_reviewed_vc

reviewed_df = get_foodtech_reviewed_vc(from_local=False)

# %% [markdown]
# - check that there are no errors in taxonomy assignments (typos)
# - recreate the taxonomy dict and associated files
# - rerun the analyses
# - consider any additional analyses
# - write up

# %%
VERSION_NAME = "September"

# %% [markdown]
# ## Deal types

# %%
for d in sorted(utils.EARLY_DEAL_TYPES):
    print(d)

# %%
for d in sorted(utils.LATE_DEAL_TYPES):
    print(d)

# %% [markdown]
# ## Reviewed taxonomy
# - Check all major categories
# - Check all sub-categories

# %%
import ast
import re


# %%
def process_reviewed_text(text: str) -> str:
    text = re.sub("‘", "'", text)
    text = re.sub("’", "'", text)
    return text


# %%
process_reviewed_text(
    "{'agritech': ['agritech (all other)'], 'food waste': ['food waste (all other)’]}"
)

# %%
for i, row in reviewed_df.iterrows():
    try:
        ast.literal_eval(process_reviewed_text(row.taxonomy_checked))
    except:
        print(row.NAME, row.taxonomy_checked)

# %%
taxonomy_assigments = reviewed_df.taxonomy_checked.apply(
    lambda x: ast.literal_eval(process_reviewed_text(x))
)

# %%
from collections import defaultdict

# %%
all_keys = set()
all_values = set()
taxonomy = defaultdict(set)
for t in taxonomy_assigments:
    all_keys = all_keys.union(set(list(t.keys())))
    all_values = all_values.union(set([t for tt in list(t.values()) for t in tt]))
    for key in t:
        taxonomy[key] = taxonomy[key].union(set(t[key]))


# %%
for key in taxonomy:
    taxonomy[key] = list(taxonomy[key])
taxonomy = dict(taxonomy)

# %%
taxonomy

# %%
df = reviewed_df[reviewed_df.taxonomy_checked != "{}"]

# %%
assert len(df[df.duplicated("id", keep=False)]) == 0

# %%
all_ids = df.id.to_list()
tax_assignments = list(
    df.taxonomy_checked.apply(lambda x: ast.literal_eval(process_reviewed_text(x)))
)

# %%
company_to_taxonomy_dict = dict(zip(all_ids, tax_assignments))

# %% [markdown]
# ## User defined taxonomy

# %%
# ids = DR.get_companies_by_labels("in-store retail & restaurant tech", "SUB INDUSTRIES").id.to_list()
# DR.company_labels.query('id in @ids').groupby(['Category', 'label_type']).count().sort_values('id').tail(30)

# %%
# label_clusters = pd.read_csv(
#     PROJECT_DIR / "outputs/foodtech/interim/dealroom_labels.csv"
# )

# %% [markdown]
# #### version 1

# %%
# Major category > Minor category > [dealroom_label, label_type]
# The minor category that has the same name as major category is a 'general' category
taxonomy = {
    "health and food": {
        "health (general)": [
            "health",
            "personal health",
            "health care",
            "wellness",
            "healthcare",
            "health and wellness",
        ],
        "diet": [
            "diet",
            "dietary supplements",
            "weight management",
            "diet personalization",
        ],
        "nutrition": [
            "nutrition",
            "nutrition solution",
            "nutrition tracking",
            "superfood",
            "healthy nutrition",
            "sports nutrition",
            "probiotics",
        ],
        "health issues": [
            "obesity",
            "diabetes",
            "disease",
            "allergies",
            "chronic disease",
            "gastroenterology",
            "cardiology",
        ],
        "health issues (other)": [
            "oncology",
            "immune system",
            "neurology",
            "mental health",
        ],
        "medicine and pharma": [
            "medical",
            "pharmaceutical",
            "therapeutics",
            "patient care",
            "drug development",
        ],
        "health tech": [
            "health platform",
            "medical devices",
            "medical device",
            "tech for patients",
            "digital healthcare",
            "health diagnostics",
            "health information",
            "medical technology",
            "healthtech",
            "digital health",
            "digital therapeutics",
        ],
    },
    "innovative food": {
        "innovative food": [
            "innovative food",
        ],
        "alt protein": [
            "enabler of alternative proteins",
            "alternative protein",
            "meat substitute",
            "dairy substitute",
        ],
        "taste": [
            "taste",
            "flavor",
        ],
        "fermentation": ["fermentation"],
        "vegan": ["vegan"],
        "plant-based": ["plant-based"],
        "insect": ["insect"],
        "algae and seafood": [
            "algae",
            "seafood",
            "seafood substitute",
        ],
        "oleaginous": ["oleaginous"],
        "dairy": [
            "dairy",
            "alternative dairy",
        ],
    },
    "biotech": {
        "biotech": [
            "biotech",
            "biotech and pharma",
            "biotechnology",
            "biotechnology in food",
            "microbiology",
            "enzymes",
            "laboratories",
            "molecular",
            "synthetic biology",
            "bioreactors",
            "mycelium technology",
        ],
        "genomics": [
            "genetics",
            "gmo",
            "genomics",
            "dna",
            "genome engineering",
            "crispr",
        ],
    },
    "logistics": {
        "logistics (general)": [
            "food logistics & delivery",
            "logistics & delivery",
            "logistic",
            "logistics",
            "logistics tech",
            "logistics solutions",
            "freight",
            "warehousing",
            "fleet management",
            "order management",
        ],
        "supply chain": [
            "supply chain management",
        ],
        "delivery": [
            "delivery",
            "food delivery platform",
            "food delivery service",
            "last-mile delivery",
            "shipping",
        ],
        "packaging": [
            "packaging and containers",
            "packaging",
            "sustainable packaging",
            "ecological packaging",
            "packaging solutions",
            "food packaging",
        ],
        "storage": [
            "storage",
        ],
        "meal kits": [
            "subscription boxes",
            "meal kits",
        ],
    },
    "cooking and catering": {
        "kitchen and cooking (general)": [
            "kitchen & cooking tech",
        ],
        "kitchen": [
            "kitchen",
            "dark kitchen",
        ],
        "restaurants and catering": [
            "catering",
            "restaurant tech",
            "restaurants management",
            "restaurant reservation",
        ],
        "cooking": [
            "cooking tech",
            "cooking",
            "chef",
            "recipes",
            "cook recipes",
            "gastronomy",
        ],
    },
    "retail and sales": {
        "retail (general)": [
            "retail",
            "shopping",
            "consumer goods",
            "point of sale",
            "group buying",
            "supermarket",
            "in-store retail tech",
            "retail tech",
            "crm & sales",
            "procurement",
            "b2b sales",
        ],
        "wholesale": [
            "wholesale",
        ],
        "online marketplace": [
            "ecommerce solutions",
            "merchant tools",
            "b2b online marketplace",
            "mobile commerce",
            "mobile shopping",
            "ecommerce sites",
        ],
        "payments": [
            "payment",
            "pay per result",
            "payments",
            "deal comparison",
            "price comparison",
            "loyalty program",
            "discount",
            "invoicing",
            "pricing",
            "auction",
            "mobile payment",
            "billing",
        ],
        "marketing": [
            "branding",
            "marketing",
        ],
    },
    "agriculture": {
        "agriculture (general)": [
            "farming",
            "crop",
            "horticulture",
            "harvesting",
            "cultivation",
            "bees and pollination",
            "fertilizer",
        ],
        "crop protection": [
            "crop protection",
            "pesticides",
            "pest control",
        ],
        "agritech": [
            "agritech",
            "novel farming",
            "vertical farming",
            "precision agriculture",
            "agricultural equipment",
            "hydroponics",
            "fertigation",
            "greenhouse",
            "regenerative agriculture",
        ],
    },
    "food waste": {
        "food waste": [
            "food waste",
            "waste solution",
            "waste reduction",
            "waste management",
        ]
    },
    "in-store retail & restaurant tech": {
        "in-store retail & restaurant tech": [
            "in-store retail & restaurant tech",
        ],
    },
}

# %% [markdown]
# #### version 2

# %%
# Major category > Minor category > [dealroom_label, label_type]
# The minor category that has the same name as major category is a 'general' category
taxonomy = {
    "health": {
        "health (all other)": [
            "health",
            "personal health",
            "health care",
            "healthcare",
            "health and wellness",
        ],
        "diet": [
            "diet",
            "weight management",
            "diet personalization",
        ],
        "dietary supplements": ["dietary supplements", "probiotics"],
        "health tech": [
            "health platform",
            "medical devices",
            "medical device",
            "tech for patients",
            "digital healthcare",
            "health diagnostics",
            "health information",
            "medical technology",
            "healthtech",
            "digital health",
            "digital therapeutics",
        ],
        "biomedical": [
            "obesity",
            "diabetes",
            "disease",
            "allergies",
            "chronic disease",
            "gastroenterology",
            "cardiology",
            "medical",
            "pharmaceutical",
            "therapeutics",
        ],
    },
    "innovative food": {
        "innovative food (all other)": [
            "innovative food",  # add special rules
        ],
        "alt protein": [
            "enabler of alternative proteins",
            "alternative protein",
            "meat substitute",
            "dairy substitute",
            "alternative dairy",
            "seafood substitute",
            "insect",
            "plant-based",
        ],
        "fermentation": ["fermentation"],
        # perhaps add mycelium
        "vegan": ["vegan"],
        "plant-based": ["plant-based"],
        "taste": [
            "taste",
            "flavor",
        ],
        "algae": ["algae"],
        "oleaginous": ["oleaginous"],
    },
    "logistics": {
        "logistics (all other)": [
            "logistic",
            "logistics",
            "logistics tech",
            "logistics solutions",
            "freight",
            "warehousing",
            "fleet management",
            "order management",
            "shipping",
        ],
        "supply chain": [
            "supply chain management",
        ],
        "delivery": [
            "delivery",
            "food delivery platform",
            "food delivery service",
            "last-mile delivery",
            "10 min delivery",
            "food logistics & delivery",  # make sure it's not too broad
            "logistics & delivery",  # make sure it's not too broad
        ],
        "meal kits": [
            "subscription boxes",
            "meal kits",
        ],
    },
    "cooking and kitchen": {
        "kitchen tech": [
            "kitchen & cooking tech",
            "cooking tech",
        ],
        "dark kitchen": [
            "dark kitchen",
        ],
    },
    "agritech": {
        "agritech (all other)": [
            "agritech",
            "novel farming",
            "hydroponics",
            "fertigation",
        ],
        "crop protection": [
            "crop protection",
            "pesticides",
            "pest control",
        ],
        "precision agriculture": ["precision agriculture"],
        "vertical farming": ["vertical farming"],
    },
    "food waste": {
        "food waste (all other)": [
            "organic waste",
            "food waste",
            "waste solution",
            "waste reduction",
            "waste management",
        ],
        "packaging": [
            "packaging and containers",
            "packaging",
            "packaging solutions",
            "food packaging",
        ],
    },
    "retail and restaurants": {
        "retail and restaurant tech (all other)": [
            "in-store retail & restaurant tech",
            "restaurant tech",
        ],
        "restaurant management": [
            # "catering",
            "restaurants management",
            "restaurant reservation",
        ],
    },
}

# %% [markdown]
# #### enabling tech

# %%
enabling_tech = {
    "robotics": [
        "robotics",
        "autonomous vehicles",
        "odense robotics",
        "odense robotics startup hub",
        "robotic process automation",
        "isensing & robotics",
        "robotic process automation",
        "industrial robotics",
        "business  robots",
        "farm robotics",
        "robotic automation",
        "seeding robot",
        "service robots",
        "weeding robot",
        "warehouse automation",
    ],
    "automation": [
        "automated technology",
        "automated process",
        "home automation",
        "automated workflow",
        "automation solutions",
        "process automatization",
    ],
    "drones": [
        "industrial drones",
        "drones",
    ],
    "machine learning": [
        "image recognition",
        "intelligent systems",
        "chatbot",
        "speech recognition",
        "facial recognition",
        "virtual assistant",
        "optical character recognition",
        "voice recognition",
    ],
    "iot": ["industrial iot"],
    "satellites": [
        "satellite data",
        "satellite imagery",
        "satellite imaging and data analytics",
        "satellite applications",
        "geopositioning",
        "aerial mapping",
        "satellite",
        "proprietary satellites for geospatial intelligence",
        "earth observation satellites",
        "satellites",
    ],
    "data analytics": [
        "analytics",
        "predictive analytics",
        "business intelligence",
        "market intelligence",
        "visualization",
        "marketing analytics",
        "behavior analytics",
        "data analytics",
        "text analytics",
        "net promoter score",
        "customer engagement analytics",
        "sentiment analysis",
        "performance tracking",
        "growth marketing",
        "data visualization analysis",
        "shopper behaviour analysis",
        "industrial analytics",
        "business analytics",
        "advanced data analytics",
        "travel analytics & software",
        "track app",
        "behavioral analysis",
    ],
    "UX": [
        "user behavior",
        "personalisation",
        "user experience",
        "personalized recommendations",
        "user friendly",
    ],
    "biotech": [
        "biotech",
        "biotech and pharma",
        "biotechnology",
        "biotechnology in food",
        "microbiology",
        "enzymes",
        "laboratories",
        "molecular",
        "synthetic biology",
        "bioreactors",
        "mycelium technology",
    ],
    "genomics": [
        "genetics",
        "gmo",
        "genomics",
        "dna",
        "genome engineering",
        "crispr",
    ],
    # "mechanical": [
    #     '3d printing',
    #     'chemical technology',
    #     'advanced materials',
    #     'process technologies',
    #     'industrial technology',
    #     'chemistry',
    #     'texture',
    #     'semiconductors',
    #     'printing',
    #     'innovative material tech',
    #     'machinery manufacturing',
    #     'material technology',
    #     'milling',
    #     'manufacturing tech',
    # ]
}


# %%
# enabling_tech

# %%
technology_clusters = [7, 12, 13, 14, 15, 18, 19, 20]
technology_tags = (
    label_clusters.query("cluster in @technology_clusters")
    .sort_values("cluster")
    .Category.to_list()
)

# %%
# technology_tags

# %% [markdown]
# #### create taxonomy dataframe

# %%
# enabling_tech.keys()

# %%
rejected_tags = [
    "pet food",
    "pet care",
    "pet",
    "veterinary",
]


# %%
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


# %%
taxonomy_df = create_taxonomy_dataframe(taxonomy)

# %%
taxonomy_df.iloc[0:20]

# %%
# taxonomy_df.to_csv(
#     PROJECT_DIR / "outputs/foodtech/interim/taxonomy_v2022_07_27.csv", index=False
# )

# %% [markdown]
# ## Refining taxonomy assignments

# %% [markdown]
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

# %%
from tqdm.notebook import tqdm

# %%
df = taxonomy_df.drop_duplicates(["Minor", "Major"])
minor_to_major = dict(zip(df.Minor, df.Major))

# %%
# major_label_types = ["INDUSTRIES", "SUB INDUSTRIES"]
# companies_to_check = list(
#     DR.company_labels.query("Category in @taxonomy_df.Category.to_list()").id.unique()
# )

# %%
# company_to_taxonomy_dict_old = company_to_taxonomy_dict.copy()

# %%
# # company_id = '890906'
# # # company_id = '1763210'
# # company_id = '1299047'
# company_to_taxonomy_dict = {}

# for company_id in tqdm(companies_to_check, total=len(companies_to_check)):

#     # Check to which taxonomy categories does the company seem to fall in
#     company_to_taxonomy = DR.company_labels.query("id == @company_id").merge(
#         taxonomy_df[["Category", "Major", "Minor", "label_type"]],
#         on=["Category", "label_type"],
#     )

#     # MAJOR CATEGORY
#     major_categories = company_to_taxonomy.Major.unique()
#     n_major = len(major_categories)
#     if n_major == 1:
#         # If there is only one major category, assign that
#         company_to_taxonomy_dict[company_id] = {major_categories[0]: []}
#     elif n_major > 1:
#         # If there are more than 1 major category assignments
#         # Check sub industries
#         df = company_to_taxonomy.query("label_type in @major_label_types")
#         if len(df) > 0:
#             # If we can use industries/sub-industries
#             company_to_taxonomy_dict[company_id] = {x: [] for x in df.Major.unique()}
#         else:
#             # If there are no industries/sub-industries
#             company_to_taxonomy_dict[company_id] = {x: [] for x in major_categories}

#     # MINOR CATEGORY
#     minor_categories = company_to_taxonomy.Minor.unique()
#     n_minor = len(minor_categories)

#     for minor_cat in minor_categories:
#         maj = minor_to_major[minor_cat]
#         if maj in company_to_taxonomy_dict[company_id]:
#             company_to_taxonomy_dict[company_id][maj].append(minor_cat)

#     for cat in company_to_taxonomy_dict[company_id]:
#         minor_cats = company_to_taxonomy_dict[company_id][cat]
#         if len(minor_cats) > 1:
#             contains_general = ["all other" in s for s in minor_cats]
#             company_to_taxonomy_dict[company_id][cat] = [
#                 cat for i, cat in enumerate(minor_cats) if not contains_general[i]
#             ]

#     # Special rules
#     if "dietary supplements" in minor_categories:
#         company_to_taxonomy_dict[company_id] = {"health": ["dietary supplements"]}


# %%
# DR.company_data[DR.company_data.NAME.str.contains("Bolt")]

# %%
company_to_taxonomy_dict["26411"]

# %% [markdown]
# #### Special assignments

# %%
# # Karma Kitchen
# company_to_taxonomy_dict["1686094"] = {
#     "logistics": ["delivery"],
#     "cooking and kitchen": ["dark kitchen"],
# }

# %% [markdown]
# ## Restart from here

# %%
company_to_taxonomy_labels = []
for company_id in all_ids:
    for cat in company_to_taxonomy_dict[company_id]:
        company_to_taxonomy_labels.append([company_id, cat, "Major"])
        for minor_cat in company_to_taxonomy_dict[company_id][cat]:
            company_to_taxonomy_labels.append([company_id, minor_cat, "Minor"])

# %%
company_to_taxonomy_df = pd.DataFrame(
    company_to_taxonomy_labels, columns=["id", "Category", "level"]
)

# %%
len(company_to_taxonomy_df.id.unique())

# %%
company_to_taxonomy_df.head(5)

# %%
# Remove categories
drop_categories = ["taste", "vegan", "algae", "oleaginous"]
company_to_taxonomy_df = company_to_taxonomy_df[
    company_to_taxonomy_df.Category.isin(drop_categories) == False
].reset_index(drop=True)

# %%
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

# %%
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

# %%
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

# %%
company_to_taxonomy_df.loc[
    company_to_taxonomy_df.query("Category == 'health tech'").index, "Category"
] = "health (all other)"


# %%
# company_to_taxonomy_df[company_to_taxonomy_df.duplicated(['id','Category'])]

# %%
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

# %%
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

# %%
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

# %%
taxonomy_df = create_taxonomy_dataframe(taxonomy)

# %%
df = taxonomy_df.drop_duplicates(["Minor", "Major"])
minor_to_major = dict(zip(df.Minor, df.Major))

# %%
company_to_taxonomy_df.groupby(["level", "Category"]).count()

# %%
# df = (
#     company_to_taxonomy_df.query("Category == 'supply chain'")
#     .merge(DR.company_data[['id', 'NAME']], how='left')
#     .merge(company_to_taxonomy_df.query("level == 'Minor'")[['id', 'Category']], on='id')
# )
# # df[df.duplicated('id', keep=False)==False]
# df

# %% [markdown]
# ### Export the outputs of processing reviewed data

# %%
output_folder = PROJECT_DIR / "outputs/foodtech/venture_capital"
from innovation_sweet_spots.utils.io import save_json, load_json

# %%
taxonomy_df.to_csv(output_folder / "vc_tech_taxonomy.csv", index=False)

# %%
save_json(minor_to_major, output_folder / "vc_tech_taxonomy_minor_to_major.json")

# %%
company_to_taxonomy_df.to_csv(output_folder / "vc_company_to_taxonomy.csv", index=False)
