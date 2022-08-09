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

# %% [markdown]
# ## Deal types

# %%
for d in sorted(utils.EARLY_DEAL_TYPES):
    print(d)

# %%
for d in sorted(utils.LATE_DEAL_TYPES):
    print(d)

# %% [markdown]
# ## Explore label embeddings

# %%
import innovation_sweet_spots.getters.dealroom as dlr
from innovation_sweet_spots.utils import cluster_analysis_utils
import innovation_sweet_spots.utils.embeddings_utils as eu

# %%
import umap
import numpy as np

# %%
v_labels = dlr.get_label_embeddings()

# %%
query = eu.QueryEmbeddings(
    vectors=v_labels.vectors, texts=v_labels.vector_ids, model=v_labels.model
)

# %%
labels = DR.labels.assign(text=lambda df: df.Category.apply(tcu.clean_dealroom_labels))

# %%
pd.set_option("max_colwidth", 200)
ids = DR.company_labels.query("Category=='compounding'").id.to_list()
DR.company_data.query("id in @ids")[["NAME", "TAGLINE", "WEBSITE"]]

# %%
i = 0
df = query.find_most_similar("retail").merge(labels).iloc[i : i + 20]
df

# %% [markdown]
# ## User defined taxonomy

# %%
# ids = DR.get_companies_by_labels("in-store retail & restaurant tech", "SUB INDUSTRIES").id.to_list()
# DR.company_labels.query('id in @ids').groupby(['Category', 'label_type']).count().sort_values('id').tail(30)

# %%
label_clusters = pd.read_csv(
    PROJECT_DIR / "outputs/foodtech/interim/dealroom_labels.csv"
)

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
enabling_tech.keys()

# %%
rejected_tags = [
    "pet food",
    "pet care",
    "pet",
    "veterinary",
]


# %%
def create_taxonomy_dataframe(
    taxonomy: pd.DataFrame, DR: wu.DealroomWrangler = None
) -> pd.DataFrame:
    """
    Create a taxonomy dataframe from a dictionary
    """
    taxonomy_df = []
    for major in taxonomy.keys():
        for minor in taxonomy[major].keys():
            for label in taxonomy[major][minor]:
                taxonomy_df.append([major, minor, label])
    taxonomy_df = pd.DataFrame(taxonomy_df, columns=["Major", "Minor", "Category"])

    if DR is not None:
        # Number of companies for each label (NB: also accounts for multiple labels of different types with the same name)
        category_counts = (
            DR.company_labels.groupby(["Category", "label_type"], as_index=False)
            .agg(counts=("id", "count"))
            .sort_values("counts", ascending=False)
        )
        taxonomy_df = taxonomy_df.merge(
            category_counts[["Category", "label_type", "counts"]]
        )
    return taxonomy_df


# %%
taxonomy_df = create_taxonomy_dataframe(taxonomy, DR)

# %%
taxonomy_df.iloc[0:20]

# %%
# taxonomy_df.to_csv("taxonomy.csv", index=False)

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
major_label_types = ["INDUSTRIES", "SUB INDUSTRIES"]
companies_to_check = list(
    DR.company_labels.query("Category in @taxonomy_df.Category.to_list()").id.unique()
)

# %%
# company_to_taxonomy_dict_old = company_to_taxonomy_dict.copy()

# %%
# company_id = '890906'
# # company_id = '1763210'
# company_id = '1299047'
company_to_taxonomy_dict = {}

for company_id in tqdm(companies_to_check, total=len(companies_to_check)):

    # Check to which taxonomy categories does the company seem to fall in
    company_to_taxonomy = DR.company_labels.query("id == @company_id").merge(
        taxonomy_df[["Category", "Major", "Minor", "label_type"]],
        on=["Category", "label_type"],
    )

    # MAJOR CATEGORY
    major_categories = company_to_taxonomy.Major.unique()
    n_major = len(major_categories)
    if n_major == 1:
        # If there is only one major category, assign that
        company_to_taxonomy_dict[company_id] = {major_categories[0]: []}
    elif n_major > 1:
        # If there are more than 1 major category assignments
        # Check sub industries
        df = company_to_taxonomy.query("label_type in @major_label_types")
        if len(df) > 0:
            # If we can use industries/sub-industries
            company_to_taxonomy_dict[company_id] = {x: [] for x in df.Major.unique()}
        else:
            # If there are no industries/sub-industries
            company_to_taxonomy_dict[company_id] = {x: [] for x in major_categories}

    # MINOR CATEGORY
    minor_categories = company_to_taxonomy.Minor.unique()
    n_minor = len(minor_categories)

    for minor_cat in minor_categories:
        maj = minor_to_major[minor_cat]
        if maj in company_to_taxonomy_dict[company_id]:
            company_to_taxonomy_dict[company_id][maj].append(minor_cat)

    for cat in company_to_taxonomy_dict[company_id]:
        minor_cats = company_to_taxonomy_dict[company_id][cat]
        if len(minor_cats) > 1:
            contains_general = ["all other" in s for s in minor_cats]
            company_to_taxonomy_dict[company_id][cat] = [
                cat for i, cat in enumerate(minor_cats) if not contains_general[i]
            ]

    # Special rules
    if "dietary supplements" in minor_categories:
        company_to_taxonomy_dict[company_id] = {"health": ["dietary supplements"]}


# %%
DR.company_data[DR.company_data.NAME.str.contains("Bolt")]

# %%
company_to_taxonomy_dict["26411"]

# %% [markdown]
# #### Special assignments

# %%
# Karma Kitchen
company_to_taxonomy_dict["1686094"] = {
    "logistics": ["delivery"],
    "cooking and kitchen": ["dark kitchen"],
}

# %%
company_to_taxonomy_labels = []
for company_id in companies_to_check:
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
company_to_taxonomy_df.groupby(["level", "Category"]).count()

# %% [markdown]
# ## Export for checking

# %%
# DR.company_data.merge(company_to_taxonomy_df, how='left').to_csv('to_check.csv')

# %%
company_to_taxonomy_dict["26411"]


# %%
def assign_tax(id_string):
    if id_string in company_to_taxonomy_dict:
        return company_to_taxonomy_dict[id_string]
    else:
        return {}


# %%
df = DR.company_data.copy().assign(taxonomy=lambda df: df.id.apply(assign_tax))

# %%
df.to_csv("to_check.csv")

# %% [markdown]
# ## Helper functions

# %% [markdown]
# ### Functions

# %%
from collections import defaultdict
import itertools


# %%
def get_category_ids_(taxonomy_df, rejected_tags, DR, column="Category"):
    category_ids = defaultdict(set)

    rejected_ids = [
        DR.get_ids_by_labels(row.Category, row.label_type)
        for i, row in DR.labels.query("Category in @rejected_tags").iterrows()
    ]
    rejected_ids = set(itertools.chain(*rejected_ids))

    for category in taxonomy_df[column].unique():
        ids = [
            DR.get_ids_by_labels(row.Category, row.label_type)
            for i, row in taxonomy_df.query(f"`{column}` == @category").iterrows()
        ]
        ids = set(itertools.chain(*ids)).difference(rejected_ids)
        category_ids[category] = ids
    return category_ids


# %%
def get_category_ids(taxonomy_df, rejected_tags, DR, column="Minor"):
    category_ids = defaultdict(set)

    rejected_ids = [
        DR.get_ids_by_labels(row.Category, row.label_type)
        for i, row in DR.labels.query("Category in @rejected_tags").iterrows()
    ]
    rejected_ids = set(itertools.chain(*rejected_ids))

    for category in taxonomy_df[column].unique():
        ids = (
            company_to_taxonomy_df.query("level == @column")
            .query("Category == @category")
            .id.to_list()
        )
        ids = set(ids).difference(rejected_ids)
        category_ids[category] = ids
    return category_ids


# %%
def get_category_ts(category_ids, DR, deal_type=utils.EARLY_DEAL_TYPES):
    ind_ts = []
    for category in category_ids:
        ids = category_ids[category]
        ind_ts.append(
            au.cb_get_all_timeseries(
                DR.company_data.query("id in @ids"),
                (
                    DR.funding_rounds.query("id in @ids").query(
                        "`EACH ROUND TYPE` in @deal_type"
                    )
                ),
                period="year",
                min_year=2010,
                max_year=2022,
            )
            .assign(year=lambda df: df.time_period.dt.year)
            .assign(Category=category)
        )
    return pd.concat(ind_ts, ignore_index=True)


# %%
def get_company_counts(category_ids: dict):
    return pd.DataFrame(
        [(key, len(np.unique(list(category_ids[key])))) for key in category_ids],
        columns=["Category", "Number of companies"],
    )


# %%
def get_deal_counts(category_ids: dict):
    category_deal_counts = []
    for key in category_ids:
        ids = category_ids[key]
        deals = DR.funding_rounds.query("id in @ids").query(
            "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
        )
        category_deal_counts.append((key, len(deals)))
    return pd.DataFrame(category_deal_counts, columns=["Category", "Number of deals"])


# %%
def get_trends(
    taxonomy_df, rejected_tags, taxonomy_level, DR, deal_type=utils.EARLY_DEAL_TYPES
):
    category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, taxonomy_level)
    company_counts = get_company_counts(category_ids)
    category_ts = get_category_ts(category_ids, DR, deal_type)

    values_title_ = "raised_amount_gbp_total"
    values_title = "Growth"
    category_title = "Category"
    colour_title = category_title
    horizontal_title = "year"

    if taxonomy_level == "Category":
        tax_levels = ["Category", "Minor", "Major"]
    if taxonomy_level == "Minor":
        tax_levels = ["Minor", "Major"]
    if taxonomy_level == "Major":
        tax_levels = ["Major"]

    return (
        utils.get_magnitude_vs_growth(
            category_ts,
            value_column=values_title_,
            time_column=horizontal_title,
            category_column=category_title,
        )
        .assign(growth=lambda df: df.Growth / 100)
        .merge(get_deal_counts(category_ids), on="Category")
        .merge(company_counts, on="Category")
        .merge(
            taxonomy_df[tax_levels].drop_duplicates(taxonomy_level),
            how="left",
            left_on="Category",
            right_on=taxonomy_level,
        )
    )


# %%
def get_deal_counts(category_ids: dict, deal_type=utils.EARLY_DEAL_TYPES):
    category_deal_counts = []
    for key in category_ids:
        ids = category_ids[key]
        deals = DR.funding_rounds.query("id in @ids").query(
            "`EACH ROUND TYPE` in @deal_type"
        )
        category_deal_counts.append((key, len(deals)))
    return pd.DataFrame(category_deal_counts, columns=["Category", "Number of deals"])


# %% [markdown]
# #### Check deal types:
# - Early vs mature
# - "Large" vs "small"

# %%
importlib.reload(utils)

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, column="Minor")
# ids = category_ids['logistics']

# %%
early_vs_late_deals = pd.concat(
    [
        (
            get_category_ts(category_ids, DR, deal_type=utils.EARLY_DEAL_TYPES).assign(
                deal_type="early"
            )
        ),
        (
            get_category_ts(category_ids, DR, deal_type=utils.LATE_DEAL_TYPES).assign(
                deal_type="late"
            )
        ),
    ],
    ignore_index=True,
)

# %%
early_vs_late_deals_5y = (
    early_vs_late_deals.query("year >= 2017 and year < 2022")
    .groupby(["Category", "deal_type"], as_index=False)
    .agg(no_of_rounds=("no_of_rounds", "sum"))
)
total_deals = early_vs_late_deals_5y.groupby("Category", as_index=False).agg(
    total_no_of_rounds=("no_of_rounds", "sum")
)
early_vs_late_deals_5y = early_vs_late_deals_5y.merge(total_deals).assign(
    fraction=lambda df: df["no_of_rounds"] / df["total_no_of_rounds"]
)

# %%
alt.Chart((early_vs_late_deals_5y.query("deal_type=='late'")),).mark_bar().encode(
    # x=alt.X('no_of_rounds:Q'),
    x=alt.X("fraction:Q"),
    y=alt.Y("Category:N", sort="-x"),
    # color='deal_type',
    tooltip=["deal_type", "no_of_rounds"],
)

# %%

# %%
cat = "diet"
alt.Chart((early_vs_late_deals.query("Category==@cat")),).mark_bar().encode(
    x=alt.X("year:O"),
    y=alt.Y("sum(no_of_rounds):Q", stack="normalize"),
    color="deal_type",
    tooltip=["deal_type", "no_of_rounds"],
)

# %%
alt
early_vs_late_deals


# %% [markdown]
# ### Figure functions

# %%
def fig_growth_vs_magnitude(
    magnitude_vs_growth,
    colour_field,
    text_field,
    legend=alt.Legend(),
    horizontal_scale="log",
):
    title_text = "Foodtech trends (2017-2021)"
    subtitle_text = [
        "Data: Dealroom. Showing data on early stage deals (eg, series funding)",
        "Late stage deals, such as IPOs, acquisitions, and debt not included.",
    ]

    fig = (
        alt.Chart(
            magnitude_vs_growth,
            width=400,
            height=400,
        )
        .mark_circle(size=50)
        .encode(
            x=alt.X(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                # scale=alt.Scale(type="linear"),
                scale=alt.Scale(type=horizontal_scale),
            ),
            y=alt.Y(
                "growth:Q",
                axis=alt.Axis(title="Growth", format="%"),
                # axis=alt.Axis(
                #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
                # ),
                # scale=alt.Scale(domain=(-.100, .300)),
                #             scale=alt.Scale(type="log", domain=(.01, 12)),
            ),
            #         size="Number of companies:Q",
            color=alt.Color(f"{colour_field}:N", legend=legend),
            tooltip=[
                "Category",
                alt.Tooltip(
                    "Magnitude", title=f"Average yearly raised amount (million GBP)"
                ),
                alt.Tooltip("growth", title="Growth", format=".0%"),
            ],
        )
        .properties(
            title={
                "anchor": "start",
                "text": title_text,
                "subtitle": subtitle_text,
                "subtitleFont": pu.FONT,
            },
        )
    )

    text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
        text=text_field
    )

    fig_final = (
        (fig + text)
        .configure_axis(
            grid=False,
            gridDash=[1, 7],
            gridColor="white",
            labelFontSize=pu.FONTSIZE_NORMAL,
            titleFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=pu.FONTSIZE_NORMAL,
            labelFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )

    return fig_final


# %%
def fig_category_growth(
    magnitude_vs_growth_filtered,
    colour_field,
    text_field,
    height=500,
):
    """ """
    fig = (
        alt.Chart(
            (
                magnitude_vs_growth_filtered.assign(
                    Increase=lambda df: df.growth > 0
                ).assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
            ),
            width=300,
            height=height,
        )
        .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1)
        .encode(
            x=alt.X(
                "growth:Q",
                axis=alt.Axis(
                    format="%",
                    title="Growth",
                    labelAlign="center",
                    labelExpr="datum.value < -1 ? null : datum.label",
                ),
                #             scale=alt.Scale(domain=(-1, 37)),
            ),
            y=alt.Y(
                "Category:N",
                sort="-x",
                axis=alt.Axis(title="Category", labels=False),
            ),
            size=alt.Size(
                "Magnitude",
                title="Yearly investment (million GBP)",
                legend=alt.Legend(orient="top"),
                scale=alt.Scale(domain=[100, 4000]),
            ),
            color=alt.Color(
                colour_field,
            ),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                "Number of companies",
                "Number of deals",
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
            ],
        )
    )

    text = (
        alt.Chart(magnitude_vs_growth_filtered)
        .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7)
        .encode(
            text=text_field,
            x="growth:Q",
            y=alt.Y("Category:N", sort="-x"),
        )
    )

    # text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
    #     text='text_label:N'
    # )

    # fig_final = (
    #     (fig + text)
    #     .configure_axis(
    #         gridDash=[1, 7],
    #         gridColor="grey",
    #         labelFontSize=pu.FONTSIZE_NORMAL,
    #         titleFontSize=pu.FONTSIZE_NORMAL,
    #     )
    #     .configure_legend(
    #         labelFontSize=pu.FONTSIZE_NORMAL - 1,
    #         titleFontSize=pu.FONTSIZE_NORMAL - 1,
    #     )
    #     .configure_view(strokeWidth=0)
    #     #     .interactive()
    # )

    return pu.configure_titles(pu.configure_axes((fig + text)), "", "")


# %%
def fig_size_vs_magnitude(
    magnitude_vs_growth_filtered,
    colour_field,
    horizontal_scale="log",
):
    fig = (
        alt.Chart(
            magnitude_vs_growth_filtered,
            width=500,
            height=450,
        )
        .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1, size=50)
        .encode(
            x=alt.X(
                "Number of companies:Q",
            ),
            y=alt.Y(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                scale=alt.Scale(type=horizontal_scale),
            ),
            color=alt.Color(colour_field),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
                "Number of companies",
                "Number of deals",
            ],
        )
    )

    return pu.configure_titles(pu.configure_axes(fig), "", "")


# %% [markdown]
# ## Checks

# %%
# major_to_industry = {
#         "health": ['health', 'INDUSTRIES'],
#         "innovative food": ['innovative food', 'SUB INDUSTRIES'],
#         "logistics": ['food logistics & delivery', 'SUB INDUSTRIES'],
#         "cooking and kitchen": ['kitchen & cooking tech', 'SUB INDUSTRIES'],
#         "agritech": ['agritech', 'SUB INDUSTRIES'],
#         "food waste": ['waste solution', 'SUB INDUSTRIES'],
#         "in-store retail & restaurant tech": ["in-store retail & restaurant tech", 'SUB INDUSTRIES'],
#     }

# %%
# taxonomy_df[['Major', 'Minor']]

# %%
# category_dict = category_major_ids

# # Dataframe to show which companies are in which major categories
# df_companies = DR.company_data[['id']].copy()
# for cat in category_dict:
#     df_companies[cat] = df_companies.id.isin(category_dict[cat])
# df_companies = df_companies.set_index('id' )

# %%
# (df_companies[list(category_dict.keys())].sum(axis=1)>1).sum()

# %%
# # Dataframe to show which companies are in which main industries / sub industries categories
# df_companies_major = DR.company_data[['id']].copy()
# for cat in category_dict:
#     ids = DR.get_ids_by_labels(major_to_industry[cat][0], major_to_industry[cat][1])
#     df_companies_major[cat] = df_companies_major.id.isin(ids)
# df_companies_major = df_companies_major.set_index('id')
# # If company has no industry / sub industry affiliation, then keep it in
# x = (df_companies_major.sum(axis=1)==0).copy()
# df_companies_major.loc[x, :] = True

# %%
# # Final dataframe that shows which major categories and industries agree
# df_companies_major_ = df_companies_major & df_companies

# %%
# # Number of companies still located in different major categories...
# (df_companies_major_[list(category_dict.keys())].sum(axis=1)>1).sum()

# %%
# df_companies_major_[df_companies_major_[list(category_dict.keys())].sum(axis=1)>1].reset_index().merge(DR.company_data[['id', 'NAME']]).sample()


# %%
category_major_ids = get_category_ids(taxonomy_df, rejected_tags, DR, column="Major")
category_minor_ids = get_category_ids(taxonomy_df, rejected_tags, DR, column="Minor")

category_ids = category_major_ids
categories = list(category_major_ids.keys())

# %%
cooc = np.zeros((len(categories), len(categories)))
cooc.shape

# %%
n_c = np.array([len(category_ids[cat_1]) for i, cat_1 in enumerate(categories)])


# %%
n_c

# %%
for i, cat_1 in enumerate(categories):
    for j, cat_2 in enumerate(categories):
        cooc[i, j] = len(category_ids[cat_1].intersection(category_ids[cat_2]))
    # cooc[i,:] /= n_c

# %%
categories

# %%
np.set_printoptions(suppress=True)
cooc

# %% [markdown]
# ## Overall stats

# %%
foodtech_ids = list(company_to_taxonomy_df.id.unique())
# foodtech_ids = DR.company_data.id.to_list()

# %%
(
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES")
    .query('announced_on > "2020-12-31" and announced_on < "2022-01-01"')
    .raised_amount_gbp.sum()
)

# %%
# - 'EACH ROUND TYPE',
# - 'EACH ROUND AMOUNT',
# - 'EACH ROUND CURRENCY',
# - 'EACH ROUND DATE',
# - 'TOTAL ROUNDS NUMBER',
# - 'EACH ROUND INVESTORS',

# %%
# DR.funding_rounds.query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES").merge(DR.company_data[['id', 'NAME']])

# %%
# DR.funding_rounds.query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES").merge(DR.company_data[['id', 'NAME']])

# %%
# DR.company_data.query("NAME == 'Obalon Therapeutics'")[['NAME', 'EACH ROUND TYPE','EACH ROUND AMOUNT']]

# %%
foodtech_ts_early = (
    au.cb_get_all_timeseries(
        DR.company_data.query("id in @foodtech_ids"),
        (
            DR.funding_rounds.query("id in @foodtech_ids").query(
                "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
            )
        ),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Early")
)

foodtech_ts_late = (
    au.cb_get_all_timeseries(
        DR.company_data.query("id in @foodtech_ids"),
        (
            DR.funding_rounds.query("id in @foodtech_ids").query(
                "`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
            )
        ),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Late")
)
foodtech_ts = pd.concat([foodtech_ts_early, foodtech_ts_late])

# %%
horizontal_label = "Year"
values_label = "Investment (bn GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    foodtech_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2022")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"sum({values_label}):Q",
            title="Raised investment (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
            # stack='normalize',
        ),
        tooltip=tooltip,
        color=alt.Color(
            "deal_type", sort=["Late", "Early"], legend=alt.Legend(title="Deal type")
        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
au.percentage_change(
    data.query("`Year`==2011 and deal_type == 'Early'")[values_label].iloc[0],
    data.query("`Year`==2021 and deal_type == 'Early'")[values_label].iloc[0],
)

# %%
au.percentage_change(
    data.query("`Year`==2020 and deal_type == 'Early'")[values_label].iloc[0],
    data.query("`Year`==2021 and deal_type == 'Early'")[values_label].iloc[0],
)

# %%
AltairSaver.save(fig, f"vJuly18_total_investment", filetypes=["html", "png"])

# %%
horizontal_label = "Year"
values_label = "Number of rounds"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.0f")]

data = (
    foodtech_ts.assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total)
    .query("time_period < 2022")
    .rename(
        columns={
            "time_period": horizontal_label,
            "no_of_rounds": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"sum({values_label}):Q",
            title="Number of rounds",
            # scale=alt.Scale(domain=[0, 1200])
            # stack='normalize',
        ),
        tooltip=tooltip,
        color=alt.Color(
            "deal_type", sort=["Late", "Late"], legend=alt.Legend(title="Deal type")
        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(fig, f"vJuly18_total_investment_rounds", filetypes=["html", "png"])

# %% [markdown]
# ## Major categories

# %%
# Initialise a Dealroom wrangler instance
importlib.reload(wu)
DR = wu.DealroomWrangler()

# %%
magnitude_vs_growth = get_trends(taxonomy_df, rejected_tags, "Major", DR)
magnitude_vs_growth

# %%
# fig_growth_vs_magnitude(
#     magnitude_vs_growth,
#     colour_field="Major",
#     text_field="Major",
#     horizontal_scale="log",
# ).interactive()

# %%
magnitude_vs_growth_plot = magnitude_vs_growth.assign(
    Magnitude=lambda df: df.Magnitude / 1e3
)

# %%
colour_field = "Major"
text_field = "Major"
horizontal_scale = "log"
horizontal_title = f"Average yearly raised amount (billion GBP)"
legend = alt.Legend()

title_text = "Foodtech trends (2017-2021)"
subtitle_text = [
    "Data: Dealroom. Showing data on early stage deals (eg, seed and series funding)",
    "Late stage deals, such as IPOs, acquisitions, and debt financing not included.",
]

fig = (
    alt.Chart(
        magnitude_vs_growth_plot,
        width=400,
        height=400,
    )
    .mark_circle(size=80)
    .encode(
        x=alt.X(
            "Magnitude:Q",
            axis=alt.Axis(title=horizontal_title),
            scale=alt.Scale(
                type=horizontal_scale,
                domain=(0.100, 20),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(title="Growth", format="%"),
        ),
        color=alt.Color(f"{colour_field}:N", legend=None),
        tooltip=[
            "Category",
            alt.Tooltip("Magnitude", title=horizontal_title),
            alt.Tooltip("growth", title="Growth", format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": title_text,
            "subtitle": subtitle_text,
            "subtitleFont": pu.FONT,
            "fontSize": 15,
        },
    )
)

text = fig.mark_text(
    align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=15
).encode(text=text_field)

fig_final = (
    (fig + text)
    .configure_axis(
        grid=False,
        gridDash=[5, 7],
        # gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig_final

# %%
AltairSaver.save(
    fig_final, f"vJuly18_growth_vs_magnitude_Major", filetypes=["html", "png"]
)

# %%
fig_category_growth(
    magnitude_vs_growth, colour_field="Major", text_field="Major", height=300
)

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, "Major")
category_ts = get_category_ts(category_ids, DR)

# %%
category_ts.head(2)

# %%
utils.get_estimates(
    category_ts,
    value_column="raised_amount_gbp_total",
    time_column="year",
    category_column="Category",
    estimate_function=au.growth,
    year_start=2020,
    year_end=2021,
)

# %% [markdown]
# #### Time series

# %%
category_ts.head(1)

# %%
category = "cooking and kitchen"
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    category_ts.assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total)
    .query("time_period < 2022")
    .query("Category == @category")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=tooltip,
    )
)
pu.configure_plots(fig)

# %%
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.funding_rounds.query("id in @ids.id.to_list()")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .groupby(["id", "NAME", "country", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)

# %%
DR.funding_rounds.id.iloc[0]

# %% [markdown]
# ### Late deals

# %%
magnitude_vs_growth_late = get_trends(
    taxonomy_df, rejected_tags, "Major", DR, deal_type=utils.LATE_DEAL_TYPES
)

# %%
fig_growth_vs_magnitude(
    magnitude_vs_growth_late,
    colour_field="Major",
    text_field="Major",
    horizontal_scale="log",
).interactive()

# %%
fig_category_growth(
    magnitude_vs_growth_late, colour_field="Major", text_field="Major", height=300
)

# %% [markdown]
# ## Minor categories (medium granularity)

# %%
magnitude_vs_growth_minor = get_trends(taxonomy_df, rejected_tags, "Minor", DR)
magnitude_vs_growth_minor

# %%
fig_growth_vs_magnitude(
    magnitude_vs_growth_minor, colour_field="Major", text_field="Minor"
).interactive()

# %%
company_to_taxonomy_df.query('Category == "fermentation"').merge(
    DR.company_data[["id", "NAME", "PROFILE URL", "TOTAL FUNDING (USD M)"]]
).sort_values("TOTAL FUNDING (USD M)", ascending=False)


# %%
fig_category_growth(magnitude_vs_growth_minor, colour_field="Major", text_field="Minor")

# %%
major_sort_order = magnitude_vs_growth.sort_values("Growth").Category.to_list()
data = magnitude_vs_growth_minor.copy()
data["Major"] = pd.Categorical(data["Major"], categories=major_sort_order)
data = data.sort_values(["Major", "growth"], ascending=False)

# %%
colour_field = "Major"
text_field = "Minor"
height = 500

fig = (
    alt.Chart(
        (
            data.assign(Increase=lambda df: df.growth > 0)
            .assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
            .assign(Magnitude=lambda df: df.Magnitude / 1e3)
        ),
        width=500,
        height=height,
    )
    .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1)
    .encode(
        x=alt.X(
            "growth:Q",
            axis=alt.Axis(
                format="%",
                title="Growth",
                labelAlign="center",
                labelExpr="datum.value < -1 ? null : datum.label",
            ),
            #             scale=alt.Scale(domain=(-1, 37)),
        ),
        y=alt.Y(
            "Category:N",
            sort=data.Minor.to_list(),
            # axis=alt.Axis(title="", labels=False),
            axis=None,
        ),
        size=alt.Size(
            "Magnitude",
            title="Avg yearly investment (£ bn)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=[0.1, 4]),
        ),
        color=alt.Color(colour_field, legend=alt.Legend(orient="left")),
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip(
                "Magnitude:Q",
                format=",.3f",
                title="Average yearly investment (billion GBP)",
            ),
            "Number of companies",
            "Number of deals",
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

text = (
    alt.Chart(data)
    .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=14)
    .encode(
        text=text_field,
        x="growth:Q",
        y=alt.Y("Category:N", sort=data.Minor.to_list(), title=""),
    )
)

final_fig = pu.configure_titles(pu.configure_axes((fig + text)), "", "")
final_fig

# %%
# df = DR.company_data[-DR.company_data.TAGLINE.isnull()]
# df[df.TAGLINE.str.contains('formula')]
# # .str.contains('formula')]

# %%
AltairSaver.save(final_fig, f"vJuly18_growth_Minor", filetypes=["html", "png"])

# %%
category = "taste"
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.funding_rounds.query("id in @ids.id.to_list()")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .groupby(["id", "NAME", "country", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)

# %%
# ids = company_to_taxonomy_df.query("Category in 'taste'").id.to_list()
# company_to_taxonomy_df[company_to_taxonomy_df.id.isin(ids)]

# %%
# DR.company_labels[DR.company_labels.Category.str.contains('lab')]

# %%
# fig_size_vs_magnitude(magnitude_vs_growth, colour_field="Major")

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, "Minor")
category_ts = get_category_ts(category_ids, DR)

# %%
short_term_trend_df = (
    utils.get_estimates(
        category_ts,
        value_column="raised_amount_gbp_total",
        time_column="year",
        category_column="Category",
        estimate_function=au.growth,
        year_start=2020,
        year_end=2021,
    )
    .set_index("Category")
    .rename(columns={"raised_amount_gbp_total": "Growth"})
    .assign(
        Magnitude=category_ts.query("year == 2021")[
            ["Category", "raised_amount_gbp_total"]
        ].set_index("Category")["raised_amount_gbp_total"]
    )
    .assign(growth=lambda df: df.Growth / 100)
    .reset_index()
    .rename(columns={"Category": "Minor"})
    .assign(Major=lambda df: df.Minor.apply(lambda x: minor_to_major[x]))
)

# %%
# category_ts.query("year == 2020

# %%
short_term_trend_df

# %%
colour_field = "Major"
text_field = "Minor"
height = 500

# data = short_term_trend_df

fig = (
    alt.Chart(
        (
            short_term_trend_df.assign(Increase=lambda df: df.growth > 0)
            .assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
            .assign(Magnitude=lambda df: df.Magnitude / 1e3)
            .query("Minor != 'logistics (all other)'")
        ),
        width=400,
        height=height,
    )
    .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1)
    .encode(
        x=alt.X(
            "growth:Q",
            axis=alt.Axis(
                format="%",
                title="Growth",
                labelAlign="center",
                labelExpr="datum.value < -1 ? null : datum.label",
            ),
            #             scale=alt.Scale(domain=(-1, 37)),
        ),
        y=alt.Y(
            "Minor:N",
            sort=data.Minor.to_list(),
            # axis=alt.Axis(title="", labels=False),
            axis=None,
        ),
        size=alt.Size(
            "Magnitude",
            title="Avg yearly investment (£ bn)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=[0.1, 4]),
        ),
        color=alt.Color(colour_field, legend=alt.Legend(orient="left")),
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            alt.Tooltip("Minor:N", title="Category"),
            alt.Tooltip(
                "Magnitude:Q",
                format=",.3f",
                title="Average yearly investment (billion GBP)",
            ),
            # "Number of companies",
            # "Number of deals",
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

text = (
    alt.Chart(short_term_trend_df.query("Minor != 'logistics (all other)'"))
    .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=14)
    .encode(
        text=text_field,
        x="growth:Q",
        y=alt.Y("Minor:N", sort=data.Minor.to_list(), title=""),
    )
)

final_fig = pu.configure_titles(pu.configure_axes((fig + text)), "", "")
final_fig

# %%
AltairSaver.save(
    final_fig, f"vJuly18_growth_Minor_2020_2021", filetypes=["html", "png"]
)

# %%
category = "delivery"
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    category_ts.assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total)
    .query("time_period < 2022")
    .query("Category == @category")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=tooltip,
    )
)
pu.configure_plots(fig)

# %% [markdown]
# ### Technology presence

# %%
# category='meal kits'
# pu.cb_investments_barplot(
#     category_ts.query("Category == @category"),
#     y_column="raised_amount_gbp_total",
#     y_label="Raised amount (million GBP)",
#     x_label="Year",
# )

# %%

# %% [markdown]
# ## Tags (most granular categories)

# %%
magnitude_vs_growth_granular = get_trends(taxonomy_df, rejected_tags, "Category", DR)

# %%
removed = ["enzymes"]
# removed = []

magnitude_vs_growth_filtered = (
    magnitude_vs_growth_granular.query("`Number of companies` > 20")
    .query("`Number of deals` > 20")
    .query("Category not in @removed")
)

# %%
fig_growth_vs_magnitude(
    magnitude_vs_growth_filtered,
    colour_field="Minor",
    text_field="Category",
).interactive()


# %%
fig_category_growth(
    magnitude_vs_growth_filtered, colour_field="Major", text_field="Category"
)

# %%
ids = get_category_ids(taxonomy_df, rejected_tags, DR, column="Minor")

# %%
# cat = list(ids.keys())[16]
# print(cat)
# ids_ = ids[cat]
# (
#     DR.company_labels.query("id in @ids_")
#     .query("Category in @technology_tags")
#     .groupby('Category', as_index=False).agg(counts=('id', 'count'))
#     .sort_values('counts', ascending=False)
# ).head(10)

# %%
# DR.get_companies_by_labels('arts & culture', 'TAGS')

# %%
# category = "meal kits"
# pu.cb_investments_barplot(
#     category_ts.query("Category == @category"),
#     y_column="raised_amount_gbp_total",
#     y_label="Raised amount (million GBP)",
#     x_label="Year",
# )

# %% [markdown]
# # Country performance

# %%
DR.funding_rounds.head(1)

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Major'"))
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)
data

# %%
fig = (
    alt.Chart(
        data,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        alt.Y(f"country:N", sort="-x", title="Country"),
        alt.X(
            f"raised_amount_gbp:Q",
            title="Investment amount (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(
    fig, "Investment amounts by country", "Early stage deals, 2017-2021"
)
fig

# %%
AltairSaver.save(fig, f"vJuly18_countries_early", filetypes=["html", "png"])

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Major'"))
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)
data

# %%
fig = (
    alt.Chart(
        data,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.Y(f"country:N", sort="-x", title="Country"),
        alt.X(
            f"raised_amount_gbp:Q",
            title="Investment amount (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(
    fig, "Investment amounts by country", "Late stage deals, 2017-2021"
)
fig

# %%
AltairSaver.save(fig, f"vJuly18_countries_late", filetypes=["html", "png"])

# %%
# countries = ['United Kingdom', 'United States', 'China', 'India', 'Singapore', 'Germany', 'France']
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Major'"))
    .groupby(["country", "Category"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    # .query("country in @countries")
    # .sort_values('raised_amount_gbp', ascending=False)
)
data_total = data.groupby(["country"], as_index=False).agg(
    total_amount=("raised_amount_gbp", "sum")
)

data = (
    data.merge(data_total, on=["country"], how="left").assign(
        percentage=lambda df: df.raised_amount_gbp / df.total_amount
    )
    # .query("country in @countries")
)

cats = ["innovative food", "health", "cooking and kitchen", "logistics"]
data_final = []
for cat in cats:
    data_final.append(
        data.copy()
        .query("Category == @cat")
        .sort_values("raised_amount_gbp", ascending=False)
        .head(8)
    )
data = pd.concat(data_final, ignore_index=True)


# %%
# data.query("Category == 'innovative food'").sort_values("raised_amount_gbp", ascending=False).head(8)

# %%
df_check = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "NAME", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Major'"))
    # .groupby(['country', 'Category'], as_index=False)
    # .agg(raised_amount_gbp=('raised_amount_gbp', 'sum'))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    # .query('country== "United Kingdom"')
    # .query('Category== "cooking and kitchen"')
)

# %%
fig = (
    alt.Chart((data.query("Category in @cats")))
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        # x=alt.X('percentage:Q', title="", axis=alt.Axis(format='%')),
        x=alt.X("raised_amount_gbp:Q", title="Investment (bn. GBP)"),
        # y=alt.Y('country:N', sort=countries, title=""),
        y=alt.Y("country:N", sort="-x", title=""),
        # color='country:N',
        facet=alt.Facet(
            "Category:N", title="", columns=2, header=alt.Header(labelFontSize=14)
        ),
        tooltip=[alt.Tooltip("raised_amount_gbp", format=".3f")],
    )
    .properties(
        width=180,
        height=180,
    )
    .resolve_scale(x="independent", y="independent")
)


fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(fig, f"vJuly18_countries_major_early", filetypes=["html", "png"])

# %%
(
    DR.company_data.merge(company_to_taxonomy_df)
    .query("Category=='health'")
    .query("country=='United Kingdom'")
    .sort_values("TOTAL FUNDING (EUR M)", ascending=False)
)[["NAME", "PROFILE URL", "TOTAL FUNDING (EUR M)"]].sum()

# %%
(
    DR.funding_rounds.merge(company_to_taxonomy_df)
    .query(
        "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES or `EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
    )
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .query("Category=='health'")
    .query("country=='United Kingdom'")
    .groupby(["NAME", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
)[["NAME", "PROFILE URL", "raised_amount_gbp"]].iloc[1:].sum()

# %%
(
    DR.funding_rounds.merge(company_to_taxonomy_df)
    .query(
        "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES or `EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
    )
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .query("Category=='logistics'")
    # .query("country=='United Kingdom'")
    .groupby(["NAME", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
)[["NAME", "PROFILE URL", "raised_amount_gbp"]]


# %% [markdown]
# ### ...

# %%
def deal_type(dealtype):
    if dealtype in utils.EARLY_DEAL_TYPES:
        return "Early"
    elif dealtype in utils.LATE_DEAL_TYPES:
        return "Late"
    else:
        return "n/a"


# %%
importlib.reload(utils)


# %%
def deal_type_no(dealtype):
    for i, d in enumerate(pu.DEAL_CATEGORIES):
        if dealtype == d:
            return i
    return 0


# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(company_to_taxonomy_df.query("level == 'Major'"), how="left")
    .copy()
    .assign(
        deal_type=lambda df: df["raised_amount_gbp"].apply(
            lambda x: utils.deal_amount_to_range(x * 1000)
        )
    )
    .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    .groupby(["Category", "deal_type", "deal_type_"], as_index=False)
    .agg(counts=("id", "count"))
)

# %%
fig = (
    alt.Chart(data.query("deal_type != 'n/a'"))
    .mark_bar()
    .encode(
        # x=alt.X('percentage:Q', title="", axis=alt.Axis(format='%')),
        # y= alt.X("sum(counts):Q", title="Number of deals"),#, stack="normalize", axis=alt.Axis(format="%")),
        y=alt.X(
            "sum(counts):Q", title="", stack="normalize", axis=alt.Axis(format="%")
        ),
        # y=alt.Y('country:N', sort=countries, title=""),
        x=alt.Y(
            "Category:N",
            title="",
            axis=alt.Axis(labelAngle=-45),
            sort=[
                "logistics",
                "food waste",
                "innovative food",
                "retail and restaurants",
                "agritech",
                "health",
            ],
        ),
        color=alt.Color("deal_type", title="Deal size", sort=pu.DEAL_CATEGORIES),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type_",
            sort="ascending",
        ),
        # color='country:N',
        # facet=alt.Facet('Category:N',title="", columns=3, header=alt.Header(labelFontSize=14)),
        tooltip=["deal_type", "counts", "sum(counts)"],
    )
    .properties(
        width=400,
        height=300,
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(fig, f"vJuly18_deal_types_not_norm", filetypes=["html", "png"])

# %% [markdown]
# ## Acquisitions

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    # .query("`EACH ROUND TYPE` == 'ACQUISITION'")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'").merge(
        company_to_taxonomy_df.query("level == 'Major'"), how="left"
    )
    # .copy()
    # .assign(deal_type=lambda df: df["raised_amount_gbp"].apply(lambda x: utils.deal_amount_to_range(x*1000)))
    # .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    # .groupby(['Category'], as_index=False).agg(counts=("id", "count"))
)

# %%
comp_names = [
    "Nestlé",
    "PepsiCo",
    "Unilever",
    "Tyson Ventures",
    "Archer Daniels Midland",
    "Cargill",
    "Danone",
    # 'Associated British Foods'
]
# comp_name = 'Nestlé' # meal kits and pet food
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
# comp_name = "The Middleby Corp"
# comp_name = 'Constellation Brands'
# comp_name = "Foodpanda"
# comp_name = "Syngenta"
# comp_name = 'Danone'
pd.set_option("max_colwidth", 200)
dfs = []
for comp_name in comp_names:
    dfs.append(
        data[
            (data["EACH ROUND INVESTORS"].isnull() == False)
            & data["EACH ROUND INVESTORS"].str.contains(comp_name)
        ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    )


# %%
comp_names_ = [
    "Nestlé",
    "PepsiCo",
    "Unilever N.V.",
    "Tyson Ventures",
    "Archer Daniels Midland Company",
    "Cargill",
    "Danone",
]

# %%
[len(df.NAME.unique()) for df in dfs]


# %%
dfs[2].query("`EACH ROUND TYPE` == 'ACQUISITION'")[["NAME"]]

# %%
pd.DataFrame(
    data={
        "company": comp_names_,
        "investments": [str(list(df.NAME.unique())) for df in dfs],
    }
)


# %%
(
    data.groupby("EACH ROUND INVESTORS", as_index=False)
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
).reset_index().iloc[1:11]

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` == 'ACQUISITION'")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(company_to_taxonomy_df.query("level == 'Major'"), how="left")
    # .copy()
    # .assign(deal_type=lambda df: df["raised_amount_gbp"].apply(lambda x: utils.deal_amount_to_range(x*1000)))
    # .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    # .groupby(['Category'], as_index=False).agg(counts=("id", "count"))
)

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, "Major")
category_ts = get_category_ts(category_ids, DR, ["ACQUISITION"])

# %%
magnitude_vs_growth = get_trends(
    taxonomy_df, rejected_tags, "Major", DR, ["ACQUISITION"]
)
magnitude_vs_growth

# %%
category_ts.query("Category == 'innovative food'")

# %%
# data.groupby('EACH ROUND INVESTORS').agg(counts=('id', 'count')).sort_values('counts', ascending=False).head(20)

# %%
# comp_name = 'Nestlé' # meal kits and pet food
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
# comp_name = "The Middleby Corp"
comp_name = "Constellation Brands"
# comp_name = "Foodpanda"
# comp_name = "Syngenta"
comp_name = "Archer"

(
    data[
        (data["EACH ROUND INVESTORS"].isnull() == False)
        & data["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
)[
    [
        "id",
        "NAME",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
        "country",
    ]
]

# %%
# comp_name = 'Nestlé'
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
(
    data[
        (data["EACH ROUND INVESTORS"].isnull() == False)
        & data["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
)[
    [
        "id",
        "NAME",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
        "country",
    ]
]

# %%
df = (
    DR.funding_rounds.query("id in @foodtech_ids")
    # .query("`EACH ROUND TYPE` == @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2010-01-01' and announced_on < '2022-01-01'").merge(
        company_to_taxonomy_df.query("level == 'Major'"), how="left"
    )
)

(
    df[
        (df["EACH ROUND INVESTORS"].isnull() == False)
        & df["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "TAGLINE", "PROFILE URL"]])
)[
    [
        "id",
        "NAME",
        "TAGLINE",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND TYPE",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
    ]
]

# %% [markdown]
# # Segmenting the categories

# %%
clusters = cluster_analysis_utils.hdbscan_clustering(v_labels.vectors)

# %%
df_clusters = (
    pd.DataFrame(clusters, columns=["cluster", "probability"])
    .astype({"cluster": int, "probability": float})
    .assign(text=v_labels.vector_ids)
    .merge(labels, how="left")
    .merge(category_counts, how="left")
    .drop_duplicates("text")
)

# %%
df_clusters

# %%
extra_stopwords = ["tech", "technology", "food"]
stopwords = cluster_analysis_utils.DEFAULT_STOPWORDS + extra_stopwords


cluster_keywords = cluster_analysis_utils.cluster_keywords(
    df_clusters.text.apply(
        lambda x: cluster_analysis_utils.simple_preprocessing(x, stopwords)
    ).to_list(),
    df_clusters.cluster.to_list(),
    11,
    max_df=0.7,
    min_df=0.1,
)

# %%
umap_viz_params = {
    "n_components": 2,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

reducer_2d = umap.UMAP(random_state=1, **umap_viz_params)
embedding = reducer_2d.fit_transform(v_labels.vectors)

# %%
embedding.shape

# %%
len(v_labels.vector_ids)

# %%
df_viz = (
    pd.DataFrame(v_labels.vector_ids, columns=["text"])
    .merge(df_clusters, how="left")
    .assign(
        x=embedding[:, 0],
        y=embedding[:, 1],
        #         cluster=df_clusters.cluster,
        #         cluster_prob=df_clusters.probability,
        cluster_name=df_clusters.cluster.apply(lambda x: str(cluster_keywords[x])),
        cluster_str=lambda df: df.cluster.astype(str),
        log_counts=lambda df: np.log10(df.counts),
    )
    .sort_values(["cluster", "counts"], ascending=False)
)

# %%
df_viz

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=600, height=500)
    .mark_circle(size=20, color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        size="log_counts",
        tooltip=list(df_viz.columns),
        #         color="label_type",
        color="cluster_str",
    )
)

# text = (
#     alt.Chart(centroids)
#     .mark_text(font=pu.FONT)
#     .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
# )

fig_final = (
    #     (fig + text)
    (fig)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Company labels"],
            "subtitle": [
                "All industry, sub-industry and tag labels.",
                "1328-d sentence embeddings > 2-d UMAP",
            ],
            "subtitleFont": pu.FONT,
        },
    )
    .interactive()
)

fig_final

# %%
from innovation_sweet_spots import PROJECT_DIR

df_viz.to_csv(PROJECT_DIR / "outputs/foodtech/interim/dealroom_labels.csv", index=False)

# %%
# c=34
# df_clusters.query("cluster == @c").label

# %%
ids = DR.company_labels.query("Category == 'taste'").id.to_list()
DR.company_data.query("id in @ids")

# %%
ids = DR.company_labels.query("Category == 'meal kits'").id.to_list()
DR.company_labels.query("id in @ids").groupby("Category").agg(
    counts=("id", "count")
).sort_values("counts", ascending=False)

# %% [markdown]
# ## Check health companies in detail

# %%
df_health = DR.get_companies_by_industry("health")

# %%
len(df_health)

# %%
df_health_labels = DR.company_tags.query("id in @df_health.id.to_list()")


# %%
# DR.company_labels.query("Category=='health'")

# %%
health_label_counts = (
    df_health_labels.groupby("TAGS")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)

# %%
health_label_counts.head(15)

# %%
chosen_cat = DR.company_labels.query("Category in @health_Weight").id.to_list()

# %%
# company_labels_list = (
#     DR.company_labels
#     .assign(Category = lambda df: df.Category.apply(tcu.clean_dealroom_labels))
#     .groupby("id")["Category"].apply(list)
# )

# %% [markdown]
# ### Inspect companies using embeddings

# %%
alt.data_transformers.disable_max_rows()

# %%
importlib.reload(dlr)
v_vectors = dlr.get_company_embeddings(
    filename="foodtech_may2022_companies_tagline_labels"
)

# %%
# Reduce the embedding to 2 dimensions
reducer_low_dim = umap.UMAP(
    n_components=2,
    random_state=333,
    n_neighbors=10,
    min_dist=0.5,
    spread=0.5,
)
embedding = reducer_low_dim.fit_transform(v_vectors.vectors)

# %%
category_ids_major = get_category_ids(taxonomy_df, rejected_tags, DR, "Major")
category_ids_minor = get_category_ids(taxonomy_df, rejected_tags, DR, "Minor")

# %%
list(category_ids_major.keys())

# %% [markdown]
# ## Select a category

# %%
major_category = "health"
ids = category_ids_major[major_category]
minor_categories = list(taxonomy[major_category].keys())

# %%
df_viz = (
    DR.company_data[["id", "NAME", "TAGLINE", "WEBSITE", "TOTAL FUNDING (EUR M)"]]
    .copy()
    .assign(x=embedding[:, 0])
    .assign(y=embedding[:, 1])
    .assign(minor="n/a")
    .query("`TOTAL FUNDING (EUR M)` > 0")
    .query("id in @ids")
    .copy()
    .merge(
        pd.DataFrame(
            DR.company_labels.groupby("id")["Category"].apply(list)
        ).reset_index()
    )
)

for cat in minor_categories:
    df_viz.loc[df_viz.id.isin(category_ids_minor[cat]), "minor"] = cat

# %%
len(DR.company_data.query("id in @ids"))

# %%
len(DR.company_data.query("`TOTAL FUNDING (EUR M)` > 0").query("id in @ids"))

# %%
# df_viz

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=500, height=500)
    .mark_circle(size=40)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=list(df_viz.columns),
        color="minor:N",
        href="WEBSITE",
        size=alt.Size("TOTAL FUNDING (EUR M)", scale=alt.Scale(range=[20, 2000])),
    )
)
fig_final = (
    (fig)
    .configure_axis(
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Landscape of companies"],
            "subtitle": "",
            #             [
            #                 "Each circle is a course; courses with similar titles will be closer on this map",
            #                 "Press Shift and click on a circle to go the course webpage",
            #             ],
            "subtitleFont": pu.FONT,
        },
    )
    .interactive()
)

fig_final

# %%
company_to_taxonomy_dict["960205"]


# %%
AltairSaver.save(fig_final, f"foodtech_companies_{major_category}", filetypes=["html"])

# %%
