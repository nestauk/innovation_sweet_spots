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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# df_per_yearTasks
# - Load in the reviewed tables and filter hits
# - Add year and funding information
# - Produce time series (absolute values and normalised to 2015)
# - Compare with cb or gtr reference 'research topic' categories
#
# Check also:
# - News articles with certain terms

# %%
import innovation_sweet_spots.analysis.analysis_utils as iss
from innovation_sweet_spots import PROJECT_DIR, logging, config
from innovation_sweet_spots.getters import gtr, crunchbase, guardian
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
import pandas as pd
import numpy as np

# %%
from itertools import groupby
import collections
from innovation_sweet_spots.analysis import discourse_utils as disc
import spacy


TAGS = ["p", "h2"]

# %%
YEARLY_COLS = [
    "no_of_projects",
    "amount_total",
    "amount_median",
    "no_of_rounds",
    "raised_amount_total",
    "raised_amount_usd_total",
    "articles",
]

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/data/results_august"

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %%

# %% [markdown]
# # Import prerequisites

# %% [markdown]
# ## GTR

# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
link_gtr_topics = gtr.get_link_table("gtr_topic")

# %%
# gtr_project_funds.category.unique()

# %%
# gtr_organisations.head()

# %%
# GTR project to funds/orgs tables
gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(gtr_projects, gtr_project_funds)
del link_gtr_funds

project_to_org = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
gtr_projects.head(1)

# %% [markdown]
# ## Crunchbase

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"

# %%
# Import Crunchbase data
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
cb_df = cb_df[cb_df.country == "United Kingdom"]
cb_df = cb_df.reset_index(drop=True)
del cb
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %%
import altair as alt

# %%
cb_categories = crunchbase.get_crunchbase_organizations_categories()

# %%
cb_df.head(1)

# %% [markdown]
# ### Hansard

# %%
from innovation_sweet_spots.getters.hansard import get_hansard_data
import innovation_sweet_spots.utils.text_pre_processing as iss_preprocess

nlp = iss_preprocess.setup_spacy_model(iss_preprocess.DEF_LANGUAGE_MODEL)
from tqdm.notebook import tqdm

hans = get_hansard_data()
hans_docs = iss.create_documents_from_dataframe(
    hans, columns=["speech"], preprocessor=iss.preprocess_text
)

# hans_docs_tokenised = []
# for doc in tqdm(nlp.pipe(hans_docs), total=len(hans_docs)):
#     hans_docs_tokenised.append(' '.join(iss_preprocess.process_text(doc)))

# %% [markdown]
# # Inspect technology categories

# %%
# REVIEWED_DOCS_PATH = OUTPUTS_DIR / 'aux/ISS_technologies_to_review_August_10.xlsx'
REVIEWED_DOCS_PATH = OUTPUTS_DIR / "aux/ISS_technologies_to_review_September_1.xlsx"


# %%
def find_hit_column(df):
    for col in df.columns:
        if "hit" in col.lower():
            return col


def get_verified_docs(sheet_names, fpath=REVIEWED_DOCS_PATH):
    dfs = pd.DataFrame()
    for SHEET_NAME in sheet_names:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME)
        hit_column = find_hit_column(df)
        df = df[df[hit_column] != 0]
        df = df[COLS]
        df["tech_category"] = SHEET_NAME
        dfs = dfs.append(df, ignore_index=True)
    return dfs


def get_verified_docs_2(sheet_names, fpath=REVIEWED_DOCS_PATH):
    dfs = pd.DataFrame()
    for SHEET_NAME in sheet_names:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME)
        hit_column = find_hit_column(df)
        df = df[df[hit_column] != 0]
        df = df.rename(columns={"short_description": "description", "name": "title"})
        df = df[COLS]
        df["tech_category"] = SHEET_NAME
        dfs = dfs.append(df, ignore_index=True)
    return dfs


def get_doc_probs(sheet_names, fpath=REVIEWED_DOCS_PATH):
    dfs = pd.DataFrame()
    for SHEET_NAME in sheet_names:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME)
        hit_column = find_hit_column(df)
        df = df[df[hit_column] != 0]
        df = df[["doc_id", "global_topic_prob", "local_topic_prob"]]
        df["tech_category"] = SHEET_NAME
        dfs = dfs.append(df, ignore_index=True)
    return dfs


def deduplicate_docs(df):
    return df.copy().drop_duplicates("doc_id").drop("tech_category", axis=1)


def add_project_data(df):
    return df.merge(
        funded_projects[
            ["project_id", "grantCategory", "leadFunder", "start", "amount", "category"]
        ],
        left_on="doc_id",
        right_on="project_id",
        how="left",
    )


def add_crunchbase_data(df):
    return df.merge(cb_df, left_on="doc_id", right_on="id", how="left")


def add_crunchbase_categories(df, doc_column="doc_id"):
    return df.merge(
        cb_categories[["organization_id", "category_name"]],
        left_on=doc_column,
        right_on="organization_id",
        how="left",
    )


def add_gtr_categories(df, doc_column="doc_id"):
    if "project_id" not in df.columns:
        df["project_id"] = df[doc_column]
    return iss.get_gtr_project_topics(df)


def normalise_timeseries(df, yearly_cols=YEARLY_COLS, ref_year=2015):
    df_norm = df.copy()
    for col in yearly_cols:
        df_norm[col] = df_norm[col] / df_norm.loc[df_norm.year == ref_year, col].iloc[0]
    return df_norm


## Guardian
def articles_table(articles):
    df = pd.DataFrame(
        data={
            "headline": [a["fields"]["headline"] for a in articles],
            "date": [a["webPublicationDate"] for a in articles],
            "section": [a["sectionName"] for a in articles],
        }
    )
    df.date = [x[0:10] for x in df.date.to_list()]
    df["year"] = df.date.apply(iss.convert_date_to_year)
    return df


# %%
# NB this is created down the line in this same notebook
FUNDS_RELIABLE = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/RELIABLE_GTR_FUNDS_3.csv"
)

# %%
FUNDS_RELIABLE = FUNDS_RELIABLE[FUNDS_RELIABLE.is_api_fund_reliable]

# %%
len(FUNDS_RELIABLE)


# %%
def get_reliable_funds(gtr_docs, FUNDS_RELIABLE):
    gtr_docs_ = gtr_docs.copy()
    gtr_docs_["amount_old"] = gtr_docs_["amount"].copy()
    gtr_docs_ = gtr_docs_.drop("amount", axis=1).merge(
        FUNDS_RELIABLE[["doc_id", "amount"]], how="left"
    )
    logging.info(
        f"{len(gtr_docs_[gtr_docs_.amount.isnull()])} documents without funding info"
    )
    gtr_docs_.loc[gtr_docs_.amount.isnull(), "category"] = "NO_FUND_INFO"
    gtr_docs_["amount"] = gtr_docs_["amount"].fillna(0)
    return gtr_docs_


# %%
# len(GTR_DOCS_ALL)

# %%
# dff = GTR_DOCS_REF_ALL[GTR_DOCS_REF_ALL.tech_category=='Batteries'].copy()

# %%
# dfff=get_reliable_funds(dff, FUNDS_RELIABLE)
# dfff.sort_values('amount_old', ascending=False)

# %%
cb_df_cat = add_crunchbase_categories(cb_df, doc_column="id")
gtr_cat = add_gtr_categories(funded_projects)

# %%
# nn = gtr_cat[gtr_cat.text!='Unclassified'].drop_duplicates('project_id').project_id.to_list()
# n_projects_with_categories = len(nn)
# n_projects_without_categories = len(gtr_cat[(gtr_cat.text=='Unclassified') & (gtr_cat.project_id.isin(dd)==False)].drop_duplicates('project_id'))
# n_projects_without_categories / len(gtr_projects)

# %%
import importlib

importlib.reload(iss)


# %%
# Find documents using substring matching
def find_docs_with_terms(terms, corpus_texts, corpus_df, return_dataframe=True):
    x = np.array([False] * len(corpus_texts))
    for term in terms:
        x = x | np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def find_docs_with_all_terms(terms, corpus_texts, corpus_df, return_dataframe=True):
    x = np.array([True] * len(corpus_texts))
    for term in terms:
        x = x & np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def get_docs_with_keyphrases(keyphrases, corpus_texts, corpus_df):
    x = np.array([False] * len(corpus_texts))
    for terms in keyphrases:
        print(terms)
        x = x | find_docs_with_all_terms(
            terms, corpus_texts, corpus_df, return_dataframe=False
        )
    return corpus_df.iloc[x]


def get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches):
    # Deduplicated versions (when combining several categories)
    gtr_docs_dedup = deduplicate_docs(gtr_docs)
    cb_doc_dedup = deduplicate_docs(cb_docs)
    # GTR data
    df_research_per_year = iss.gtr_funding_per_year(
        gtr_docs_dedup, min_year=2007, max_year=2021
    )
    # CB data
    df_deals = iss.get_cb_org_funding_rounds(cb_doc_dedup, cb_funding_rounds)
    df_deals_per_year = iss.get_cb_funding_per_year(df_deals, max_year=2021)
    df_cb_orgs_founded_per_year = iss.cb_orgs_founded_by_year(
        cb_doc_dedup, max_year=2021
    )
    # Guardian data
    df_articles_per_year = iss.get_guardian_mentions_per_year(
        guardian_articles, max_year=2021
    )
    # Hansard
    speeches_per_year = iss.get_hansard_mentions_per_year(
        speeches, max_year=2021
    ).rename(columns={"mentions": "speeches"})

    df_per_year = (
        df_research_per_year.merge(df_deals_per_year, how="left")
        .merge(df_cb_orgs_founded_per_year, how="left")
        .merge(df_articles_per_year, how="left")
        .merge(speeches_per_year, how="left")
    )
    return df_per_year


def aggregate_guardian_articles(category_articles, CATEGORY_NAMES):
    articles = []
    for key in CATEGORY_NAMES:
        articles += category_articles[key]
    aggregated_articles = [article for sublist in articles for article in sublist]
    return aggregated_articles


def aggregate_hansard_speeches(CATEGORY_NAMES):
    s_list = [categories_keyphrases_hans[key] for key in CATEGORY_NAMES]
    s_list = [s for ss in s_list for s in ss]
    speeches = get_docs_with_keyphrases(s_list, corpus_texts=hans_docs, corpus_df=hans)
    return speeches


# %% [markdown]
# ## Keywords

# %% [markdown]
# NB: Main heat pump manufacturers are not from the UK
# - https://thegreenergroup.com/news/ashp-manufacturers/

# %%
# Params
CATEGORY_NAMES = [
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Hydrogen boilers",
    "Biomass boilers",
    "Building insulation",
    "Energy management",
    "Radiators",
]
COLS = ["doc_id", "title", "description", "source"]

# Somereference categories
CB_REFERENCE_CATEGORIES = [
    "pet",
    "artificial intelligence",
    "wind energy",
    "solar",
    "biofuel",
    "biomass energy",
    "nuclear",
    "fuel cell",
    "electric vehicle",
]
GTR_REFERENCE_CATEGORIES = [
    "Solar Technology",
    "Wind Power",
    "Bioenergy",
    "Fuel Cell Technologies",
]

# %%
# categories_keyphrases = {
#     # Heating subcategories
#     'Heat pumps': [['heat pump']],
#     'Geothermal energy': [['geothermal', 'energy'], ['geotermal', 'heat']],
#     'Solar thermal': [['solar thermal']],
#     'Waste heat': [['waste heat'], ['heat recovery']],
#     'Heat storage': [['heat stor'], ['thermal energy stor'], ['thermal stor'], ['heat batter']],
#     'District heating': [['heat network'], ['district heat']],
#     'Electric boilers': [['electric', 'boiler'], ['electric heat']],
#     'Biomass boilers': [['pellet', 'boiler'], ['biomass', 'boiler']],
#     'Hydrogen boilers': [['hydrogen', 'boiler']],
#     'Micro CHP': [['combined heat power', 'micro'], ['micro', 'chp'] ,['mchp']]
#     'Building insulation': [['insulat', 'build'], ['insulat', 'hous'], ['insulat', 'retrofit'], ['cladding', 'hous'], ['cladding', 'build'], ['glazing', 'window'], ['glazed', 'window']],
#     'Radiators': [['radiator']],
#     'Energy management': [['energy management', 'build'], ['energy management', 'domestic'], ['energy management', 'hous'], ['thermostat'], ['smart meter'], ['smart home', 'heat'], ['demand response', 'heat']],
# }


# %%
# for key in categories_keyphrases:
#     print(f'{key}: {categories_keyphrases[key]}')

# %%
categories_keyphrases_hans = {
    # Heating subcategories
    "Heat pumps": [["heat pump"]],
    "Geothermal energy": [["geothermal", "energy"], ["geotermal", "heat"]],
    "Solar thermal": [["solar thermal"]],
    "Waste heat": [["waste heat"], ["heat recovery"]],
    "Heat storage": [
        ["heat stor"],
        ["thermal energy stor"],
        ["thermal stor"],
        ["heat batter"],
    ],
    "District heating": [["heat network"], ["district heat"]],
    "Electric boilers": [["electric", "boiler"], ["electric heat"]],
    "Biomass heating": [
        ["pellet", "boiler"],
        ["biomass", "boiler"],
        ["biomass", "heat"],
    ],
    "Hydrogen boilers": [["hydrogen", "boiler"], ["hydrogen", "heat"]],
    "Hydrogen heating": [["hydrogen", "boiler"], ["hydrogen", "heat"]],
    "Micro CHP": [["combined heat power", "micro"], ["micro", "chp"], ["mchp"]],
    "Building insulation": [
        ["insulat", "build"],
        ["insulat", "hous"],
        ["insulat", "retrofit"],
        ["insulat", "home"],
        ["glazing", "window"],
        ["glazed", "window"],
    ],
    "Radiators": [["radiator"]],
    "Energy management": [
        ["energy management", "build"],
        ["energy management", "domestic"],
        ["energy management", "hous"],
        ["energy management", "home"],
        ["thermostat"],
        ["smart meter"],
        ["smart home"],
        ["demand reponse", "heat"],
        ["demand response", "energy"],
        ["load shift", "heat"],
        ["load shift", "energy"],
    ],
}

# %%
# Search terms for public discourse analysis
search_terms = {
    "Heat pumps": ["heat pump", "heat pumps"],
    "Geothermal energy": ["geothermal energy", "geothermal heat", "geothermal heating"],
    "Solar thermal": ["solar thermal"],
    "District heating": [
        "district heating",
        "district heat",
        "heat network",
        "heat networks",
    ],
    "Building insulation": [
        "home insulation",
        "building insulation",
        "house insulation",
        "retrofitting insulation",
    ],
    "Energy management": [
        "smart meter",
        "smart meters",
        "smart thermostat",
        "smart thermostats",
        "home energy management",
        "household energy management",
        "building energy management",
        "domestic energy management",
        "energy demand response",
        "heating demand response",
        "heat demand response",
        "heat load shifting",
        "energy load shifting",
        "demand response",
    ],
    "Hydrogen boilers": [
        "hydrogen boiler",
        "hydrogen boilers",
        "hydrogen-ready boiler",
        "hydrogen-ready boilers",
        "hydrogen heating",
        "hydrogen heat",
    ],
    "Biomass heating": [
        "biomass boiler",
        "biomass boilers",
        "biomass heating",
        "biomass heat",
    ],
    "Electric boilers": ["electric heating", "electric boiler", "electric boilers"],
    "Hydrogen heating": [
        "hydrogen boiler",
        "hydrogen boilers",
        "hydrogen-ready boiler",
        "hydrogen-ready boilers",
        "hydrogen heating",
        "hydrogen heat",
    ],
    "Micro CHP": [
        "micro chp",
        "micro-combined heat and power",
        "micro combined heat and power",
        "micro-chp",
    ],
    "Heat storage": [
        "heat storage",
        "heat store",
        "thermal storage",
        "thermal energy storage",
    ],
}

# %%
# USE_CACHED_GUARDIAN = False
USE_CACHED_GUARDIAN = True

# %%
# Get guardian articles
category_articles = {}
for category in search_terms:
    category_articles[category] = [
        guardian.search_content(search_term, use_cached=USE_CACHED_GUARDIAN)
        for search_term in search_terms[category]
    ]

# %% [markdown]
# ## Yearly time series

# %%
REF_YEAR = 2016


# %%
def process_gtr_docs(df):
    gtr_docs = add_project_data(df[df.source == "gtr"])
    # Deduplicate
    gtr_docs_dedup = gtr_docs.groupby(["title", "description"]).sum().reset_index()
    gtr_docs_ = (
        gtr_docs.drop("amount", axis=1)
        .merge(gtr_docs_dedup, on=["title", "description"])
        .sort_values("start")
    )
    gtr_docs_ = gtr_docs_.drop_duplicates(["title", "description"], keep="first")
    return gtr_docs_


def dedup_gtr_docs(gtr_docs):
    gtr_docs_dedup = gtr_docs.groupby(["title", "description"]).sum().reset_index()
    gtr_docs_ = (
        gtr_docs.drop("amount", axis=1)
        .merge(
            gtr_docs_dedup[["title", "description", "amount"]],
            on=["title", "description"],
        )
        .sort_values("start")
    )
    gtr_docs_ = gtr_docs_.drop_duplicates(["title", "description"], keep="first")
    return gtr_docs_


def process_guardian_articles(guardian_articles):
    guardian_articles_ = [
        g
        for g in guardian_articles
        if g["sectionName"]
        in [
            "Environment",
            "Guardian Sustainable Business",
            "Technology",
            "Science",
            "Business",
            "Money",
            "Cities",
            "Politics",
            "Opinion",
            "Global Cleantech 100",
            "The big energy debate",
        ]
    ]
    dff = iss.articles_table(guardian_articles_, sort_by_date=False)
    is_dupl = dff.reset_index().duplicated("headline").to_list()
    guardian_articles_ = [
        g for i, g in enumerate(guardian_articles_) if is_dupl[i] == False
    ]
    return guardian_articles_


# %%
# pd.set_option('max_colwidth', 150)
# gtr_projects[gtr_projects.title.str.contains('Peak Stoves')]

# %%
# cb_df[cb_df.name.str.contains('JFS Home Farm Biogas')]

# %%
manual_cb_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-11-at-2021-08-24-17-28-6e9a63e0.csv"
)
manual_build_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-7-at-2021-08-24-22-59-20819493_buildings.csv"
)
manual_build_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-7-at-2021-08-30-16-06-b6d91b02_buildings.csv"
)

# %%
NEW_CB_NAMES = [
    "Heat storage",
    "Electric boilers",
    "Heat pumps",
    "District heating",
    "Building insulation",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
]

# %%
# CATEGORY_NAMES = ['Heat pumps', 'Geothermal energy', 'Solar thermal', 'District heating', 'Hydrogen boilers', 'Biomass boilers', 'Building insulation', 'Energy management']
CATEGORY_NAMES = [
    "Micro CHP",
    "Heat storage",
    #     'Electric boilers',
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Building insulation",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
]
# CATEGORY_NAMES = ['Heat pumps']
YEARLY_STATS = {}
YEARLY_STATS_NORM = {}
GTR_DOCS_ALL = pd.DataFrame()
CB_DOCS_ALL = pd.DataFrame()
for cat in CATEGORY_NAMES:
    df = get_verified_docs([cat])
    if cat in NEW_CB_NAMES:
        df = pd.concat(
            [
                df,
                get_verified_docs_2(
                    [cat],
                    fpath=PROJECT_DIR
                    / "outputs/data/results_august/ISS_technologies_to_review_August_27_checked.xlsx",
                ),
            ]
        ).drop_duplicates("doc_id")

    df_add = pd.concat(
        [
            manual_cb_review[manual_cb_review.sentiment == cat],
            manual_build_review[manual_build_review.sentiment == cat],
        ]
    )[["doc_id", "title", "description", "source"]]
    df_add["tech_category"] = cat
    df = pd.concat([df, df_add]).drop_duplicates("doc_id")
    # Extract GTR and CB into separate dataframes
    gtr_docs = process_gtr_docs(df)
    gtr_docs = get_reliable_funds(gtr_docs, FUNDS_RELIABLE)
    cb_docs = add_crunchbase_data(df[df.source == "cb"])
    # Guardian articles
    guardian_articles = aggregate_guardian_articles(category_articles, [cat])
    guardian_articles = process_guardian_articles(guardian_articles)
    # Speeches
    speeches = aggregate_hansard_speeches([cat])
    # Yearly stats
    df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
    df_per_year_norm = normalise_timeseries(
        iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
        ref_year=REF_YEAR,
    )
    YEARLY_STATS[cat] = df_per_year
    YEARLY_STATS_NORM[cat] = df_per_year_norm
    GTR_DOCS_ALL = GTR_DOCS_ALL.append(gtr_docs, ignore_index=True)
    CB_DOCS_ALL = CB_DOCS_ALL.append(cb_docs, ignore_index=True)


# %%
cat = "Heating & Building Energy Efficiency"
# Combined heating and building efficiency
gtr_docs = GTR_DOCS_ALL.copy().drop_duplicates("doc_id")
cb_docs = CB_DOCS_ALL.copy().drop_duplicates("doc_id")
# Guardian articles
guardian_articles = aggregate_guardian_articles(category_articles, CATEGORY_NAMES)
guardian_articles = process_guardian_articles(guardian_articles)
# Speeches
speeches = aggregate_hansard_speeches(CATEGORY_NAMES)
# Yearly stats
df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
df_per_year_norm = normalise_timeseries(
    iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
    ref_year=REF_YEAR,
)
YEARLY_STATS[cat] = df_per_year
YEARLY_STATS_NORM[cat] = df_per_year_norm

gtr_docs["tech_category"] = "Heating & Building Energy Efficiency"
cb_docs["tech_category"] = "Heating & Building Energy Efficiency"
GTR_DOCS_ALL_HEAT_BUILD = gtr_docs.copy()
CB_DOCS_ALL_HEAT_BUILD = cb_docs.copy()

# %%
# CB_DOCS_ALL[CB_DOCS_ALL.tech_category=='Geothermal energy'].sort_values('title')

# %%
GTR_DOCS_ALL.to_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_GTR.csv",
    index=False,
)
CB_DOCS_ALL.to_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_CB.csv", index=False
)

# %%
GTR_DOCS_ALL.duplicated(["doc_id", "tech_category"]).sum()

# %%
CATEGORY_NAMES = [
    "Micro CHP",
    "Heat storage",
    #     'Electric boilers',
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Hydrogen heating",
    "Biomass heating",
]
cat = "Heating (all)"
# Combined heating and building efficiency
gtr_docs = (
    GTR_DOCS_ALL[GTR_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)]
    .copy()
    .drop_duplicates("doc_id")
)
cb_docs = pd.concat(
    [
        CB_DOCS_ALL[CB_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)].copy(),
        #     add_crunchbase_data(heating_cb_orgs[['doc_id', 'title', 'description', 'source']])
    ]
).drop_duplicates("doc_id")

# Guardian articles
guardian_articles = aggregate_guardian_articles(category_articles, CATEGORY_NAMES)
guardian_articles = process_guardian_articles(guardian_articles)
# Speeches
speeches = aggregate_hansard_speeches(CATEGORY_NAMES)
# Yearly stats
df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
df_per_year_norm = normalise_timeseries(
    iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
    ref_year=REF_YEAR,
)
YEARLY_STATS[cat] = df_per_year
YEARLY_STATS_NORM[cat] = df_per_year_norm

gtr_docs["tech_category"] = "Heating (all)"
cb_docs["tech_category"] = "Heating (all)"
GTR_DOCS_ALL_HEAT = gtr_docs.copy()
CB_DOCS_ALL_HEAT = cb_docs.copy()

# %%
CATEGORY_NAMES = [
    "Building insulation",
    "Energy management",
]
cat = "Building Energy Efficiency"
# Combined heating and building efficiency
gtr_docs = (
    GTR_DOCS_ALL[GTR_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)]
    .copy()
    .drop_duplicates("doc_id")
)
cb_docs = pd.concat(
    [
        CB_DOCS_ALL[CB_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)].copy(),
        #     add_crunchbase_data(heating_cb_orgs[['doc_id', 'title', 'description', 'source']])
    ]
).drop_duplicates("doc_id")

# Guardian articles
guardian_articles = aggregate_guardian_articles(category_articles, CATEGORY_NAMES)
guardian_articles = process_guardian_articles(guardian_articles)
# Speeches
speeches = aggregate_hansard_speeches(CATEGORY_NAMES)
# Yearly stats
df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
df_per_year_norm = normalise_timeseries(
    iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
    ref_year=REF_YEAR,
)
YEARLY_STATS[cat] = df_per_year
YEARLY_STATS_NORM[cat] = df_per_year_norm

gtr_docs["tech_category"] = "Building Energy Efficiency"
cb_docs["tech_category"] = "Building Energy Efficiency"
GTR_DOCS_ALL_BUILDINGS = gtr_docs.copy()
CB_DOCS_ALL_BUILDINGS = cb_docs.copy()

# %%
heating_cb_orgs = manual_cb_review[
    (manual_cb_review.category == "Heating (all other)")
    & (manual_cb_review.sentiment == "Keep")
    & (manual_cb_review.doc_id.isin(CB_DOCS_ALL.doc_id.to_list()) == False)
]

# %%
CATEGORY_NAMES = [
    "Electric boilers",
]
cat = "Heating (other)"
# Combined heating and building efficiency
gtr_docs = (
    GTR_DOCS_ALL[GTR_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)]
    .copy()
    .drop_duplicates("doc_id")
)
cb_docs = pd.concat(
    [
        CB_DOCS_ALL[CB_DOCS_ALL.tech_category.isin(CATEGORY_NAMES)].copy(),
        add_crunchbase_data(
            heating_cb_orgs[["doc_id", "title", "description", "source"]]
        ),
        get_verified_docs_2(
            [cat],
            fpath=PROJECT_DIR
            / "outputs/data/results_august/ISS_technologies_to_review_August_27_checked.xlsx",
        ),
    ]
).drop_duplicates("doc_id")
cb_docs = cb_docs[cb_docs.doc_id.isin(CB_DOCS_ALL_HEAT.doc_id.to_list()) == False]

# Guardian articles
guardian_articles = aggregate_guardian_articles(category_articles, CATEGORY_NAMES)
guardian_articles = process_guardian_articles(guardian_articles)
# Speeches
speeches = aggregate_hansard_speeches(CATEGORY_NAMES)
# Yearly stats
df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
df_per_year_norm = normalise_timeseries(
    iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
    ref_year=REF_YEAR,
)
YEARLY_STATS[cat] = df_per_year
YEARLY_STATS_NORM[cat] = df_per_year_norm

gtr_docs["tech_category"] = "Heating (other)"
cb_docs["tech_category"] = "Heating (other)"
GTR_DOCS_ALL_HEAT_OTHER = gtr_docs.copy()
CB_DOCS_ALL_HEAT_OTHER = cb_docs.copy()

# %%
# CB_DOCS_ALL_HEAT_OTHER

# %%
# ALL_DFS = []
# for cat in CATEGORY_NAMES:
#     ALL_DFS.append(get_doc_probs([cat]))

# %%
# from innovation_sweet_spots.utils.io import save_list_of_terms
# save_list_of_terms(list(GTR_DOCS_ALL.doc_id.unique()), PROJECT_DIR / 'outputs/data/results_august/check_doc_id_all.txt')\

# %%
# GTR_DOCS_ALL['year'] = GTR_DOCS_ALL.start.apply(iss.convert_date_to_year)
# df=GTR_DOCS_ALL[GTR_DOCS_ALL.tech_category.isin(['Heat pumps', 'Hydrogen heating', 'Building insulation'])][['doc_id', 'title', 'description', 'source', 'tech_category', 'year']]
# df.to_csv('ISS_example_gtr_data_August17.csv', index=False)

# %%
df = (
    gtr_docs[["title", "amount", "start"]]
    .sort_values("start")
    .drop_duplicates("title", keep="first")
)
df["year"] = df.start.apply(iss.convert_date_to_year)
df["amount"] = df["amount"] / 1000
df = df[["title", "amount", "year"]]
df = df.rename(columns={"amount": "amount (1000s)"})
df


# %% [markdown]
# #### Define growth calculation

# %%
def get_growth_and_level(cat, variable, year_1=2016, year_2=2020, window=3):
    df = YEARLY_STATS[cat].copy()
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    if df_ma.loc[year_1, variable] != 0:
        growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    else:
        growth_rate = np.nan
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


# %%
def get_growth_and_level_2(df, variable, year_1=2016, year_2=2020, window=3):
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


# %%
variable = "no_of_projects"
# variable='amount_total'
# variable='articles'
# variable='speeches'
# variable='no_of_rounds'


# %%
from innovation_sweet_spots.utils.visualisation_utils import COLOUR_PAL


# %%
def plot_matrix(
    variable, category_names, x_label, y_label="Growth", year_1=2016, year_2=2020
):

    df_stats = pd.DataFrame(
        [
            get_growth_and_level(cat, variable, year_1, year_2) + tuple([cat])
            for cat in category_names
        ],
        columns=["growth", variable, "tech_category"],
    )

    points = (
        alt.Chart(
            df_stats,
            height=350,
            width=350,
        )
        .mark_circle(size=35)
        .encode(
            alt.Y("growth:Q", title=y_label),
            alt.X(f"{variable}:Q", title=x_label),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(
                    domain=category_names, range=COLOUR_PAL[0 : len(category_names)]
                ),
                legend=None,
            ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=7, size=15).encode(
        text="tech_category"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig


# %%
CATEGORY_NAMES_ = [
    "Heat pumps",
    "Heat storage",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Building insulation",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
]
iss.nicer_axis(
    plot_matrix(
        variable="no_of_projects",
        category_names=CATEGORY_NAMES_,
        x_label="Avg number of projects per year",
    )
)

# %%
# iss.nicer_axis(plot_matrix(variable='amount_total', category_names=CATEGORY_NAMES_, x_label='Avg yearly amount (£1000s)'))

# %%
# aggregate_guardian_articles(category_articles, ['Hydrogen boilers'])

# %%
# iss.get_guardian_mentions_per_year(aggregate_guardian_articles(category_articles, ['Hydrogen boilers']))

# %%
# iss.nicer_axis(plot_matrix(variable='articles', category_names=CATEGORY_NAMES, x_label='Avg news articles per year', year_1=2017))

# %%
# iss.nicer_axis(plot_matrix(variable='speeches', category_names=CATEGORY_NAMES, x_label='Avg speeches per year'))

# %%
# iss.nicer_axis(plot_matrix(variable='no_of_projects', x_label='Avg number of projects per year'))

# %%
# speeches = aggregate_hansard_speeches(['Building insulation'])

# %%
# speeches_docs = iss.create_documents_from_dataframe(speeches, columns=["speech"], preprocessor=iss.preprocess_text)

# %%
# sentences = iss.get_sentences_with_term('insulat', speeches_docs)
# sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# for i, row in sentiment_df.iloc[0:5].iterrows():
#     print(row.compound, row.sentences, end="\n\n")

# %%
# for i, row in sentiment_df.sort_values("compound").iloc[0:5].iterrows():
#     print(row.compound, row.sentences, end="\n\n")

# %%
# YEARLY_STATS_df = pd.DataFrame()
# YEARLY_STATS_NORM_df = pd.DataFrame()
# for cat in CATEGORY_NAMES:
#     df_per_year = YEARLY_STATS[cat].copy()
#     df_per_year['tech_category'] = cat
#     YEARLY_STATS_df = YEARLY_STATS_df.append(df_per_year, ignore_index=True)

#     df_per_year_norm = YEARLY_STATS_NORM[cat].copy()
#     df_per_year_norm['tech_category'] = cat
#     YEARLY_STATS_NORM_df=YEARLY_STATS_NORM_df.append(df_per_year_norm, ignore_index=True)

# %%
# viz_cols = ['raised_amount_usd_total', 'no_of_projects', 'articles']
# df_per_year_melt = pd.melt(df_per_year_norm, id_vars=['year'], value_vars=viz_cols)
# alt.Chart(df_per_year_melt, width=450, height=200).mark_line(size=2.5).encode(
#     x='year:O',
#     y='value',
#     color='variable',
# )

# %%
# cat = 'Energy management'
# df_ = YEARLY_STATS[cat]
# df = iss_topics.get_moving_average(df_, window=3, rename_cols=False)
# ww=350
# hh=150

# base = (
#     alt.Chart(df, width=ww, height=hh)
#     .encode(
#         alt.X(
#             'year:O',
#             axis=alt.Axis(title=None, labels=False)))
# )

# fig_projects = (
#     base
#     .mark_line(color='#2F1847', size=2.5)
#     .encode(
#         alt.Y(
#             'no_of_projects',
#             axis=alt.Axis(title='Number of projects', titleColor='#2F1847')
#         )
#     )
# )

# fig_money = (
#     base
#     .mark_line(color='#C62E65', size=2.5)
#     .encode(
#         alt.Y(
#             'amount_total',
#             axis=alt.Axis(title='Funding amount (£1000s)', titleColor='#C62E65')
#         )
#     )
# )
# fig_gtr = alt.layer(fig_projects, fig_money).resolve_scale(y = 'independent')

# fig_rounds = (
#     base
#     .mark_line(color='#1D3354', size=2.5)
#     .encode(
#         alt.Y(
#             'no_of_rounds',
#             axis=alt.Axis(title='Number of rounds', titleColor='#1D3354')
#         )
#     )
# )
# fig_amount_raised = (
#     base
#     .mark_line(color='#4CB7BD', size=2.5)
#     .encode(
#         alt.Y(
#             'raised_amount_usd_total',
#             axis=alt.Axis(title='Amount raised ($1000s)', titleColor='#4CB7BD')
#         )
#     )
# )
# fig_crunchbase = alt.layer(fig_rounds, fig_amount_raised).resolve_scale(y = 'independent')

# fig_articles = (
#     alt.Chart(df, width=ww, height=hh)
#     .mark_line(color='#624763', size=2.5)
#     .encode(
#         x=alt.X(
#             'year:O',
#             axis=alt.Axis(title='Year')
#         ),
#         y=alt.Y(
#             'articles',
#             axis=alt.Axis(title='Number of news articles', titleColor='#624763')
#         )
#     )
# )
# fig_speeches = (
#     alt.Chart(df, width=ww, height=hh)
#     .mark_line(color='#F9B3D1', size=2.5)
#     .encode(
#         x=alt.X(
#             'year:O',
#             axis=alt.Axis(title='Year')
#         ),
#         y=alt.Y(
#             'speeches',
#             axis=alt.Axis(title='Parliament speeches', titleColor='#F9B3D1')
#         )
#     )
# )

# fig_discourse = alt.layer(fig_articles, fig_speeches).resolve_scale(y = 'independent')
# fig = iss.nicer_axis(alt.vconcat(fig_gtr, fig_crunchbase, fig_discourse))
# fig

# %%
# alt_save.save_altair(fig, f"asf_showntell_{'_'.join(cat.lower().split())}_tseries", driver)

# %%
# alt_save.save_altair

# %%
# w=3
# iss.show_time_series(iss_topics.get_moving_average(mentions, window=w), y=f'mentions_sma{w}')

# %%
# speeches.groupby(['year', 'major_heading']).agg(counts=('speech', 'count')).sort_values('counts', ascending=False)

# %%
# w=1
# iss.show_time_series(iss_topics.get_moving_average(mentions, window=w), y=f'mentions_sma{w}')

# %%
# # viz_cols = ['no_of_projects', 'amount_total', 'no_of_rounds', 'raised_amount_usd_total', 'articles']
# # viz_cols = ['no_of_projects', 'amount_total', 'articles']
# viz_cols = ['raised_amount_usd_total', 'no_of_projects', 'articles']
# df_per_year_melt = pd.melt(df_per_year_norm, id_vars=['year'], value_vars=viz_cols)
# alt.Chart(df_per_year_melt, width=450, height=200).mark_line(size=2.5).encode(
#     x='year:O',
#     y='value',
#     color='variable',
# )

# %% [markdown]
# ### Check organisations

# %%
project_orgs = iss.get_gtr_project_orgs(gtr_docs, project_to_org)
funded_orgs = iss.get_org_stats(project_orgs)
funded_orgs.amount_total = funded_orgs.amount_total / 1000
funded_orgs.reset_index().rename(columns={"amount_total": "total amount (1000s)"})

# %%
articles_table(guardian_articles).query("year==2019")

# %%
academic_org_terms = ["university", "college", "institute", "royal academy", "research"]
gov_org_terms = ["dept", "department", "council", "comittee", "authority", "agency"]


def term_indicators(df, terms, col_name, name_col="name"):
    df = df.copy()
    x = [False] * len(df)
    for term in terms:
        x = x | df[name_col].str.lower().str.contains(term)
    df[col_name] = x
    return df


# %%
df = term_indicators(project_orgs, academic_org_terms, "is_academic")
df["year"] = df.start.apply(iss.convert_date_to_year)

# %%
alt.Chart(df).mark_bar().encode(
    x="year:O",
    y=alt.Y("count(project_id)", stack="normalize"),
    color="is_academic",
    order=alt.Order("is_academic", sort="ascending"),
)

# %% [markdown]
# # Stats and narratives

# %%
# heat_pump_articles = {
#     'Heat pumps': [guardian.search_content(search_term, use_cached = False, adjusted_parameters = {'from-date': "2000-01-01"})

# %%
# a = guardian.search_content(, use_cached = False, adjusted_parameters = {'from-date': "2000-01-01"})
# aggregate_guardian_articles(category_articles, [cat])

# %%
# DF_MANUAL_REVIEW_1 = (
#     pd.read_csv('/Users/karliskanders/Downloads/project-7-at-2021-08-20-11-17-039f3681.csv')
#     .rename(columns={'category': 'tech_category'})
#     .query('sentiment=="Keep"')
# )


# %%
# CATEGORY_NAMES = [
#     'Micro CHP',
#     'Heat storage',
#     'Electric boilers',
#     'Heat pumps',
#     'Geothermal energy',
#     'Solar thermal',
#     'District heating',
#     'Building insulation',
#     'Energy management',
#     'Hydrogen heating',
#     'Biomass heating',
# ]

# GTR_DOCS_ALL = pd.DataFrame()
# CB_DOCS_ALL = pd.DataFrame()
# for cat in CATEGORY_NAMES:
#     df = get_verified_docs([cat])
#     # Extract GTR and CB into separate dataframes
#     gtr_docs = add_project_data(df[df.source=='gtr'])
#     cb_docs = add_crunchbase_data(df[df.source=='cb'])
#     GTR_DOCS_ALL = GTR_DOCS_ALL.append(gtr_docs, ignore_index=True)
#     CB_DOCS_ALL = CB_DOCS_ALL.append(cb_docs, ignore_index=True)

# %%
# CB_DOCS_ALL.groupby('tech_category').count()

# %%
# GTR_DOCS_ALL.groupby('tech_category').count()

# %% [markdown]
# ## Reference categories

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
DF_REF = iss_io.load_pickle(
    PROJECT_DIR / "outputs/data/results_august/reference_category_data_2.p"
)

# %%
cat = "Solar"
DF_REF[cat].head(1)

# %%
cat = "Hydrogen & Fuel Cells"
DF_REF[cat].head(1)

# %%
# DF_REF["Hydrogen & Fuel Cells"] =
cat = "Hydrogen & Fuel Cells"
DF_REF[cat] = DF_REF[cat][DF_REF[cat].cluster_keywords.str.contains("stars") == False]
DF_REF[cat] = DF_REF[cat][
    DF_REF[cat].cluster_keywords.str.contains("standard_model") == False
]


# %%
def aggregate_guardian_articles_2(category_articles, CATEGORY_NAMES):
    articles = []
    for key in CATEGORY_NAMES:
        articles += category_articles[key]
    aggregated_articles = [article for sublist in articles for article in sublist]
    return aggregated_articles


def aggregate_hansard_speeches_2(s_list):
    speeches = get_docs_with_keyphrases(s_list, corpus_texts=hans_docs, corpus_df=hans)
    return speeches


# %%
# gtr_docs = add_project_data(df[df.source=='gtr'])
# cb_docs = add_crunchbase_data(df[df.source=='cb'])
# guardian_articles = aggregate_guardian_articles(category_articles, [cat])
# speeches = aggregate_hansard_speeches([cat])
# # guardian_articles_ = [g for g in guardian_articles if g['sectionName'] not in ['World news', 'Australia news']]
# df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)

# %%
narrow_ref_keywords = {
    "Batteries": [["battery"], ["batteries"]],
    "Solar": [
        ["solar energy"],
        ["solar", "renewable"],
        ["solar cell"],
        ["solar panel"],
        ["photovoltaic"],
        ["perovskite"],
        ["pv cell"],
    ],
    "Carbon Capture & Storage": [
        [" ccs "],
        ["carbon storage"],
        ["carbon capture"],
        ["co2 capture"],
        ["carbon abatement"],
    ],
    "Bioenergy": [
        ["biomass", "energy"],
        ["bioenergy"],
        ["biofuel"],
        ["syngas"],
        ["biogas"],
        ["biochar"],
        ["biodiesel"],
        ["torrefaction"],
        ["pyrolisis"],
        ["ethanol", "fuel"],
        ["butanol", "fuel"],
    ],
    "Nuclear": [
        ["nuclear", "energy"],
        ["radioactive", "waste"],
        ["nuclear", "fission"],
        ["fission", "reactor"],
        ["nuclear", "fusion"],
        ["fusion", "reactor"],
        ["tokamak"],
    ],
    "Hydrogen & Fuel Cells": [[" hydrogen "], ["fuel cell"], ["sofc"]],
    "Wind & Offshore": [
        ["wind energy"],
        ["wind", "renewable"],
        ["wind turbine"],
        ["wind farm"],
        ["wind generator"],
        ["wind power"],
        ["wave", "renewable"],
        ["tidal energy"],
        ["tidal turbine"],
        ["offshore energy"],
        ["offshore wind"],
        ["onshore wind"],
    ],
}

# %%
REF_TERMS = {
    "Solar": [
        "solar cell",
        "solar cells",
        "solar panel",
        "solar panels",
        "photovoltaic",
        "perovskite",
        "pv cell",
    ],
    "Wind & Offshore": [
        "wind energy",
        "renewable wind",
        "wind farm",
        "wind farms",
        "wind turbine",
        "wind turbines",
        "tidal energy",
        "offshore wind",
        "onshore wind",
    ],
    "Hydrogen & Fuel Cells": [
        "hydrogen energy",
        "fuel cell",
        "fuel cells",
        "hydrogen production",
        "solid oxide fuel cell",
    ],
    "Batteries": ["battery", "batteries"],
    "Carbon Capture & Storage": [
        "carbon capture and storage",
        "carbon capture",
        "carbon capture & storage",
        "co2 capture",
    ],
    "Bioenergy": [
        "biomass energy",
        "bioenergy",
        "biofuel",
        "syngas",
        "biogas",
        "biochar",
        "biodiesel",
        "torrefaction",
        "pyrolisis",
    ]
    #         ['wind', 'renewable'],
    #         ['wind turbine'],
    #         ['wind farm'],
    #         ['wind generator'],
    #         ['wind power'],
    #         ['wave', 'renewable'],
    #         ['tidal energy'],
    #         ['tidal turbine'],
    #         ['offshore energy'],
    #         ['offshore wind'],
    #         ['onshore wind']
}

# %% [markdown]
# ## Get reference data

# %%
GTR_DOCS_REF_ALL = pd.DataFrame()
CB_DOCS_REF_ALL = pd.DataFrame()

THRESH_TOPIC_PROB = 0.1

# %%
# cat='Batteries'
# DF_REF[cat] = DF_REF[cat].rename(columns={'category': 'tech_category'})
# DF_REF[cat]['tech_category']=cat
# gtr_docs = add_project_data(DF_REF[cat][DF_REF[cat].source=='gtr'])
# gtr_docs = dedup_gtr_docs(gtr_docs)
# gtr_docs=gtr_docs[(gtr_docs.manual_ok==1) | (gtr_docs.topic_probs>THRESH_TOPIC_PROB)]
# gtr_docs = get_reliable_funds(gtr_docs, FUNDS_RELIABLE)

# %%
for cat in [
    "Solar",
    "Wind & Offshore",
    "Hydrogen & Fuel Cells",
    "Batteries",
    "Carbon Capture & Storage",
    "Bioenergy",
]:
    # cat = 'Wind & Offshore'
    # cat='Solar'
    # cat='Batteries'
    # cat='Hydrogen & Fuel Cells'
    DF_REF[cat] = DF_REF[cat].rename(columns={"category": "tech_category"})
    DF_REF[cat]["tech_category"] = cat
    gtr_docs = add_project_data(DF_REF[cat][DF_REF[cat].source == "gtr"])
    gtr_docs = dedup_gtr_docs(gtr_docs)
    gtr_docs = gtr_docs[
        (gtr_docs.manual_ok == 1) | (gtr_docs.topic_probs > THRESH_TOPIC_PROB)
    ]

    gtr_docs = get_reliable_funds(gtr_docs, FUNDS_RELIABLE)
    #     gtr_docs = process_gtr_docs(DF_REF[cat][DF_REF[cat].source=='gtr'])
    cb_docs_ = pd.concat(
        [
            DF_REF[cat][DF_REF[cat].source == "cb"],
            get_verified_docs_2(
                [cat],
                fpath=PROJECT_DIR
                / "outputs/data/results_august/ISS_technologies_to_review_August_27_reference_checked.xlsx",
            ),
        ]
    ).drop_duplicates("doc_id")
    cb_docs = add_crunchbase_data(cb_docs_)

    category_articles[cat] = [
        guardian.search_content(search_term, use_cached=False)
        for search_term in REF_TERMS[cat]
    ]
    guardian_articles = aggregate_guardian_articles(category_articles, [cat])
    guardian_articles_ = process_guardian_articles(guardian_articles)
    #     guardian_articles_ = [g for g in guardian_articles if g['sectionName'] not in ['World news', 'Australia news']]
    #     guardian_articles_ = [g for g in guardian_articles if g['sectionName'] in ['Environment']]
    #     dff = iss.articles_table(guardian_articles_, sort_by_date=False)
    #     is_dupl = dff.reset_index().duplicated('headline').to_list()
    #     guardian_articles_ = [g for i, g in enumerate(guardian_articles_) if is_dupl[i]==False]

    speeches = aggregate_hansard_speeches_2(narrow_ref_keywords[cat])

    df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles_, speeches)
    df_per_year_norm = normalise_timeseries(
        iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
        ref_year=REF_YEAR,
    )
    YEARLY_STATS[cat] = df_per_year
    YEARLY_STATS_NORM[cat] = df_per_year_norm

    GTR_DOCS_REF_ALL = GTR_DOCS_REF_ALL.append(gtr_docs, ignore_index=True)
    CB_DOCS_REF_ALL = CB_DOCS_REF_ALL.append(cb_docs, ignore_index=True)

# %%
print(THRESH_TOPIC_PROB)
for cat in [
    "Solar",
    "Wind & Offshore",
    "Hydrogen & Fuel Cells",
    "Batteries",
    "Carbon Capture & Storage",
    "Bioenergy",
]:
    print(cat)
    print(get_growth_and_level(cat, "no_of_projects", year_1=2016, year_2=2020))
    print(get_growth_and_level(cat, "amount_total", year_1=2016, year_2=2020))


# %%
print(THRESH_TOPIC_PROB)
for cat in [
    "Solar",
    "Wind & Offshore",
    "Hydrogen & Fuel Cells",
    "Batteries",
    "Carbon Capture & Storage",
    "Bioenergy",
]:
    print(cat)
    print(get_growth_and_level(cat, "no_of_projects", year_1=2016, year_2=2020))
    print(get_growth_and_level(cat, "amount_total", year_1=2016, year_2=2020))

# %%
print(THRESH_TOPIC_PROB)
for cat in [
    "Solar",
    "Wind & Offshore",
    "Hydrogen & Fuel Cells",
    "Batteries",
    "Carbon Capture & Storage",
    "Bioenergy",
]:
    print(cat)
    print(get_growth_and_level(cat, "no_of_projects", year_1=2016, year_2=2020))
    print(get_growth_and_level(cat, "amount_total", year_1=2016, year_2=2020))


# %%
# GTR_DOCS_REF_ALL[GTR_DOCS_REF_ALL.category=='NO_FUND_INFO']

# %%
CB_DOCS_ALL_ = pd.concat(
    [
        CB_DOCS_ALL,
        CB_DOCS_REF_ALL,
        CB_DOCS_ALL_HEAT,
        CB_DOCS_ALL_BUILDINGS,
        CB_DOCS_ALL_HEAT_BUILD,
        CB_DOCS_ALL_HEAT_OTHER,
    ]
)
GTR_DOCS_ALL_ = pd.concat(
    [
        GTR_DOCS_ALL,
        GTR_DOCS_REF_ALL,
        GTR_DOCS_ALL_HEAT,
        GTR_DOCS_ALL_BUILDINGS,
        GTR_DOCS_ALL_HEAT_BUILD,
        GTR_DOCS_ALL_HEAT_OTHER,
    ]
)

# %%
# gtr_docs_ = gtr_docs.sort_values('start').drop_duplicates('doc_id', keep='first').merge(gtr_docs.groupby('doc_id').agg(amount=('amount', 'sum')).reset_index())
# len(gtr_docs)

# %%
list(YEARLY_STATS.keys())

# %%
iss_io.save_pickle(
    YEARLY_STATS,
    PROJECT_DIR
    / "outputs/data/results_august/yearly_stats_all_categories_2021_Funds.csv",
)

# %%
# iss_io.save_pickle(YEARLY_STATS, PROJECT_DIR / 'outputs/data/results_august/yearly_stats_all_categories_2021_Funds.p')

# %%
cat = "Batteries"


def plot_tseries(cat, ma_window=3):
    df_ = YEARLY_STATS[cat]
    df = iss_topics.get_moving_average(df_, window=ma_window, rename_cols=False)
    ww = 350
    hh = 150

    base = alt.Chart(df, width=ww, height=hh).encode(
        alt.X("year:O", axis=alt.Axis(title=None, labels=False))
    )

    fig_projects = base.mark_line(color="#2F1847", size=2.5).encode(
        alt.Y(
            "no_of_projects",
            axis=alt.Axis(title="Number of projects", titleColor="#2F1847"),
        )
    )

    fig_money = base.mark_line(color="#C62E65", size=2.5).encode(
        alt.Y(
            "amount_total",
            axis=alt.Axis(title="Funding amount (£1000s)", titleColor="#C62E65"),
        )
    )
    fig_gtr = alt.layer(fig_projects, fig_money).resolve_scale(y="independent")

    fig_rounds = base.mark_line(color="#1D3354", size=2.5).encode(
        alt.Y(
            "no_of_rounds",
            axis=alt.Axis(title="Number of rounds", titleColor="#1D3354"),
        )
    )
    fig_amount_raised = base.mark_line(color="#4CB7BD", size=2.5).encode(
        alt.Y(
            "raised_amount_usd_total",
            axis=alt.Axis(title="Amount raised ($1000s)", titleColor="#4CB7BD"),
        )
    )
    fig_crunchbase = alt.layer(fig_rounds, fig_amount_raised).resolve_scale(
        y="independent"
    )

    fig_articles = (
        alt.Chart(df, width=ww, height=hh)
        .mark_line(color="#624763", size=2.5)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(title="Year")),
            y=alt.Y(
                "articles",
                axis=alt.Axis(title="Number of news articles", titleColor="#624763"),
            ),
        )
    )
    fig_speeches = (
        alt.Chart(df, width=ww, height=hh)
        .mark_line(color="#F9B3D1", size=2.5)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(title="Year")),
            y=alt.Y(
                "speeches",
                axis=alt.Axis(title="Parliament speeches", titleColor="#F9B3D1"),
            ),
        )
    )

    fig_discourse = alt.layer(fig_articles, fig_speeches).resolve_scale(y="independent")
    fig = iss.nicer_axis(alt.vconcat(fig_gtr, fig_crunchbase, fig_discourse))
    return fig


# %%
plot_tseries("Heat pumps", ma_window=3)

# %%
alt_save.save_altair(
    fig, f"reference_categories_{'_'.join(cat.lower().split())}_tseries", driver
)

# %%
importlib.reload(iss)

# %%
dff = iss.articles_table(guardian_articles_)
y = "2020"
dff[(dff.date >= y) & (dff.date < str(int(y) + 1))].sample(10).sort_values("date")

# %%
# guardian_articles[-8]

# %% [markdown]
# ## Check funds

# %%
add_project_data(DF_REF[cat][DF_REF[cat].source == "gtr"]).sort_values(
    "amount", ascending=False
).head(5)

# %%
link_gtr_funds = gtr.get_link_table("gtr_funds")

# %%
# gtr_funds.groupby('category').count()

# %%
gtr_funds.head(1)

# %%
link_gtr_funds.head(1)

# %%
# link_gtr_funds[link_gtr_funds.project_id=='A223E948-8642-41A4-B9EE-2B758D58CDB2']

# %%
proj_funds = gtr_funds[
    gtr_funds.id.isin(
        link_gtr_funds[
            link_gtr_funds.project_id == "A223E948-8642-41A4-B9EE-2B758D58CDB2"
        ].id.to_list()
    )
].sort_values("amount")
proj_funds


# %% [markdown]
# ## Get all funds and indicate reliable funds

# %%
def get_fund_from_api_response(r):
    if r.json()["totalSize"] != 0:
        return r.json()["fund"][0]["valuePounds"]["amount"]
    else:
        return 0


# %%
import requests
from time import sleep

# %%
all_selected_doc_ids = GTR_DOCS_ALL_.doc_id.unique()

# %%
GTR_DOCS_ALL_x = GTR_DOCS_ALL_[GTR_DOCS_ALL_.amount == 0]
all_selected_doc_ids_x = GTR_DOCS_ALL_x.doc_id.unique()

# %%
len(all_selected_doc_ids_x)

# %%
len(all_selected_doc_ids)

# %%
all_responses = []
for doc_id in tqdm(all_selected_doc_ids_x, total=len(all_selected_doc_ids_x)):
    r = requests.get(
        f"https://gtr.ukri.org/gtr/api/projects/{doc_id}/funds",
        headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
    )
    all_responses.append(r)
    sleep(0.21)

# %%
# len(all_responses)
# all_responses_first_batch = all_responses.copy()

# %%
# all_responses_total = all_responses_first_batch + all_responses

# %%
# iss_io.save_pickle(all_responses_total, PROJECT_DIR / 'outputs/data/results_august/gtr_funding_api_responses.p')

# %%
# iss_io.save_pickle(all_responses, PROJECT_DIR / 'outputs/data/results_august/gtr_funding_api_responses_additional.p')

# %%
df = pd.read_csv(PROJECT_DIR / "outputs/data/results_august/RELIABLE_GTR_FUNDS.csv")
all_selected_doc_ids = df.doc_id.to_list() + list(all_selected_doc_ids_x)
n_first_iter = len(df.doc_id.to_list())

# %%
all_responses_total = iss_io.load_pickle(
    PROJECT_DIR / "outputs/data/results_august/gtr_funding_api_responses.p"
)

# %%
all_responses_total += iss_io.load_pickle(
    PROJECT_DIR / "outputs/data/results_august/gtr_funding_api_responses_additional.p"
)


# %%
all_responses_total[0].json()["fund"][0]["valuePounds"]["amount"]

# %%
len(all_responses_total)

# %%
n_funds = [r.json()["totalSize"] for r in all_responses_total]

# %%
np.unique(n_funds)

# %%
all_funds = [get_fund_from_api_response(r) for r in all_responses_total]

# %%
df_all_funds = pd.DataFrame(
    data={"doc_id": all_selected_doc_ids, "api_funds": all_funds}
)
# df_all_funds=df_all_funds.merge(GTR_DOCS_ALL_[['doc_id', 'amount']].drop_duplicates('doc_id'), how='left')

# %%
def get_all_fund_items(project_id):
    funds_ids = link_gtr_funds[link_gtr_funds.project_id == project_id].id.to_list()
    gtr_project_to_fund = gtr_funds[gtr_funds.id.isin(funds_ids)].query(
        "category=='INCOME_ACTUAL'"
    )
    # Columns to use in downstream analysis
    #     fund_cols = ["project_id", "id", "rel", "category", "amount", "currencyCode"]
    return gtr_project_to_fund


def check_if_api_fund_present(project_id, api_fund):
    df = get_all_fund_items(project_id)
    if len(df[df.amount == api_fund]) != 0:
        return True
    else:
        return False


# %%
check_if_api_fund_present("933A2702-FBC3-4FBA-91FC-6AD2046502A6", 80737)

# %%
is_fund_reliable = []
for i, row in tqdm(df_all_funds.iterrows(), total=len(df_all_funds)):
    is_fund_reliable.append(check_if_api_fund_present(row.doc_id, row.api_funds))

# %%
# df_all_funds.merge(GTR_DOCS_ALL_[['doc_id', 'amount']].drop_duplicates('doc_id', how='left')
df_all_funds["is_api_fund_reliable"] = is_fund_reliable
df_all_funds["amount"] = df_all_funds["api_funds"]

# %%
# df_all_funds.to_csv(PROJECT_DIR / 'outputs/data/results_august/RELIABLE_GTR_FUNDS_2.csv', index=False)

# %%
## Additional funds

# %%
df_all_funds_x = df_all_funds.iloc[n_first_iter:]

# %%
is_fund_reliable = []
for i, row in tqdm(df_all_funds_x.iterrows(), total=len(df_all_funds_x)):
    is_fund_reliable.append(check_if_api_fund_present(row.doc_id, row.api_funds))

# %%
# df_all_funds.merge(GTR_DOCS_ALL_[['doc_id', 'amount']].drop_duplicates('doc_id', how='left')
df_all_funds_x["is_api_fund_reliable"] = is_fund_reliable
df_all_funds_x["amount"] = df_all_funds["api_funds"]

# %%
# df = pd.read_csv(PROJECT_DIR / 'outputs/data/results_august/RELIABLE_GTR_FUNDS.csv')
# df = pd.concat([df, df_all_funds_x], ignore_index=True).drop_duplicates('doc_id', keep='first')
# df.to_csv(PROJECT_DIR / 'outputs/data/results_august/RELIABLE_GTR_FUNDS_2.csv', index=False)

# %% [markdown]
# #### Another funding session

# %%
df_base = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/RELIABLE_GTR_FUNDS_2.csv"
)

# %%
df_new = GTR_DOCS_ALL_[GTR_DOCS_ALL_.category == "NO_FUND_INFO"].drop_duplicates(
    "doc_id"
)
df_new_doc = df_new.doc_id.to_list()

# %%
all_responses = []
for doc_id in tqdm(df_new_doc, total=len(df_new_doc)):
    r = requests.get(
        f"https://gtr.ukri.org/gtr/api/projects/{doc_id}/funds",
        headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
    )
    all_responses.append(r)
    sleep(0.21)

# %%
all_funds = [get_fund_from_api_response(r) for r in all_responses]
df_all_funds = pd.DataFrame(data={"doc_id": df_new_doc, "api_funds": all_funds})
is_fund_reliable = []
for i, row in tqdm(df_all_funds.iterrows(), total=len(df_all_funds)):
    is_fund_reliable.append(check_if_api_fund_present(row.doc_id, row.api_funds))
# df_all_funds.merge(GTR_DOCS_ALL_[['doc_id', 'amount']].drop_duplicates('doc_id', how='left')
df_all_funds["is_api_fund_reliable"] = is_fund_reliable
df_all_funds["amount"] = df_all_funds["api_funds"]

# %%
# df = pd.concat([df_base, df_all_funds], ignore_index=True).drop_duplicates('doc_id', keep='first')
# df.to_csv(PROJECT_DIR / 'outputs/data/results_august/RELIABLE_GTR_FUNDS_3.csv', index=False)

# %%
# len(df_base)

# %%
# df_all_funds.is_api_fund_reliable

# %% [markdown]
# ### Inspect specific cases

# %%
# df_all_funds_ = df_all_funds.merge(GTR_DOCS_ALL_[['doc_id', 'amount']].drop_duplicates('doc_id', how='left')

# %%
# xdf = df_all_funds[df_all_funds.doc_id.isin(GTR_DOCS_ALL.doc_id.to_list())]

# %%
len(df_all_funds[df_all_funds.difference == 0])

# %%
df_all_funds.api_funds

# %%
len(df_all_funds)

# %%
GTR_DOCS_ALL_[GTR_DOCS_ALL_.doc_id == "FAD8F0C3-4698-4162-A2F9-81A540E4FF49"].iloc[
    0
].title

# %%

# %%
r = requests.get(
    f"https://gtr.ukri.org/gtr/api/projects/FAD8F0C3-4698-4162-A2F9-81A540E4FF49	/funds",
    headers={"Accept": "application/vnd.rcuk.gtr.json-v6"},
).json()
r

# %% [markdown]
# ## Rename categories

# %%
category_names_new = {
    "Building insulation": "Insulation & retrofit",
    "Building Energy Efficiency": "Energy efficiency & management",
    "Heating (all)": "Low carbon heating",
    "Hydrogen & Fuel Cells": "Hydrogen & fuel cells",
    "Wind & Offshore": "Wind & offshore",
    "Carbon Capture & Storage": "Carbon capture & storage",
    "Heating & Building Energy Efficiency": "LCH & EEM",
    "Energy efficiency & management": "EEM",
}


# %%
YEARLY_STATS_backup = YEARLY_STATS.copy()
GTR_DOCS_ALL_backup = GTR_DOCS_ALL_.copy()
CB_DOCS_ALL_backup = CB_DOCS_ALL_.copy()

# %%
for key in list(YEARLY_STATS.keys()):
    if key in category_names_new:
        YEARLY_STATS[category_names_new[key]] = YEARLY_STATS[key]


# %%
def change_name(x):
    if x in category_names_new:
        return category_names_new[x]
    else:
        return x


# %%
GTR_DOCS_ALL_.tech_category = GTR_DOCS_ALL_.tech_category.apply(change_name)
CB_DOCS_ALL_.tech_category = CB_DOCS_ALL_.tech_category.apply(change_name)

# %%
GTR_DOCS_ALL_.tech_category.unique()

# %%
iss_io.save_pickle(
    YEARLY_STATS,
    PROJECT_DIR
    / "outputs/data/results_august/FINAL_TABLES_yearly_stats_all_categories_2021_Funds.p",
)

# %%
GTR_DOCS_ALL_.to_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_GTR.csv", index=False
)
CB_DOCS_ALL_.to_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_CB.csv", index=False
)

# %% [markdown]
# # Trajectories

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
YEARLY_STATS = iss_io.load_pickle(
    PROJECT_DIR
    / "outputs/data/results_august/FINAL_TABLES_yearly_stats_all_categories_2021_Funds.p"
)
GTR_DOCS_ALL_ = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_GTR.csv"
)
CB_DOCS_ALL_ = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_CB.csv"
)


# %%
GTR_DOCS_REF_ALL.tech_category.unique()

# %%
# CATEGORY_NAMES_ = [
#     'Heat pumps', 'Geothermal energy', 'Solar thermal', 'District heating',
#     'Building insulation', 'Energy management',
#     'Hydrogen heating', 'Biomass heating',
#     'Solar', 'Wind & offshore', 'Batteries', 'Hydrogen & fuel cells'
# ]
# CATEGORY_NAMES_ = list(YEARLY_STATS.keys())
# y=2016
# iss.nicer_axis(plot_matrix(
#     variable='no_of_projects',
#     category_names=CATEGORY_NAMES_,
#     x_label='Avg number of projects per year',
#     year_1=y,
#     year_2=y+4
# ))

# %%
list(YEARLY_STATS.keys())


# %%
def get_year_by_year_stats(variable):
    df_stats_all = pd.DataFrame()
    for cat in list(YEARLY_STATS.keys()):
        for y in range(2011, 2021):
            yy = 4
            #         if y==2009: yy=1;
            #         if y==2010: yy=2;
            #         if y==2011: yy=2;
            #         if y>2010: yy=4;
            df_stats = pd.DataFrame(
                [
                    get_growth_and_level(cat, variable, y - yy, y) + tuple([cat, y])
                    for cat in [cat]
                ],
                columns=["growth", variable, "tech_category", "year"],
            )
            df_stats_all = df_stats_all.append(df_stats, ignore_index=True)
    #     df_stats_all.loc[df_stats_all.tech_category==cat, variable] = df_stats_all.loc[df_stats_all.tech_category==cat, variable] / df_stats_all[(df_stats_all.tech_category==cat)].iloc[0][variable]
    return df_stats_all


# %%
# df_stats.head(20)

# %%
# df_stats_all

# %% [markdown]
# ### Establish baseline

# %%
gtr_projects_ = (
    gtr_projects[["project_id", "title", "abstractText"]]
    .rename(columns={"project_id": "doc_id", "abstractText": "description"})
    .copy()
)
gtr_projects_["source"] = "gtr"
gtr_projects_ = process_gtr_docs(gtr_projects_)
gtr_projects_["year"] = gtr_projects_.start.apply(iss.convert_date_to_year)
gtr_total_projects = gtr_projects_.groupby("year").count().reset_index()

# %%
gtr_total_projects_amounts = gtr_projects_.groupby("year").sum().reset_index()

# %%
gtr_total_projects_amounts

# %%
gtr_total_projects_amounts.head(1)

# %%
get_growth_and_level_2(gtr_total_projects, "project_id", year_1=2016, year_2=2020)

# %%
get_growth_and_level_2(gtr_total_projects_amounts, "amount", year_1=2016, year_2=2020)

# %% [markdown]
# #### Define visuals

# %%
PLOT_LCH_CATEGORY_DOMAIN = [
    "Heat pumps",
    "Biomass heating",
    "Hydrogen heating",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Heat storage",
    "Insulation & retrofit",
    "Energy management",
]

PLOT_LCH_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    #     '#bab0ac',
]
COLOR_PAL_LCH = (PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS)

PLOT_REF_CATEGORY_DOMAIN = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

PLOT_REF_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    #     '#bab0ac',
]

COLOR_PAL_REF = (PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS)


# %%
def plot_matrix(
    variable,
    category_names,
    x_label,
    y_label="Growth",
    year_1=2016,
    year_2=2020,
    color_pal=(PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS),
):

    df_stats = pd.DataFrame(
        [
            get_growth_and_level(cat, variable, year_1, year_2) + tuple([cat])
            for cat in category_names
        ],
        columns=["growth", variable, "tech_category"],
    )

    points = (
        alt.Chart(
            df_stats,
            height=350,
            width=350,
        )
        .mark_circle(size=35)
        .encode(
            alt.Y("growth:Q", title=y_label),
            alt.X(f"{variable}:Q", title=x_label),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(domain=color_pal[0], range=color_pal[1]),
                #             color=alt.Color(
                #                 'tech_category',
                #                 scale=alt.Scale(domain=category_names,
                #                                 range=COLOUR_PAL[0:len(category_names)]),
                legend=None,
            ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=7, size=15).encode(
        text="tech_category"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig


def plot_matrix_trajectories(
    variable,
    category_names,
    x_label,
    y_label,
    ww=350,
    hh=350,
    color_pal=(PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS),
):
    points = (
        alt.Chart(
            df_stats_all[df_stats_all.tech_category.isin(cats)],
            height=hh,
            width=ww,
        )
        .mark_line(size=3)
        .encode(
            alt.Y("growth:Q", title=y_label),
            alt.X(f"{variable}:Q", title=x_label),
            order="year",
            #             color=alt.Color('tech_category'),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(domain=color_pal[0], range=color_pal[1]),
            ),
            #             color=alt.Color(
            #                 'tech_category',
            #                 scale=alt.Scale(domain=category_names, range=COLOUR_PAL[0:len(category_names)]),
            #                 legend=None
            #             ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=4, size=9).encode(
        text="year"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig


# %%
plot_tseries("District heating")


# %%
def plot_tseries(cat, ma_window=3):
    df_ = YEARLY_STATS[cat]
    df = iss_topics.get_moving_average(df_, window=ma_window, rename_cols=False)
    ww = 350
    hh = 150

    base = alt.Chart(df, width=ww, height=hh).encode(
        alt.X("year:O", axis=alt.Axis(title=None, labels=False))
    )

    fig_projects = base.mark_line(color="#2F1847", size=2.5).encode(
        alt.Y(
            "no_of_projects",
            axis=alt.Axis(title="Number of projects", titleColor="#2F1847"),
        )
    )

    fig_money = base.mark_line(color="#C62E65", size=2.5).encode(
        alt.Y(
            "amount_total",
            axis=alt.Axis(title="Funding amount (£1000s)", titleColor="#C62E65"),
        )
    )
    fig_gtr = alt.layer(fig_projects, fig_money).resolve_scale(y="independent")

    fig_rounds = base.mark_line(color="#1D3354", size=2.5).encode(
        alt.Y(
            "no_of_rounds",
            axis=alt.Axis(title="Number of rounds", titleColor="#1D3354"),
        )
    )
    fig_amount_raised = base.mark_line(color="#4CB7BD", size=2.5).encode(
        alt.Y(
            "raised_amount_usd_total",
            axis=alt.Axis(title="Amount raised ($1000s)", titleColor="#4CB7BD"),
        )
    )
    fig_crunchbase = alt.layer(fig_rounds, fig_amount_raised).resolve_scale(
        y="independent"
    )

    fig_articles = (
        alt.Chart(df, width=ww, height=hh)
        .mark_line(color="#624763", size=2.5)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(title="Year")),
            y=alt.Y(
                "articles",
                axis=alt.Axis(title="Number of news articles", titleColor="#624763"),
            ),
        )
    )
    fig_speeches = (
        alt.Chart(df, width=ww, height=hh)
        .mark_line(color="#F9B3D1", size=2.5)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(title="Year")),
            y=alt.Y(
                "speeches",
                axis=alt.Axis(title="Parliament speeches", titleColor="#F9B3D1"),
            ),
        )
    )

    fig_discourse = alt.layer(fig_articles, fig_speeches).resolve_scale(y="independent")
    fig = iss.nicer_axis(alt.vconcat(fig_gtr, fig_crunchbase, fig_discourse))
    return fig


# %%
# len(cb_df)

# %%
# cb_df[(cb_df.roles.isnull()==False) & cb_df.roles.str.contains('investor')]

# %%
# cb_df.columns

# %%
# dff = iss.get_cb_org_funding_rounds(cb_df, cb_funding_rounds).drop_duplicates('funding_round_id')
# # dff[dff.announced_on>'2000']
# dff

# %%
def plot_bars(variable, cat, y_label, bar_color="#2F1847"):
    df_ = YEARLY_STATS[cat]
    ww = 350
    hh = 150
    base = alt.Chart(df_, width=ww, height=hh).encode(
        alt.X(
            "year:O",
            #             axis=alt.Axis(title=None, labels=False)
        )
    )
    fig_projects = base.mark_bar(color=bar_color).encode(
        alt.Y(variable, axis=alt.Axis(title=y_label, titleColor=bar_color))
    )
    return iss.nicer_axis(fig_projects)


# %%
def plot_bars_2020(variable, cat, y_label, bar_color="#2F1847"):
    df_ = YEARLY_STATS[cat]
    df_ = df_[df_.year <= 2020]
    ww = 350
    hh = 150
    base = alt.Chart(df_, width=ww, height=hh).encode(
        alt.X(
            "year:O",
            #             axis=alt.Axis(title=None, labels=False)
        )
    )
    fig_projects = base.mark_bar(color=bar_color).encode(
        alt.Y(variable, axis=alt.Axis(title=y_label, titleColor=bar_color))
    )
    return iss.nicer_axis(fig_projects)


# %%
# GTR_DOCS_REF_ALL[GTR_DOCS_REF_ALL.tech_category=='Hydrogen & Fuel Cells']

# %%
# funded_projects_dedup = dedup_gtr_docs(funded_projects)

# %%
cat = "Hydrogen & fuel cells"
plot_bars_2020("no_of_projects", cat, "Number of projects").properties(title=cat)

# %% [markdown]
# ## LCH

# %%
list(YEARLY_STATS.keys())

# %%
YEARLY_STATS["Solar"].head(1)

# %%
variable = "no_of_projects"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
x_label = "Average number of projects per year"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
alt_save.save_altair(fig, f"_key_findings_Heating_{variable}", driver)

# %%
variable = "amount_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
x_label = "Average funding amount per year (£1000s)"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
alt_save.save_altair(fig, f"_key_findings_Heating_{variable}", driver)

# %% [markdown]
# ##### LCH: Trajectories

# %%
variable = "no_of_projects"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of projects per year"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
    ]
)
cats = sorted(
    [
        "Heat pumps",
        "Hydrogen heating",
        "Biomass heating",
        "Insulation & retrofit",
        "Energy management",
    ]
)
iss.nicer_axis(
    plot_matrix_trajectories(
        variable, cats, x_label, y_label, ww=400, hh=400
    ).interactive()
)

# %% [markdown]
# ## Reference categories

# %%
# variable = 'no_of_projects'
# # variable = 'no_of_rounds'
# # variable = 'articles'
# # variable = 'speeches'
# df_stats_all = get_year_by_year_stats(variable)
# y_label='Growth'
# x_label='Avg number of projects per year'
# cats = ['Batteries', 'Hydrogen & fuel cells',
#         'Carbon capture & storage', 'Bioenergy',
#         'Solar', 'LCH & EEM',
#         'Low carbon heating', 'Wind & offshore']
# fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
# fig

# %%
variable = "no_of_projects"
# variable = 'no_of_rounds'
# variable = 'articles'
# variable = 'speeches'
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of projects per year"
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
alt_save.save_altair(fig, f"_key_findings_Reference_{variable}", driver)

# %%
# variable = 'no_of_projects'
# variable = 'no_of_rounds'
# variable = 'articles'
# variable = 'speeches'
variable = "amount_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Average funding amount per year (£1000s)"
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
alt_save.save_altair(fig, f"_key_findings_Reference_{variable}", driver)

# %%
cat = "Hydrogen & fuel cells"
variable = "amount_total"
y_label = ""
fig = plot_bars(variable, cat, y_label).properties(title=cat)
fig

# %%
cat = "Insulation & retrofit"
variable = "amount_total"
# variable = 'no_of_projects'
y_label = ""
fig = plot_bars_2020(variable, cat, y_label).properties(title=cat)
fig

# %%
# iss.nicer_axis(plot_matrix_trajectories(variable, cats, x_label, y_label, ww=400, hh=400))

# %%
variable = "no_of_projects"
# variable = 'no_of_rounds'
# variable = 'articles'
# variable = 'speeches'
df_stats_all = get_year_by_year_stats(variable)
iss.nicer_axis(
    plot_matrix_trajectories(
        variable, cats, x_label, y_label, ww=400, hh=400, color_pal=COLOR_PAL_REF
    )
)

# %% [markdown]
# #### Barplots and details

# %%
cat = "Micro CHP"
variable = "no_of_projects"
y_label = "Number of projects"
fig = plot_bars(variable, cat, y_label).properties(title=cat)
fig


# %%
# variable = 'no_of_projects'
# y_label = 'Number of projects'
# for cat in YEARLY_STATS:
#     fig=plot_bars(variable, cat, y_label).properties(title=cat)
#     alt_save.save_altair(fig, f"user_meet_{variable}_{cat}.png", driver)

# %%
# variable = 'no_of_rounds'
# y_label = 'Number of rounds'
# for cat in YEARLY_STATS:
#     fig=plot_bars(variable, cat, y_label, bar_color='#0398fc').properties(title=cat)
#     alt_save.save_altair(fig, f"user_meet_{variable}_{cat}.png", driver)

# %%
def plot_bars_2(variable, cat, y_label, bar_color="#2F1847"):
    df_ = YEARLY_STATS[cat]
    ww = 350
    hh = 150
    base = alt.Chart(df_, width=ww, height=hh).encode(
        alt.X(
            "year:O",
            #             axis=alt.Axis(title=None, labels=False)
        )
    )
    fig_projects = base.mark_bar(color=bar_color).encode(
        alt.Y(variable, axis=alt.Axis(title=y_label, titleColor=bar_color))
    )
    return fig_projects


# %%
# for cat in YEARLY_STATS:
#     variable = 'no_of_projects'
#     y_label = 'Number of projects'
#     fig_1=plot_bars_2(variable, cat, y_label).properties(title=cat)
#     variable = 'no_of_rounds'
#     y_label = 'Number of rounds'
#     fig_2=plot_bars_2(variable, cat, y_label, bar_color='#0398fc').properties(title=cat)
#     fig=iss.nicer_axis(alt.vconcat(fig_1, fig_2))
#     alt_save.save_altair(fig, f"user_meet_projects_rounds_{cat}.png", driver)

# %%
# for cat in YEARLY_STATS:
#     variable = 'articles'
#     y_label = 'Number of articles'
#     fig_1=plot_bars_2(variable, cat, y_label).properties(title=cat)
#     variable = 'speeches'
#     y_label = 'Number of speeches'
#     fig_2=plot_bars_2(variable, cat, y_label, bar_color='#0398fc').properties(title=cat)
#     fig=iss.nicer_axis(alt.vconcat(fig_1, fig_2))
#     alt_save.save_altair(fig, f"user_meet_articles_speeches_{cat}.png", driver)

# %%
df = GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category == "Heat pumps"][
    ["title", "description", "start"]
]
df["year"] = df.start.apply(iss.convert_date_to_year)
df = df.drop("start", axis=1)

# %%
y = 2016
df[df.year.isin(list(range(y, y + 2)))].drop_duplicates("title").sort_values("title")

# %%
variable = "amount_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg yearly funding (£1000s)"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
alt_save.save_altair(fig, f"user_meet_Heating_funding.png", driver)

# %% [markdown]
# ### Project funding

# %%
variable = "amount_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg yearly funding (£1000s)"
cats = [
    "Batteries",
    "Hydrogen & Fuel Cells",
    "Carbon Capture & Storage",
    "Bioenergy",
    "Solar",
    "Heating & Building Energy Efficiency",
    "Heating (all)",
    "Wind & Offshore",
]
iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))

# %%
variable = "amount_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg yearly funding (£1000s)"
cats = [
    "Batteries",
    "Hydrogen & Fuel Cells",
    "Carbon Capture & Storage",
    "Bioenergy",
    "Solar",
    "Heating & Building Energy Efficiency",
    "Heating (all)",
    "Wind & Offshore",
]
iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))

# %%
# variable = 'no_of_projects'
# y_label='Growth'
# x_label='Avg number of projects per year'
# cats = ['Batteries', 'Hydrogen & Fuel Cells',
#         'Carbon Capture & Storage', 'Bioenergy',
#         'Solar', 'Heating & Building Energy Efficiency',
#         'Heating (all)', 'Wind & Offshore']
# plot_matrix(variable, cats, x_label, y_label)

# %%
# GTR_DOCS_ALL[(GTR_DOCS_ALL.tech_category=='Building insulation')
#              & (GTR_DOCS_ALL.start>='2012')
#              & (GTR_DOCS_ALL.start<'2013')
#             ]

# %%
# GTR_DOCS_REF_ALL[GTR_DOCS_REF_ALL.]

# %%
# funded_projects[funded_projects.title=='Automated Construction Project - 105 Sumner Street']

# %%
# GTR_DOCS_ALL_HEAT_BUILD.sort_values('amount', ascending=False)

# %%
# iss_io.save_pickle(YEARLY_STATS, PROJECT_DIR / 'outputs/data/results_august/categories.p')

# %%
# YEARLY_STATS['Batteries']

# %%
# variable = 'articles'
# df_stats_all = get_year_by_year_stats(variable)
# y_label='Growth'
# x_label='Avg number of projects per year'
# cats = ['Batteries', 'Hydrogen & Fuel Cells',
#         'Carbon Capture & Storage', 'Bioenergy',
#         'Solar', 'Heating & Building Energy Efficiency',
#        'Heating (all)', 'Wind & Offshore']
# plot_matrix(variable, cats, x_label, y_label)

# %%
cat = "EEM"
plot_bars("amount_total", cat, "Number of projects").properties(title=cat)

# %%
funded_projects[funded_projects.title == "Pozibot"]

# %%
# GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category=='Heating (all)'].sort_values('amount').tail(10)

# %%
df = GTR_DOCS_ALL_.copy()
df["year"] = GTR_DOCS_ALL_["start"].apply(iss.convert_date_to_year)
df["amount"] = df["amount"] / 1000
df = df[df.tech_category == "Insulation & retrofit"]

# %%
iss.show_time_series_points(df, y="amount", ymax=10000)

# %%
# df.tail(15)

# %% [markdown]
# ## Private investment

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & Offshore",
    "EEM",
    "Heating (other)",
]

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]
variable = "no_of_rounds"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of deals per year"
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
variable = "raised_amount_gbp_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg amount invested per year ($1000)"
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
cat = "Low carbon heating"
print(get_growth_and_level(cat, "raised_amount_usd_total", year_1=2016, year_2=2020))
print(get_growth_and_level(cat, "no_of_rounds", year_1=2016, year_2=2020))

# %%
cat = "EEM"
print(get_growth_and_level(cat, "raised_amount_usd_total", year_1=2016, year_2=2020))
print(get_growth_and_level(cat, "no_of_rounds", year_1=2016, year_2=2020))

# %%
cat = "Hydrogen & fuel cells"
print(get_growth_and_level(cat, "raised_amount_usd_total", year_1=2016, year_2=2020))
print(get_growth_and_level(cat, "no_of_rounds", year_1=2016, year_2=2020))

# %%
cat = "Carbon capture & storage"
print(get_growth_and_level(cat, "raised_amount_usd_total", year_1=2016, year_2=2020))
print(get_growth_and_level(cat, "no_of_rounds", year_1=2016, year_2=2020))

# %%
cat = "Heating (other)"
print(get_growth_and_level(cat, "raised_amount_usd_total", year_1=2016, year_2=2020))
print(get_growth_and_level(cat, "no_of_rounds", year_1=2016, year_2=2020))

# %%
cat = "Low carbon heating"
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)

# %%
# dff[dff.announced_on>='2016'].sort_values('raised_amount')

# %%
dff[dff.announced_on >= "2016"]["raised_amount_usd"].median()

# %%
# CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category=='Heat pumps']

# %%
cat = "EEM"
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[dff.announced_on >= "2016"]["raised_amount_usd"].median()

# %%

# %%
# dff[dff.announced_on>='2016'].sort_values('announced_on').tail(50)

# %%
cat = "LCH & EEM"
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)

# %%
dff.raised_amount.isnull().sum() / len(dff)

# %%
cat = "Wind & offshore"
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[dff.announced_on >= "2016"]["raised_amount_usd"].median()
# dff[dff.announced_on>='2016'].sort_values('raised_amount')

# %%
cb_selected_cat = CB_DOCS_ALL_.drop_duplicates("doc_id").copy()
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff["year"] = dff["announced_on"].apply(iss.convert_date_to_year)
dff.groupby("year").sum()

# %%
# dff[dff.announced_on>='2016'].sort_values('raised_amount')

# %% [markdown]
# ### Barplots

# %%
cat = "EEM"
variable = "no_of_rounds"
# variable = 'raised_amount_usd_total'
y_label = ""
fig = plot_bars(variable, cat, y_label).properties(title=cat)
fig

# %%
# variable = 'no_of_rounds'
variable = "raised_amount_usd_total"
y_label = ""
fig = plot_bars(variable, cat, y_label).properties(title=cat)
fig

# %%
variable = "raised_amount_usd_total"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
# x_label = "Avg number of deals per year"
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
        "Micro CHP",
    ]
)

# %%
variable = "no_of_rounds"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of deals per year"
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
df_all_yearly_stats = pd.DataFrame()
for key in YEARLY_STATS:
    df_ = YEARLY_STATS[key].copy()
    df_["tech_category"] = key
    df_all_yearly_stats = df_all_yearly_stats.append(df_, ignore_index=True)

df_all_yearly_stats_norm = pd.DataFrame()
for key in YEARLY_STATS_NORM:
    df_ = YEARLY_STATS_NORM[key].copy()
    df_["tech_category"] = key
    df_all_yearly_stats_norm = df_all_yearly_stats_norm.append(df_, ignore_index=True)

# %%
# GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category=='Heating (all)'].duplicated('doc_id').sum()

# %%
# iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False)

# %%
df_all_yearly_stats.head(1)

# %%
fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(no_of_rounds)", title="Number of deals"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
        "Micro CHP",
    ]
)

fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(raised_amount_usd_total)", title="Total amount raised ($1000s)"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(no_of_rounds)", title="Number of deals"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%

# %%
# cats = ['Batteries', 'Hydrogen & Fuel Cells',
#         'Carbon Capture & Storage', 'Bioenergy',
#         'Solar', 'Heating (all)', 'Building Energy Efficiency', 'Wind & Offshore', 'Heating (other)']
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]

fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(raised_amount_usd_total)", title="Total amount raised ($1000s)"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
REF_ORDER = [
    "Batteries",
    "Bioenergy",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Low carbon heating",
    "EEM",
    "Solar",
    "Wind & offshore",
]

# %%
PLOT_REF_CATEGORY_DOMAIN_2 = [
    "Batteries",
    "Bioenergy",
    "Carbon capture & storage",
    "EEM",
    "Hydrogen & fuel cells",
    "Low carbon heating",
    "Solar",
    "Wind & offshore",
    "Heating (other)"
    #     'LCH & EEM',
]

PLOT_REF_CATEGORY_COLOURS_2 = [
    "#4c78a8",
    "#72b7b2",
    "#e45756",
    "#ff9da6",
    "#f58518",
    "#b279a2",
    "#54a24b",
    "#9d755d",
    "#eeca3b",
    #      '#bab0ac',
]

COLOR_PAL_REF_2 = (PLOT_REF_CATEGORY_DOMAIN_2, PLOT_REF_CATEGORY_COLOURS_2)

# %%

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "EEM",
    "Wind & offshore",
]
fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "year:O", title="Year", scale=alt.Scale(domain=list(range(2006, 2021)))
        ),
        y=alt.Y("sum(amount_total)", title="Research funding (£1000s)"),
        color=alt.Color(
            "tech_category",
            legend=alt.Legend(title="Technology"),
            scale=alt.Scale(domain=COLOR_PAL_REF_2[0], range=COLOR_PAL_REF_2[1]),
            sort=COLOR_PAL_REF_2[0],
        ),
    )
)
iss.nicer_axis(fig)

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Heating (other)",
    "Low carbon heating",
    "EEM",
    "Wind & offshore",
]
fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "year:O", title="Year", scale=alt.Scale(domain=list(range(2007, 2021)))
        ),
        y=alt.Y("sum(raised_amount_usd_total)", title="Total amount raised ($1000s)"),
        color=alt.Color(
            "tech_category",
            legend=alt.Legend(title="Technology"),
            scale=alt.Scale(domain=COLOR_PAL_REF_2[0], range=COLOR_PAL_REF_2[1]),
            sort=COLOR_PAL_REF_2[0],
        ),
    )
)
iss.nicer_axis(fig)

# %%
dff = GTR_DOCS_ALL_.drop_duplicates("doc_id").copy()
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
# dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff["year"] = dff["start"].apply(iss.convert_date_to_year)
dff.groupby("year").sum()

# %%
fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_line(size=3.0)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(no_of_projects)", title="Number of projects"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
    ]
)

fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_line(size=3.0)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("articles", title="Number of projects"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
cats = [
    "Batteries",
    "Hydrogen & Fuel Cells",
    "Carbon Capture & Storage",
    "Bioenergy",
    "Solar",
    "Building Energy Efficiency",
    "Wind & Offshore",
    "Heating (other)",
    "Heating (all)",
]

fig = (
    alt.Chart(
        df_all_yearly_stats_norm[df_all_yearly_stats_norm.tech_category.isin(cats)],
        width=500,
    )
    .mark_line(size=3.0)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("articles", title="Number of projects"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %% [markdown]
# ### Check funding rounds

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]

# %%
cat = "Low carbon heating"
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category==cat]
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
# dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
# dff

# %%
df_counts = (
    cb_selected_cat.groupby("tech_category")
    .agg(no_of_companies=("doc_id", "count"))
    .sort_values("no_of_companies")
    .reset_index()
)

# %%
no_of_deals = []
no_of_companies_with_deals = []
for cat in cats:
    cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
    dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
    dff = dff[dff.announced_on >= "2010"]
    no_of_companies_with_deals.append(len(dff.name.unique()))
    no_of_deals.append(len(dff))

# %%
df = pd.DataFrame(
    data={
        "tech_category": cats,
        "no_of_deals": no_of_deals,
        "no_of_companies_with_deals": no_of_companies_with_deals,
    }
).merge(df_counts)
# df['ratio'] = df['no_of_companies_with_deals'] / df['no_of_companies']
# df['ratio'] = df['no_of_deals'] / df['no_of_companies']
df

# %%
alt.Chart(df.sort_values("no_of_companies")).mark_bar().encode(
    y=alt.Y("tech_category", sort="-x"), x="no_of_companies", tooltip=["tech_category"]
)

# %%
alt.Chart(df.sort_values("no_of_companies")).mark_bar().encode(
    y=alt.Y("tech_category", sort="-x"), x="no_of_deals", tooltip=["tech_category"]
)

# %%
alt.Chart(df.sort_values("no_of_companies_with_deals")).mark_bar().encode(
    y=alt.Y("tech_category", sort="-x"), x="no_of_deals", tooltip=["tech_category"]
)

# %%
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]


fig = (
    alt.Chart(cb_selected_cat, width=500)
    .mark_bar()
    .encode(
        y=alt.X("tech_category", title="fraction of companies"),
        x=alt.Y("count(doc_id)", stack="normalize"),
        color="employee_count",
    )
)
iss.nicer_axis(fig)

# %%
dff = iss.get_cb_org_funding_rounds(
    CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == "Building insulation"], cb_funding_rounds
)
dff[dff.name == "Q-Bot"][
    ["name", "announced_on", "investment_type", "raised_amount_usd"]
]

# %%
# fig=(
#     alt.Chart(
#         CB_DOCS_ALL_.groupby('tech_category').count()
#         df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)],
#         width=500
#     )
#     .mark_bar()
#     .encode(
#     x=alt.X('year:O', title='Year'),
#     y=alt.Y('sum(amount_total)', title='Research funding (£1000s)'),
#     color='tech_category'
# )
# )
# iss.nicer_axis(fig)

# %%
CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == ""]

# %%
GTR_DOCS_ALL[GTR_DOCS_ALL.title.str.contains("Q-Bot")]

# %%
# iss.get_cb_org_funding_rounds(CB_DOCS_ALL_.drop_duplicates('doc_id'), cb_funding_rounds).groupby('investment_type').count()

# %%
CB_DOCS_ALL_.groupby("tech_category").count()

# %%
cat = "Heat pumps"
# cat = 'Hydrogen heating'
# cat = 'Insulation & retrofit'
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[dff.announced_on >= "2016"]["raised_amount_usd"].median()


# %%
dff[dff.announced_on >= "2016"].sort_values("raised_amount")

# %%
cat = "Low carbon heating"
# cat = 'Hydrogen heating'
# cat = 'Insulation & retrofit'
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
# cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
# dff[dff.announced_on>='2016']['raised_amount_usd'].median()
dff[dff.announced_on >= "2016"].groupby("investment_type").count().sort_values("org_id")

# %%
cat = "Insulation & retrofit"
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[
    dff.announced_on >= "2016"
]  # .groupby('investment_type').count().sort_values('org_id')

# %% [markdown]
# #### Export example businesses

# %%
GTR_DOCS_ALL_.tech_category.unique()

# %%
# GTR_DOCS_ALL_HEAT_OTHER

# %%
CB_DOCS_ALL_[
    CB_DOCS_ALL_.tech_category.isin(
        [
            "Heat pumps",
            "Biomass heating",
            "Hydrogen heating",
            "Geothermal energy",
            "Solar thermal",
            "District heating",
            "Heat storage",
            "Micro CHP",
            "Heating (other)",
            "Insulation & retrofit",
            "Energy management",
        ]
    )
].sort_values("tech_category")[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "tech_category",
        "city",
        "country",
        "employee_count",
        "founded_on",
        "total_funding",
        "total_funding_currency_code",
        "num_funding_rounds",
        "last_funding_on",
        "homepage_url",
        "cb_url",
    ]
].to_csv(
    PROJECT_DIR / "outputs/data/results_august/LCH_EEM_companies.csv", index=False
)

# %%
GTR_DOCS_ALL_[
    GTR_DOCS_ALL_.tech_category.isin(
        [
            "Heat pumps",
            "Biomass heating",
            "Hydrogen heating",
            "Geothermal energy",
            "Solar thermal",
            "District heating",
            "Heat storage",
            "Micro CHP",
            "Insulation & retrofit",
            "Energy management",
        ]
    )
].sort_values("tech_category")[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "tech_category",
        "grantCategory",
        "leadFunder",
        "start",
        "amount",
    ]
].to_csv(
    PROJECT_DIR / "outputs/data/results_august/LCH_EEM_projects.csv", index=False
)

# %%

# %% [markdown]
# # Discourse signals and contexts in greater detail
#
# - Extract speeches for a particular topic
# - Extract articles for a particular topic

# %% [markdown]
# ### Number of articles

# %%
# sentences = iss.get_guardian_sentences_with_term('batter', guardian_articles_, field="body")
# sentiment_df = iss.get_sentence_sentiment(sentences)
# for i, row in (
#     sentiment_df.sort_values("compound", ascending=True).iloc[0:10].iterrows()
# ):
#     print(np.round(row.compound, 2), row.sentences, end="\n\n")

# %%
variable = "articles"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of articles per year"
# x_label='Average number of projects per year'
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label, year_1=2016))
fig

# %%
# Baseline growth for articles

# %%

# %%
plot_bars("articles", "District heating", "articles")

# %%
variable = "articles"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of articles per year"
iss.nicer_axis(
    plot_matrix_trajectories(
        variable, cats, x_label, y_label, ww=400, hh=400
    ).interactive()
)

# %%
variable = "articles"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of articles per year"
# x_label='Average number of projects per year'
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label, year_1=2016))
fig

# %%

# %% [markdown]
# ## Parliament speeches

# %%
variable = "speeches"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of speeches per year"
x_label = "Average number of projects per year"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label, year_1=2018))
fig

# %%
plot_bars("speeches", "District heating", "")

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & Offshore",
    "EEM",
    "Heating (other)",
]
variable = "speeches"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of speeches per year"
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
plot_bars("speeches", "Solar", "")

# %% [markdown]
# ### Reference categories

# %%
# Articles
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]
variable = "articles"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of articles per year"
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
# Speeches
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]
variable = "speeches"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of speeches per year"
fig = iss.nicer_axis(
    plot_matrix(variable, cats, x_label, y_label, color_pal=COLOR_PAL_REF)
)
fig

# %%
plot_bars("articles", "Wind & offshore", "")

# %%
cats = [
    "Heat pumps",
    "Hydrogen heating",
    "District heating",
    "Energy management",
    "Building insulation",
    "Hydrogen & Fuel Cells",
]
# cats = ['District heating']
# cat = cats[1]

# %%
dfs = {}

# %%
for cat in cats:
    if cat in categories_keyphrases_hans:
        speeches = aggregate_hansard_speeches([cat])
        keyphrases = categories_keyphrases_hans[cat]
    else:
        speeches = aggregate_hansard_speeches_2(narrow_ref_keywords[cat])
        keyphrases = narrow_ref_keywords[cat]

    hans_term_docs = iss.create_documents_from_dataframe(
        speeches, columns=["speech"], preprocessor=iss.preprocess_text
    )
    all_terms = np.unique([s for s_list in keyphrases for s in s_list])
    all_sents = [iss.get_doc_sentences_with_term(s, hans_term_docs) for s in all_terms]

    all_sents_merge = []
    for i in range(len(all_sents[0])):
        s = []
        for j in range(len(all_terms)):
            s += all_sents[j][i]
        all_sents_merge.append(list(np.unique(s)))

    speeches_ = speeches.copy()
    speeches_["sentences"] = all_sents_merge
    speeches_statistics = (
        speeches.groupby(["year", "major_heading", "minor_heading"])
        .agg(counts=("id", "count"))
        .reset_index()
    )

    dfs[cat] = speeches_
    dfs[f"{cat} - stats"] = speeches_statistics

# %%
for cat in ["Heat pumps", "Hydrogen heating", "Hydrogen & Fuel Cells"]:
    dfs[cat].to_csv(
        PROJECT_DIR
        / f'outputs/data/results_august/Hansard_data_{"_".join(cat.split())}.csv'
    )

# %%
dfs[cat]

# %%
with pd.ExcelWriter(
    PROJECT_DIR / "outputs/data/results_august/examples_Hansard_speeches.xlsx"
) as writer:
    for key in dfs:
        dfs[key].to_excel(writer, sheet_name=key)

# %%
## TODO: EXTRACT GUARDIAN ARTICLES IN THE SAME WAY
## TODO: SAVE SPEECHES AND GIVE TO JYL
dfs = {}
for cat in cats:
    guardian_articles = aggregate_guardian_articles(category_articles, [cat])
    guardian_articles_ = process_guardian_articles(guardian_articles)
    guardian_articles_table = iss.articles_table(guardian_articles_).sort_values("date")

    dfs[cat] = guardian_articles_table

# %%
guardian_articles_[0]

# %%
with pd.ExcelWriter(
    PROJECT_DIR / "outputs/data/results_august/examples_Guardian_articles.xlsx"
) as writer:
    for key in dfs:
        dfs[key].to_excel(writer, sheet_name=key)

# %%
importlib.reload(iss)

# %%
guardian_articles = aggregate_guardian_articles(category_articles, CATEGORY_NAMES)
guardian_articles = process_guardian_articles(guardian_articles)

# %%
guardian_articles = aggregate_guardian_articles(category_articles, [cat])
guardian_articles_ = process_guardian_articles(guardian_articles)

# %%
guardian_articles = aggregate_guardian_articles(category_articles, ["Heat pumps"])
guardian_articles_ = process_guardian_articles(guardian_articles)

# %%

# %%
# guardian_articles_

# %%
# with pd.ExcelWriter(PROJECT_DIR / 'outputs/data/results_august/examples_Guardian_articles.xlsx') as writer:
#     for key in dfs:
#         dfs[key].to_excel(writer, sheet_name=key)

# %% [markdown]
# # Networks

# %%
import innovation_sweet_spots.pipeline.network_analysis as iss_net
import innovation_sweet_spots.utils.altair_network as alt_net
import networkx as nx

# %%
proj = GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category == "Heat pumps"].copy()
proj["year"] = proj.start.apply(iss.convert_date_to_year)

# %%
pd.set_option("max_colwidth", 150)
proj.sort_values("amount").tail(50)

# %%
importlib.reload(iss)
iss.show_time_series_points(proj, y="amount")

# %%

# %%
project_orgs = iss.get_gtr_project_orgs(proj, project_to_org)
funded_orgs = iss.get_org_stats(project_orgs)
funded_orgs.amount_total = funded_orgs.amount_total / 1000
funded_orgs.reset_index().rename(columns={"amount_total": "total amount (1000s)"}).head(
    15
)

# %%
org_list = (
    pd.DataFrame(project_orgs.groupby(["project_id", "name"]).count().index.to_list())
    .groupby(0)[1]
    .apply(lambda x: list(x))
    .to_list()
)

# %%
graph = iss_net.make_network_from_coocc(org_list, spanning=False)

# %%
nodes = (
    pd.DataFrame(nx.layout.spring_layout(graph, seed=1))
    .T.reset_index()
    .rename(columns={"index": "node", 0: "x", 1: "y"})
)
df = funded_orgs.reset_index().rename(columns={"name": "node"})
df["node_name"] = df["node"]
nodes = nodes.merge(df)

# %%
nodes["is_university"] = nodes.node_name.str.contains(
    "University"
) | nodes.node_name.str.contains("College")
# nodes = nodes[nodes.is_university]

# %%
alt.data_transformers.disable_max_rows()
net_plot = alt_net.plot_altair_network(
    nodes,
    graph=graph,
    node_label="node",
    node_size="no_of_projects",
    node_size_title="number of projects",
    edge_weight_title="number of projects",
    title=f"Collaboration network",
    node_color="is_university",
    node_color_title="University",
)
net_plot.interactive()

# %%
alt_save.save_altair(net_plot, "district_heating_network", driver)

# %%
###

# %%

# %%
GTR_DOCS_ALL_.merge()

# %%

# %%
df = GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category.isin(CATEGORY_NAMES_cooc)]

# %%
# co_oc_matrix = np.zeros()

# %% [markdown]
# # LCH co-occurrences

# %%
CATEGORY_NAMES_lch = [
    "Micro CHP",
    "Heat storage",
    #     'Electric boilers',
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Insulation & retrofit",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
]

# %%
proj = GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category == "Heat pumps"].copy()
proj["year"] = proj.start.apply(iss.convert_date_to_year)

proj_other = GTR_DOCS_ALL_[
    GTR_DOCS_ALL_.tech_category.isin(CATEGORY_NAMES_lch)
    & (GTR_DOCS_ALL_.tech_category != "Heat pumps")
].copy()
proj_other["year"] = proj.start.apply(iss.convert_date_to_year)

# %%
proj_cooc = proj[["doc_id", "year", "title", "description"]].merge(
    proj_other[["doc_id", "tech_category"]], on="doc_id", how="left"
)
proj_cooc = proj_cooc.fillna("none")

# %%
proj_cooc_counts = (
    proj_cooc.groupby(["doc_id", "title", "description", "year"]).count().reset_index()
)
proj_cooc_counts["more_than_one"] = proj_cooc_counts["tech_category"] > 1

# %%
proj_cooc_counts.groupby("year").sum()

# %% [markdown]
# # Fuel poverty

# %%
df_fp = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/fuel_poverty_docs_checked.csv"
)
df_fp = df_fp[df_fp.hit != 0]

# %%
df_lch_eem_gtr = GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category.isin(["LCH & EEM"])]
fp_gtr = df_lch_eem_gtr[df_lch_eem_gtr.doc_id.isin(df_fp.doc_id.to_list())].copy()

# %%
fp_gtr.amount.sum() / df_lch_eem_gtr.amount.sum()

# %%
fp_gtr.amount.sum() / df_lch_eem_gtr.amount.sum()

# %%
df = iss.gtr_funding_per_year(fp_gtr, min_year=2007, max_year=2021)
df

# %%
get_growth_and_level_2(df, "no_of_projects")

# %%
get_growth_and_level_2(df, "amount_total")

# %%
fp_gtr.sort_values("amount").tail(5)

# %%
fp_gtr.sort_values("amount").iloc[-1].description

# %%
df_lch_eem_cb = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(["LCH & EEM"])]
fp_cb = df_lch_eem_cb[df_lch_eem_cb.doc_id.isin(df_fp.doc_id.to_list())]

# %%
df_fp[df_fp.source == "cb"]

# %%
iss.get_cb_org_funding_rounds(fp_cb, cb_funding_rounds)

# %%
### Examples

# %%
GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category == "Geothermal "]

# %%
df = YEARLY_STATS["Energy management"]
y = df[df.year.isin(list(range(2016, 2021)))].no_of_projects
print(np.min(y))
print(np.max(y))
m = np.mean(y)
m

# %%
get_level

# %%
df

# %%
s = np.std(df[df.year.isin(list(range(2016, 2021)))].no_of_projects)
s

# %%
s / m

# %%
get_growth_and_level("Energy management", "raised_amount_usd_total")

# %%
cb_df[cb_df]

# %%
orgs = [
    "Bud",
    "Capitalise.com",
    "Coconut",
    "Fluidly",
    "Tomato Pay",
    "Funding Options",
    "Iwoca",
    "Teller",
    "Akoni",
    "Bokio",
    "Finpoint",
    "Funding Circle",
    "OpenWrks",
    "Swoop Funding",
]

# %%
df = cb_df[cb_df.name.isin(orgs)][["name"]]

# %%
df.name.sort_values()

# %%
sorted(orgs)

# %%
