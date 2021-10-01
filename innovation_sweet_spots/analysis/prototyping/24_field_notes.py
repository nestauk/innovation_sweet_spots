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

# %% [markdown]
# # Import prerequisites

# %% [markdown]
# ### Hansard

# %%
from innovation_sweet_spots.getters.hansard import get_hansard_data
import innovation_sweet_spots.utils.text_pre_processing as iss_preprocess

nlp = iss_preprocess.setup_spacy_model(iss_preprocess.DEF_LANGUAGE_MODEL)
from tqdm.notebook import tqdm

# hans_docs_tokenised = []
# for doc in tqdm(nlp.pipe(hans_docs), total=len(hans_docs)):
#     hans_docs_tokenised.append(' '.join(iss_preprocess.process_text(doc)))

# %%
hans = get_hansard_data(n_rows_to_skip=1000000, start_year=2000)
hans_docs = iss.create_documents_from_dataframe(
    hans, columns=["speech"], preprocessor=iss.preprocess_text
)

# %% [markdown]
# # Inspect technology categories

# %%
REVIEWED_DOCS_PATH = OUTPUTS_DIR / "aux/ISS_technologies_to_review_August_10.xlsx"


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


def get_yearly_stats(guardian_articles, speeches):
    # Guardian data
    df_articles_per_year = iss.get_guardian_mentions_per_year(
        guardian_articles, min_year=2000, max_year=2021
    )
    # Hansard
    speeches_per_year = iss.get_hansard_mentions_per_year(
        speeches, min_year=2000, max_year=2021
    ).rename(columns={"mentions": "speeches"})

    df_per_year = df_articles_per_year.merge(speeches_per_year, how="left")
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


# %%
hans.head(1)

# %% [markdown]
# ## Keywords

# %%
categories_keyphrases_hans = {
    # Heating subcategories
    "Heat pumps": [["heat pump"]],
    "Hydrogen heating": [["hydrogen", "boiler"], ["hydrogen", "heat"]],
    "Hydrogen & Fuel Cells": [[" hydrogen "], ["fuel cell"], ["sofc"]],
}

# %%
# Search terms for public discourse analysis
search_terms = {
    "Heat pumps": ["heat pump", "heat pumps"],
    "Hydrogen heating": [
        "hydrogen boiler",
        "hydrogen boilers",
        "hydrogen-ready boiler",
        "hydrogen-ready boilers",
        "hydrogen heating",
        "hydrogen heat",
    ],
    "Hydrogen & Fuel Cells": [
        "hydrogen energy",
        "fuel cell",
        "fuel cells",
        "hydrogen production",
        "solid oxide fuel cell",
    ],
}

# %%
guardian.search_content(adjusted_parameters)

# %%
# Get guardian articles
adjusted_param = {"from-date": "2007-01-01"}
category_articles = {}
for category in search_terms:
    category_articles[category] = [
        guardian.search_content(
            search_term,
            use_cached=False,
            save_to_cache=False,
            adjusted_parameters=adjusted_param,
        )
        for search_term in search_terms[category]
    ]

# %%
# Get guardian articles
adjusted_param = {"from-date": "2000-01-01"}
category_articles_2000 = {}
for category in search_terms:
    category_articles[category] = [
        guardian.search_content(
            search_term,
            use_cached=False,
            save_to_cache=False,
            adjusted_parameters=adjusted_param,
        )
        for search_term in search_terms[category]
    ]

# %% [markdown]
# ## Yearly time series

# %%
REF_YEAR = 2016


# %%
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
# CATEGORY_NAMES = ['Heat pumps', 'Geothermal energy', 'Solar thermal', 'District heating', 'Hydrogen boilers', 'Biomass boilers', 'Building insulation', 'Energy management']
CATEGORY_NAMES = [
    "Heat pumps",
    "Hydrogen heating",
    "Hydrogen & Fuel Cells",
]
# CATEGORY_NAMES = ['Heat pumps']
YEARLY_STATS = {}
YEARLY_STATS_NORM = {}
for cat in CATEGORY_NAMES:
    # Guardian articles
    guardian_articles = aggregate_guardian_articles(category_articles, [cat])
    guardian_articles = process_guardian_articles(guardian_articles)
    # Speeches
    speeches = aggregate_hansard_speeches([cat])
    # Yearly stats
    df_per_year = get_yearly_stats(guardian_articles, speeches)
    YEARLY_STATS[cat] = df_per_year

# %%
get_yearly_stats


# %%
def get_growth_and_level(cat, variable, year_1=2016, year_2=2020):
    df = YEARLY_STATS[cat].copy()
    df_ma = iss_topics.get_moving_average(df, window=3, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


# %%
from innovation_sweet_spots.utils.visualisation_utils import COLOUR_PAL

# %%
YEARLY_STATS["Hydrogen heating"]

# %%

# %%

# %%
speeches = aggregate_hansard_speeches(["Hydrogen heating"])
speeches_docs = iss.create_documents_from_dataframe(
    speeches, columns=["speech"], preprocessor=iss.preprocess_text
)

# %%
# speeches

# %%
sentences = iss.get_sentences_with_term("heat pump", speeches_docs)

# %%

# %%
i = 0
speeches.iloc[0]

# %%
# speeches_docs[-10]

# %%
for s in sentences:
    print(s)
    print()

# %% [markdown]
#

# %%
sentences = iss.get_sentences_with_term("insulat", speeches_docs)
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# for i, row in sentiment_df.iloc[0:5].iterrows():
#     print(row.compound, row.sentences, end="\n\n")

# %%
# for i, row in sentiment_df.sort_values("compound").iloc[0:5].iterrows():
#     print(row.compound, row.sentences, end="\n\n")

# %%
import altair as alt

# %%
YEARLY_STATS_df = pd.DataFrame()
YEARLY_STATS_NORM_df = pd.DataFrame()
for cat in CATEGORY_NAMES:
    df_per_year = YEARLY_STATS[cat].copy()
    df_per_year["tech_category"] = cat
    YEARLY_STATS_df = YEARLY_STATS_df.append(df_per_year, ignore_index=True)

#     df_per_year_norm = YEARLY_STATS_NORM[cat].copy()
#     df_per_year_norm['tech_category'] = cat
#     YEARLY_STATS_NORM_df=YEARLY_STATS_NORM_df.append(df_per_year_norm, ignore_index=True)

# %%
# viz_cols = ['articles', 'tech_category']
# df_per_year_melt = pd.melt(df_per_year, id_vars=['year'], value_vars=viz_cols)
df_viz = YEARLY_STATS_df[YEARLY_STATS_df.year > 2001]
df_viz = df_viz[df_viz.tech_category.isin(["Hydrogen heating", "Heat pumps"])]
# df_viz=df_viz[df_viz.tech_category.isin(['Heat pumps'])]
(
    alt.Chart(df_viz, width=450, height=200).mark_line(size=3)
    #  .mark_line(size=2.5)
    .encode(
        x="year:O",
        y="speeches",
        color="tech_category",
    )
)

# %%
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
alt_save.save_altair(
    fig, f"asf_showntell_{'_'.join(cat.lower().split())}_tseries", driver
)

# %%
alt_save.save_altair

# %%
# w=3
# iss.show_time_series(iss_topics.get_moving_average(mentions, window=w), y=f'mentions_sma{w}')

# %%
# speeches.groupby(['year', 'major_heading']).agg(counts=('speech', 'count')).sort_values('counts', ascending=False)

# %%
# w=1
# iss.show_time_series(iss_topics.get_moving_average(mentions, window=w), y=f'mentions_sma{w}')

# %%
# viz_cols = ['no_of_projects', 'amount_total', 'no_of_rounds', 'raised_amount_usd_total', 'articles']
# viz_cols = ['no_of_projects', 'amount_total', 'articles']
viz_cols = ["raised_amount_usd_total", "no_of_projects", "articles"]
df_per_year_melt = pd.melt(df_per_year_norm, id_vars=["year"], value_vars=viz_cols)
alt.Chart(df_per_year_melt, width=450, height=200).mark_line(size=2.5).encode(
    x="year:O",
    y="value",
    color="variable",
)

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
DF_MANUAL_REVIEW_1 = (
    pd.read_csv(
        "/Users/karliskanders/Downloads/project-7-at-2021-08-20-11-17-039f3681.csv"
    )
    .rename(columns={"category": "tech_category"})
    .query('sentiment=="Keep"')
)


# %%
CATEGORY_NAMES = [
    "Micro CHP",
    "Heat storage",
    "Electric boilers",
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Building insulation",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
]

GTR_DOCS_ALL = pd.DataFrame()
CB_DOCS_ALL = pd.DataFrame()
for cat in CATEGORY_NAMES:
    df = get_verified_docs([cat])
    # Extract GTR and CB into separate dataframes
    gtr_docs = add_project_data(df[df.source == "gtr"])
    cb_docs = add_crunchbase_data(df[df.source == "cb"])
    GTR_DOCS_ALL = GTR_DOCS_ALL.append(gtr_docs, ignore_index=True)
    CB_DOCS_ALL = CB_DOCS_ALL.append(cb_docs, ignore_index=True)

# %%
CB_DOCS_ALL.groupby("tech_category").count()

# %%
GTR_DOCS_ALL.groupby("tech_category").count()

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
    #     gtr_docs = process_gtr_docs(DF_REF[cat][DF_REF[cat].source=='gtr'])
    cb_docs = add_crunchbase_data(DF_REF[cat][DF_REF[cat].source == "cb"])
    gtr_docs = gtr_docs[(gtr_docs.manual_ok == 1) | (gtr_docs.topic_probs > 0.1)]

    category_articles[cat] = [
        guardian.search_content(search_term) for search_term in REF_TERMS[cat]
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
CB_DOCS_ALL_ = pd.concat(
    [
        CB_DOCS_ALL,
        CB_DOCS_REF_ALL,
        CB_DOCS_ALL_HEAT,
        CB_DOCS_ALL_HEAT_BUILD,
        CB_DOCS_ALL_HEAT_OTHER,
    ]
)
GTR_DOCS_ALL_ = pd.concat(
    [
        GTR_DOCS_ALL,
        GTR_DOCS_REF_ALL,
        GTR_DOCS_ALL_HEAT,
        GTR_DOCS_ALL_HEAT_BUILD,
        GTR_DOCS_ALL_HEAT_OTHER,
    ]
)

# %%

# %%
# gtr_docs_ = gtr_docs.sort_values('start').drop_duplicates('doc_id', keep='first').merge(gtr_docs.groupby('doc_id').agg(amount=('amount', 'sum')).reset_index())
# len(gtr_docs)

# %%
list(YEARLY_STATS.keys())

# %%
iss_io.save_pickle(
    YEARLY_STATS,
    PROJECT_DIR / "outputs/data/results_august/yearly_stats_all_categories.csv",
)

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

# %%
sentences = iss.get_guardian_sentences_with_term(
    "batter", guardian_articles_, field="body"
)
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# for i, row in (
#     sentiment_df.sort_values("compound", ascending=True).iloc[0:10].iterrows()
# ):
#     print(np.round(row.compound, 2), row.sentences, end="\n\n")

# %% [markdown]
# # Trajectories

# %%
CATEGORY_NAMES_ = [
    "Heat pumps",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Building insulation",
    "Energy management",
    "Hydrogen heating",
    "Biomass heating",
    "Solar",
    "Wind & Offshore",
    "Batteries",
    "Hydrogen & Fuel Cells",
]
CATEGORY_NAMES_ = list(YEARLY_STATS.keys())
y = 2016
iss.nicer_axis(
    plot_matrix(
        variable="no_of_projects",
        category_names=CATEGORY_NAMES_,
        x_label="Avg number of projects per year",
        year_1=y,
        year_2=y + 4,
    )
)

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
df_stats.head(20)

# %%
# df_stats_all

# %%
gtr_projects_ = gtr_projects[["project_id", "start"]].copy()
gtr_projects_["year"] = gtr_projects_.start.apply(iss.convert_date_to_year)


# %%
# gtr_projects_.groupby('year').count()

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
    variable, category_names, x_label, y_label, ww=350, hh=350
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
            color=alt.Color("tech_category"),
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
plot_tseries("Building insulation")


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
# GTR_DOCS_REF_ALL[GTR_DOCS_REF_ALL.tech_category=='Hydrogen & Fuel Cells']

# %%
# funded_projects_dedup = dedup_gtr_docs(funded_projects)

# %% [markdown]
# ### Number of projects

# %%
list(YEARLY_STATS.keys())

# %%
YEARLY_STATS["Solar"].head(1)

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
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
alt_save.save_altair(fig, f"user_meet_Heating.png", driver)

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
        #         'Heat storage',
        #                'Solar thermal',
        "Building insulation",
        "Energy management",
    ]
)
iss.nicer_axis(
    plot_matrix_trajectories(
        variable, cats, x_label, y_label, ww=400, hh=400
    ).interactive()
)

# %% [markdown]
# #### Reference categories

# %%
variable = "no_of_projects"
df_stats_all = get_year_by_year_stats(variable)
y_label = "Growth"
x_label = "Avg number of projects per year"
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
fig = iss.nicer_axis(plot_matrix(variable, cats, x_label, y_label))
fig

# %%
iss.nicer_axis(
    plot_matrix_trajectories(variable, cats, x_label, y_label, ww=400, hh=400)
)

# %% [markdown]
# #### Barplots and details

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
cat = "Carbon Capture & Storage"
plot_bars("amount_total", cat, "Number of projects").properties(title=cat)

# %%
funded_projects[funded_projects.title == "Pozibot"]

# %%
GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category == "Heating (all)"].sort_values(
    "amount"
).tail(10)

# %% [markdown]
# ### Private investment

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
cats = [
    "Batteries",
    "Hydrogen & Fuel Cells",
    "Carbon Capture & Storage",
    "Bioenergy",
    "Solar",
    "Heating & Building Energy Efficiency",
    "Wind & Offshore",
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
# cat = 'Wind & Offshore'
# plot_bars('no_of_rounds', cat, 'Number of deals').properties(title=cat)

# %%
# cats = sorted(['Heat pumps', 'Biomass heating', 'Hydrogen heating',
#         'Geothermal energy', 'Solar thermal', 'District heating',
#         'Heat storage', 'Building insulation', 'Energy management'])# CB_DOCS_ALL_[CB_DOCS_ALL_.name.str.contains('Loowatt')]

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

# %%

fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("sum(amount_total)", title="Research funding (£1000s)"),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%

# %%
cb_df[cb_df.name == "Integrated Environmental Solutions"].long_description.iloc[0]

# %% [markdown]
# ### Check funding rounds

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
# .groupby('tech_category').count()

# %%
