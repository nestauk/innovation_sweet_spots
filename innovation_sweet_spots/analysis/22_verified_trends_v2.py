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
gtr_organisations.head()

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
cb_df_cat = add_crunchbase_categories(cb_df, doc_column="id")
gtr_cat = add_gtr_categories(funded_projects)

# %%
nn = (
    gtr_cat[gtr_cat.text != "Unclassified"]
    .drop_duplicates("project_id")
    .project_id.to_list()
)
n_projects_with_categories = len(nn)
n_projects_without_categories = len(
    gtr_cat[
        (gtr_cat.text == "Unclassified") & (gtr_cat.project_id.isin(dd) == False)
    ].drop_duplicates("project_id")
)
n_projects_without_categories / len(gtr_projects)

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
REVIEWED_DOCS_PATH = OUTPUTS_DIR / "aux/ISS_technologies_to_review_August_10.xlsx"
COLS = ["doc_id", "title", "description", "source"]

# Some reference categories
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
for key in categories_keyphrases:
    print(f"{key}: {categories_keyphrases[key]}")

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
        ["glazing", "window"],
        ["glazed", "window"],
    ],
    "Radiators": [["radiator"]],
    "Energy management": [
        ["energy management", "build"],
        ["energy management", "domestic"],
        ["energy management", "hous"],
        ["thermostat"],
        ["smart meter"],
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
        "retrofitting",
    ],
    "Energy management": [
        "smart meter",
        "smart meters",
        "smart thermostat",
        "home energy management",
        "household energy management",
        "building energy management",
        "domestic energy management",
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
        "combined heat and power",
    ],
    "Heat storage": ["heat storage"],
}

# %%
# Get guardian articles
category_articles = {}
for category in search_terms:
    category_articles[category] = [
        guardian.search_content(search_term) for search_term in search_terms[category]
    ]


# %% [markdown]
# ### Hansard data

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


# %% [markdown]
# ## Yearly time series

# %%
REF_YEAR = 2015


# %%
def get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches):
    # Deduplicated versions (when combining several categories)
    gtr_docs_dedup = deduplicate_docs(gtr_docs)
    cb_doc_dedup = deduplicate_docs(cb_docs)
    # GTR data
    df_research_per_year = iss.gtr_funding_per_year(
        gtr_docs_dedup, min_year=2007, max_year=2020
    )
    # CB data
    df_deals = iss.get_cb_org_funding_rounds(cb_doc_dedup, cb_funding_rounds)
    df_deals_per_year = iss.get_cb_funding_per_year(df_deals)
    df_cb_orgs_founded_per_year = iss.cb_orgs_founded_by_year(
        cb_doc_dedup, max_year=2020
    )
    # Guardian data
    df_articles_per_year = iss.get_guardian_mentions_per_year(
        guardian_articles, max_year=2020
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


# %%
# CATEGORY_NAMES = ['Heat pumps', 'Geothermal energy', 'Solar thermal', 'District heating', 'Hydrogen boilers', 'Biomass boilers', 'Building insulation', 'Energy management']
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
# CATEGORY_NAMES = ['Heat pumps']
YEARLY_STATS = {}
YEARLY_STATS_NORM = {}
GTR_DOCS_ALL = pd.DataFrame()
for cat in CATEGORY_NAMES:
    df = get_verified_docs([cat])
    # Extract GTR and CB into separate dataframes
    gtr_docs = add_project_data(df[df.source == "gtr"])
    cb_docs = add_crunchbase_data(df[df.source == "cb"])
    guardian_articles = aggregate_guardian_articles(category_articles, [cat])
    speeches = aggregate_hansard_speeches([cat])
    # guardian_articles_ = [g for g in guardian_articles if g['sectionName'] not in ['World news', 'Australia news']]
    df_per_year = get_yearly_stats(gtr_docs, cb_docs, guardian_articles, speeches)
    df_per_year_norm = normalise_timeseries(
        iss_topics.get_moving_average(df_per_year, window=3, rename_cols=False),
        ref_year=REF_YEAR,
    )
    YEARLY_STATS[cat] = df_per_year
    YEARLY_STATS_NORM[cat] = df_per_year_norm
    GTR_DOCS_ALL = GTR_DOCS_ALL.append(gtr_docs, ignore_index=True)

# %%
# ALL_DFS = []
# for cat in CATEGORY_NAMES:
#     ALL_DFS.append(get_doc_probs([cat]))

# %%
from innovation_sweet_spots.utils.io import save_list_of_terms

save_list_of_terms(
    list(GTR_DOCS_ALL.doc_id.unique()),
    PROJECT_DIR / "outputs/data/results_august/check_doc_id_all.txt",
)
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

# %%
plt1 = iss.show_time_series_fancier(
    iss.gtr_funding_per_year(gtr_docs, min_year=2007),
    y="no_of_projects",
    show_trend=True,
)
iss.nicer_axis(plt1)

# %%
# iss.show_time_series_points(gtr_docs, y="amount", ymax=5000)

# %%
color = (alt.Color("species", legend=None),)

# %%
YEARLY_STATS["Heat pumps"].head(1)


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
variable = "no_of_projects"
# variable='amount_total'
# variable='articles'
# variable='speeches'
# variable='no_of_rounds'


# %%
from innovation_sweet_spots.utils.visualisation_utils import COLOUR_PAL


# %%
def plot_matrix(variable, category_names, x_label, y_label="Growth", year_1=2016):

    df_stats = pd.DataFrame(
        [
            get_growth_and_level(cat, variable, year_1) + tuple([cat])
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
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Building insulation",
    "Energy management",
]
iss.nicer_axis(
    plot_matrix(
        variable="no_of_projects",
        category_names=CATEGORY_NAMES_,
        x_label="Avg number of projects per year",
    )
)

# %%
iss.nicer_axis(
    plot_matrix(
        variable="amount_total",
        category_names=CATEGORY_NAMES_,
        x_label="Avg yearly amount (£1000s)",
    )
)

# %%
# aggregate_guardian_articles(category_articles, ['Hydrogen boilers'])

# %%
iss.get_guardian_mentions_per_year(
    aggregate_guardian_articles(category_articles, ["Hydrogen boilers"])
)

# %%
iss.nicer_axis(
    plot_matrix(
        variable="articles",
        category_names=CATEGORY_NAMES,
        x_label="Avg news articles per year",
        year_1=2017,
    )
)

# %%
iss.nicer_axis(
    plot_matrix(
        variable="speeches",
        category_names=CATEGORY_NAMES,
        x_label="Avg speeches per year",
    )
)

# %%
iss.nicer_axis(
    plot_matrix(variable="no_of_projects", x_label="Avg number of projects per year")
)

# %%
speeches = aggregate_hansard_speeches(["Building insulation"])

# %%
speeches_docs = iss.create_documents_from_dataframe(
    speeches, columns=["speech"], preprocessor=iss.preprocess_text
)

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
YEARLY_STATS_df = pd.DataFrame()
YEARLY_STATS_NORM_df = pd.DataFrame()
for cat in CATEGORY_NAMES:
    df_per_year = YEARLY_STATS[cat].copy()
    df_per_year["tech_category"] = cat
    YEARLY_STATS_df = YEARLY_STATS_df.append(df_per_year, ignore_index=True)

    df_per_year_norm = YEARLY_STATS_NORM[cat].copy()
    df_per_year_norm["tech_category"] = cat
    YEARLY_STATS_NORM_df = YEARLY_STATS_NORM_df.append(
        df_per_year_norm, ignore_index=True
    )

# %%
viz_cols = ["raised_amount_usd_total", "no_of_projects", "articles"]
df_per_year_melt = pd.melt(df_per_year_norm, id_vars=["year"], value_vars=viz_cols)
alt.Chart(df_per_year_melt, width=450, height=200).mark_line(size=2.5).encode(
    x="year:O",
    y="value",
    color="variable",
)

# %%
cat = "Hydrogen boilers"
df_ = YEARLY_STATS[cat]
df = iss_topics.get_moving_average(df_, window=3, rename_cols=False)
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
    alt.Y("no_of_rounds", axis=alt.Axis(title="Number of rounds", titleColor="#1D3354"))
)
fig_amount_raised = base.mark_line(color="#4CB7BD", size=2.5).encode(
    alt.Y(
        "raised_amount_usd_total",
        axis=alt.Axis(title="Amount raised ($1000s)", titleColor="#4CB7BD"),
    )
)
fig_crunchbase = alt.layer(fig_rounds, fig_amount_raised).resolve_scale(y="independent")

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
            "speeches", axis=alt.Axis(title="Parliament speeches", titleColor="#F9B3D1")
        ),
    )
)

fig_discourse = alt.layer(fig_articles, fig_speeches).resolve_scale(y="independent")
fig = iss.nicer_axis(alt.vconcat(fig_gtr, fig_crunchbase, fig_discourse))
fig

# %%
alt_save.save_altair(
    fig, f"asf_showntell_{'_'.join(cat.lower().split())}_tseries", driver
)

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

# %%
# term_indicators(funded_orgs.reset_index(), academic_org_terms, 'is_academic').iloc[251:300]

# %%
# project_orgs[project_orgs.name=='Institute of Mental Health']

# %%
# project_orgs

# %%
# gtr_docs.sort_values('amount', ascending=False)

# %%
nlp = spacy.load("en_core_web_sm")

# %%
w = 1
df_plot = df_per_year_norm
col = "articles"
fig = iss.show_time_series(
    iss_topics.get_moving_average(df_per_year_norm, window=w), y=f"{col}_sma{w}"
)
col = "amount_total"
fig + iss.show_time_series(
    iss_topics.get_moving_average(df_per_year_norm, window=w), y=f"{col}_sma{w}"
)

# %%
# stripplot =  alt.Chart(source, width=40).mark_circle(size=8).encode(
#     x=alt.X(
#         'jitter:Q',
#         title=None,
#         axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
#         scale=alt.Scale(),
#     ),
#     y=alt.Y('IMDB_Rating:Q'),
#     color=alt.Color('Major_Genre:N', legend=None),
#     column=alt.Column(
#         'Major_Genre:N',
#         header=alt.Header(
#             labelAngle=-90,
#             titleOrient='top',
#             labelOrient='bottom',
#             labelAlign='right',
#             labelPadding=3,
#         ),
#     ),
# ).transform_calculate(
#     # Generate Gaussian jitter with a Box-Muller transform
#     jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
# ).configure_facet(
#     spacing=0
# ).configure_view(
#     stroke=None
# )

# stripplot

# %%
# alt.Chart(df_per_year_norm).mark_line().encode(
#     x='year:O',
#     y='price',
#     color='symbol',
#     strokeDash='symbol',
# )

# %%
# pd.set_option('max_colwidth', 300)
# ref_category = CB_REFERENCE_CATEGORIES[1]
# print(ref_category)
# cb_ref = cb_df_cat[cb_df_cat.category_name==ref_category]
# cb_ref = cb_ref[cb_ref.id.isin(cb_docs.doc_id.to_list())==False]


# ref_category = GTR_REFERENCE_CATEGORIES[3]
# print(ref_category)
# gtr_ref = gtr_cat[gtr_cat.text==ref_category]
# gtr_ref = gtr_ref[gtr_ref.project_id.isin(gtr_docs.doc_id.to_list())==False]

# fund_rounds = iss.get_cb_org_funding_rounds(cb_ref, cb_funding_rounds)
# funding_per_year = iss.get_cb_funding_per_year(fund_rounds)
# funding_per_year

# %%
# # Further Guardian processing...
# # group returned articles by year
# sorted_articles = sorted(aggregated_articles, key = lambda x: x['webPublicationDate'][:4])
# articles_by_year = collections.defaultdict(list)
# for k,v in groupby(sorted_articles,key=lambda x:x['webPublicationDate'][:4]):
#     articles_by_year[k] = list(v)

# # Extract article metadata
# metadata = disc.get_article_metadata(articles_by_year, fields_to_extract=['id', 'webUrl'])
# # Extract article text
# article_text = disc.get_article_text_df(articles_by_year, TAGS, metadata)

# #

# %%
# articles_table(aggregated_articles).sample(20).sort_values('date')

# %%
(
    sentences_by_year,
    processed_articles_by_year,
    sentence_records,
) = disc.get_sentence_corpus(sample_articles, nlp)

# %%
# sample_articles = article_text
# sentences_by_year, processed_articles_by_year, sentence_records = disc.get_sentence_corpus(sample_articles, nlp)

# %%
# Flat list of sentences that contain search terms
term_sentences = disc.combine_flat_sentence_mentions(search_terms, sentence_records)

# %%
aggregated_sentiment_all_terms = dict()
sentence_sentiment_all_terms = dict()
for term in search_terms:
    aggregated_sentiment, sentence_sentiment = disc.agg_term_sentiments(
        term, term_sentences
    )
    aggregated_sentiment_all_terms[term] = aggregated_sentiment
    sentence_sentiment_all_terms[term] = sentence_sentiment

# %%
# Example sentences and sentiment for a given term (e.g. heat)
sentence_sentiment_all_terms["district heating"].tail()

# %%
