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

# %% [markdown]
# # Food tech: Hansard parliamentary debate trends
#

# %%
from innovation_sweet_spots.getters import hansard
import importlib
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_hansard_corpus
import utils
import innovation_sweet_spots.analysis.query_terms as query_terms
import altair as alt
import pandas as pd
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots import PROJECT_DIR
from ast import literal_eval
from innovation_sweet_spots.utils import plotting_utils as pu
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.utils import chart_trends

# %%
# Plotting utils
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

VERSION_NAME = "Report_Hansard"

# %% [markdown]
# ## Prepare search terms

# %%
# Noisy terms (based on checking Guardian articles)
terms_to_remove = [
    "supply chain",
    "delivery platform",
    "meal box",
    "smart kitchen",
    "novel food",
    "kitchen",
]

# %%
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

df_search_terms_table = get_foodtech_search_terms(from_local=False)

# %%
# Process search terms (lemmatise etc) for querying the Hansard corpus
df_search_terms = (
    df_search_terms_table.query("Terms not in @terms_to_remove")
    .assign(Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma))
    .pipe(utils.process_foodtech_terms)
)

# %%
# Search term dictionary
tech_area_terms = utils.compile_term_dict(df_search_terms)

# %% [markdown]
# ## Load Hansard speeches data

# %%
# Dataframe with parliamentary debates
df_debates = hansard.get_debates().drop_duplicates("id", keep="first")
assert len(df_debates.id.unique()) == len(df_debates)
len(df_debates)

# %% [markdown]
# # Keyword search

# %%
# Remove the last there tech areas (food terms, innovation terms and food technology terms)
tech_areas_to_check = list(tech_area_terms.keys())[:-3]

# %%
# Double check the aras
tech_areas_to_check

# %%
# Preprocessed Hansard corpus for keyword search (same data as in the table, but lemmatised etc)
hansard_corpus = get_hansard_corpus()
# Use query util for keyword search
Query_hansard = QueryTerms(corpus=hansard_corpus)

# %%
## Get speeches with food terms
food_hits = Query_hansard.find_matches(
    tech_area_terms["Food terms"], return_only_matches=True
)

# %%
## Get speeches with innovation terms
innovation_hits = Query_hansard.find_matches(
    tech_area_terms["Innovation terms"], return_only_matches=True
)

# %%
## Get speeches with technology terms
gtr_query_results, gtr_all_hits = query_terms.get_document_hits(
    Query_hansard, tech_area_terms, tech_areas_to_check, food_hits
)

# %%
# Check how many hits we found
len(gtr_query_results)

# %% [markdown]
# ### Additional filtering: Check that terms are in the same sentence
#
# A check for cases where the search term is a combiation of multiple terms, eg "food, reformulation".
# This will require that both "food" and "reformulation" are mentioned in the same sentnece.

# %%
# Add debate speech texts to the query results
hansard_query_results = gtr_query_results.merge(
    df_debates[["id", "speech", "speakername", "year"]], on="id"
)

# Create a column with found terms
hansard_query_results["found_terms_list"] = hansard_query_results.found_terms.apply(
    literal_eval
)
# Add non-processed search terms to the table (we'll use those to search in the full debate text)
hansard_query_results_ = (
    # Create a row for each search term
    hansard_query_results.explode("found_terms_list").astype({"found_terms_list": str})
    # Add the non-processed version of the term to the table
    .merge(
        df_search_terms[["terms_processed", "Terms"]].astype({"terms_processed": str}),
        left_on="found_terms_list",
        right_on="terms_processed",
    )
)
# Check that the non-processed terms are in the same sentence
has_terms_in_same_sentence = [
    utils.check_articles_for_comma_terms(row.speech, row.Terms)
    for i, row in hansard_query_results_.iterrows()
]

# %%
# Filter out the hits where terms are not in the same sentence
hansard_query_results_filtered = (
    hansard_query_results_[has_terms_in_same_sentence]
    .copy()
    .merge(
        df_search_terms[["Tech area", "Sub Category", "Category"]].drop_duplicates(
            "Tech area"
        ),
        how="left",
        left_on="tech_area",
        right_on="Tech area",
    )
    .drop_duplicates(["id", "Sub Category"])
)

# %%
len(hansard_query_results_filtered)
# Should at this point export thist table and start a new notebook

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Baseline number of speeches

# %%
hansard_baseline = df_debates.groupby("year", as_index=False).agg(
    total_counts=("id", "count")
)

# %%
alt.Chart(hansard_baseline).mark_line().encode(x="year", y="total_counts")

# %% [markdown]
# ## Check number of mentions per category

# %%
(
    hansard_query_results_filtered.astype({"year": int})
    .query("year >= 2017 and year < 2022")
    .groupby(["Category"])
    .agg(counts=("id", "count"))
    .reset_index()
)

# %%
hansard_query_results_filtered.astype({"year": int}).query(
    "year >= 2017 and year < 2022"
).groupby(["Category", "Sub Category"]).agg(counts=("id", "count")).reset_index()

# %%
hansard_query_results_filtered.groupby(
    ["Category", "Sub Category", "Tech area", "Terms"]
).agg(counts=("id", "count")).reset_index()

# %% [markdown]
# ## Time series charts

# %%
ts_category = (
    hansard_query_results_filtered.groupby(["year", "Category"])
    .agg(counts=("id", "count"))
    .reset_index()
    .merge(hansard_baseline, how="left", on="year")
    .assign(fraction=lambda df: df.counts / df.total_counts)
    .astype({"year": str})
)

# %%
ts_category

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line(size=3, interpolate="monotone")
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("fraction:Q", sort="-x", scale=alt.Scale(type=scale)),
        # size=alt.Size('magnitude'),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
data = (
    ts_category.copy()
    .query("Category == 'Food waste'")
    .assign(fraction=lambda df: df.fraction * 100)
)

fig = pu.ts_smooth_incomplete(
    data,
    ["Food waste"],
    "fraction",
    "Proportion of speeches (%)",
    "Category",
    amount_div=1,
)
pu.configure_plots(fig)

# %%
data = (
    ts_category.copy()
    .query("Category == 'Health'")
    .assign(fraction=lambda df: df.fraction * 100)
)

fig = pu.ts_smooth_incomplete(
    data,
    ["Health"],
    "fraction",
    "Proportion of speeches (%)",
    "Category",
    amount_div=1,
)
pu.configure_plots(fig)

# %%
data = (
    ts_category.copy()
    .query("Category == 'Innovative food'")
    .assign(fraction=lambda df: df.fraction * 100)
)

fig = pu.ts_smooth_incomplete(
    data,
    ["Innovative food"],
    "fraction",
    "Proportion of speeches (%)",
    "Category",
    amount_div=1,
)
pu.configure_plots(fig)

# %%
ts_category.query("Category == 'Logistics'")

# %% [markdown]
# ## Trends analysis

# %%
categories_to_check = ts_category.Category.unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    # Impute empty years
    ts_category_ = au.impute_empty_periods(
        ts_category.query("Category == @tech_area").assign(
            period=lambda df: pd.to_datetime(df.year)
        ),
        "period",
        "Y",
        2000,
        2021,
    ).assign(year=lambda df: df.period.dt.year, Category=tech_area)

    df = ts_category_.query("Category == @tech_area").drop(
        ["Category", "total_counts", "counts"], axis=1
    )[["year", variable]]
    df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
    magnitude_growth.append(
        [
            df_trends.query('trend == "magnitude"').iloc[0][variable],
            df_trends.query('trend == "growth"').iloc[0][variable],
            tech_area,
        ]
    )
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
chart_trends.estimate_trend_type(
    magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
)

# %% [markdown]
# ### Export results

# %%
# Export
(
    chart_trends.estimate_trend_type(
        magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
    ).to_csv(
        PROJECT_DIR / f"outputs/foodtech/trends/hansard_{VERSION_NAME}_Categories.csv",
        index=False,
    )
)

# %%
hansard_query_results_filtered.to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/hansard_hits_v2022_11_22.csv",
    index=False,
)

# %% [markdown]
# ## Check sentences with specific terms

# %%
import re
import ast


def extract_mention_sentence(text, search_term, any_term=False):
    text = re.sub("hon\.", "hon", text)
    text = re.sub("Hon\.", "Hon", text)
    sents = text.lower().split(".")
    sents_with_terms = []
    for sent in sents:
        if any_term:
            has_mention = False
            for s in search_term:
                has_mention = has_mention or (s in sent)
        else:
            has_mention = True
            for s in search_term:
                has_mention = has_mention and (s in sent)
        if has_mention == True:
            sents_with_terms.append(sent)
    return sents_with_terms


# %%
search_term = ["food environment"]
hits = Query_hansard.find_matches([search_term], return_only_matches=True)
hits = hits.merge(df_debates[["id", "speech", "speakername", "year"]], on="id")

# %%
hits.groupby("year").agg(counts=("id", "count"))

# %%
hits["sents"] = hits.speech.apply(
    lambda x: extract_mention_sentence(x, search_term, any_term=True)
)

# %%
# for i, row in hits.query("year > '2015'").iterrows():
#     print(row.year, row.speech)
#     print("")
#     print("")

# %% [markdown]
# ## Check specific terms

# %%
df = Query_hansard.find_matches([["deliveroo"]], return_only_matches=True)

# %%
df.head(1)

# %%
df = df.merge(df_debates[["id", "speech", "speakername", "year"]], on="id")

# %%
k = -19
print(df.iloc[k].year)
print(df.iloc[k].speech)
