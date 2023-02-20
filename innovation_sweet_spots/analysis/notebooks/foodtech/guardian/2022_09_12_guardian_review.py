# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysing guardian article trends

# %%
from innovation_sweet_spots.getters._foodtech import get_guardian_searches
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
from innovation_sweet_spots.utils import plotting_utils as pu
import altair as alt
import pandas as pd
import numpy as np
from innovation_sweet_spots import PROJECT_DIR
import utils
import importlib
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pd_au
from innovation_sweet_spots.analysis import analysis_utils as au

# %% [markdown]
# # Plotting utils

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %% [markdown]
# # Load and check search results

# %% [markdown]
# ## Collect article ids and search terms

# %%
# Fetch list of search terms from the google sheet
df_search_terms = get_foodtech_search_terms(from_local=False).assign(
    Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma)
)

## Fetch public discourse analysis interim results
pd_path = PROJECT_DIR / "outputs/foodtech/interim/public_discourse"
# Fetch the aggregated search term query data
df_search_results = pd.read_csv(pd_path / "Food_terms_table_V2.csv")
assert df_search_results.id.duplicated().sum() == 0
# Fetch the search term precision assessment data
df_precision = pd.read_csv(pd_path / "Food_terms_precision.csv")

# %%
## Check terms that didn't return any articles
# Columns not related to seach terms
cols_to_drop = ["id", "text", "date", "year", "URL", "Headline", "Unnamed: 0"]
df_counts = (
    pd.DataFrame(df_search_results.drop(cols_to_drop, axis=1).sum())
    .rename(columns={0: "counts"})
    .reset_index()
    .rename(columns={"index": "Terms"})
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)

df_counts[df_counts.counts == 0].sort_values(["Sub Category", "Tech area"])

# %%
## Create table linking article ids to tech categories

# Reshape the dataframe and link article IDs to categories, sub categories and tech areas
non_search_term_columns = ["year", "text", "date", "URL", "Headline", "Unnamed: 0"]
df_id_to_term_all = (
    pd.melt(
        df_search_results.drop(
            set(non_search_term_columns).difference({"year"}), axis=1
        ),
        id_vars=["id", "year"],
    )
    .query("value==1")
    .rename(columns={"variable": "Terms"})
    .drop("value", axis=1)
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)

# Deal with terms that haven't been linked to a category yet (all about reformulation)
new_terms = list(df_id_to_term_all[df_id_to_term_all.Category.isnull()].Terms.unique())
print(new_terms)
df_id_to_term_all.loc[
    df_id_to_term_all.Terms.isin(new_terms), "Category"
] = "Innovative food"
df_id_to_term_all.loc[
    df_id_to_term_all.Terms.isin(new_terms), "Sub Category"
] = "Reformulation"
df_id_to_term_all.loc[
    df_id_to_term_all.Terms.isin(new_terms), "Tech area"
] = "Reformulation"


# %% [markdown]
# ## Add extra articles
#
# These will be added to the interim results

# %%
def create_id_term_table(
    term: str,
    category: str,
    sub_category: str,
    tech_area: str,
) -> pd.DataFrame:
    """Fetches articles from cache"""
    articles, _ = pd_au.get_guardian_articles(
        search_terms=[term],
        use_cached=True,
        allowed_categories=[],
        query_identifier="",
        save_outputs=False,
    )

    return pd.DataFrame(
        data={
            "id": articles.id.to_list(),
            "year": articles.year.to_list(),
            "Terms": term,
            "Category": category,
            "Sub Category": sub_category,
            "Tech area": tech_area,
        }
    )


# %%
extra_terms = [
    # Extra: Fetch also articles featuring obesity / 'biomedical' terms
    {
        "term": "obesity",
        "category": "Health",
        "sub_category": "Biomedical",
        "tech_area": "Biomedical",
    },
    {
        "term": "obese",
        "category": "Health",
        "sub_category": "Biomedical",
        "tech_area": "Biomedical",
    },
    {
        "term": "overweight",
        "category": "Health",
        "sub_category": "Biomedical",
        "tech_area": "Biomedical",
    },
    # Extra: Fetch also articles featuring food and reformulation terms
    {
        "term": "food,reformulation",
        "category": "Innovative food",
        "sub_category": "Reformulation",
        "tech_area": "Reformulation",
    },
    # Extra: Fetch also articles featuring retail and restaurant technology terms
    {
        "term": "restaurant,technology",
        "category": "Restaurants and retail",
        "sub_category": "Restaurants",
        "tech_area": "Restaurants",
    },
    {
        "term": "restaurants,technology",
        "category": "Restaurants and retail",
        "sub_category": "Restaurants",
        "tech_area": "Restaurants",
    },
    {
        "term": "retail,technology",
        "category": "Restaurants and retail",
        "sub_category": "Retail",
        "tech_area": "Retail",
    },
    {
        "term": "supermarket,technology",
        "category": "Restaurants and retail",
        "sub_category": "Retail",
        "tech_area": "Retail",
    },
    {
        "term": "supermarkets,technology",
        "category": "Restaurants and retail",
        "sub_category": "Retail",
        "tech_area": "Retail",
    },
    # Extra: Fetch also articles featuring cooking and kitchen tech
    {
        "term": "kitchen,technology",
        "category": "Cooking and kitchen",
        "sub_category": "Kitchen tech",
        "tech_area": "Kitchen tech",
    },
    {
        "term": "food preparation,technology",
        "category": "Cooking and kitchen",
        "sub_category": "Kitchen tech",
        "tech_area": "Kitchen tech",
    },
]

# %%
extra_tables = [create_id_term_table(**extra_term) for extra_term in extra_terms]


# %%
df_id_to_term_all_extra = pd.concat(
    [df_id_to_term_all] + extra_tables, ignore_index=True
)


# %% [markdown]
# ## Define filtering parameters

# %%
# Set a lower limit for accepting search terms
PRECISION_THRESHOLD = 0.5

# Search terms to exclude
terms_to_manually_remove = [
    "supply chain",
]

# Extra *required* keywords to use for filtering
food_terms_categories = ["Food technology terms", "Food terms"]
food_terms = df_search_terms.query("Category in @food_terms_categories").Terms.to_list()

# innovation_terms_categories = ['Innovation terms']
# innovation_terms = df_search_terms.query("Category in @innovation_terms_categories").Terms.to_list()
innovation_terms = [
    "innovation",
    "innovative",
    "innovat",
    "novel",
    "research",
    "technology",
    "tech",
]

# Extra *banned* keywords to use for removing articles
banned_terms = ["australia"]

# %%
# List of terms to remove
imprecise_terms = df_precision[
    (df_precision.proportion_correct <= PRECISION_THRESHOLD)
    & (df_precision.terms.str.contains(",") == False)
].terms.to_list()

# %%
terms_to_remove = terms_to_manually_remove + imprecise_terms
terms_to_remove

# %% [markdown]
# ## Add article texts

# %%
# To speed-up, only fetch article texts for terms that will be included
search_terms = list(
    set(sorted(list(df_id_to_term_all_extra.Terms.unique()))).difference(
        set(terms_to_remove)
    )
)


# %%
id_to_text = []
for term in search_terms:
    articles, metadata = pd_au.get_guardian_articles(
        search_terms=[term],
        use_cached=True,
        allowed_categories=[],
        query_identifier="",
        save_outputs=False,
    )
    id_to_text.append(articles[["id", "text"]])


# %%
df_id_to_term_all_extra_text = df_id_to_term_all_extra.merge(
    pd.concat(id_to_text, ignore_index=False), how="left", on="id"
)


# %% [markdown]
# ## Article filtering: search terms & counts
# - Remove imprecise terms
# - Check number of terms mentioned in an article

# %%
# Count the number of term mentions per article &
# check if article hasa term that's in the list of removed terms
n_terms_per_article = (
    df_id_to_term_all_extra_text.drop_duplicates(["id", "Terms"])
    .groupby("id", as_index=False)
    .agg(counts=("Terms", "count"))
)

# Filter articles according to criteria
df_id_to_term_filters = (
    df_id_to_term_all_extra_text.copy()
    .assign(has_imprecise_term=lambda df: df.Terms.isin(terms_to_remove))
    .merge(n_terms_per_article, on="id", how="left")
    # NB: Most important line
    .assign(keep=lambda df: (df.has_imprecise_term == False) & (df.counts > 0))
    # .assign(keep=lambda df: (df.has_imprecise_term == False))
)

# Final list of articles to keep for further analysis
df_id_to_term_filtered = (
    df_id_to_term_filters.query("keep == True").drop_duplicates(["Terms", "id"]).copy()
)

len(df_id_to_term_filtered)

# %%
len(df_id_to_term_filtered.id.unique())

# %% [markdown]
# ## Article filtering: extra keywords

# %%
df_id_to_term_filtered["text_lower"] = df_id_to_term_filtered.text.str.lower()

# %%
has_any_term = df_id_to_term_filtered.text_lower.str.contains(food_terms[0])
for term in food_terms:
    has_any_term = has_any_term | df_id_to_term_filtered.text_lower.str.contains(term)
df_id_to_term_filtered["has_food_terms"] = has_any_term

# %%
has_any_term = df_id_to_term_filtered.text_lower.str.contains(innovation_terms[0])
for term in innovation_terms:
    has_any_term = has_any_term | df_id_to_term_filtered.text_lower.str.contains(term)
df_id_to_term_filtered["has_innovation_terms"] = has_any_term

# %%
df_id_to_term_filtered_ = df_id_to_term_filtered[
    df_id_to_term_filtered.has_innovation_terms & df_id_to_term_filtered.has_food_terms
].copy()

# %%
len(df_id_to_term_filtered_)

# %%
1-(26393/42659)

# %%
# df_id_to_term.query("`Sub Category` == 'Alt protein'").to_csv(PROJECT_DIR / 'outputs/foodtech/interim/test_guardian_articles.csv', index=False)

# %%
# df_id_to_term_filtered_.to_csv(
#     PROJECT_DIR
#     / "outputs/foodtech/interim/public_discourse/guardian_interim_articles.csv",
#     index=False,
# )

# %% [markdown]
# ## Additional filtering for comma terms

# %%
# importlib.reload(utils)

# %%
# df_id_to_term_filtered_ = pd.read_csv(PROJECT_DIR / 'outputs/foodtech/interim/public_discourse/guardian_interim_articles.csv')


# %%
# # Testing the functionality
# txt = df_id_to_term[df_id_to_term.Terms.str.contains(',')].iloc[1267].text
# utils.check_articles_for_comma_terms(txt, 'proteins,market')

# %%
has_terms_in_same_sentence = [
    utils.check_articles_for_comma_terms(row.text, row.Terms)
    for i, row in df_id_to_term_filtered_.iterrows()
]

# %%
assert len(has_terms_in_same_sentence) == len(df_id_to_term_filtered_)

# %%
df_id_to_term_filtered_["has_terms_in_same_sent"] = has_terms_in_same_sentence

# %%
txt = (
    df_id_to_term_filtered_.query(
        "Terms == 'restaurant,technology' and has_terms_in_same_sent == True"
    )
    .iloc[35]
    .text
)
utils.find_sentences_with_terms(txt, ["restaurant", "technology"])

# %%
df_id_to_term = df_id_to_term_filtered_[
    df_id_to_term_filtered_.has_terms_in_same_sent
].copy()

# %%
len(df_id_to_term)

# %%
1-(12866/26393)

# %% [markdown]
# ### Check category counts

# %%
counts_df = (
    df_id_to_term.drop_duplicates(["Category", "id"])
    .groupby(["Category"], as_index=False)
    .agg(counts=("id", "count"))
)
counts_df

# %%
counts_df = (
    df_id_to_term.drop_duplicates(["Terms", "id"])
    .groupby(["Category", "Sub Category"], as_index=False)
    .agg(counts=("id", "count"))
)
counts_df

# %% [markdown]
# ## Export for reviewing

# %%
cols = [
    "id",
    "URL",
    "year",
    "Terms",
    "counts",
    "Category",
    "Sub Category",
    "Tech area",
    "has_food_terms",
    "has_innovation_terms",
    "has_terms_in_same_sent",
]

df_export = (
    df_id_to_term.copy()
    .assign(URL=lambda df: "https://www.theguardian.com/" + df.id)
    .drop_duplicates(["id", "Sub Category"])
)[cols]

# %%
df_id_to_term["Sub Category"].unique()

# %%
len(df_export)

# %%
df_export.to_csv(
    PROJECT_DIR / "outputs/foodtech/public_discourse/guardian_to_check_FINAL_n1.csv",
    index=False,
)

# %%
df_id_to_term_copy = df_id_to_term.copy()

# %%
df_id_to_term.drop("text_lower", axis=1).query('`Sub Category`=="Delivery"').to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_delivery.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query(
    '`Sub Category`=="Reformulation"'
).to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_reformulation.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query('`Category`=="Food waste"').to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_food_waste.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query('`Category`=="Health"').to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_health.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query('`Sub Category`=="Restaurants"').to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_restaurants.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query('`Sub Category`=="Retail"').to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_retail.csv",
    index=False,
)
df_id_to_term.drop("text_lower", axis=1).query(
    '`Category`=="Cooking and kitchen"'
).to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_cooking_kitchen.csv",
    index=False,
)

# %%
df_id_to_term.head(1)

# %% [markdown]
# # Checking technology trends
# - Combining terms by categories
# - Time series with terms and categories
# - Calculate magnitude and growth

# %%
from innovation_sweet_spots.getters import google_sheets

importlib.reload(google_sheets)
df_id_to_term_reviewed = google_sheets.get_foodtech_guardian(from_local=False)

# %%
df_id_to_term = (
    df_id_to_term_reviewed.copy()
    .query("checked_subcategory != '-'")
    .assign(
        **{
            "Category": lambda df: df.checked_category,
            "Sub Category": lambda df: df.checked_subcategory,
        }
    )
    .astype({"year": int})
)

# %%
df_id_to_term.year.unique()

# %%
df_id_to_text = (
    df_id_to_term[["id", "year", "Category", "Sub Category"]]
    .merge(df_id_to_term_copy[["id", "text"]], on="id", how="left")
    .drop_duplicates(["id", "Sub Category"])
    .fillna("")
)

# %%

# %% [markdown]
# ## Charts

# %%
from innovation_sweet_spots import PROJECT_DIR

# Total article counts per year
guardian_baseline = pd.read_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_baseline.csv"
)


def get_ts(df_id_to_term, category="Category"):
    """build time series"""
    return (
        df_id_to_term.drop_duplicates(["id", category], keep="first")
        .groupby(["year", category])
        .agg(counts=("id", "count"))
        .reset_index()
        .merge(
            guardian_baseline.rename(columns={"counts": "total_counts"}),
            on="year",
            how="left",
        )
        .assign(fraction=lambda df: df.counts / df.total_counts)
    )


# %%
ts_category = get_ts(df_id_to_term, "Category")

# %%
ts_subcategory = get_ts(df_id_to_term, "Sub Category").merge(
    df_search_terms[["Category", "Sub Category"]].drop_duplicates("Sub Category"),
    how="left",
)

# %%
# ts_tech_area = get_ts(df_id_to_term, "Tech area").merge(
#     df_search_terms[["Category", "Sub Category", "Tech area"]].drop_duplicates(
#         "Tech area"
#     ),
#     how="left",
# )

# %%
# ts_search_term = get_ts(df_id_to_term, "Terms").merge(
#     df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]].drop_duplicates(
#         "Terms"
#     ),
#     how="left",
# )

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line()
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        # size=alt.Size('magnitude'),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
alt.Chart(ts_subcategory, width=200, height=100).mark_line(size=3).encode(
    x="year:O",
    y=alt.Y(
        "counts:Q",
    ),
    color="Sub Category:N",
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Sub Category", "counts", "year"],
).resolve_scale(y="independent")

# %%
# alt.Chart(ts_tech_area, width=200, height=100).mark_line().encode(
#     x="year:O",
#     y=alt.Y(
#         "fraction:Q",
#     ),
#     color=alt.Color("Tech area:N", scale=alt.Scale(scheme="dark2")),
#     facet=alt.Facet("Category:N", columns=2),
#     tooltip=["Tech area", "counts", "year"],
# ).resolve_scale(y="independent")

# %%
# alt.Chart(ts_search_term, width=200, height=100).mark_line(
#     point=alt.OverlayMarkDef()
# ).encode(
#     x="year:O",
#     y=alt.Y(
#         "fraction:Q",
#     ),
#     color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
#     facet=alt.Facet("Category:N", columns=2),
#     tooltip=["Terms", "counts"],
# ).resolve_scale(
#     y="independent"
# )

# %%
# alt.Chart(ts_search_term, width=200, height=100).mark_line().encode(
#     x="year:O",
#     y=alt.Y(
#         "fraction:Q",
#     ),
#     color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
#     facet=alt.Facet("Category:N", columns=2),
#     tooltip=["Terms", "counts"],
# ).resolve_scale(y="independent")

# %%
# import importlib


# importlib.reload(au)

# %%
# Export articles to check

# %%
# df_to_check = (
#     df_id_to_term.sort_values(["year", "Tech area", "Sub Category", "Category"])
#     .drop(["text", "text_lower"], axis=1)
#     .drop_duplicates(["Tech area", "id"])
#     .merge(df_search_results[["id", "Headline", "URL"]], how="left", on="id")
# )
# df_to_check.to_csv(
#     PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_to_check_V3.csv"
# )

# %% [markdown]
# ## Magnitude and growth trends

# %%
from innovation_sweet_spots.utils import chart_trends

# %%
categories_to_check = ts_category.Category.unique()

# %%
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = ts_category.query("Category == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
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
domain = [
    "Health",
    "Innovative food",
    "Logistics",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
]
range_ = pu.NESTA_COLOURS[0 : len(domain)]

# %%
magnitude_growth_df

# %%
baseline_magnitude_growth = au.estimate_magnitude_growth(guardian_baseline, 2017, 2021)
baseline_growth = (
    baseline_magnitude_growth.query("trend == 'growth'").iloc[0].counts / 100
)

# %%
baseline_growth

# %%
magnitude_growth_df.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=0.45,
    y_limit=4,
    mid_point=0.09,
    baseline_growth=0,
    values_label="Proportion of articles (%)",
    text_column="tech_area",
)
fig

# %%
# fig = chart_trends.mangitude_vs_growth_chart(
#     data=magnitude_growth_df,
#     x_limit=0.2,
#     y_limit=4,
#     mid_point=0.053,
#     baseline_growth=0,
#     values_label="Proportion of articles (%)",
#     text_column="tech_area",
# )
# fig

# %%
variable = "counts"
magnitude_growth = []
for tech_area in categories_to_check:
    df = ts_category.query("Category == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
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
)

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=130,
    y_limit=3,
    mid_point=42,
    baseline_growth=-0.25,
    values_label="Average number of articles",
    text_column="tech_area",
)
fig

# %%
(
    alt.Chart(magnitude_growth_df)
    .mark_bar()
    .encode(
        x=alt.X("growth"),
        y=alt.Y("tech_area"),
    )
)

# %%
AltairSaver.save(
    fig_final, f"Guardian_articles_magnitude_growth", filetypes=["html", "svg", "png"]
)

# %%
ts_subcategory.head(1)

# %%
cats = ["Reformulation", "Alt protein"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df.query("year > 2010")).mark_line(size=3, interpolate="monotone")
    # .mark_bar(size=5, interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_InnovativeFood", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Delivery"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_Delivery", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Restaurants", "Retail"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df).mark_line(
        size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3]
    )
    # .mark_bar(size=5, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
cats = ["Kitchen tech"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
categories_to_check

# %%
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = ts_subcategory.query("Sub Category == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
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

# %% [markdown]
# g.document_mentions## Checking obesity/health trends

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pdau

banned_terms = ["Australia"]


def get_g_ts(g):
    """Get time series from a search query"""
    return (
        g.document_mentions.rename(columns={"documents": "counts"})
        .merge(
            guardian_baseline.rename(columns={"counts": "total_counts"}),
            on="year",
            how="left",
        )
        .assign(fraction=lambda df: df.counts / df.total_counts)
    )


def get_queries_ts(queries, required_terms=None):
    """Get time series for all specified search queries"""
    ts_df = []
    for query_id in queries:
        required_terms = queries[query_id] if required_terms is None else required_terms
        # REQUIRED_TERMS = search_terms
        g = pdau.DiscourseAnalysis(
            search_terms=queries[query_id],
            required_terms=required_terms,
            banned_terms=banned_terms,
            use_cached=True,
            query_identifier=query_id,
        )
        ts_df.append(get_g_ts(g).assign(query=query_id))
    ts_df = pd.concat(ts_df, ignore_index=True)
    return ts_df


# %%
# {query_id: search terms}
queries = {
    "obesity": ["obesity", "obese"],
    "overweight": ["overweight"],
    "healthy_eating": [
        "healthy food",
        "healthy foods",
        "healthy eating",
        "healthy meal",
        "healthy meals",
    ],
    "food_environment": ["food environment", "food environments", "obesogenic"],
    # "food_desert": ["food desert"],
}


ts_df = get_queries_ts(queries)


# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("counts:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig

# %%
ts_df_obesity = ts_df.query("query=='obesity'")
# ts_df_obesity = ts_df.query("query=='food_environment'")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_obesity)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_Obesity", filetypes=["html", "svg", "png"]
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_obesity)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q",
            title="Percentage of articles",
            axis=alt.Axis(
                format="%",
                tickCount=5,
            ),
        ),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format=".2%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_proportion_per_year_Obesity", filetypes=["html", "svg", "png"]
)

# %%
au.estimate_magnitude_growth(ts_df_obesity, 2017, 2021)

# %%
ts_df_environment = ts_df.query("query=='food_environment'")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_environment)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
au.estimate_magnitude_growth(ts_df_environment, 2017, 2021)

# %% [markdown]
# ### Overlap of obesity and food tech articles

# %%
query_id = "obesity"
g = pdau.DiscourseAnalysis(
    search_terms=queries[query_id],
    required_terms=queries[query_id],
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
obesity_ids = list(g.metadata.keys())
years = [int(g.metadata[key]["webPublicationDate"][0:4]) for key in g.metadata]
obesity_ids = set(
    [x for i, x in enumerate(obesity_ids) if ((years[i] < 2022) and (years[i] >= 2017))]
)


# %%
len(obesity_ids)

# %%
foodtech_ids = set(df_id_to_term.query("year >= 2017 and year < 2022").id.to_list())

# %%
common_articles = len(obesity_ids.intersection(foodtech_ids))

# %%
common_articles / len(obesity_ids)

# %%
common_articles / len(foodtech_ids)

# %%
for cat in df_id_to_term.Category.unique():
    foodtech_ids = set(
        df_id_to_term.query("year >= 2017 and year < 2022")
        .query("Category==@cat")
        .id.to_list()
    )
    common_articles = len(obesity_ids.intersection(foodtech_ids))
    print(cat)
    print(common_articles / len(foodtech_ids))

# %%
query_id = "healthy_eating"
g = pdau.DiscourseAnalysis(
    search_terms=queries[query_id],
    required_terms=[],
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
ids = set(list(g.metadata.keys()))

# %%
for cat in df_id_to_term.Category.unique():
    foodtech_ids = set(df_id_to_term.query("Category==@cat").id.to_list())
    common_articles = len(ids.intersection(foodtech_ids))
    print(cat)
    print(common_articles / len(foodtech_ids))

# %% [markdown]
# ## Check mentions of Obesity AND food environment related terms

# %%
food_environment_terms = [
    "food environment",
    "advertising",
    "marketing",
    "advert",
    "takeaway service",
    "meal takeaway",
    "food takeaway",
    "fast food",
    "junk food",
    "ultra processed food",
    "food desert",
    "food system",
    "inequality",
    "inequalities",
    "poverty",
    "obesogenic",
]

# %%
ts_df = get_queries_ts({"obesity": queries["obesity"]})

# %%
ts_df_ = get_queries_ts(
    {"obesity": queries["obesity"]}, required_terms=food_environment_terms
)

# %%
ts_df_["query"] = "obesity_x_food_env"

# %%
ts_df_combined = pd.concat([ts_df, ts_df_])


# %%
df = ts_df.merge(ts_df_[["counts", "year"]], on="year").assign(
    fraction_environment=lambda df: df.counts_y / df.counts_x
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction_environment:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "fraction_environment", "query"],
    )
)
fig

# %%
