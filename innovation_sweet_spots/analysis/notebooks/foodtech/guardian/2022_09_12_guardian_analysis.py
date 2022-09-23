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
# # Analysing guardian article trends

# %%
from innovation_sweet_spots.getters._foodtech import get_guardian_searches
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
from innovation_sweet_spots.utils import plotting_utils as pu
import altair as alt
import pandas as pd
import numpy as np


# %% [markdown]
# # Load and check search results

# %%
def remove_space_after_comma(text):
    """util function to process search terms with comma"""
    return ",".join([s.strip() for s in text.split(",")])


# %%
# Fetch search terms
df_search_terms = get_foodtech_search_terms()
df_search_terms["Terms"] = df_search_terms["Terms"].apply(remove_space_after_comma)

# %%
# Get Guardian search results
df_search_results = get_guardian_searches()

# %%
# Get article counts for each term
cols_to_drop = ["id", "text", "date", "year", "URL", "Headline", "Unnamed: 0"]

df_counts = (
    pd.DataFrame(df_search_results.drop(cols_to_drop, axis=1).sum())
    .rename(columns={0: "counts"})
    .sort_values("counts", ascending=False)
    .reset_index()
    .rename(columns={"index": "Terms"})
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)

# %%
# Terms without any hits
df_counts[df_counts.counts == 0].sort_values("Sub Category")

# %%
# Most popular search terms
scale = "log"
# scale = 'linear'

fig = (
    alt.Chart(df_counts[df_counts.counts != 0])
    .mark_circle(size=60)
    .encode(
        x=alt.X("counts:Q", scale=alt.Scale(type=scale)),
        y=alt.Y("Terms:N", sort="-x"),
        color="Sub Category",
    )
)

# %%
fig

# %% [markdown]
# # Checking technology trends
# - Combining terms by categories
# - Time series with terms and categories
# - Calculate magnitude and growth

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
terms_to_remove = ["supply chain"]

# %%
assert df_search_results.id.duplicated().sum() == 0

# %%
# Link articles to categories, sub categories and tech areas
non_search_term_columns = ["year", "text", "date", "URL", "Headline", "Unnamed: 0"]
df_id_to_term = df_search_results.drop(
    set(non_search_term_columns).difference({"year"}), axis=1
).copy()

df_id_to_term = (
    pd.melt(df_id_to_term, id_vars=["id", "year"])
    .query("value==1")
    .rename(columns={"variable": "Terms"})
    .drop("value", axis=1)
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)
df_id_to_term = df_id_to_term[
    df_id_to_term.Terms.isin(terms_to_remove) == False
].reset_index(drop=True)

# %%
df_id_to_term

# %%
ts_category = get_ts(df_id_to_term, "Category")

# %%
ts_subcategory = get_ts(df_id_to_term, "Sub Category").merge(
    df_search_terms[["Category", "Sub Category"]].drop_duplicates("Sub Category"),
    how="left",
)

# %%
ts_tech_area = get_ts(df_id_to_term, "Tech area").merge(
    df_search_terms[["Category", "Sub Category", "Tech area"]].drop_duplicates(
        "Tech area"
    ),
    how="left",
)

# %%
ts_search_term = get_ts(df_id_to_term, "Terms").merge(
    df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]].drop_duplicates(
        "Terms"
    ),
    how="left",
)

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
alt.Chart(ts_tech_area, width=200, height=100).mark_line().encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Tech area:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Tech area", "counts", "year"],
).resolve_scale(y="independent")

# %%
alt.Chart(ts_search_term, width=200, height=100).mark_line(
    point=alt.OverlayMarkDef()
).encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Terms", "counts"],
).resolve_scale(
    y="independent"
)

# %%
alt.Chart(ts_search_term, width=200, height=100).mark_line().encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Terms", "counts"],
).resolve_scale(y="independent")

# %%
import importlib
from innovation_sweet_spots.analysis import analysis_utils as au

importlib.reload(au)

# %%
(
    df_id_to_term.sort_values(["year", "Tech area", "Sub Category", "Category"])
    .merge(df_search_results[["id", "Headline", "URL"]])
    .to_csv(
        PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_to_check.csv"
    )
)

# %% [markdown]
# ## Checking obesity/health trends

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
        tooltip=["year", "counts", "query"],
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
