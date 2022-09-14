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


# %%
def remove_space_after_comma(text):
    return ",".join([s.strip() for s in text.split(",")])


# %%
df_search_terms = get_foodtech_search_terms()

# %%
df_search_terms["Terms"] = df_search_terms["Terms"].apply(remove_space_after_comma)

# %%
df_search_terms.tail(10)

# %%
df_search_results = get_guardian_searches()

# %%
# Get counts for each term
cols_to_drop = ["id", "text", "date", "year", "URL", "Headline", "Unnamed: 0"]

# %%
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
df_counts[df_counts.counts == 0].sort_values("Sub Category")

# %%
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
# # To do
# - Combine terms by categories
# - Calculate magnitude and growth
# - Visualise
# - Time series with terms, within each main category

# %%
from innovation_sweet_spots import PROJECT_DIR

guardian_baseline = pd.read_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_baseline.csv"
)

# %%
terms_to_remove = ["supply chain"]

# %%
assert df_search_results.id.duplicated().sum() == 0

# %%
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
def get_ts(df_id_to_term, category="Category"):
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
alt.Chart(ts_subcategory, width=200, height=100).mark_area().encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color="Sub Category:N",
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Sub Category", "counts", "year"],
).resolve_scale(y="independent")

# %%
alt.Chart(ts_tech_area, width=200, height=100).mark_area().encode(
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
magnitude_growth = (
    au.ts_magnitude_growth_(df_ts, year_start=2017, year_end=2021)
    .sort_values("growth")
    .reset_index()
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]],
        left_on="index",
        right_on="Terms",
        how="left",
    )
)
magnitude_growth

# %%
# # scale = 'log'
# scale = 'linear'

# fig = (
#     alt.Chart(magnitude_growth)
#     .mark_circle(size=60)
#     .encode(
#         x=alt.X('growth:Q', scale=alt.Scale(type=scale)),
#         y=alt.Y('index:N', sort='-x'),
#         # size=alt.Size('magnitude'),
#         color='Category',
#     )
# )

# %%
# df_search_results.head(2)

# %%
(
    df_id_to_term.sort_values(["year", "Tech area", "Sub Category", "Category"])
    .merge(df_search_results[["id", "Headline", "URL"]])
    .to_csv(
        PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_to_check.csv"
    )
)

# %%
df_search_results

# %% [markdown]
# ## Obesity

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pdau

# %%
query_id = "obesity"
search_terms = ["obesity", "obese"]

# query_id = "overweight"
# search_terms = ["overweight"]

# query_id = "healthy_eating"
# search_terms = ["healthy food", "healthy foods", "healthy eating", "healthy meal", "healthy meals"]

# query_id = "food_environment"
# search_terms = ["food environment", "food environments"]

REQUIRED_TERMS = search_terms
banned_terms = ["Australia"]

g = pdau.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
df = (
    g.document_mentions.rename(columns={"documents": "counts"})
    .merge(
        guardian_baseline.rename(columns={"counts": "total_counts"}),
        on="year",
        how="left",
    )
    .assign(fraction=lambda df: df.counts / df.total_counts)
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(df)
    .mark_area()
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=['year', 'counts', 'Category'],
    )
)
fig

# %%
