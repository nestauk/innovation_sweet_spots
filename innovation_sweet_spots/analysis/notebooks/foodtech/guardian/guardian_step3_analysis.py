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
# # Food tech: Guardian news discourse trends
# ## Step 3: Analysis and charts
#
# - Data has been fetched from The Guardian (step 1)
# - Articles have been lightly reviewed and a few have been manually added (step 2)
# - This notebook (step 3) produces charts for the report
#

# %% [markdown]
# ### Loading dependencies

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters._foodtech import get_guardian_searches
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
from innovation_sweet_spots.utils import plotting_utils as pu
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pd_au
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.getters import google_sheets
from innovation_sweet_spots.utils import chart_trends
from innovation_sweet_spots import PROJECT_DIR
import utils

import altair as alt
import pandas as pd
import numpy as np
import importlib

# %%
# Plotting utils
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
VERSION_NAME = "Report_Guardian"

# %% [markdown]
# ### Loading data

# %%
# Loading the identified and partially reviewed list of relevant Guardian articles
df_id_to_term_reviewed = google_sheets.get_foodtech_guardian(from_local=False)
# Process the reviewed Guardian data
df_id_to_term = (
    df_id_to_term_reviewed.copy()
    # Remove irrelevant articles
    .query("checked_subcategory != '-'")
    # Create category and sub-category columns
    .assign(
        **{
            "Category": lambda df: df.checked_category,
            "Sub Category": lambda df: df.checked_subcategory,
        }
    ).astype({"year": int})
)


# %%
# Fetch list of search terms from the google sheet
df_search_terms = get_foodtech_search_terms(from_local=False).assign(
    Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma)
)

# %%
# Total article counts per year
guardian_baseline = pd.read_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_baseline.csv"
)

# %% [markdown]
# ### Check the number of articles in each category

# %%
# Number of articles per subcategory
df_counts = (
    df_id_to_term.query("year > 2016 and year < 2022")
    .drop_duplicates(["id", "Category", "Sub Category"])
    .groupby(["Category", "Sub Category"], as_index=False)
    .agg(counts=("id", "count"))
)
df_counts

# %%
# Number of articles per tech area (more granular than subcategory)
# Note: Empty rows in for 'tech area' indicate manually added articles
df_tech_area_counts = (
    df_id_to_term.query("year > 2016 and year < 2022")
    .drop_duplicates(["id", "Category", "Sub Category", "Tech area"])
    .groupby(["Category", "Sub Category", "Tech area"], as_index=False)
    .agg(counts=("id", "count"))
)
df_tech_area_counts

# %% [markdown]
# ## Generate time series
#

# %%
# Create time series showing the number of articles per category
ts_category = utils.get_ts(df_id_to_term, guardian_baseline, "Category")
# Create time series showing the number of articles per subcategory
ts_subcategory = utils.get_ts(df_id_to_term, guardian_baseline, "Sub Category").merge(
    df_search_terms[["Category", "Sub Category"]].drop_duplicates("Sub Category"),
    how="left",
)
# Create time series showing the number of articles per tech area
ts_tech_area = utils.get_ts(df_id_to_term, guardian_baseline, "Tech area").merge(
    df_search_terms[["Category", "Sub Category", "Tech area"]].drop_duplicates(
        "Tech area"
    ),
    how="left",
)

# %%
# Check category time series
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line()
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
# Check subcategory time series
alt.Chart(ts_subcategory, width=200, height=100).mark_line(size=3).encode(
    x="year:O",
    y=alt.Y(
        "counts:Q",
    ),
    color="Sub Category:N",
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Sub Category", "counts", "year"],
).resolve_scale(y="independent")

# %% [markdown]
# ## Magnitude and growth trends: Major categories

# %% [markdown]
# ### Variable = Proportion of articles

# %%
importlib.reload(utils)

# %%
# Produce magnitude and growth tables
magnitude_growth_df = utils.get_magnitude_growth(
    ts_category, "fraction", "Category"
).assign(
    # Report percentages
    magnitude=lambda df: df.magnitude
    * 100
)
magnitude_growth_df

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=0.45,
    y_limit=4,
    mid_point=magnitude_growth_df.magnitude.median(),
    baseline_growth=0,
    values_label="Proportion of articles (%)",
    text_column="Category",
)
fig

# %%
AltairSaver.save(
    fig,
    f"Guardian_{VERSION_NAME}_magnitude_growth_proportion",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# #### Export major category trends

# %%
# Export trends
(
    magnitude_growth_df.to_csv(
        PROJECT_DIR / f"outputs/foodtech/trends/guardian_{VERSION_NAME}_Category.csv",
        index=False,
    )
)

# %% [markdown]
# ### Variable = Number of articles

# %%
baseline_magnitude_growth = au.estimate_magnitude_growth(guardian_baseline, 2017, 2021)
baseline_growth = (
    baseline_magnitude_growth.query("trend == 'growth'").iloc[0].counts / 100
)
baseline_growth

# %%
magnitude_growth_counts_df = utils.get_magnitude_growth(
    ts_category, "counts", "Category"
)

# %%
magnitude_growth_counts_df

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_counts_df,
    x_limit=400,
    y_limit=3,
    mid_point=magnitude_growth_counts_df.magnitude.median(),
    baseline_growth=-0.25,
    values_label="Average number of articles",
    text_column="Category",
)
fig

# %% [markdown]
# ## Subcategory trends

# %%
magnitude_growth_df_subcategory = utils.get_magnitude_growth(
    ts_subcategory, "fraction", "Sub Category"
).assign(
    # Report percentages
    magnitude=lambda df: df.magnitude
    * 100
)

# %%
magnitude_growth_df_subcategory

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df_subcategory,
    x_limit=0.12,
    y_limit=7.5,
    mid_point=magnitude_growth_df_subcategory.magnitude.median(),
    baseline_growth=0,
    values_label="Average number of articles",
    text_column="Sub Category",
)
fig.interactive()

# %% [markdown]
# ### Export the trends tables

# %%
magnitude_growth_df

# %%
magnitude_growth_df_subcategory

# %%
(
    magnitude_growth_df_subcategory.merge(
        df_search_terms[["Category", "Sub Category"]].drop_duplicates(), how="left"
    ).to_csv(
        PROJECT_DIR
        / f"outputs/foodtech/trends/guardian_{VERSION_NAME}_SubCategory.csv",
        index=False,
    )
)

# %%
trends_combined = (
    pd.concat(
        [
            magnitude_growth_df,
            magnitude_growth_df_subcategory.merge(
                df_search_terms[["Category", "Sub Category"]].drop_duplicates(),
                how="left",
            ),
        ]
    )
    .fillna("n/a (category level)")
    .sort_values(["Category", "Sub Category"])
)
trends_combined.to_csv(
    PROJECT_DIR / f"outputs/foodtech/trends/guardian_{VERSION_NAME}_all.csv",
    index=False,
)


# %% [markdown]
# ## Time series charts

# %%
cats = ["Reformulation", "Alt protein"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

scale = "linear"

fig = (
    alt.Chart(ts_df.query("year >= 2000"))
    .mark_line(size=3, interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"Guardian_{VERSION_NAME}_articles_per_year_InnovativeFood",
    filetypes=["html", "svg", "png"],
)

# %%
cats = [
    "Delivery",
    # "Meal kits",
    # "Supply chain",
    # "Personalised nutrition",
    # "Restaurants",
    # "Retail",
]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        tooltip=["year", "counts"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"Guardian_{VERSION_NAME}_articles_per_year_Delivery",
    filetypes=["html", "svg", "png"],
)

# %%
cats = ["Food waste"]
ts_df = ts_category.query("`Category` in @cats")

# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Category:N"),
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"Guardian_{VERSION_NAME}_articles_per_year_Food_waste",
    filetypes=["html", "svg", "png"],
)

# %%
cats = ["Health"]
ts_df = ts_category.query("`Category` in @cats")

# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"Guardian_{VERSION_NAME}_articles_per_year_Health",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ## Checking technology and obesity overlaps
#
# Check whether articles that mention obesity terms also mention technologies

# %%
# Fetch articles between 2017 and 2021
df = df_id_to_term.query("year > 2016 and year < 2022")
# Fetch articles that mention obesity and overweight terms
obesity_ids = set(df[df.Terms.isin(["obesity", "obese", "overweight"])].id.to_list())
# Fetch articles that are not in the health category, and hence are about other food technologies
not_health_ids = set(df[df.Category.isin(["Health"]) == False].id.to_list())

# %%
len(obesity_ids)

# %%
len(not_health_ids)

# %%
# Articles that mention obesity terms but are not explicitly about health
len(not_health_ids.intersection(obesity_ids)) / len(not_health_ids)

# %%
# Articles among the obesity term articles that also mention technologies
len(not_health_ids.intersection(obesity_ids)) / len(obesity_ids)

# %%
# Check the overlap articles
overlap_ids = not_health_ids.intersection(obesity_ids)
len(overlap_ids)

# %%
df_id_to_term.query("id in @overlap_ids").groupby(["Category", "Sub Category"]).agg(
    counts=("id", "count")
)

# %%
df_id_to_term.query("id in @overlap_ids").iloc[0].URL

# %% [markdown]
# ## Checking articles about alternative proteins
#
# Check what types of alternative proteins are mentioned

# %%
n_alt_protein = len(
    df_id_to_term.query("year > 2016 and year < 2022")
    .query("`Sub Category` == 'Alt protein'")
    .drop_duplicates("id")
)
n_alt_protein

# %%
# Fraction of Alternative protein articles mentioning 'lab meat'
df_tech_area_counts.query("`Tech area` == 'Lab meat'").counts.iloc[0] / n_alt_protein

# %%
# Fraction of Alternative protein articles mentioning 'plant-based meat and protein'
df_tech_area_counts.query("`Tech area` == 'Plant-based'").counts.iloc[0] / n_alt_protein

# %%
# Fraction of Alternative protein articles mentioning 'fermentation' technologies
df_tech_area_counts.query("`Tech area` == 'Fermentation'").counts.iloc[
    0
] / n_alt_protein
