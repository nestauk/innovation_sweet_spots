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
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis of the Guardian articles

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

figure_folder = alt_save.FIGURE_PATH + "/foodtech"
# Folder for data tables
tables_folder = figure_folder + "/tables"

# %%
importlib.reload(utils);

# %% [markdown]
# ## Helper functions

# %%
from typing import Iterable
from innovation_sweet_spots.utils import plotting_utils as pu
import innovation_sweet_spots.utils.google_sheets as gs

def export_chart(data_: pd.DataFrame, cats: Iterable[str], chart_number: str, chart_name: str, fig, category_column: str="Sub Category"):
    """ Prepares table for plotting with Flourish, saves it locally and on Google Sheets, and exports altair plot """
    # Prepare the table
    df = pu.prepare_ts_table_for_flourish(
        data_,
        cats,
        category_column,
        max_complete_year=2021,
        values_column="percentage",
        values_label="Percentage of articles",
    )
    #  Upload the prepared table to google sheet
    gs.upload_to_google_sheet(
        df,
        google_sheet_id=utils.REPORT_TABLES_SHEET,
        wks_name=chart_number,
        overwrite=True,
    )
    # Export the chart
    AltairSaver.save(
        fig, chart_name, filetypes=["html", "svg", "png"]
    )
    pu.export_table(df, chart_name, tables_folder)


# %% [markdown]
# ### Loading data

# %%
# Loading the identified and partially reviewed list of relevant Guardian articles
df_id_to_term_reviewed = google_sheets.get_foodtech_guardian(from_local=False)


# %%
df_id_to_term_reviewed

# %%
# Fetch list of search terms from the google sheet
df_search_terms = get_foodtech_search_terms(from_local=False).assign(
    Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma)
)

# %%
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
df_counts = (
    df_id_to_term.query("year > 2016 and year < 2022")
    .groupby(["Category", "Sub Category"], as_index=False)
    .agg(counts=("id", "count"))
)
df_counts

# %%
# Note:Empty rows in for 'tech area' indicate manually added articles
df_counts = (
    df_id_to_term.query("year > 2016 and year < 2022")
    .groupby(["Category", "Sub Category", "Tech area"], as_index=False)
    .agg(counts=("id", "count"))
)
df_counts

# %% [markdown]
# ### Check articles about alternative proteins

# %%
n_alt_protein = len(
    df_id_to_term.query("year > 2016 and year < 2022")
    .query("`Sub Category` == 'Alt protein'")
    .drop_duplicates("id")
)
n_alt_protein

# %%
df_counts.query("`Tech area` == 'Lab meat'").counts.iloc[0] / n_alt_protein

# %%
df_counts.query("`Tech area` == 'Plant-based'").counts.iloc[0] / n_alt_protein

# %%
df_counts.query("`Tech area` == 'Fermentation'").counts.iloc[0] / n_alt_protein

# %% [markdown]
# ## Checking technology trends
# - Combining terms by categories
# - Time series with categories and subcategories

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
ts_tech_area = get_ts(df_id_to_term, "Tech area").merge(
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

# %%
categories_to_check = ts_category.Category.unique()
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
    magnitude_growth, columns=["magnitude", "growth", "Category"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
domain = [
    "Health",
    "Innovative food",
    "Logistics and delivery",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
]
range_ = pu.NESTA_COLOURS[0 : len(domain)]

# %%
chart_trends.estimate_trend_type(
    magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
)

# %%
(
    chart_trends.estimate_trend_type(
        magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
    ).to_csv(
        PROJECT_DIR / f"outputs/foodtech/trends/guardian_{VERSION_NAME}_Category.csv",
        index=False,
    )
)


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
    text_column="Category",
)
fig

# %%

magnitude_growth_df.magnitude.describe(percentiles=[0.25, 0.5, 0.75])

# %%
# Baseline
baseline_rule = (
    alt.Chart(pd.DataFrame({"x": [0.092123]}))
    .mark_rule(strokeDash=[5, 7], size=1, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

baseline_rule_1= (
    alt.Chart(pd.DataFrame({"x": [0.045914]}))
    .mark_rule(strokeDash=[2, 2], size=.5, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

baseline_rule_2 = (
    alt.Chart(pd.DataFrame({"x": [0.127115]}))
    .mark_rule(strokeDash=[2, 2], size=.5, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

fig_guidelines = fig + baseline_rule + baseline_rule_1 + baseline_rule_2
fig_guidelines

# %%
# %%
AltairSaver.save(
    fig_guidelines, f"Ch5_Fig39_magnitude_growth_news", filetypes=["html", "svg", "png"]
)

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
magnitude_growth_df.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=400,
    y_limit=3,
    mid_point=72,
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
# AltairSaver.save(
#     fig, f"Guardian_articles_magnitude_growth", filetypes=["html", "svg", "png"]
# )

# %%
ts_subcategory.head(1)

# %% [markdown]
# ### Time series

# %%
cats = ["Reformulation", "Alt protein"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

data_ = ts_df.query("year >= 2000")
fig = (
    alt.Chart(data_).mark_line(size=3, interpolate="monotone")
    # .mark_bar(size=5, interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Percentage of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Name the chart
chart_number = "Ch5-Fig41"
chart_name = f"{chart_number}_ts_subcategory_Innovative_food"
export_chart(data_.assign(percentage = lambda df: df.fraction * 100), cats, chart_number, chart_name, fig)


# %%
cats = [
    "Delivery",
    "Meal kits",
    "Supply chain",
    "Personalised nutrition",
    "Restaurants",
    "Retail",
]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

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
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
cats = ["Delivery"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# scale = 'log'
scale = "linear"
data_ = ts_df.query("year >= 2000")

fig = (
    alt.Chart(data_)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Name the chart
chart_number = "Ch5-Fig40"
chart_name = f"{chart_number}_ts_subcategory_Delivery"
export_chart(data_.assign(percentage = lambda df: df.fraction * 100), cats, chart_number, chart_name, fig)


# %%
cats = ["Restaurants", "Retail"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

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
cats = ["Kitchen tech", "Dark kitchen"]
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
cats = ["Food waste"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# scale = 'log'
scale = "linear"
data_ = ts_df.query("year >= 2000")

fig = (
    alt.Chart(data_)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Name the chart
chart_number = "Ch5-Fig42"
chart_name = f"{chart_number}_ts_subcategory_Food_waste"
export_chart(data_.assign(percentage = lambda df: df.fraction * 100), cats, chart_number, chart_name, fig)


# %%
cats = ["Health"]
ts_df = ts_category.query("`Category` in @cats")

# scale = 'log'
scale = "linear"
data_ = ts_df.query("year >= 2000")
fig = (
    alt.Chart(data_)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q", title="Proportion of articles", axis=alt.Axis(format=".2%")
        ),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Name the chart
chart_number = "Ch5-Fig44"
chart_name = f"{chart_number}_ts_subcategory_Health"
export_chart(data_.assign(percentage = lambda df: df.fraction * 100), cats, chart_number, chart_name, fig)


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
        y=alt.Y("fraction:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
cats = ["Meal kits", "Supply chain"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

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
categories_to_check = ts_subcategory["Sub Category"].unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = ts_subcategory.query("`Sub Category` == @tech_area").drop("Category", axis=1)[
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
magnitude_growth_df_subcategory = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "Sub Category"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
magnitude_growth_df_subcategory

# %%
(
    chart_trends.estimate_trend_type(
        magnitude_growth_df_subcategory.merge(
            df_search_terms[["Category", "Sub Category"]].drop_duplicates(), how="left"
        ),
        magnitude_column="magnitude",
        growth_column="growth",
    ).to_csv(
        PROJECT_DIR
        / f"outputs/foodtech/trends/guardian_{VERSION_NAME}_SubCategory.csv",
        index=False,
    )
)


# %%
magnitude_growth_df_subcategory.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df_subcategory,
    x_limit=0.12,
    y_limit=7.5,
    mid_point=0.0183,
    baseline_growth=0,
    values_label="Average number of articles",
    text_column="Sub Category",
)
fig.interactive()

# %%
magnitude_growth_df

# %%
magnitude_growth_df_subcategory

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


# %%
# categories_to_check = ts_tech_area['Tech area'].unique()
# variable = "fraction"
# magnitude_growth = []
# for tech_area in categories_to_check:
#     print(tech_area)
#     try:
#         df = ts_tech_area.query("`Tech area` == @tech_area").drop("Category", axis=1)[
#             ["year", variable]
#         ]
#         df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
#         magnitude_growth.append(
#             [
#                 df_trends.query('trend == "magnitude"').iloc[0][variable],
#                 df_trends.query('trend == "growth"').iloc[0][variable],
#                 tech_area,
#             ]
#         )
#     except:
#         pass
# magnitude_growth_df = pd.DataFrame(
#     magnitude_growth, columns=["magnitude", "growth", "tech_area"]
# ).assign(
#     growth=lambda df: df.growth / 100,
#     magnitude=lambda df: df.magnitude * 100,
# )

# %%
# magnitude_growth_df.sort_values('magnitude')

# %%
df = df_id_to_term.query("year > 2016 and year < 2022")
health_ids = set(df[df.Terms.isin(["obesity", "obese", "overweight"])].id.to_list())
not_health_ids = set(df[df.Category.isin(["Health"]) == False].id.to_list())

# %%
len(health_ids)

# %%
len(not_health_ids)

# %%
len(not_health_ids.intersection(health_ids)) / len(not_health_ids)

# %%
len(not_health_ids.intersection(health_ids)) / len(health_ids)

# %%
len(not_health_ids.intersection(health_ids))

# %%
len(not_health_ids)

# %%
len(not_health_ids.intersection(health_ids)) / len(not_health_ids)

# %%
overlap_ids = not_health_ids.intersection(health_ids)

# %%
df_id_to_term.query("id in @overlap_ids").groupby(["Category", "Sub Category"]).agg(
    counts=("id", "count")
)

# %%
df_id_to_term.query("id in @overlap_ids").iloc[0].URL

# %%
df_id_to_term

# %%
dd
