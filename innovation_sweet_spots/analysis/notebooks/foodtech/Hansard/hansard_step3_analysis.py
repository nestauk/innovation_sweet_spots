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
# # Food tech: Hansard parliamentary debate trends
#

# %%
from innovation_sweet_spots.getters import hansard
import altair as alt
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils import plotting_utils as pu
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.utils import chart_trends
import innovation_sweet_spots.utils.altair_save_utils as alt_save

# %%
# Plotting utils
import innovation_sweet_spots.utils.altair_save_utils as alt_save
figure_folder = alt_save.FIGURE_PATH + "/foodtech"
# Folder for data tables
tables_folder = figure_folder + "/tables"

AltairSaver = alt_save.AltairSaver(path=figure_folder)

# %% [markdown]
# ## Load Hansard speeches data

# %%
# Dataframe with parliamentary debates
df_debates = hansard.get_debates().drop_duplicates("id", keep="first")
assert len(df_debates.id.unique()) == len(df_debates)
len(df_debates)

# %% [markdown]
# ## Load Hansard query results filtered

# %%
# Hansard filtered query results from hansard_step3_query_terms notebook
hansard_query_results_filtered = pd.read_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/hansard_hits_v2022_11_22.csv"
)
len(hansard_query_results_filtered)

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Plotting setup

# %%
AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

VERSION_NAME = "Report_Hansard"

# %% [markdown]
# ## Baseline number of speeches

# %%
# Plot number of speeches per year
hansard_baseline = df_debates.groupby("year", as_index=False).agg(
    total_counts=("id", "count")
)

alt.Chart(hansard_baseline).mark_line().encode(x="year", y="total_counts")

# %% [markdown]
# ## Check number of mentions per category

# %%
# Mentions per category
(
    hansard_query_results_filtered.astype({"year": int})
    .query("year >= 2017 and year < 2022")
    .groupby(["Category"])
    .agg(counts=("id", "count"))
    .reset_index()
    .sort_values("counts", ascending=False)
)

# %%
# Mentions per category/subcategory 2017-2022
hansard_query_results_filtered.astype({"year": int}).query(
    "year >= 2017 and year < 2022"
).groupby(["Category", "Sub Category"]).agg(counts=("id", "count")).reset_index()

# %%
# Mentions per category/subcategory/tech area/term
hansard_query_results_filtered.groupby(
    ["Category", "Sub Category", "Tech area", "Terms"]
).agg(counts=("id", "count")).reset_index()

# %% [markdown]
# ## Time series charts

# %%
# Calculate speech mentions related to category as a fraction of total speeches in a year
ts_category = (
    hansard_query_results_filtered.groupby(["year", "Category"])
    .agg(counts=("id", "count"))
    .reset_index()
    .merge(hansard_baseline.astype({"year": int}), how="left", on="year")
    .assign(fraction=lambda df: df.counts / df.total_counts)
    .astype({"year": str})
)

# %%
# Plot fraction of speeches per year for all categories
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line(size=3, interpolate="monotone")
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("fraction:Q", sort="-x", scale=alt.Scale(type=scale)),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
import utils


# %%
def plot_proportion_of_speeches_over_time(
    category: str
) -> alt.vegalite.v4.api.LayerChart:
    """Plot proportion of speeches over
    time for specified category
    """
    data = (
        ts_category.copy()
        .query(f"Category == '{category}'")
        .assign(fraction=lambda df: df.fraction * 100)
    )

    fig = pu.ts_smooth_incomplete(
        data,
        [category],
        "fraction",
        "Proportion of speeches (%)",
        "Category",
        amount_div=1,
    )
    return pu.configure_plots(fig), data

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
        values_label="Percentage of speeches",
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


# %%
#  Plot food waste proportion of speeches over time
cats = ["Food waste"]
fig, data_ = plot_proportion_of_speeches_over_time(category=cats[0])
fig

# %%
# Name the chart
chart_number = "Ch5-Fig43"
chart_name = f"{chart_number}_ts_category_Food_waste"
export_chart(data_.assign(percentage = lambda df: df.fraction), cats, chart_number, chart_name, fig, "Category")


# %%
#  Plot food waste proportion of speeches over time
cats = ["Health"]
fig, data_ = plot_proportion_of_speeches_over_time(category=cats[0])
fig

# %%
# Name the chart
chart_number = "Ch5-Fig45"
chart_name = f"{chart_number}_ts_category_Health"
export_chart(data_.assign(percentage = lambda df: df.fraction), cats, chart_number, chart_name, fig, "Category")


# %%
#  Plot innovative food proportion of speeches over time
plot_proportion_of_speeches_over_time(category="Innovative food")

# %%
#  Plot logistics proportion of speeches over time
plot_proportion_of_speeches_over_time(category="Logistics")

# %% [markdown]
# ## Trends analysis

# %%
# Calculate magnitude and growth for categories
categories_to_check = ts_category.Category.unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
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
).assign(growth=lambda df: df.growth / 100, magnitude=lambda df: df.magnitude * 100)

# %%
# Add estimated trend types
magnitude_growth_df = chart_trends.estimate_trend_type(
    magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
)

# %% [markdown]
# ### Export results

# %%
# Export magnitude and growth table
magnitude_growth_df.to_csv(
    PROJECT_DIR / f"outputs/foodtech/trends/hansard_{VERSION_NAME}_Categories.csv",
    index=False,
)
