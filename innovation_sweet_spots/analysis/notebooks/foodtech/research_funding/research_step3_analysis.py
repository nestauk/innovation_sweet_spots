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
# # Food tech: Research funding trends
# ## Step 3: Analysis and charts
#
# - Data from UKRI's Gateway to Research and NIHR Open Data platform
# - Keyword search results (step 1) have been reviewed and a final table of projects has been processed and saved (step 2)
# - This notebook (step 3) uses the final reviewed project table to produce charts

# %% [markdown]
# ### Import dependencies

# %%
from innovation_sweet_spots.getters import google_sheets
from innovation_sweet_spots.getters import gtr_2022 as gtr
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.utils import chart_trends
import pandas as pd
import utils

pd.options.display.float_format = "{:.3f}".format
import importlib

# %%
import utils

# %% [markdown]
# ### Plotting utils

# %%
# Utils for making and saving charts
import altair as alt
import innovation_sweet_spots.utils.altair_save_utils as alt_save
from innovation_sweet_spots.utils import plotting_utils as pu
from pathlib import Path
import altair

figure_folder = Path(alt_save.FIGURE_PATH + "/foodtech")
AltairSaver = alt_save.AltairSaver(path=figure_folder)
# Note this will open a chrome window, leave it open

# %%
# Figure version name
fig_version_name = "Report_GTR_NIHR"
# Folder for data tables
tables_folder = figure_folder / "tables/"
tables_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Load data

# %%
# Data folder
data_dir = PROJECT_DIR / "outputs/foodtech/research_funding/"
# Taxonomy of innovation categories
taxonomy_df = pd.read_csv(data_dir / "research_funding_tech_taxonomy.csv")
# Reviewed final table of research projects
research_project_funding = pd.read_csv(data_dir / "research_funding_projects.csv")

# %%
# Check number of projects
len(research_project_funding)

# %%
# Get GtR project data
gtr_df = gtr.get_gtr_projects()
# Get GtR project data with accurate funds
gtr_projects = gtr.get_wrangled_projects()

# %%
# Get NIHR project data
NIHR_DIR = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
nihr_df = pd.read_csv(NIHR_DIR)

# %% [markdown]
# ## Baseline funding growth
#
# Estimating baseline funding growth of all projects combined. This will be used as a reference to compare other growth figures.

# %%
# Relevant columns
cols = ["project_id", "title", "description", "amount", "start_date", "funder"]

# Prepare table with all UKRI projects' funding
ukri_df_ref = (
    gtr_projects.merge(
        gtr_df[["id", "abstractText"]], left_on="project_id", right_on="id", how="left"
    )
    .rename(columns={"fund_start": "start_date", "abstractText": "description"})
    .assign(funder="ukri")
)[cols]

# Prepare table with all NIHR projects' funding
nihr_df_ref = (
    nihr_df.assign(
        amount=lambda df: df.award_amount_m.astype(float) * 1e6,
        project_id=lambda df: df.recordid,
        funder="nihr",
    ).rename(columns={"project_title": "title", "scientific_abstract": "description"})
)[cols]

# Combine UKRI and NIHR funding data
funding_ref = pd.concat([nihr_df_ref, ukri_df_ref], ignore_index=True)

# %%
# Check total number of projects
len(funding_ref)

# %%
# Deal with duplicate projects between NIHR and UKRI by
# not allowing projects in NIHR and UKRI with the exact same title
# and keeping the project with larger funding amount
project_ids_to_remove = []
for i, row in ukri_df_ref.merge(nihr_df_ref, on="title").iterrows():
    if row.amount_x > row.amount_y:
        project_ids_to_remove.append(row.project_id_y)
    else:
        project_ids_to_remove.append(row.project_id_x)

# %%
# Remove duplicate projects
funding_ref = funding_ref.query("project_id not in @project_ids_to_remove")

# %%
funding_ref

# %%
# Calculate reference time series
df_reference = au.gtr_get_all_timeseries_period(
    funding_ref,  # .query("funder == 'nihr'"),
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)

# %%
df_reference

# %%
# Estimate the magnitude and growth of baseline funding
baseline = au.estimate_magnitude_growth(
    (
        df_reference.assign(year=lambda df: df.time_period.dt.year).drop(
            "time_period", axis=1
        )
    ),
    2017,
    2021,
)
baseline

# %%
# Calculate baseline growth percentage to be used later
baseline_growth_decimal = (
    float(baseline.query("trend == 'growth'")["amount_total"]) / 100
)
baseline_growth_decimal


# %% [markdown]
# The total amount of funding has grown by about 11.4% between 2017 and 2021

# %% [markdown]
# ## Total food tech and obesity research funding growth
#
# Combined growth all relevant projects

# %%
# Function for plotting funding over time
def funding_over_time_chart(
    data: pd.DataFrame,
    vertical_axis_values: str,
    vertical_axis_label: str,
    horizontal_axis_values: str,
    horizontal_axis_label: str,
    horizontal_tooltip_label: str,
    y_scale_upper: int,
) -> altair.Chart:
    return (
        alt.Chart(data, width=400, height=250)
        .mark_bar(color=pu.NESTA_COLOURS[0])
        .encode(
            x=alt.X(f"{horizontal_axis_values}:O", title=horizontal_axis_label),
            y=alt.Y(
                f"{vertical_axis_values}:Q",
                title=vertical_axis_label,
                scale=alt.Scale(domain=(0, y_scale_upper)),
            ),
            tooltip=[
                alt.Tooltip(
                    f"{horizontal_axis_values}:O", title=horizontal_tooltip_label
                ),
                alt.Tooltip(
                    f"{vertical_axis_values}:Q", title=vertical_axis_label, format=".3f"
                ),
            ],
        )
    )


# Define chart variables
vertical_axis_values = "amount_total"
vertical_axis_label = "Research funding (£ millions)"
horizontal_axis_values = "year"
horizontal_axis_label = ""
horizontal_tooltip_label = "Year"

# %%
# Calculate time series
ts_total = au.gtr_get_all_timeseries_period(
    research_project_funding.drop_duplicates("project_id"),
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)

# Prepare chart data
data = (
    ts_total.copy()
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(amount_total=lambda df: df.amount_total / 1e3)
    .query("year < 2022")
)

fig = funding_over_time_chart(
    data,
    vertical_axis_values,
    vertical_axis_label,
    horizontal_axis_values,
    horizontal_axis_label,
    horizontal_tooltip_label,
    y_scale_upper=120,
)

fig = pu.configure_plots(fig, chart_title="Food tech research funding over time")
fig

# %%
chart_name = f"v{fig_version_name}_total_funding_per_year"
AltairSaver.save(fig, chart_name, filetypes=["html", "svg", "png"])
data.to_csv(tables_folder / f"{chart_name}.csv", index=False)

# %%
# Magnitude and growth of the total funding
au.estimate_magnitude_growth(
    (data.assign(year=lambda df: df.time_period.dt.year).drop("time_period", axis=1)),
    2017,
    2021,
)

# %% [markdown]
# Total funding has grown by about 51.6% ~ 52% between 2017 and 2021

# %% [markdown]
# ### Total food tech research funding growth (excluding 'health')

# %%
# Calculate time series
ts_total = au.gtr_get_all_timeseries_period(
    research_project_funding.drop_duplicates("project_id").query('Category!="Health"'),
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)

# Prepare chart data
data = (
    ts_total.copy()
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(amount_total=lambda df: df.amount_total / 1e3)
    .query("year < 2022")
)

fig = funding_over_time_chart(
    data,
    vertical_axis_values,
    vertical_axis_label,
    horizontal_axis_values,
    horizontal_axis_label,
    horizontal_tooltip_label,
    y_scale_upper=30,
)

fig = pu.configure_plots(
    fig, chart_title="Food tech (excl. 'health') research funding over time"
)
fig

# %%
au.estimate_magnitude_growth(
    (data.assign(year=lambda df: df.time_period.dt.year).drop("time_period", axis=1)),
    2017,
    2021,
)

# %% [markdown]
# Total funding has grown by about 51.6% ~ 52% between 2017 and 2021. Excluding health that growth increases to 70% suggesting health is growing slower than the rest of the food tech sector.

# %%
au.percentage_change(
    data.query("`year`==2017")["amount_total"].iloc[0],
    data.query("`year`==2021")["amount_total"].iloc[0],
)

# %%
chart_name = f"v{fig_version_name}_total_funding_per_year_wout_health"
AltairSaver.save(fig, chart_name, filetypes=["html", "svg", "png"])
data.to_csv(tables_folder / f"{chart_name}.csv", index=False)

# %% [markdown]
# ## Funders
#
# Checking projects funded by Innovate UK and other research councils

# %%
# Get leading funder names
df_funders = (
    research_project_funding
    # merge funders from GtR data (NIHR does not indicate names of non-NIHR funders)
    .merge(
        gtr_df[["id", "leadFunder"]], left_on="project_id", right_on="id", how="left"
    )
    # 2017-2021
    .query('start_date > "2016-12-31" and start_date < "2022-01-01"').query(
        "Category != 'Health'"
    )
)

# %%
# Total non-health funding
total_funding = (
    df_funders.query("Category != 'Health'").drop_duplicates("project_id").amount.sum()
)

# %%
# Check Innovate UK fraction of total non-health fundnig
df_funders_sums = (
    df_funders.query("Category != 'Health'")
    .drop_duplicates("project_id")
    .groupby("leadFunder")
    .sum()
)
df_funders_sums / total_funding

# %% [markdown]
# Approximately 46.7 percent of non-health projects are funded by Innovate UK

# %% [markdown]
# ## Main innovation categories

# %%
research_project_funding.Category.unique()

# %%
major_categories_to_check = [
    "Health",
    "Innovative food",
    "Logistics",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
]

# %% [markdown]
# ### Major category totals

# %%
# Get total funding 2017-2021 by major innovation category (excl. Social)
category_funding_df = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    # Remove 'social' category, which is not discussed in detail in report
    .query("Category != 'Social'")
    #  Remove duplicates
    .drop_duplicates(["project_id", "Category"])
    .groupby(["Category"])
    .agg(amount_total=("amount", "sum"), counts=("project_id", "count"))
    .assign(amount_total=lambda df: df.amount_total / 1e6)
    .reset_index()
)

# %%
category_funding_df

# %%
# Get total funding 2017-2021 for all categories
total_funding = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .drop_duplicates(["project_id"])
    .amount.sum()
) / 1e6
print(f"Total funding is {total_funding.round(2)} million")

# %%
# Calculate percentage of health related projects
category_funding_df.query("Category == 'Health'").amount_total / total_funding

# %% [markdown]
# Health-related projects correspond to approximately 66% of total funding 2017-2021

# %%
# Plot research funding and number of projects for each category
order = category_funding_df.sort_values(
    "amount_total", ascending=False
).Category.to_list()

fig_1 = (
    alt.Chart(category_funding_df, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title=""),
        x=alt.X("amount_total", title="Research funding (£ millions)"),
    )
)

fig_2 = (
    alt.Chart(category_funding_df, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title="", axis=alt.Axis(labels=False)),
        x=alt.X("counts", title="Number of research projects"),
    )
)

final_fig = pu.configure_plots(fig_1)

# %%
final_fig

# %%
AltairSaver.save(
    final_fig,
    f"v{fig_version_name}_2017_2021_sum_major",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ### Major category trends

# %%
category_ts = utils.get_time_series(
    research_project_funding,
    major_categories_to_check,
    taxonomy_level="Category",
    id_column="project_id",
)
category_magnitude_growth = utils.get_magnitude_vs_growth(
    category_ts, major_categories_to_check, verbose=True
)
category_amount_magnitude_growth = utils.get_magnitude_vs_growth_plot(
    category_magnitude_growth, "amount_total"
)

# %%
chart_trends.estimate_trend_type(
    category_amount_magnitude_growth, magnitude_column="magnitude"
)

# %%
au.moving_average(
    category_ts.query('Category == "Innovative food"').assign(
        year=lambda df: df.time_period.dt.year
    )
)

# %% [markdown]
# ### Major category trends chart

# %%
category_amount_magnitude_growth

# %%
# In domain list, health is last item in the list so that
# if domain_excl_health is used instead of domain
# the colours will remain the same.
domain = [
    "Innovative food",
    "Logistics",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
    "Health",
]
range_ = pu.NESTA_COLOURS[0 : len(domain)]

# %%
mid_point = category_amount_magnitude_growth.magnitude.median()

# %%
chart_trends._epsilon = 0.075
domain_excl_health = domain[:-1]
range_excl_health = pu.NESTA_COLOURS[0 : len(domain_excl_health)]

# %%
baseline_growth_decimal

# %%
# Plot major categories magnitude vs. growth chart
fig = chart_trends.mangitude_vs_growth_chart(
    category_amount_magnitude_growth,
    x_limit=45,
    y_limit=2.5,
    mid_point=mid_point,
    baseline_growth=baseline_growth_decimal,
    values_label="Average new funding per year (£ millions)",
    text_column="Category",
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_major_magnitude_vs_growth_colour",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot major categories magnitude vs. growth chart without Health category
fig = chart_trends.mangitude_vs_growth_chart(
    category_amount_magnitude_growth.query("Category != 'Health'"),
    x_limit=10,
    y_limit=2.5,
    mid_point=mid_point,
    baseline_growth=baseline_growth_decimal,
    values_label="Average new funding per year (£ millions)",
    text_column="Category",
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_major_magnitude_vs_growth_colour_woutHealth",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ### Time series

# %%
# Plot major projects timeseries without health
fig = (
    alt.Chart(
        (
            category_ts.assign(year=lambda df: df.time_period.dt.year).query(
                "Category != 'Health'"
            )
        ),
        width=400,
    )
    .mark_line(interpolate="monotone", size=3)
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("no_of_projects:Q", title="Number of new projects"),
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(domain=domain_excl_health, range=range_excl_health),
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_major_ts_projects_without_health",
    filetypes=["html", "svg", "png"],
)


# %%
def chart_funding_ts(
    category_ts,
    domain,
    colours,
    taxonomy_level="Category",
    excluded_categories=["Health"],
):
    fig1 = (
        alt.Chart(
            (
                category_ts.assign(year=lambda df: df.time_period.dt.year)
                .assign(amount_total=lambda df: df.amount_total / 1e3)
                .query(f"`{taxonomy_level}` not in @excluded_categories")
                .query("year < 2022")
            ),
            width=400,
        )
        .mark_line(
            interpolate="monotone",
            size=3,
            # interpolate='cardinal',
        )
        .encode(
            x=alt.X(
                "year:O", scale=alt.Scale(domain=list(range(2010, 2022))), title=""
            ),
            y=alt.Y("amount_total:Q", title="Research funding (million GBP)"),
            color=alt.Color(
                f"{taxonomy_level}:N",
                title="Category",
                scale=alt.Scale(domain=domain, range=colours),
                # legend=alt.Legend(orient="top", columns=2),
            ),
            tooltip=["year", "amount_total"],
        )
    )
    return pu.configure_plots(fig1)


# %%
# Plot major funding timeseries without health
fig_final = chart_funding_ts(
    category_ts, domain=domain_excl_health, colours=range_excl_health
)
fig_final

# %%
AltairSaver.save(
    fig_final,
    f"v{fig_version_name}_major_ts_funding_without_health",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot major funding timeseries with health
fig_final = chart_funding_ts(
    category_ts, excluded_categories=[], domain=domain, colours=range_
)
fig_final

# %%
AltairSaver.save(
    fig_final, f"v{fig_version_name}_major_ts_funding", filetypes=["html", "svg", "png"]
)

# %% [markdown]
# ## Subcategory trends

# %%
# Calculate number of projects for each sub category
yearly_projects_minor = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .assign(year=lambda df: df.start_date.apply(lambda x: x[0:4]))
    .groupby(["Sub Category"])
    .agg(counts=("project_id", "count"))
    .reset_index()
)

yearly_projects_minor.sort_values("counts", ascending=False)

# %%
categories_to_check = [
    "Biomedical",
    "Alt protein",
    "Supply chain",
    "Retail",
    "Dark kitchen",
    "Meal kits",
    "Kitchen tech",
    "Dietary supplements",
    "Diet",
    "Delivery",
    "Packaging",
    "Personalised nutrition",
    "Restaurants",
    "Reformulation",
    "Innovative food (other)",
    "Waste reduction",
]

# %%
# Calculate research funding timeseries data for the subcategories
subcategory_ts = utils.get_time_series(
    research_project_funding,
    categories_to_check,
    taxonomy_level="Sub Category",
    id_column="project_id",
).merge(taxonomy_df, how="left")

# Calculate research funding magnitude and growth for the subcategories
subcategory_magnitude_growth = utils.get_magnitude_vs_growth(
    subcategory_ts, categories_to_check, taxonomy_level="Sub Category", verbose=True
).merge(taxonomy_df, how="left")

# Select research funding amounts magnitude and growth for the subcategories
subcategory_amount_magnitude_growth = utils.get_magnitude_vs_growth_plot(
    subcategory_magnitude_growth, "amount_total"
).merge(taxonomy_df, how="left")

# %%
subcategory_amount_magnitude_growth

# %%
(
    chart_trends.estimate_trend_type(
        subcategory_amount_magnitude_growth, magnitude_column="magnitude"
    ).to_csv(
        PROJECT_DIR
        / f"outputs/foodtech/trends/research_{fig_version_name}_SubCategories.csv",
        index=False,
    )
)

# %%
subcategory_ts_2022 = utils.get_time_series(
    research_project_funding,
    categories_to_check,
    taxonomy_level="Sub Category",
    id_column="project_id",
    max_year=2022,
).merge(taxonomy_df, how="left")


# %% [markdown]
# ## Growth plots

# %%
major_sort_order = category_amount_magnitude_growth.sort_values("growth")[
    "Category"
].to_list()

data = subcategory_amount_magnitude_growth.merge(taxonomy_df, how="left")
data["Category"] = pd.Categorical(data["Category"], categories=major_sort_order)
data = data.sort_values(["Category", "growth"], ascending=False)
data = data.merge(yearly_projects_minor, how="left")

# %%
# Plot subcategory growth
colour_field = "Category"
text_field = "Sub Category"
height = 500

fig = (
    alt.Chart(data, width=500, height=height)
    .mark_circle(color=pu.NESTA_COLOURS[0], opacity=0.7, size=40)
    .encode(
        x=alt.X(
            "growth:Q",
            axis=alt.Axis(
                format="%",
                title="Growth",
                labelAlign="center",
                labelExpr="datum.value < -1 ? null : datum.label",
                tickCount=6,
            ),
        ),
        y=alt.Y("Sub Category:N", sort=data["Sub Category"].to_list(), axis=None),
        size=alt.Size(
            "magnitude",
            title="Avg yearly funding (£ million)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(range=[50, 500]),
        ),
        color=alt.Color(
            colour_field,
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=domain, range=range_),
        ),
        tooltip=["growth", "magnitude", "Sub Category"],
    )
)

text = (
    alt.Chart(data)
    .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=14)
    .encode(
        text=text_field,
        x="growth:Q",
        y=alt.Y("Sub Category:N", sort=data["Sub Category"].to_list(), title=""),
    )
)

final_fig = pu.configure_titles(pu.configure_axes((fig + text)), "", "")
final_fig

# %%
AltairSaver.save(
    final_fig, f"v{fig_version_name}_minor_growth", filetypes=["html", "svg", "png"]
)

# %%
data.magnitude.median()

# %%
# chart_trends._epsilon = 0.075
fig = chart_trends.mangitude_vs_growth_chart(
    data,
    x_limit=6,
    y_limit=10,
    mid_point=1.2,
    baseline_growth=0.11417,
    values_label="Average new funding per year (£ millions)",
    text_column="Sub Category",
)
fig.interactive()

# %% [markdown]
# ## Time series plots

# %%
# Plot funding over time for each sub category
alt.Chart(subcategory_ts, width=200, height=100).mark_line(
    size=2.5, interpolate="monotone"
).encode(
    x="year:O",
    y=alt.Y("amount_total:Q", title="Funding (m GBP)"),
    color=alt.Color("Sub Category:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Sub Category", "amount_total", "year"],
).resolve_scale(
    y="independent"
)

# %%
# Plot funding for subcategories Delivery and Supply chain

importlib.reload(pu)

# %%
subcategory_ts_2022.head(1)

# %%
# fig = pu.ts_funding_projects(subcategory_ts, ['Delivery', 'Supply chain'], height=100)
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Delivery", "Supply chain"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig

# %%
importlib.reload(pu)
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Delivery", "Supply chain"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_ts_subcategory_Logistics",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot funding for subcategories Packaging and Waste reduction
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Waste reduction", "Packaging"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig


# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_ts_subcategory_Food_waste",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot funding for subcategories Alt protein, Innovative food and Reformulation
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Alt protein", "Innovative food (other)", "Reformulation"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig


# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_ts_subcategory_Innovative_food",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot funding for subcategories Restaurants and Retail
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Restaurants", "Retail"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig


# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_ts_subcategory_Restaurants_retail",
    filetypes=["html", "svg", "png"],
)

# %%
# Plot funding for subcategory Kitchen tech
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts_2022,
        ["Kitchen tech"],
        "amount_total",
        "Funding (£ millions)",
        height=125,
    )
)
fig


# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_ts_subcategory_Kitchen_tech",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ## Combining categories and subcategories

# %%
trends_combined = (
    pd.concat([category_amount_magnitude_growth, subcategory_amount_magnitude_growth])
    .fillna("n/a (category level)")
    .sort_values(["Category", "Sub Category"])
)
trends_combined.to_csv(
    PROJECT_DIR / f"outputs/foodtech/trends/research_{fig_version_name}_all.csv",
    index=False,
)

# %% [markdown]
# ## Checking alt protein

# %%
ukri_df_reviewed = google_sheets.get_foodtech_reviewed_gtr(from_local=True).query(
    "tech_area_checked!='-'"
)

# %%
# See number of projects and funding between 2017 and 2022 for subcategories Alt protein, Plant-based, Fermentation, Lab meat
cats = ["Alt protein", "Plant-based", "Fermentation", "Lab meat"]
(
    ukri_df_reviewed[["id", "tech_area_checked"]]
    .rename(columns={"id": "project_id"})
    .merge(gtr_projects[["project_id", "fund_start", "amount"]])
    .query("fund_start > '2016-12-31' and fund_start < '2022-01-01'")
    .query("tech_area_checked in @cats")
    .groupby("tech_area_checked")
    .agg(counts=("project_id", "count"), amount=("amount", "sum"))
    .assign(amount=lambda df: df.amount / 1e6)
)

# %%
# See number of projects and funding since 2022 for subcategories Alt protein, Plant-based, Fermentation, Lab meat
cats = ["Alt protein", "Plant-based", "Fermentation", "Lab meat"]
(
    ukri_df_reviewed[["id", "tech_area_checked"]]
    .rename(columns={"id": "project_id"})
    .merge(gtr_projects[["project_id", "fund_start", "amount"]])
    .query("fund_start >= '2022-01-01'")
    .query("tech_area_checked in @cats")
    .groupby("tech_area_checked")
    .agg(counts=("project_id", "count"), amount=("amount", "sum"))
    .assign(amount=lambda df: df.amount / 1e6)
)

# %%
# See number of projects and funding since 2017 for subcategories Alt protein, Plant-based, Fermentation, Lab meat
cats = ["Alt protein", "Plant-based", "Fermentation", "Lab meat"]
(
    ukri_df_reviewed[["id", "tech_area_checked"]]
    .rename(columns={"id": "project_id"})
    .merge(gtr_projects[["project_id", "fund_start", "title", "amount"]])
    .query("fund_start >= '2017-01-01'")
    .query("tech_area_checked in @cats")
    .groupby("tech_area_checked")
    .agg(counts=("project_id", "count"), amount=("amount", "sum"))
    .assign(amount=lambda df: df.amount / 1e6)
)

# %%
# Find projects since 2017 for subcategories Alt protein, Plant-based, Fermentation, Lab meat
cats = ["Alt protein", "Plant-based", "Fermentation", "Lab meat"]
df = (
    ukri_df_reviewed[["id", "tech_area_checked"]]
    .rename(columns={"id": "project_id"})
    .merge(gtr_projects[["project_id", "fund_start", "title", "amount"]])
    .query("fund_start >= '2017-01-01'")
    .query("tech_area_checked in @cats")
)

# %%
# View Plant-based projects
df.query("tech_area_checked == 'Plant-based'").sort_values("fund_start")

# %%
# See number of projects and funding since 2017 for subcategories Reformulation, Alt protein, Innovative food
cats = ["Reformulation", "Alt protein", "Innovative food (other)"]
(
    research_project_funding.query("start_date >= '2017-01-01'")
    .query("start_date < '2022-01-01'")
    .query("`Sub Category` in @cats")
    .groupby("Sub Category")
    .agg(counts=("project_id", "count"), amount=("amount", "sum"))
    .assign(amount=lambda df: df.amount / 1e6)
)

# %%
# See number of projects and funding since 2022 for subcategories Reformulation, Alt protein, Innovative food
cats = ["Reformulation", "Alt protein", "Innovative food (other)"]
(
    research_project_funding.query("start_date >= '2022-01-01'")
    .query("`Sub Category` in @cats")
    .groupby("Sub Category")
    .agg(counts=("project_id", "count"), amount=("amount", "sum"))
    .assign(amount=lambda df: df.amount / 1e6)
)

# %%
# View most recent Reformulation projects
research_project_funding.query("`Sub Category` == 'Reformulation'").sort_values(
    "start_date"
).tail(10)

# %%
