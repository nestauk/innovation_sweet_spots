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
# # UKRI and NIHR funding trends

# %%
from innovation_sweet_spots.getters import google_sheets
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
from innovation_sweet_spots.getters import gtr_2022 as gtr
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis import analysis_utils as au

import pandas as pd
import importlib
import numpy as np

# %%
pd.options.display.float_format = "{:.3f}".format

# %% [markdown]
# ## Plotting utils

# %%
# Functionality for saving charts
import altair as alt
import innovation_sweet_spots.utils.altair_save_utils as alt_save
from innovation_sweet_spots.utils import plotting_utils as pu

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
# Figure version name
fig_version_name = "September_GtrNihr"

# %% [markdown]
# # Load inputs

# %%
outputs_dir = PROJECT_DIR / "outputs/foodtech/research_funding/"
taxonomy_df = pd.read_csv(outputs_dir / "research_funding_tech_taxonomy.csv")
research_project_funding = pd.read_csv(outputs_dir / "research_funding_projects.csv")

# %%
len(research_project_funding)

# %%
gtr_df = gtr.get_gtr_projects()

# %%
NIHR_DIR = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
nihr_df = pd.read_csv(NIHR_DIR)

# %%
gtr_projects = gtr.get_wrangled_projects()

# %% [markdown]
# # Baseline funding

# %%
cols = ["project_id", "title", "description", "amount", "start_date", "funder"]

# UKRI funding
ukri_df_ref = (
    gtr_projects.merge(
        gtr_df[["id", "abstractText"]], left_on="project_id", right_on="id", how="left"
    )
    .rename(
        columns={
            "fund_start": "start_date",
            "abstractText": "description",
        }
    )
    .assign(funder="ukri")
)[cols]

# NIHR funding
nihr_df_ref = (
    nihr_df.assign(
        amount=lambda df: df.award_amount_m.astype(float) * 1e6,
        project_id=lambda df: df.recordid,
        funder="nihr",
    ).rename(
        columns={
            "project_title": "title",
            "scientific_abstract": "description",
        }
    )
)[cols]

# Combining funding data
funding_ref = pd.concat([nihr_df_ref, ukri_df_ref], ignore_index=True)

# %%
len(funding_ref)

# %%
# check which project ids to remove
project_ids_to_remove = []
for i, row in ukri_df_ref.merge(nihr_df_ref, on="title").iterrows():
    if row.amount_x > row.amount_y:
        project_ids_to_remove.append(row.project_id_y)
    else:
        project_ids_to_remove.append(row.project_id_x)

# %%
funding_ref = funding_ref.query("project_id not in @project_ids_to_remove")

# %%
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
au.estimate_magnitude_growth(
    (
        df_reference.assign(year=lambda df: df.time_period.dt.year).drop(
            "time_period", axis=1
        )
    ),
    2017,
    2021,
)

# %%
df = gtr.get_gtr_funds()

# %%
y = 2020
df_ = (
    df.query(f"start >= '{y}-01-01' and start < '{y+1}-01-01'")
    .sort_values("amount", ascending=False)
    .drop_duplicates("project_id", keep="first")
)
df_.amount.sum() / 1e6

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Total funding

# %%
vertical_axis_values = "amount_total"
vertical_axis_label = "Research funding (million GBP)"
horizontal_axis_values = "year"
horizontal_axis_label = ""
horizontal_tooltip_label = "Year"

# %%
ts_total = au.gtr_get_all_timeseries_period(
    research_project_funding.drop_duplicates("project_id"),
    # research_project_funding.query('funder=="nihr"'),
    # research_project_funding
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)

data = (
    ts_total.copy()
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(amount_total=lambda df: df.amount_total / 1e3)
    .query("year < 2022")
)

fig = (
    alt.Chart(data, width=400, height=250)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X(f"{horizontal_axis_values}:O", title=horizontal_axis_label),
        y=alt.Y(
            f"{vertical_axis_values}:Q",
            title=vertical_axis_label,
            scale=alt.Scale(domain=(0, 120)),
        ),
        tooltip=[
            alt.Tooltip(f"{horizontal_axis_values}:O", title=horizontal_tooltip_label),
            alt.Tooltip(
                f"{vertical_axis_values}:Q", title=vertical_axis_label, format=".3f"
            ),
        ],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{fig_version_name}_total_funding_per_year", filetypes=["html", "svg", "png"]
)

# %%
au.estimate_magnitude_growth(
    (data.assign(year=lambda df: df.time_period.dt.year).drop("time_period", axis=1)),
    2017,
    2021,
)

# %%
au.percentage_change(
    data.query("`year`==2017")["amount_total"].iloc[0],
    data.query("`year`==2021")["amount_total"].iloc[0],
)

# %%
# research_project_funding.query("start_date >= '2013-01-01' and start_date < '2014-01-01'").sort_values('amount',ascending=False).groupby('consolidated_category').sum()


# %% [markdown]
# ## Total funding (without Health)

# %%
ts_total = au.gtr_get_all_timeseries_period(
    research_project_funding.drop_duplicates("project_id").query('Category!="Health"'),
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)

data = (
    ts_total.copy()
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(amount_total=lambda df: df.amount_total / 1e3)
    .query("year < 2022")
)

fig = (
    alt.Chart(data, width=400, height=250)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X(f"{horizontal_axis_values}:O", title=horizontal_axis_label),
        y=alt.Y(
            f"{vertical_axis_values}:Q",
            title=vertical_axis_label,
            scale=alt.Scale(domain=(0, 30)),
        ),
        tooltip=[
            alt.Tooltip(f"{horizontal_axis_values}:O", title=horizontal_tooltip_label),
            alt.Tooltip(
                f"{vertical_axis_values}:Q", title=vertical_axis_label, format=".3f"
            ),
        ],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
au.estimate_magnitude_growth(
    (data.assign(year=lambda df: df.time_period.dt.year).drop("time_period", axis=1)),
    2017,
    2021,
)

# %%
au.percentage_change(
    data.query("`year`==2017")["amount_total"].iloc[0],
    data.query("`year`==2021")["amount_total"].iloc[0],
)

# %%
AltairSaver.save(
    fig,
    f"v{fig_version_name}_total_funding_per_year_wout_health",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ## Major categories

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
# ### Major category sums

# %%
yearly_funding_df = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .query("Category != 'Social'")
    .drop_duplicates(["project_id", "Category"])
    .assign(year=lambda df: df.start_date.apply(lambda x: x[0:4]))
    .groupby(["Category", "year"])
    .agg(amount_total=("amount", "sum"))
    .reset_index()
    .groupby(["Category"])
    .agg(amount_total=("amount_total", "mean"))
    .assign(amount_total=lambda df: df.amount_total / 1e6)
    .reset_index()
)

# %%
yearly_funding_df

# %%
(
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .query('Category == "Health"')
    .amount.sum()
) / 1e6

# %%
(
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    ).amount.sum()
) / 1e6

# %%
181.68204095999997 / 279.37967556

# %%
yearly_projects = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .query("Category != 'Social'")
    .drop_duplicates(["project_id", "Category"])
    .assign(year=lambda df: df.start_date.apply(lambda x: x[0:4]))
    .groupby(["Category", "year"])
    .agg(counts=("project_id", "count"))
    .reset_index()
    .groupby(["Category"])
    .agg(counts=("counts", "mean"))
    .reset_index()
)

yearly_projects

# %%
order = yearly_funding_df.sort_values(
    "amount_total", ascending=False
).Category.to_list()

fig_1 = (
    alt.Chart(yearly_funding_df, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title=""),
        x=alt.X("amount_total", title="Research funding (million GBP)"),
    )
)

fig_2 = (
    alt.Chart(yearly_projects, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title="", axis=alt.Axis(labels=False)),
        x=alt.X("counts", title="Number of research projects"),
    )
)

final_fig = pu.configure_plots(fig_1 | fig_2)

# %%
final_fig

# %%
AltairSaver.save(
    final_fig,
    f"v{fig_version_name}_2017_2021_average_major",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ## Major category trends

# %%
import utils

importlib.reload(utils)

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
category_amount_magnitude_growth

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
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

colour_field = "Category"
text_field = "Category"
horizontal_scale = "linear"
# horizontal_scale = "log"
horizontal_title = f"Average yearly funding (million GBP)"
legend = alt.Legend()

title_text = "Research funding trends (2017-2021)"
subtitle_text = [
    # "Data: Dealroom. Showing data on early stage deals (eg, seed and series funding)",
    # "Late stage deals, such as IPOs, acquisitions, and debt financing not included.",
]

fig = (
    alt.Chart(
        category_amount_magnitude_growth,
        width=400,
        height=400,
    )
    .mark_circle(size=80)
    .encode(
        x=alt.X(
            "magnitude:Q",
            axis=alt.Axis(
                title=horizontal_title,
                tickCount=10,
                labelFlush=False,
            ),
            scale=alt.Scale(
                type=horizontal_scale,
                domain=(0.2, 40),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(
                title="Growth",
                format="%",
                tickCount=5,
            ),
            scale=alt.Scale(
                domain=(-1, 2.5),
            ),
        ),
        color=alt.Color(
            f"{colour_field}:N",
            legend=None,
            scale=alt.Scale(domain=domain, range=range_),
        ),
        tooltip=[
            alt.Tooltip("Category", title="Category"),
            alt.Tooltip("magnitude", title=horizontal_title),
            alt.Tooltip("growth", title="Growth", format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": title_text,
            "subtitle": subtitle_text,
            "subtitleFont": pu.FONT,
            "fontSize": 15,
        },
    )
)

text = fig.mark_text(
    align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=15
).encode(text=text_field)

yrule = (
    alt.Chart(pd.DataFrame({"y": [0.11417]}))
    .mark_rule(strokeDash=[5, 7], size=1)
    .encode(y="y:Q")
)

fig_final = (
    (fig + yrule + text)
    .configure_axis(
        grid=False,
        gridDash=[5, 7],
        # gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig_final

# %%
AltairSaver.save(
    fig_final,
    f"v{fig_version_name}_major_magnitude_vs_growth",
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ### Time series

# %%
category_ts.head(1)

# %%
fig = (
    alt.Chart(
        (
            category_ts.assign(year=lambda df: df.time_period.dt.year).query(
                "Category != 'Health'"
            )
        ),
        width=400,
    )
    .mark_line(
        interpolate="monotone",
        size=3,
    )
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("no_of_projects:Q", title="Number of new projects"),
        color=alt.Color("Category:N", scale=alt.Scale(domain=domain, range=range_)),
    )
)
# fig
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
    category_ts, taxonomy_level="Category", excluded_categories=["Health"]
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
                scale=alt.Scale(domain=domain, range=range_),
                # legend=alt.Legend(orient="top", columns=2),
            ),
            tooltip=["year", "amount_total"],
        )
    )
    return pu.configure_plots(fig1)


# %%
fig_final = chart_funding_ts(category_ts)
fig_final

# %%
AltairSaver.save(
    fig_final,
    f"v{fig_version_name}_major_ts_funding_without_health",
    filetypes=["html", "svg", "png"],
)

# %%
fig_final = chart_funding_ts(category_ts, excluded_categories=[])
fig_final

# %%
AltairSaver.save(
    fig_final, f"v{fig_version_name}_major_ts_funding", filetypes=["html", "svg", "png"]
)

# %% [markdown]
# ## Sub-category trends

# %%
# research_project_funding_ = research_project_funding.copy()
# for cat in ["Fermentation", "Lab meat", "Plant-based"]:
#     research_project_funding_.loc[
#         research_project_funding_.consolidated_category == cat, "consolidated_category"
#     ] = "Alt protein"
# research_project_funding_.loc[
#     research_project_funding_.consolidated_category == "Innovative food",
#     "consolidated_category",
# ] = "Innovative food (other)"


# %%
yearly_projects_minor = (
    research_project_funding.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .assign(year=lambda df: df.start_date.apply(lambda x: x[0:4]))
    .groupby(["Sub Category"])
    .agg(counts=("project_id", "count"))
    # .reset_index()
    # .groupby(['Category'])
    # .agg(counts = ('counts', 'mean'))
    .reset_index()
)

yearly_projects_minor

# %%
categories_to_check = [
    "Biomedical",
    "Alt protein",
    "Supply chain",
    "Retail",
    # "Fermentation",
    # "Dark kitchen",
    # "Plant-based",
    # "Lab meat",
    # "Meal kits",
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
importlib.reload(utils)

# %%
subcategory_ts = utils.get_time_series(
    research_project_funding,
    categories_to_check,
    taxonomy_level="Sub Category",
    id_column="project_id",
).merge(taxonomy_df, how="left")
subcategory_magnitude_growth = utils.get_magnitude_vs_growth(
    subcategory_ts, categories_to_check, taxonomy_level="Sub Category", verbose=True
).merge(taxonomy_df, how="left")
subcategory_amount_magnitude_growth = utils.get_magnitude_vs_growth_plot(
    subcategory_magnitude_growth, "amount_total"
).merge(taxonomy_df, how="left")

# %% [markdown]
# #### Growth plots

# %%
major_sort_order = category_amount_magnitude_growth.sort_values("growth")[
    "Category"
].to_list()

data = subcategory_amount_magnitude_growth.merge(taxonomy_df, how="left")
data["Category"] = pd.Categorical(data["Category"], categories=major_sort_order)
data = data.sort_values(["Category", "growth"], ascending=False)
data = data.merge(yearly_projects_minor, how="left")

# %%
colour_field = "Category"
text_field = "Sub Category"
height = 500

fig = (
    alt.Chart(
        data,
        # (
        # data.assign(Increase=lambda df: df.growth > 0)
        # .assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
        # .assign(Magnitude=lambda df: df.magnitude / 1e3)
        # ),
        width=500,
        height=height,
    )
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
            #             scale=alt.Scale(domain=(-1, 37)),
        ),
        y=alt.Y(
            "Sub Category:N",
            sort=data["Sub Category"].to_list(),
            # axis=alt.Axis(title="", labels=False),
            axis=None,
        ),
        size=alt.Size(
            "magnitude",
            title="Avg yearly funding (£ million)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(range=[50, 500]),
            # scale=alt.Scale(domain=[0.1, 4]),
        ),
        color=alt.Color(
            colour_field,
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=domain, range=range_),
        ),
        tooltip=["growth", "magnitude", "Sub Category"],
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        # tooltip=[
        #     alt.Tooltip("Category:N", title="Category"),
        #     alt.Tooltip(
        #         "Magnitude:Q",
        #         format=",.3f",
        #         title="Average yearly investment (billion GBP)",
        #     ),
        #     "Number of companies",
        #     "Number of deals",
        #     alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        # ],
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
# fig

# %%
AltairSaver.save(
    final_fig, f"v{fig_version_name}_minor_growth", filetypes=["html", "svg", "png"]
)

# %% [markdown]
# #### Time series plots

# %%
# tech_area_ts_minor = tech_area_ts.copy().assign(
#     amount_total=lambda df: df.amount_total / 1e3
# )

# %%
subcategory_ts

# %%
alt.Chart(subcategory_ts, width=200, height=100).mark_line(
    size=2.5,
    interpolate="monotone",
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
cats = ["Reformulation", "Alt protein", "Innovative food (other)"]


# %%
fig = (
    alt.Chart(
        (
            subcategory_ts.query("`Sub Category` in @cats").assign(
                amount_total=lambda df: df.amount_total / 1e3
            )
        ),
        width=400,
    )
    .mark_line(
        interpolate="monotone",
        size=3,
    )
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("amount_total:Q", title="Funding (million GBP)"),
        color=alt.Color(
            "Sub Category:N"
        ),  # , scale=alt.Scale(domain=domain, range=range_)),
    )
)
# fig
fig = pu.configure_plots(fig)
fig

# %%
# alt.Chart(tech_area_ts_minor, width=200, height=100).mark_line(
#     size=2.5,
#     interpolate="monotone",
# ).encode(
#     x="year:O",
#     y=alt.Y(
#         "no_of_projects:Q",
#     ),
#     color=alt.Color("tech_area:N", scale=alt.Scale(scheme="dark2")),
#     facet=alt.Facet("Category:N", columns=2),
#     tooltip=["tech_area", "no_of_projects", "year"],
# ).resolve_scale(
#     y="independent"
# )

# %%
(
    research_project_funding_.query('consolidated_category == "Reformulation"')
    .query('start_date >= "2019-01-01" and start_date < "2020-01-01"')
    .sort_values("amount", ascending=False)
)[["title", "amount", "start_date"]]

# %%
