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
importlib.reload(au)

# %%
# Functionality for saving charts
import altair as alt
import innovation_sweet_spots.utils.altair_save_utils as alt_save
from innovation_sweet_spots.utils import plotting_utils as pu

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
# ukri_df_reviewed = google_sheets.get_foodtech_reviewed_gtr(from_local=False)
# ukri_df_reviewed.merge(gtr_projects[['project_id', 'amount', 'fund_start']], left_on='id', right_on='project_id', how='left').to_csv('gtr_extra_data.csv', index=False)

# %%
# Figure version name
fig_version_name = "September_GtrNihr"

# %%
gtr_df = gtr.get_gtr_projects()

# %% [markdown]
# # Process and combine GtR and NIHR documents

# %%
ukri_df_reviewed = google_sheets.get_foodtech_reviewed_gtr(from_local=False).query(
    "tech_area_checked!='-'"
)
nihr_df_reviewed = google_sheets.get_foodtech_reviewed_nihr(from_local=False).query(
    "tech_area_checked!='-'"
)

# %%
NIHR_DIR = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
nihr_df = pd.read_csv(NIHR_DIR)

# %%
gtr_projects = gtr.get_wrangled_projects()

# %%
pd.set_option("max_colwidth", 200)
tit = "Extraction and processing of Nucleotides"
gtr_projects[gtr_projects.title.str.contains(tit) == True]

# %%
ukri_df = (
    ukri_df_reviewed.drop(["amount", "fund_start"], axis=1)
    .merge(
        gtr_projects,
        left_on=["id", "title"],
        right_on=["project_id", "title"],
        how="left",
    )
    .rename(
        columns={
            "abstractText": "description",
            "fund_start": "start_date",
            "fund_end": "end_date",
            "leadOrganisationDepartment": "lead_organisation",
            "grantCategory": "programme_grant_category",
        }
    )
    .assign(funder="gtr")
)[
    [
        "project_id",
        "title",
        "description",
        "lead_organisation",
        "funder",
        "programme_grant_category",
        "amount",
        "start_date",
        "tech_area_checked",
    ]
]


# %%
nihr_df_reviewed_ = (
    nihr_df_reviewed.merge(
        nihr_df[["recordid", "contracted_organisation", "start_date", "end_date"]],
        left_on="id",
        right_on="recordid",
    )
    .rename(
        columns={
            "id": "project_id",
            "project_title": "title",
            "scientific_abstract": "description",
            "contracted_organisation": "lead_organisation",
            "programme": "programme_grant_category",
        }
    )
    .assign(amount=lambda df: df.award_amount_m.astype(float) * 1e6)
    .assign(funder="nihr")
)[
    [
        "project_id",
        "title",
        "description",
        "lead_organisation",
        "organisation_type",
        "funder",
        "programme_grant_category",
        "amount",
        "start_date",
        "tech_area_checked",
    ]
]


# %%
search_terms = get_foodtech_search_terms()
cols = ["Category", "Sub Category", "Tech area"]
taxonomy_df = search_terms.drop_duplicates(cols)[cols]
taxonomy_df = pd.concat(
    [
        taxonomy_df,
        pd.DataFrame(
            data={
                "Category": ["Innovative food", "General"],
                "Sub Category": ["Reformulation", "General"],
                "Tech area": ["Reformulation", "General"],
            }
        ),
    ],
    ignore_index=True,
)


# %%
research_project_funding = pd.concat([ukri_df, nihr_df_reviewed_]).merge(
    taxonomy_df, left_on="tech_area_checked", right_on="Tech area", how="left"
)

# %%
category_consolidation_dict = {
    "Fat": "Reformulation",
    "Sugar": "Reformulation",
    "Fiber": "Reformulation",
    "Delivery apps": "Delivery",
    "Food waste": "Waste reduction",
}

# %%
research_project_funding[
    "consolidated_category"
] = research_project_funding.tech_area_checked.copy()

for key in category_consolidation_dict:
    research_project_funding.loc[
        research_project_funding.consolidated_category == key, "consolidated_category"
    ] = category_consolidation_dict[key]

# %%
# Remove the only duplicate between reviewed UKRI and NIHR projects
research_project_funding = research_project_funding.query(
    "project_id != '9B59448A-300F-4352-9B17-65ACE7AEACCB'"
).copy()

# %%
research_project_funding = research_project_funding[
    -research_project_funding.start_date.isnull()
].copy()

# %%
research_project_funding.loc[
    research_project_funding.consolidated_category == "Diet", "Category"
] = "Health"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Waste reduction", "Category"
] = "Food waste"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Social", "Category"
] = "Social"

# %%
cols = ["Category", "consolidated_category"]
taxonomy_df = (
    research_project_funding.drop_duplicates(cols)[cols]
    .sort_values(cols)
    .reset_index(drop=True)
)
taxonomy_df

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
# data = (
#     df_reference
#     .copy()
#     .assign(year=lambda df:df.time_period.dt.year)
#     .assign(amount_total = lambda df: df.amount_total / 1e3)
#     # .query("year < 2022")
# )

# fig = (
#     alt.Chart(data, width=400, height=250)
#     .mark_bar(color=pu.NESTA_COLOURS[0])
#     .encode(
#         x=alt.X('year:O', title=''),
#         y=alt.Y('amount_total:Q', title='Research funding (million GBP)'),#, scale=alt.Scale(domain=(0,120))),
#         tooltip = [alt.Tooltip('year', title='Year'), alt.Tooltip('amount_total', title='Amount (million GBP)', format=".3f"), 'no_of_projects'],
#     )
# )
# fig = pu.configure_plots(fig)
# fig

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

# %%
df_total = au.gtr_get_all_timeseries_period(
    research_project_funding,
    # research_project_funding.query('funder=="nihr"'),
    # research_project_funding.query('Category!="Health"'),
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)


# %%
data = (
    df_total.copy()
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(amount_total=lambda df: df.amount_total / 1e3)
    .query("year < 2022")
)

fig = (
    alt.Chart(data, width=400, height=250)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X("year:O", title=""),
        y=alt.Y(
            "amount_total:Q",
            title="Research funding (million GBP)",
            scale=alt.Scale(domain=(0, 120)),
        ),
        tooltip=[
            alt.Tooltip("year", title="Year"),
            alt.Tooltip("amount_total", title="Amount (million GBP)", format=".3f"),
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
# ## Major categories

# %%
categories_to_check = [
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
# ### Major category trends

# %%
tech_area_ts = []
for tech_area in categories_to_check:
    df = research_project_funding.query("Category == @tech_area")
    df_ts = (
        au.gtr_get_all_timeseries_period(
            df,
            period="year",
            min_year=2010,
            max_year=2022,
            start_date_column="start_date",
        )
        .assign(tech_area=tech_area, year=lambda df: df.time_period.dt.year)
        .query("time_period <= '2021'")
    )
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = au.ts_magnitude_growth(
        tech_area_ts.query("tech_area == @tech_area").drop("tech_area", axis=1),
        2017,
        2021,
    ).drop("index")
    magnitude_growth.append(df.assign(tech_area=tech_area))
magnitude_growth = pd.concat(magnitude_growth, ignore_index=False).reset_index()

# %%
pd.options.display.float_format = "{:.3f}".format
magnitude_growth_plot = (
    magnitude_growth.sort_values(["index", "magnitude"], ascending=False)
    .assign(magnitude=lambda df: df.magnitude / 1000)
    .assign(growth=lambda df: df.growth / 100)
    .query("index=='amount_total'")
)

# %%
magnitude_growth_plot

# %%
au.moving_average(
    tech_area_ts.query('tech_area == "Innovative food"').assign(
        year=lambda df: df.time_period.dt.year
    )
)

# %% [markdown]
# ### Major category trends chart

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

colour_field = "tech_area"
text_field = "tech_area"
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
        magnitude_growth_plot,
        width=400,
        height=400,
    )
    .mark_circle(size=80)
    .encode(
        x=alt.X(
            "magnitude:Q",
            axis=alt.Axis(
                title=horizontal_title,
                tickCount=5,
            ),
            scale=alt.Scale(
                type=horizontal_scale,
                domain=(0, 40),
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
            alt.Tooltip("tech_area", title="Category"),
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
tech_area_ts.head(1)

# %%
fig = (
    alt.Chart(
        (
            tech_area_ts.assign(year=lambda df: df.time_period.dt.year).query(
                "tech_area != 'Health'"
            )
        ),
        width=400,
    )
    .mark_line(
        interpolate="monotone",
        size=2.5,
    )
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("no_of_projects:Q", title="Number of new projects"),
        color=alt.Color("tech_area:N", scale=alt.Scale(domain=domain, range=range_)),
    )
)
# fig
pu.configure_plots(fig)


# %%
def chart_funding_ts(excluded_categories=["Health"]):
    fig1 = (
        alt.Chart(
            (
                tech_area_ts.assign(year=lambda df: df.time_period.dt.year)
                .assign(amount_total=lambda df: df.amount_total / 1e3)
                .query("tech_area not in @excluded_categories")
                .query("year < 2022")
            ),
            width=400,
        )
        .mark_line(
            interpolate="monotone",
            size=2.5,
            # interpolate='cardinal',
        )
        .encode(
            x=alt.X(
                "year:O", scale=alt.Scale(domain=list(range(2010, 2022))), title=""
            ),
            y=alt.Y("amount_total:Q", title="Research funding (million GBP)"),
            color=alt.Color(
                "tech_area:N",
                title="Category",
                scale=alt.Scale(domain=domain, range=range_),
                legend=alt.Legend(orient="top", columns=2),
            ),
            tooltip=["year", "amount_total"],
        )
    )
    return pu.configure_plots(fig1)


# %%
fig_final = chart_funding_ts()
fig_final

# %%
AltairSaver.save(
    fig_final,
    f"v{fig_version_name}_major_ts_funding_without_health",
    filetypes=["html", "svg", "png"],
)

# %%
fig_final = chart_funding_ts([])
fig_final

# %%
AltairSaver.save(
    fig_final, f"v{fig_version_name}_major_ts_funding", filetypes=["html", "svg", "png"]
)

# %% [markdown]
# ## Subcategories

# %%
research_project_funding_ = research_project_funding.copy()
for cat in ["Fermentation", "Lab meat", "Plant-based"]:
    research_project_funding_.loc[
        research_project_funding_.consolidated_category == cat, "consolidated_category"
    ] = "Alt protein"
research_project_funding_.loc[
    research_project_funding_.consolidated_category == "Innovative food",
    "consolidated_category",
] = "Innovative food (other)"


# %%
yearly_projects_minor = (
    research_project_funding_.query(
        'start_date >= "2017-01-01" and start_date < "2022-01-01"'
    )
    .assign(year=lambda df: df.start_date.apply(lambda x: x[0:4]))
    .groupby(["consolidated_category"])
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
    # "Lab meat",
    "Alt protein",
    "Supply chain",
    # "Plant-based",
    "Retail",
    # "Fermentation",
    # "Dark kitchen",
    "Kitchen tech",
    "Dietary supplements",
    "Diet",
    "Delivery",
    "Packaging",
    "Personalised nutrition",
    # "Meal kits",
    "Restaurants",
    "Reformulation",
    "Innovative food (other)",
    "Waste reduction",
]

# %%
## Produce magnitude and growth for detailed/minor categories
tech_area_ts = []
for tech_area in categories_to_check:
    df = research_project_funding_.query("consolidated_category == @tech_area")
    df_ts = (
        au.gtr_get_all_timeseries_period(
            df,
            period="year",
            min_year=2010,
            max_year=2022,
            start_date_column="start_date",
        )
        .assign(tech_area=tech_area, year=lambda df: df.time_period.dt.year)
        .query("time_period <= '2021'")
    )
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = au.ts_magnitude_growth(
        tech_area_ts.query("tech_area == @tech_area").drop("tech_area", axis=1),
        2017,
        2021,
    ).drop("index")
    magnitude_growth.append(df.assign(tech_area=tech_area))
magnitude_growth = pd.concat(magnitude_growth, ignore_index=False).reset_index()

pd.options.display.float_format = "{:.3f}".format
magnitude_growth_plot_minor = (
    magnitude_growth.sort_values(["index", "magnitude"], ascending=False)
    .assign(magnitude=lambda df: df.magnitude / 1000)
    .assign(growth=lambda df: df.growth / 100)
    .query("index=='amount_total'")
    # .query("index=='no_of_projects'")
)
tech_area_ts = tech_area_ts.merge(
    taxonomy_df, left_on="tech_area", right_on="consolidated_category", how="left"
)


# %%
# magnitude_growth_plot.sort_values("magnitude", ascending=False).reset_index(drop=True).assign(growth=lambda df: df.growth*100, magnitude= lambda df: df.magnitude*1000)


# %%
magnitude_growth_plot_minor.head(2)

# %%
magnitude_growth_plot.head(1)

# %% [markdown]
# #### Growth plots

# %%
taxonomy_df_ = taxonomy_df.copy()
taxonomy_df_.loc[
    taxonomy_df.consolidated_category == "Innovative food", "consolidated_category"
] = "Innovative food (other)"
# taxonomy_df_

# %%
major_sort_order = magnitude_growth_plot.sort_values("growth").tech_area.to_list()

data = magnitude_growth_plot_minor.merge(
    taxonomy_df_, left_on="tech_area", right_on="consolidated_category", how="left"
)
data["Category"] = pd.Categorical(data["Category"], categories=major_sort_order)
data = data.sort_values(["Category", "growth"], ascending=False)
data = data.merge(yearly_projects_minor, how="left")

# %%
# data

# %%
colour_field = "Category"
text_field = "tech_area"
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
            "tech_area:N",
            sort=data.tech_area.to_list(),
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
        y=alt.Y("tech_area:N", sort=data.tech_area.to_list(), title=""),
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
tech_area_ts_minor = tech_area_ts.copy().assign(
    amount_total=lambda df: df.amount_total / 1e3
)

# %%
alt.Chart(tech_area_ts_minor, width=200, height=100).mark_line(
    size=2.5,
    interpolate="monotone",
).encode(
    x="year:O",
    y=alt.Y(
        "amount_total:Q",
    ),
    color=alt.Color("tech_area:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["tech_area", "amount_total", "year"],
).resolve_scale(
    y="independent"
)

# %%
alt.Chart(tech_area_ts_minor, width=200, height=100).mark_line(
    size=2.5,
    interpolate="monotone",
).encode(
    x="year:O",
    y=alt.Y(
        "no_of_projects:Q",
    ),
    color=alt.Color("tech_area:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["tech_area", "no_of_projects", "year"],
).resolve_scale(
    y="independent"
)

# %%
(
    research_project_funding_.query('consolidated_category == "Reformulation"')
    .query('start_date >= "2019-01-01" and start_date < "2020-01-01"')
    .sort_values("amount", ascending=False)
)[["title", "amount", "start_date"]]

# %%
