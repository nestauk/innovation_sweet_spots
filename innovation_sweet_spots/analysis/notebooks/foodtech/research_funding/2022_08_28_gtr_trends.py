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
from innovation_sweet_spots.getters import gtr_2022 as gtr

# %%
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd

NIHR_DIR = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
nihr_df = pd.read_csv(NIHR_DIR)

# %%
ukri_df_reviewed = google_sheets.get_foodtech_reviewed_gtr(from_local=False).query(
    "tech_area_checked!='-'"
)
nihr_df_reviewed = google_sheets.get_foodtech_reviewed_nihr(from_local=False).query(
    "tech_area_checked!='-'"
)


# %% [markdown]
# ## Process GtR documents

# %%
gtr_projects = gtr.get_wrangled_projects()

# %%
gtr_projects.info()

# %%
ukri_df_reviewed.info()

# %%
ukri_df = (
    ukri_df_reviewed.merge(
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
ukri_df.info()

# %% [markdown]
# ## Process NIHR documents

# %%
nihr_df.info()

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


# %% [markdown]
# ## Combine datasets

# %%
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

search_terms = get_foodtech_search_terms()
cols = ["Category", "Sub Category", "Tech area"]
taxonomy_df = search_terms.drop_duplicates(cols)[cols]

# %%
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
# research_project_funding

# %%
(
    research_project_funding.groupby(["tech_area_checked"])
    .sum()
    .sort_values("amount", ascending=False)
)


# %%
from innovation_sweet_spots.analysis import analysis_utils as au
import importlib

importlib.reload(au)

# %%
au.gtr_get_all_timeseries_period(
    research_project_funding,
    period="year",
    min_year=2010,
    max_year=2022,
    start_date_column="start_date",
)


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

# %%
tech_area_ts = []
for tech_area in categories_to_check:
    df = research_project_funding.query("Category == @tech_area")
    df_ts = au.gtr_get_all_timeseries_period(
        df, period="year", min_year=2010, max_year=2022, start_date_column="start_date"
    ).assign(tech_area=tech_area)
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

# %%
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = au.ts_magnitude_growth(
        tech_area_ts.query("tech_area == @tech_area"), 2017, 2021
    ).drop("index")
    magnitude_growth.append(df.assign(tech_area=tech_area))

magnitude_growth = pd.concat(magnitude_growth, ignore_index=False).reset_index()

# %%
pd.options.display.float_format = "{:.3f}".format
magnitude_growth_plot = (
    magnitude_growth.sort_values(["index", "magnitude"], ascending=False)
    .assign(growth=lambda df: df.growth / 100)
    .query("index=='amount_total'")
)

# %%
magnitude_growth_plot

# %% [markdown]
# ### Defining figure

# %%
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

colour_field = "tech_area"
text_field = "tech_area"
# horizontal_scale = "linear"
horizontal_title = f"Average yearly funding (GBP)"
legend = alt.Legend()

title_text = "Foodtech trends (2017-2021)"
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
            axis=alt.Axis(title=horizontal_title),
            scale=alt.Scale(
                # type=horizontal_scale,
                domain=(0, 90_000),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(title="Growth", format="%"),
        ),
        color=alt.Color(f"{colour_field}:N", legend=None),
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

fig_final = (
    (fig + text)
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

# %% [markdown]
# ## Subcategories

# %%
categories_to_check = [
    "Biomedical",
    "Lab meat",
    "Alt protein",
    "Supply chain",
    "Plant-based",
    "Retail",
    "Delivery apps",
    "Fermentation",
    "Fat",
    "Dark kitchen",
    "Kitchen tech",
    "Dietary supplements",
    "Insects",
    "Food waste",
    "Diet",
    "Packaging",
    "Sugar",
    "Personalised nutrition",
    "Meal kits",
    "Restaurants",
    "Reformulation",
    "Innovative food",
    "Suply chain",
]

# %%
research_project_funding[
    "consolidated_category"
] = research_project_funding.tech_area_checked.copy()
research_project_funding.loc[
    research_project_funding.consolidated_category == "Fat", "consolidated_category"
] = "Reformulation"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Sugar", "consolidated_category"
] = "Reformulation"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Suply chain",
    "consolidated_category",
] = "Supply chain"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Delivery apps",
    "consolidated_category",
] = "Delivery"


# %%
tech_area_ts = []
for tech_area in categories_to_check:
    df = research_project_funding.query("tech_area_checked == @tech_area")
    df_ts = au.gtr_get_all_timeseries_period(
        df, period="year", min_year=2010, max_year=2022, start_date_column="start_date"
    ).assign(tech_area=tech_area)
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

# %%
magnitude_growth = []
for tech_area in categories_to_check:
    df = au.ts_magnitude_growth(
        tech_area_ts.query("tech_area == @tech_area"), 2017, 2021
    ).drop("index")
    magnitude_growth.append(df.assign(tech_area=tech_area))

magnitude_growth = pd.concat(magnitude_growth, ignore_index=False).reset_index()

# %%
pd.options.display.float_format = "{:.3f}".format
magnitude_growth_plot = (
    magnitude_growth.sort_values(["index", "magnitude"], ascending=False)
    .assign(growth=lambda df: df.growth / 100)
    .query("index=='amount_total'")
)

# %%
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

colour_field = "tech_area"
text_field = "tech_area"
# horizontal_scale = "linear"
horizontal_title = f"Average yearly funding (GBP)"
legend = alt.Legend()

title_text = "Foodtech trends (2017-2021)"
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
            axis=alt.Axis(title=horizontal_title),
            scale=alt.Scale(
                # type=horizontal_scale,
                domain=(0, 90_000),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(title="Growth", format="%"),
        ),
        color=alt.Color(f"{colour_field}:N", legend=None),
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

fig_final = (
    (fig + text)
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

fig_final.interactive()

# %%
pd.options.display.float_format = "{:.3f}".format
magnitude_growth_plot = (
    magnitude_growth.sort_values(["index", "magnitude"], ascending=False)
    .assign(growth=lambda df: df.growth / 100)
    .query("index=='no_of_projects'")
)

# %%
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

colour_field = "tech_area"
text_field = "tech_area"
# horizontal_scale = "linear"
horizontal_title = f"Average yearly funding (GBP)"
legend = alt.Legend()

title_text = "Foodtech trends (2017-2021)"
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
            axis=alt.Axis(title=horizontal_title),
            scale=alt.Scale(
                # type=horizontal_scale,
                # domain=(0, 90_000),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(title="Growth", format="%"),
        ),
        color=alt.Color(f"{colour_field}:N", legend=None),
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

fig_final = (
    (fig + text)
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

fig_final.interactive()

# %% [markdown]
# Very, very rough and ready trends on research funding (using the mostly not-reviewed data) but looks interesting - biomedical at the top
# Supply chain is probably overestimated
# Packaging is mixing together sustainable packaging with shelf life
# Food waste is probably mixing agtech
# Retial is interesting! what's going on there?
# personalised nutrition a lot of attention there
# kitchen tech?
# some interesting movement around delivery apps,
# alt protein and such appears quite modest, but
#
# Overall it looks like innovation funding is going to emerging areas?
