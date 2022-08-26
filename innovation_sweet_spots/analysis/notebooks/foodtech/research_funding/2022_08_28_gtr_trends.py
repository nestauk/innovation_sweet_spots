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
        "tech_area",
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
        "tech_area",
    ]
]


# %% [markdown]
# ## Combine datasets

# %%
research_project_funding = pd.concat([ukri_df, nihr_df_reviewed_])

# %%
(
    research_project_funding.groupby(["tech_area"])
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


# %%
tech_area_ts = []
for tech_area in research_project_funding.tech_area.unique():
    df = research_project_funding.query("tech_area == @tech_area")
    df_ts = au.gtr_get_all_timeseries_period(
        df, period="year", min_year=2010, max_year=2022, start_date_column="start_date"
    ).assign(tech_area=tech_area)
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

# %%
magnitude_growth = []
for tech_area in research_project_funding.tech_area.unique():
    df = au.ts_magnitude_growth(
        tech_area_ts.query("tech_area == @tech_area"), 2017, 2021
    ).drop("index")
    magnitude_growth.append(df.assign(tech_area=tech_area))

magnitude_growth = pd.concat(magnitude_growth, ignore_index=False).reset_index()

# %%
pd.options.display.float_format = "{:.3f}".format
magnitude_growth.sort_values(["index", "magnitude"], ascending=False).query(
    "index=='amount_total'"
)

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

# %%
