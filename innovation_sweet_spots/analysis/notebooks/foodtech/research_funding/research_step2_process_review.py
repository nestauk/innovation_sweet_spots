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
# # Combine reviewed UKRI and NIHR funding tables

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
# Simplify the Sub Categories
research_project_funding = research_project_funding.copy()
for cat in ["Fermentation", "Lab meat", "Plant-based"]:
    research_project_funding.loc[
        research_project_funding.consolidated_category == cat, "consolidated_category"
    ] = "Alt protein"
research_project_funding.loc[
    research_project_funding.consolidated_category == "Innovative food",
    "consolidated_category",
] = "Innovative food (other)"

# %%
cols = ["Category", "consolidated_category"]
taxonomy_df = (
    research_project_funding.drop_duplicates(cols)[cols]
    .sort_values(cols)
    .reset_index(drop=True)
    .rename(columns={"consolidated_category": "Sub Category"})
)
taxonomy_df

# %%
outputs_dir = PROJECT_DIR / "outputs/foodtech/research_funding/"

# %%
taxonomy_df.to_csv(outputs_dir / "research_funding_tech_taxonomy.csv", index=False)

# %%
cols = [
    "project_id",
    "title",
    "description",
    "lead_organisation",
    "funder",
    "programme_grant_category",
    "amount",
    "start_date",
    "consolidated_category",
]

# %%
research_project_final = (
    research_project_funding[cols]
    .rename(columns={"consolidated_category": "Sub Category"})
    .merge(taxonomy_df, how="left")
    .drop_duplicates(["project_id", "Sub Category"])
)

# %%
research_project_final.to_csv(
    outputs_dir / "research_funding_projects.csv", index=False
)

# %%
len(research_project_final)
