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
# # Organising GtR project data
#
# Start and end dates, and funding amounts

# %%
from innovation_sweet_spots import logging, PROJECT_DIR
from innovation_sweet_spots.getters import gtr_2022 as gtr
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib
from innovation_sweet_spots.getters import gtr_api


# %%
output_filepath = (
    PROJECT_DIR / "inputs/data/gtr_2022_august/gtr_projects-wrangled_project_data.csv"
)

# %%
importlib.reload(gtr_api)

# %%
importlib.reload(wu)
GtR = wu.GtrWrangler()
GtR.gtr = gtr

# %%
gtr_projects = GtR.gtr.get_gtr_projects()
ukri_funding_df = GtR.get_project_funds_n(gtr_projects)

# %%
# Deduplicate project funding entries
funding_dedup = ukri_funding_df.drop_duplicates(["project_id", "amount"])
# Check which projects have multiple different funding amounts
duplicated_funds = funding_dedup.duplicated("project_id", keep=False)

# Collect those that are not duplicated
projects_with_unambiguous_funding = funding_dedup[duplicated_funds == False]
# double check the duplicates
assert len(projects_with_unambiguous_funding) == len(
    projects_with_unambiguous_funding.project_id.unique()
)

# Projects with ambiguous funding data
ambiguous_projects = funding_dedup[duplicated_funds].project_id.unique()
logging.info(
    f"There are {len(ambiguous_projects)} projects with ambiguous funding data"
)

# Fetch the correct data from the API
filepath = PROJECT_DIR / "inputs/data/misc/gtr_api_calls/gtr_projects_2022_08_26_v1.csv"
api_funds = gtr_api.get_funds_for_projects(
    ambiguous_projects, filepath, column_names=["i", "project_id", "amount"]
)

# Project id and funding table
project_funding_final_df = pd.concat(
    [
        projects_with_unambiguous_funding[["project_id", "amount"]],
        api_funds[["project_id", "amount"]],
    ],
    ignore_index=True,
)

# double check the duplicates
assert project_funding_final_df.duplicated("project_id").sum() == 0


# %%
# Find project start and end dates
project_start_end = GtR.get_start_end_dates(
    gtr_projects.assign(project_id=lambda df: df.id), ukri_funding_df
)

project_data = project_start_end[["project_id", "fund_start", "fund_end"]].merge(
    project_funding_final_df, how="left"
)

projects_with_missing_data = project_data[
    project_data.amount.isnull()
].project_id.to_list()


# %%
filepath = PROJECT_DIR / "inputs/data/misc/gtr_api_calls/gtr_projects_2022_08_26_v2.csv"
api_funds = gtr_api.get_funds_for_projects(projects_with_missing_data, filepath)


# %%
project_data_ = project_data[
    project_data.project_id.isin(projects_with_missing_data) == False
]
project_data_ = (
    pd.concat(
        [project_data_, api_funds[["project_id", "fund_start", "fund_end", "amount"]]],
        ignore_index=True,
    )
    .merge(gtr_projects[["id", "title"]], left_on="project_id", right_on="id")
    .drop("id", axis=1)
    .astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
)
project_data_.to_csv(output_filepath, index=False)


# %%
assert project_data_.duplicated("project_id").sum() == 0

# %%
project_data_.info()

# %%
