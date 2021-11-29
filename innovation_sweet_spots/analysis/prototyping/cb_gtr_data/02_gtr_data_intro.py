# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GtR data exploration
# - Load data
# - Check organisations participating in a particular project
#
# Prerequisite: Fetching GtR data by running `make fetch-daps1` from the main repo directory

# %% [markdown]
# ## Import data and dependencies

# %%
from innovation_sweet_spots.getters import gtr
import innovation_sweet_spots.analysis.analysis_utils as iss
import altair as alt

# %%
import importlib

importlib.reload(gtr)

# %%
## Import GtR data

# Projects
gtr_projects = gtr.get_gtr_projects()
# Funding (with start and end dates)
gtr_funds = gtr.get_gtr_funds()
# Organisations
gtr_organisations = gtr.get_gtr_organisations()
# Research topics
gtr_topics = gtr.get_gtr_topics()

# Links table between projects and organisations
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
# Links table between projects and research topics
link_gtr_topics = gtr.get_link_table("gtr_topic")
# Links between projects and funds (might be useful to find end-dates of project funding)
link_gtr_funds = gtr.get_link_table("gtr_funds")

# Add funding data to projects (note, I'm using another funding table which
# I've obtained directly via GtR API and is less ambiguous)
gtr_project_funds = gtr.add_funding_data(gtr_projects, gtr.get_gtr_funds_api())
# Add organisation data to projects
gtr_project_organisations = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
gtr_projects.head(5)

# %% [markdown]
# ## Check a particular project

# %%
check_project = "106DA39B-4A30-4D98-BB8B-42779D7CEE41"

# %%
gtr_project_funds[gtr_project_funds.project_id == check_project]

# %%
# Organisations participating in the project
gtr_project_organisations.query(f"project_id=='{check_project}'")
