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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Crunchbase data exploration
# - Load data
# - Find companies with a specific industry label
# - Characterise investment into these companies
# - Additionally filter companies using keywords
#
# Prerequisite: Fetching CB data by running `make fetch-daps1` from the main repo directory

# %%
from innovation_sweet_spots.getters import crunchbase
import altair as alt
import itertools

from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
import innovation_sweet_spots.analysis.analysis_utils as au

CB = CrunchbaseWrangler()

# %% [markdown]
# ## Company industries
# Explore the different company categories (also called industries in Crunchbase website) that exist in the Crunchbase dataset

# %%
# A list of the 700+ categories/industries
CB.industries[0:5]

# %%
# Example: Check if parenting is in the list of categories
"parenting" in CB.industries

# %%
# A list of broader industry groups
CB.industry_groups[0:5]

# %%
# Check which broader group does 'parenting' industry belong to
CB.industry_to_group["parenting"]

# %%
# Check which other narrower categories/industries are in the same broader group
CB.group_to_industries["community and lifestyle"]

# %% [markdown]
# ## Find companies within specific industries

# %%
CB = CrunchbaseWrangler()

# %%
# Define the industries of interest
industries_names = ["parenting"]

# Get companies within the industry (might take a minute the first time)
companies = CB.get_companies_in_industries(industries_names)

# %% [markdown]
# ### Number of companies per country

# %%
# Number of companies per country
companies.groupby("country").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(10)

# %% [markdown]
# ### What other industries are these companies in?

# %%
# Get all industries of the selected companies
company_industries = CB.get_company_industries(companies)

industry_counts = (
    company_industries.groupby("industry")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)
industry_counts.head(10)

# %% [markdown]
# ### Further filtering
# Select parenting companies only related to the broader group 'software'

# %%
# List of additional filtering industries
filtering_industries = CB.group_to_industries["software"]
filtering_industries[0:10]

# %%
filtered_companies = CB.select_companies_by_industries(companies, filtering_industries)


# %%
filtered_companies[["name", "country"]]

# %%
