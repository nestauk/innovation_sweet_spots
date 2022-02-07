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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selecting projects and companies by categories

# %%
import innovation_sweet_spots.analysis.query_categories as qc
import importlib

importlib.reload(qc)

# %%
# Define research topics and industries of interest
GTR_RESEARCH_TOPICS = ["Wind Power", "Energy - Marine & Hydropower"]
CRUNCHBASE_INDUSTRIES = ["wind energy", "renewable energy"]

# %% [markdown]
# # Check research projects
#
# GtR research topic categories can be accessed and viewed via `GtrWrangler`. In this particular example, you can use the `GtrWrangler` instance already initialised by the`query_categories` module. Alternatively, you can load a new instance:
#
# ```python
# from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler
# GtrWrangler().gtr_topics_list
# ```

# %%
qc.GTR.gtr_topics_list[0:5]

# %% [markdown]
# Example of a query where the research topic has been specified incorrectly

# %%
query_df_wrong = qc.query_gtr_categories(["Wind"], return_only_matches=False)

# %% [markdown]
# Correct example of checking two categories, and adding additional project data to the results

# %%
query_df = qc.query_gtr_categories(GTR_RESEARCH_TOPICS, return_only_matches=True)
query_df.head(5)

# %%
qc.GTR.add_project_data(query_df, "id", ["title"])

# %% [markdown]
# # Check companies
#
# Crunchbase industries can be accessed and viewed via `CrunchbaseWrangler`. In this particular example, you can use the `CrunchbaseWrangler` instance already initialised by the`query_categories` module. Alternatively, you can load a new instance:
#
# ```python
# from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
# CrunchbaseWrangler().industries
# ```

# %%
qc.CB.industries[0:5]

# %% [markdown]
# Example of a query where the industry has been specified incorrectly

# %%
query_df_wrong = qc.query_cb_categories(["wind"], return_only_matches=True)

# %% [markdown]
# Correct example of checking an industry, and adding additional company data to the results

# %%
query_df = qc.query_cb_categories(CRUNCHBASE_INDUSTRIES, return_only_matches=True)
query_df.head(5)

# %%
qc.CB.add_company_data(query_df, "id", ["name", "country", "homepage_url"])

# %%
