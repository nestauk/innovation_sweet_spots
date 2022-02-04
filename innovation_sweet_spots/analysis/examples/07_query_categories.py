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
# # Selecting projects and companies by keywords and key phrases

# %%
import innovation_sweet_spots.analysis.query_categories as qc

# %%
importlib.reload(qc)

# %% [markdown]
# # Check research projects

# %%
CATEGORIES = ["Wind Power", "Energy - Marine & Hydropower"]

# %%
query_df = qc.query_gtr_categories(CATEGORIES, return_only_matches=True)

# %%
query_df_wrong = qc.query_gtr_categories(["Wind"], return_only_matches=False)

# %%
qc.GTR.add_project_data(query_df, "id", ["title"])

# %% [markdown]
# # Check companies

# %%

# %%
