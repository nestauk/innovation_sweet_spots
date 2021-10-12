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
# # Prep data for network analysis

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters import crunchbase, gtr
import innovation_sweet_spots.analysis.analysis_utils as iss

import pandas as pd

# %%
INPUTS_DIR = PROJECT_DIR / "outputs/finals/"

# %% [markdown]
# # Get GTR and CB data

# %% [markdown]
# ## GTR data

# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
link_gtr_topics = gtr.get_link_table("gtr_topic")

gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(gtr_projects, gtr_project_funds)
del link_gtr_funds

# %%
project_to_org = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
# gtr_columns = ["title", "abstractText", "techAbstractText"]
# gtr_docs = iss.create_documents_from_dataframe(gtr_projects, gtr_columns, preprocessor=iss.preprocess_text)

# %% [markdown]
# ## CB data

# %%
# Import Crunchbase data
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
cb_df = cb_df.reset_index(drop=True)
del cb
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %% [markdown]
# ## LCH and EEM projects and companies

# %%
# Input the curated set of Low carbon heating (LCH)
# and Energy efficiency and management (EEM) companies/projects
proj = pd.read_csv(INPUTS_DIR / "ISS_projects.csv")
companies = pd.read_csv(INPUTS_DIR / "ISS_companies.csv")

# %% [markdown]
# # Prep datasets

# %% [markdown]
# ## Research projects

# %%
proj_ = proj.copy()
proj_["project_id"] = proj_["doc_id"]

# %%
# Add project organisations to the project table
project_orgs = iss.get_gtr_project_orgs(proj_, project_to_org)

# %%
project_orgs.to_csv(INPUTS_DIR / "ISS_projects_organisations.csv", index=False)

# %% [markdown]
# ## Companies

# %%
# importlib.reload(iss)

# %%
companies_ = companies.copy()
companies_["id"] = companies_["doc_id"]
companies_["name"] = companies_["title"]

# %%
# Add deals and investors to the companies table
df_deals = iss.get_cb_org_funding_rounds_full_info(companies_, cb_funding_rounds)
df_deals = df_deals.sort_values("announced_on")
df_investors = iss.get_funding_round_investors(df_deals, cb_investments)

# %%
df_investors.to_csv(INPUTS_DIR / "ISS_companies_investors.csv", index=False)

# %%
cols = ["investor_name", "investor_id"]
df_investor_data = df_investors.drop_duplicates(cols)[cols].merge(
    cb_investors, left_on="investor_id", right_on="id", how="left"
)

# %%
df_investor_data.to_csv(INPUTS_DIR / "ISS_investor_data.csv", index=False)
