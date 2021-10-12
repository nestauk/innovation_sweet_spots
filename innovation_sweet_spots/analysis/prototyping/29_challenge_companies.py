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
# # Producing graphs for blogs

# %%
from innovation_sweet_spots import PROJECT_DIR, logging, config
from innovation_sweet_spots.getters import gtr, crunchbase, guardian
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.topic_analysis as iss_topics

# %%
import pandas as pd
import numpy as np
import altair as alt

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"

# %%
# Import Crunchbase data
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
# cb_df = cb_df[cb_df.country == "United Kingdom"]
cb_df = cb_df.reset_index(drop=True)
del cb

# %%
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %%
cb_df.info()

# %% [markdown]
# ## Pick up data

# %%
companies = pd.read_csv(PROJECT_DIR / "inputs/data/misc/requests/Vidal_crunchbase.csv")


# %%
companies.head(5)

# %%
orgs_df = companies.drop_duplicates("company")
orgs = list(orgs_df.company.unique())
df = cb_df[cb_df.name.isin(orgs)].sort_values("name")[
    ["name", "id", "country", "homepage_url"]
]
len(orgs)

# %%
len(df)

# %%
set(orgs).difference(set(df.name.unique()))

# %%
cb_df_ = cb_df[-cb_df.homepage_url.isnull()]
cb_df_[cb_df_.homepage_url.str.contains("creditdata")]

# %%
cb_df_[cb_df_.homepage_url.str.contains("creditdata")].name

# %%
df.name.unique()

# %%
len(df.name.unique())

# %%
df

# %%
cb_df[cb_df.name == "Plum Fintech"][["name", "id", "country", "homepage_url"]]

# %%
df.to_csv(
    PROJECT_DIR / "inputs/data/misc/requests/Vidal_crunchbase_check.csv", index=False
)

# %%
orgs_new = [
    "Beam",
    "Hastee",
    "IncomeMAX CIC",
    "Money Dashboard",
    "Udrafter",
    "Evolution Devices",
    "MYOLYN",
    "ItalDesign",
    "Foot++",
    "Human in Motion Robotics",
    "Solo Expenses",
]
len(orgs_new)

# %%
df_new = cb_df[cb_df.name.isin(orgs_new)].sort_values("name")[
    ["name", "id", "country", "homepage_url", "cb_url"]
]
df_new.to_csv(
    PROJECT_DIR / "inputs/data/misc/requests/Vidal_crunchbase_check_2.csv", index=False
)

# %%
df = pd.read_csv(PROJECT_DIR / "inputs/data/misc/requests/Vidal_crunchbase_checked.csv")
df = cb_df[cb_df.id.isin(df.id.to_list())]
df.to_csv(
    PROJECT_DIR / "outputs/data/misc/requests/vidal_oct2021_cb_companies.csv",
    index=False,
)

# %%
# request_orgs = pd.DataFrame(data=sorted(orgs), columns=['name'])
# request_orgs['is_found']=1
# request_orgs.merge(df, how='left')

# %%
df_deals = iss.get_cb_org_funding_rounds_full_info(df, cb_funding_rounds)
df_deals = df_deals.sort_values("announced_on")
df_deals.to_csv(
    PROJECT_DIR / "outputs/data/misc/requests/vidal_oct2021_cb_companies_deals.csv",
    index=False,
)

# %%
df_investors = iss.get_funding_round_investors(df_deals, cb_investments)
df_investors.to_csv(
    PROJECT_DIR
    / "outputs/data/misc/requests/vidal_oct2021_cb_companies_investments.csv",
    index=False,
)


# %% [markdown]
# ## ASF companies

# %%
cb_df_[cb_df_.legal_name.str.contains("Sort Holdings")][
    ["name", "country", "homepage_url", "id"]
]

# %%
cb_df_ = cb_df_[-cb_df_.legal_name.isnull()]

# %%
cb_df_[cb_df_.name.str.contains("NestEgg")][["name", "id", "country", "homepage_url"]]

# %%
companies = pd.read_csv(
    PROJECT_DIR
    / "inputs/data/misc/requests/[Karlis copy] DSR Automation - CrunchBase.csv"
)
companies.head(4)

# %%
orgs_df = companies.drop_duplicates("name")
orgs = list(orgs_df.name.unique())
df = cb_df[cb_df.name.isin(orgs)].sort_values("name")[
    ["name", "id", "country", "homepage_url"]
]
df = companies.merge(df, on="name", how="left")
len(orgs)

# %%
df.to_csv(
    PROJECT_DIR / "inputs/data/misc/requests/ASF_crunchbase_check.csv", index=False
)

# %%
# df = pd.read_csv(PROJECT_DIR / 'inputs/data/misc/requests/ASF_crunchbase_checked.csv')
df = pd.read_csv(
    PROJECT_DIR
    / "inputs/data/misc/requests/[Karlis copy] DSR Automation - CrunchBase_checked.csv"
)
df = df[-df.id.isnull()]
df = cb_df[cb_df.id.isin(df.id.to_list())]

# %%
# df

# %%
df.to_csv(
    PROJECT_DIR / "outputs/data/misc/requests/dsr_automation_companies.csv", index=False
)

# %%
# request_orgs = pd.DataFrame(data=sorted(orgs), columns=['name'])
# request_orgs['is_found']=1
# request_orgs.merge(df, how='left')

# %%
df_deals = iss.get_cb_org_funding_rounds_full_info(df, cb_funding_rounds)
df_deals = df_deals.sort_values("announced_on")
df_deals.to_csv(
    PROJECT_DIR / "outputs/data/misc/requests/dsr_automation_companies_deals.csv",
    index=False,
)

# %%
df_investors = iss.get_funding_round_investors(df_deals, cb_investments)
df_investors.to_csv(
    PROJECT_DIR / "outputs/data/misc/requests/dsr_automation_companies_investments.csv",
    index=False,
)
