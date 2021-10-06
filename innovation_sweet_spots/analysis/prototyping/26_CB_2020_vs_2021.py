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
# # Compare CB data versions

# %%
from innovation_sweet_spots.getters import crunchbase
import altair as alt
import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
import pandas as pd

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb"
cb_2020 = crunchbase.get_crunchbase_orgs_full()

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"
cb_2021 = crunchbase.get_crunchbase_orgs_full()

# %% [markdown]
# ## Checks
# - Number of UK companies
# - Number of UK deals

# %% [markdown]
# ### Companies

# %%
cb_2020_uk = cb_2020[cb_2020.country == "United Kingdom"]

# %%
len(cb_2020_uk)

# %%
cb_2021_uk = cb_2021[cb_2021.country == "United Kingdom"]

# %%
len(cb_2021_uk)

# %% [markdown]
# There are 98,000+ companies in the newest database version, and 75,000+ in the Oct 2020 version

# %%
cb_2020_uk.info()


# %%
def foundation_year(df, column, descriptor):
    df_ = df[-df[column].isnull()].copy()
    df_["year"] = df_[column].apply(iss.convert_date_to_year)
    df_ = df_.groupby("year").agg(counts=("id", "count")).reset_index()
    df_["descriptor"] = descriptor
    return df_


# %%
company_counts = pd.concat(
    [
        foundation_year(cb_2020_uk, "founded_on", "v2020"),
        foundation_year(cb_2021_uk, "founded_on", "v2021"),
    ]
)

# %%
fig = (
    alt.Chart(company_counts)
    .mark_line(clip=True)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(domain=list(range(2001, 2022)))),
        #         x='descriptor',
        #         x='year',
        y="counts",
        color="descriptor",
        #         column='year'
    )
)
fig

# %%
df = company_counts.pivot(index="year", columns=["descriptor"]).fillna(0)
df = (
    ((df[("counts", "v2021")] / df[("counts", "v2020")]) - 1)
    .reset_index()
    .rename(columns={0: "difference"})
)

# %%
fig = (
    alt.Chart(df)
    .mark_line(clip=True)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(domain=list(range(1990, 2022)))),
        y=alt.Y("difference:Q", scale=alt.Scale(domain=(0, 1))),
    )
)
fig

# %%
cols = ["id", "name", "short_description", "long_description", "founded_on"]
cb_2020_uk = cb_2020_uk.copy()
cb_2020_uk["version"] = "v2020"
cb_combined = cb_2021_uk[cols].merge(cb_2020_uk[cols + ["version"]], how="left")
cb_new = cb_combined[cb_combined.version.isnull() == True]


# %%
def check_phrase(search_term):
    check = cb_new.short_description.str.lower().str.contains(
        search_term
    ) | cb_new.long_description.str.lower().str.contains(search_term)
    return cb_new[check]


# %%
search_terms = [
    "heating",
    "heat pump",
    "hydrogen",
    "heat stor",
    "solar",
    " wind ",
    "renewable energy",
]
[len(check_phrase(s)) for s in search_terms]

# %% [markdown]
# ### Categories

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb"
cb_2020 = crunchbase.get_crunchbase_category_groups()
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"
cb_2021 = crunchbase.get_crunchbase_category_groups()

# %%
len(cb_2020), len(cb_2021)

# %% [markdown]
# ### Deals

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb"
cb_2020 = crunchbase.get_crunchbase_orgs_full()
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"
cb_2021 = crunchbase.get_crunchbase_orgs_full()

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb"
cb_2020_deals = crunchbase.get_crunchbase_funding_rounds()
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"
cb_2021_deals = crunchbase.get_crunchbase_funding_rounds()

# %%
len(cb_2020), len(cb_2021)

# %%
cb_2020_uk = cb_2020[cb_2020.country == "United Kingdom"]
cb_2021_uk = cb_2021[cb_2021.country == "United Kingdom"]

# %%
uk_deals_2020 = cb_2020_uk.merge(cb_2020_deals, left_on="id", right_on="org_id")
uk_deals_2021 = cb_2021_uk.merge(cb_2021_deals, left_on="id", right_on="org_id")

# %%
len(uk_deals_2020), len(uk_deals_2021)

# %%
cols = ["id", "name", "short_description", "long_description", "founded_on"]
cb_2020_uk = cb_2020_uk.copy()
cb_2020_uk["version"] = "v2020"
cb_combined = cb_2021_uk[cols].merge(cb_2020_uk[cols + ["version"]], how="left")
cb_new = cb_combined[cb_combined.version.isnull() == True]

# %%
cb_new.to_csv(crunchbase.CB_PATH.parent / "cb_2021/new_cb_uk_orgs.csv", index=False)

# %%
