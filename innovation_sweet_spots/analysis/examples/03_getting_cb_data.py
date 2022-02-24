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
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
from innovation_sweet_spots.getters import crunchbase as cb
import altair as alt
import itertools

from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
import innovation_sweet_spots.analysis.analysis_utils as au
import innovation_sweet_spots.utils.plotting_utils as pu

# Functionality for saving charts
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver()

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

# %%
fig = (
    alt.Chart((industry_counts.reset_index().query("counts > 10")))
    .mark_bar()
    .encode(
        alt.X("counts"),
        alt.Y("industry", sort="-x"),
    )
)
fig

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

# %% [markdown]
# ## Characterise investment into these companies

# %% [markdown]
# ### Companies attracting most funding

# %%
au.sort_companies_by_funding(filtered_companies)[
    ["name", "country", "homepage_url", "total_funding_usd"]
].head(10)


# %% [markdown]
# ### Investment trends across years

# %%
funding_rounds = CB.get_funding_rounds(filtered_companies)
funding_rounds.head(5)

# %%
period = "Y"
yearly_funding = au.cb_investments_per_period(
    funding_rounds, period=period, min_year=2005, max_year=2021
)

fig = pu.time_series(
    yearly_funding,
    x_column="time_period",
    y_column="raised_amount_gbp_total",
    period=period,
)
fig


# %%
fig = pu.time_series(
    yearly_funding, x_column="time_period", y_column="no_of_rounds", period=period
)
fig

# %%
deal_types = [
    "angel",
    "grant",
    "pre-seed",
    "seed",
    "series_a",
    "series_b",
    "series_c",
    "series_d",
    "series_unknown",
]

pu.cb_deal_types(funding_rounds.query("year>2009"), deal_types=deal_types)


# %%
# Individual investment deals
company_industries = CB.get_company_industries(
    filtered_companies, return_lists=True
).reset_index()

fig = pu.cb_deals_per_year(
    filtered_companies, funding_rounds.query("year>2009"), company_industries
)
fig


# %%
AltairSaver.save(fig, "test", filetypes=["html"])

# %% [markdown]
# ## Keyword search for companies
#
# NB: Presently works only for UK companies

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_pilot_crunchbase_corpus
from toolz import pipe


# %%
Query = QueryTerms(corpus=get_pilot_crunchbase_corpus())

# %%
SEARCH_TERMS = [["toddler"], ["infant"], ["baby"], ["preschool"]]

# %%
query_df = Query.find_matches(SEARCH_TERMS, return_only_matches=True)

# %%
# Add information about company industries (should make this neater)
query_df_data = pipe(
    query_df,
    lambda x: CB.add_company_data(x, id_column="id", columns=["name"]),
    lambda x: CB.get_company_industries(x, return_lists=True).reset_index(),
    lambda x: CB.add_company_data(x, id_column="id", columns=["homepage_url"]),
).merge(query_df)


# %%
query_df_data

# %% [markdown]
# ## Find persons working in specific companies

# %%
# Get a sample of Crunchbase organisations
cb_orgs = cb.get_crunchbase_orgs(100)

# Find people associate with these organisations
cb_org_persons = CB.get_company_persons(cb_orgs)

# %%
cb_org_persons[["name", "person_name", "linkedin_url"]]

# %% [markdown]
# ### Fetching person university degrees

# %%
CB.get_person_degrees(cb_org_persons).head(3)

# %% [markdown]
# ### Fetching company education data

# %%
df = CB.get_company_education_data(cb_orgs)
df

# %%
