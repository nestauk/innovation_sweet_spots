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
import innovation_sweet_spots.analysis.analysis_utils as iss
import altair as alt
import itertools

# %% [markdown]
# ## Load data

# %%
# Full table of companies (this might take a minute)
CB = crunchbase.get_crunchbase_orgs_full()
CB = CB[-CB.id.duplicated()]

# %%
# Table with investors
CB_investors = crunchbase.get_crunchbase_investors()
# Table with investments
CB_investments = crunchbase.get_crunchbase_investments()
# Table with investment rounds (one round can have several investments)
CB_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %% [markdown]
# ### Company categories / industries
# Explore the different company categories that exist in the Crunchbase dataset

# %%
# Table of company categories (also called "industries") and broader "category groups"
CB_category_groups = crunchbase.get_crunchbase_category_groups()
# Table of companies and their categories
CB_org_categories = crunchbase.get_crunchbase_organizations_categories()

# %%
# Get a list of the 700+ categories/industries
industries = CB_category_groups["name"].sort_values().to_list()

# %%
industries[0:5]

# %%
# Get a list of the broader category groups
CB_category_groups["category_groups"] = CB_category_groups[
    "category_groups_list"
].apply(lambda x: x.split(",") if type(x) is str else [])
unique_broad_categories = sorted(
    list(set(itertools.chain(*CB_category_groups["category_groups"].to_list())))
)

# %%
unique_broad_categories[0:5]

# %%
# Get a mapping from categories/industries to broader groups
industries_to_broader_group = dict(
    zip(CB_category_groups.name, CB_category_groups.category_groups.to_list())
)

# %%
# Get a mapping from broader categories to categories/industries
df = (
    CB_category_groups.explode("category_groups")
    .groupby("category_groups")
    .agg(categories=("name", lambda x: x.tolist()))
    .reset_index()
)
broader_group_to_industries = dict(zip(df.category_groups, df.categories.to_list()))

# %%
# Example: Check if parenting is in the list of categories
"Parenting" in categories

# %%
# Check which broader category group does 'Parenting' belong to
industries_to_broader_group["Parenting"]

# %%
# Check which other narrower categories/industries are in the broader group
broader_group_to_industries["Community and Lifestyle"]

# %% [markdown]
# ## Find companies within specific industries

# %%
# Define the industry of interest (note lower case)
industries_names = ["parenting"]

# %%
# Get identifiers of companies belonging to the industry
company_ids = CB_org_categories[
    CB_org_categories.category_name.isin(industries_names)
].organization_id

# %%
# Get companies within the industry
companies = CB[CB.id.isin(company_ids)]

# %% [markdown]
# ### Number of companies per country

# %%
# Number of companies per country
companies.groupby("country").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(10)

# %% [markdown]
# ### What other industries are these companies in

# %%
# Get all industries of the selected companies
company_industries = CB_org_categories[
    CB_org_categories.organization_id.isin(companies.id.to_list())
]
industry_counts = (
    company_categories.groupby("category_name")
    .agg(counts=("organization_id", "count"))
    .sort_values("counts", ascending=False)
)
industry_counts.head(10)

# %% [markdown]
# ### Use additional industries to filter companies

# %%
# List of additional industries
filtering_industries = ["apps"] + [
    s.lower() for s in broader_group_to_industries["Software"]
]


# %%
def is_string_in_list(list_of_strings, list_to_check):
    return True in [s in list_to_check for s in list_of_strings]


# %%
# List of categories/industries for each company
company_industries_list = company_industries.groupby(["organization_id"]).agg(
    categories=("category_name", lambda x: x.tolist())
)
# Filter using filtering_categories
is_in_filtering_industries = company_industries_list["categories"].apply(
    lambda x: is_string_in_list(filtering_industries, x)
)
filtered_ids = company_industries_list[is_in_filtering_industries].index.to_list()

# %%
companies_filtered = companies[companies.id.isin(filtered_ids)]

# %%
companies_filtered[["name", "country"]]

# %% [markdown]
# ## Characterise investment into these companies

# %% [markdown]
# ### Companies attracting most funding

# %%
companies_to_check = companies_filtered
# companies_to_check = companies

# %%
# Remove companies without funding information
companies_ = companies_to_check[(-companies_to_check.total_funding_usd.isnull())].copy()
# Covert all funding values to float
companies_.total_funding_usd = companies_.total_funding_usd.astype(float)
# Sort companies according to total funding
companies_ = companies_.sort_values("total_funding_usd", ascending=False)

# %%
# Fraction of companies with funding information
len(companies_) / len(companies_to_check)

# %%
# Companies with the most funding
companies_[["name", "country", "homepage_url", "total_funding_usd"]].head(10)

# %% [markdown]
# ### Investment trends across years

# %%
company_funding_rounds = iss.get_cb_org_funding_rounds(
    companies_to_check, CB_funding_rounds
)

# %%
company_funding_rounds.head(5)

# %%
# Get number of rounds and raised amount (thousands) per year
yearly_investments = iss.get_cb_funding_per_year(company_funding_rounds, max_year=2021)

# %%
# Amount of funding
iss.show_time_series_fancier(yearly_investments, "raised_amount_gbp_total")

# %%
# Number of investment rounds/deals
iss.show_time_series_fancier(yearly_investments, "no_of_rounds")

# %%
# Types of deals
alt.Chart(company_funding_rounds).mark_bar().encode(
    x=alt.X("year:O", title="Year"),
    y=alt.Y("count(investment_type)", title="Number of deals"),
    color="investment_type",
)

# %%
# Individual investment rounds/deals
df = (
    company_funding_rounds.copy()
    .merge(
        companies[["id", "country", "short_description", "long_description"]],
        left_on="org_id",
        right_on="id",
        how="left",
    )
    .merge(
        company_industries_list.reset_index(),
        left_on="org_id",
        right_on="organization_id",
        how="left",
    )
)
df = df[-df.raised_amount_gbp.isnull()]
df = df[df.raised_amount_gbp > 0]
ymax = df["raised_amount_gbp"].max()
fig = (
    alt.Chart(df, width=800, height=400)
    .mark_point(opacity=0.8, size=20, clip=False, color="#6295c4")
    .encode(
        alt.X("year:O"),
        alt.Y("raised_amount_gbp:Q", scale=alt.Scale(type="log")),
        tooltip=[
            "year",
            "name",
            "country",
            "short_description",
            "long_description",
            "investment_type",
            "categories",
            "raised_amount_gbp",
        ],
    )
)
fig

# %%
# Export altair figure (a little bit involved)
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %%
# This will save figures in outputs/figures/{html, svg, png} folders
fig_name = "cb_explore_investments"
alt_save.save_altair(fig, fig_name, driver)

# %% [markdown]
# ### Number of companies founded per year

# %%
iss.cb_orgs_founded_by_year(companies, max_year=2021)

# %% [markdown]
# ## Filter companies using keywords

# %%
# Create company text "documents" using specified columns (might take a minute if many companies)
cb_columns = ["name", "short_description", "long_description"]
cb_documents = iss.create_documents_from_dataframe(
    companies_to_check,
    cb_columns,
    preprocessor=iss.preprocess_text_clean  # Use this for more involved preprocessing (e.g. removing punctuation, lematising)
    #    preprocessor=iss.preprocess_text # Use this for very light preprocessing
)

# %%
cb_documents[0:2]

# %%
# Each document corresponds to a row in the provided company table
assert len(cb_documents) == len(companies_to_check)

# %%
# Filter companies with 'baby' in their text description
terms = ["baby"]
orgs_with_term = iss.find_docs_with_all_terms(terms, cb_documents, companies_to_check)
orgs_with_term[["name", "country", "homepage_url"]].head(10)

# %%
# Filter companies with 'baby' AND 'learn' in their text description
terms = ["baby", "learn"]
orgs_with_term = iss.find_docs_with_all_terms(terms, cb_documents, companies_to_check)
orgs_with_term[["name", "country", "homepage_url"]].head(10)

# %%
# Find companies that have any of the specified keyphrases/terms
terms = [
    ["baby", "learn"],
    ["infant", "learn"],
    ["early years", "learn"],
]
orgs_with_terms = iss.get_docs_with_keyphrases(terms, cb_documents, companies_to_check)
orgs_with_terms[["name", "country", "homepage_url"]].head(10)

# %%
