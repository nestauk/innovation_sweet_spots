# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.google_sheets import (
    download_google_sheet,
    upload_to_google_sheet,
)
import pandas as pd

import importlib
import utils

importlib.reload(utils)

keep_columns = ["name", "homepage_url", "cb_url", "user"]


# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

CB = CrunchbaseWrangler()

# %% [markdown]
# ## Import parenting tech project results


# %%
def get_parent_tech_df():
    # Get the final list from parenting tech project (download the data from Google Sheets)
    parent_tech_df = download_google_sheet(
        utils.GOOGLE_SHEET_ID, utils.GOOGLE_WORKSHEET_NAME
    )

    # Remove the first row of the dataframe, and set the second row as the header
    df = parent_tech_df.reset_index().copy()
    df = df.iloc[2:]
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df[keep_columns]
    df["relevant"] = 1
    return df


# %%
parent_tech_df = get_parent_tech_df()


# %%
parent_tech_df.head(2)


# %%
# Import both review tables and join them up with the final list of companies
parent_tech_reviewed_df = pd.concat(
    [
        pd.read_csv(utils.PARENT_TECH_DIR / utils.CHILDREN_COMPANIES_CSV),
        pd.read_csv(utils.PARENT_TECH_DIR / utils.PARENTING_COMPANIES_CSV),
    ],
    ignore_index=True,
)


# %%
cols = ["id", "name", "cb_url", "homepage_url", "relevancy", "comment"]
# Merge parent_tech_df and parent_tech_reviewed_df dataframes
parent_tech_df_ = (
    parent_tech_df.merge(parent_tech_reviewed_df[cols], on="cb_url", how="outer")
    .assign(
        # Combine the name_x and name_y columns
        company_name=lambda x: x["name_x"].fillna(x["name_y"]),
        homepage_url=lambda x: x["homepage_url_x"].fillna(x["homepage_url_y"]),
    )
    .rename(
        columns={
            "user": "category",
            "id": "cb_id",
            "relevancy": "relevancy_comment",
            "comment": "other_comments",
        }
    )
    .drop(columns=["name_x", "name_y", "homepage_url_x", "homepage_url_y"])
    .reindex(
        columns=[
            "category",
            "company_name",
            "cb_url",
            "homepage_url",
            "relevancy_comment",
            "other_comments",
            "relevant",
            "cb_id",
        ]
    )
    # Remove duplicated ids, keeping those that have a relevancy evaluation
    # (note that in this project we used a binary categorisation into 'Parent' or 'Children')
    .sort_values(["company_name", "relevancy_comment"])
    .drop_duplicates(["cb_id"], keep="first")
    # Fill null values in the 'relevant' column with 0
    .fillna({"relevant": 0})
)
parent_tech_df_


# %%
parent_tech_df_.to_csv(
    utils.PROJECT_INPUTS_DIR / "parenting_tech_proj.csv", index=False
)


# %% [markdown]
# ## Import the FamTech list

# %%
famtech_df = (
    pd.read_csv(utils.PROJECT_INPUTS_DIR / utils.FAMTECH)
    .rename(columns={"Name": "company_name", "Name URL": "cb_url"})
    .assign(category="Famtech")
)


# %%
famtech_df.to_csv(utils.PROJECT_INPUTS_DIR / "famtech.csv", index=False)


# %% [markdown]
# ## Join up the manual tables

# %%
childcare_industry_df = (
    pd.read_csv(utils.PROJECT_INPUTS_DIR / "test_childcare_industry_hits.csv")
    .assign(source="crunchbase_categories", category="Child care")
    .rename(columns={"id": "cb_id", "name": "company_name"})
)[["cb_id", "company_name", "cb_url", "homepage_url", "source", "category"]]

# %%
# Import the three dataframes
holoniq_df_with_cb = (
    pd.read_csv(
        PROJECT_DIR / "inputs/data/misc/2023_childcare/holoniq_taxonomy_with_cb.csv"
    )
    .assign(source="holoniq")
    .rename(columns={"websites": "homepage_url"})
)
famtech_df = (
    pd.read_csv(utils.PROJECT_INPUTS_DIR / "famtech.csv")
    .assign(source="famtech_list")
    .drop(["Type", "Description", "CB Rank"], axis=1)
)
parenting_tech_df = pd.read_csv(
    utils.PROJECT_INPUTS_DIR / "parenting_tech_proj.csv"
).assign(source="parenting_tech_proj")
df = pd.concat(
    [holoniq_df_with_cb, famtech_df, parenting_tech_df, childcare_industry_df]
)

# Get all unique company names from the three dataframes
df = df[(~df["cb_url"].duplicated()) | df["cb_url"].isna()]

# Replace null values in 'category' column with 'Rejected'
df["category"] = df["category"].fillna("Rejected")


# %%
df.to_csv(utils.PROJECT_INPUTS_DIR / "init_company_list.csv", index=False)


# %%
cb_cols = [
    "id",
    "name",
    "homepage_url",
    "cb_url",
    "country_code",
    "region",
    "city",
    "total_funding_usd",
    "last_funding_on",
    "founded_on",
    "closed_on",
    "short_description",
    "long_description",
    "rank",
    "country",
]


# %%
# Merge the initial company list with Crunchbase data (country, region, city, funding, etc.)
# Ignore the null values (but keep in the final dataframe)
company_list = pd.concat(
    [
        (
            df[-df.cb_url.isnull()]
            .merge(CB.cb_organisations[cb_cols], on=["cb_url"], how="left")
            # combine homepage_url_x and homepage_url_y columns
            .assign(
                homepage_url=lambda x: x["homepage_url_x"].fillna(x["homepage_url_y"])
            )
            .drop(columns=["homepage_url_x", "homepage_url_y"])
        ),
        df[df.cb_url.isnull()],
    ],
    ignore_index=True,
)

# %%
funding_rounds = CB.get_funding_rounds(
    (company_list[company_list.cb_id.notnull()].drop_duplicates("cb_id")),
    org_id_column="cb_id",
)


# %% [markdown]
# ### Investibility rating
#
# - Last round type
# - Last round size
# - Last round date
# - Raised funding since 2020
# - Number of rounds since 2020
# - Total raised funding


# %%
def investibility_indicator(
    df: pd.DataFrame, funding_threshold_gbp: float = 1
) -> pd.DataFrame:
    """Add an investibility indicator to the dataframe"""
    df["investible"] = funding_threshold_gbp / df["funding_since_2020_gbp"]
    return df


# %%
# Get last rounds for each org_id
last_rounds = (
    funding_rounds.sort_values("announced_on_date")
    .groupby("org_id")
    .last()
    .reset_index()
    .rename(
        columns={
            "org_id": "cb_id",
            "announced_on_date": "last_round_date",
            "raised_amount_gbp": "last_round_gbp",
            "raised_amount_usd": "last_round_usd",
            "post_money_valuation_usd": "last_valuation_usd",
            "investor_count": "last_round_investor_count",
            "cb_url": "deal_url",
        }
    )
    # convert funding to millions
    .assign(last_round_gbp=lambda x: x.last_round_gbp / 1e3)
    .assign(last_round_usd=lambda x: x.last_round_usd / 1e3)
    .assign(last_valuation_usd=lambda x: x.last_valuation_usd / 1e6)
)[
    [
        "cb_id",
        "last_round_date",
        "investment_type",
        "last_round_gbp",
        "last_round_usd",
        "last_valuation_usd",
        "last_round_investor_count",
        "deal_url",
    ]
]


# Get rounds since 2020
last_rounds_since_2020 = (
    funding_rounds[funding_rounds.announced_on_date >= "2020-01-01"]
    .groupby("org_id")
    .agg({"raised_amount_gbp": "sum", "funding_round_id": "count"})
    # convert funding to millions
    .assign(raised_amount_gbp=lambda x: x.raised_amount_gbp / 1e3)
    .reset_index()
    .rename(
        columns={
            "org_id": "cb_id",
            "raised_amount_gbp": "funding_since_2020_gbp",
            "funding_round_id": "funding_rounds_since_2020",
        }
    )
)

# Total funding
total_funding = (
    funding_rounds.groupby("org_id")
    .agg({"raised_amount_gbp": "sum"})
    .reset_index()
    .rename(columns={"org_id": "cb_id", "raised_amount_gbp": "total_funding_gbp"})
    # convert funding to millions
    .assign(total_funding_gbp=lambda x: x.total_funding_gbp / 1e3)
)


# %%
# merge all the funding dataframes to company_list
company_list_funding = (
    company_list.drop(["total_funding_usd", "last_funding_on", "id", "name"], axis=1)
    .merge(last_rounds, on="cb_id", how="left")
    .merge(last_rounds_since_2020, on="cb_id", how="left")
    .merge(total_funding, on="cb_id", how="left")
)
company_list_funding = investibility_indicator(company_list_funding)
# change the order of columns
company_list_funding = company_list_funding[
    [
        "category",
        "company_name",
        "homepage_url",
        "source",
        "cb_id",
        "cb_url",
        "country",
        "region",
        "city",
        "rank",
        "short_description",
        "long_description",
        "rank",
        "last_round_date",
        "investment_type",
        "last_round_gbp",
        "last_round_usd",
        "last_valuation_usd",
        "last_round_investor_count",
        "deal_url",
        "funding_since_2020_gbp",
        "funding_rounds_since_2020",
        "total_funding_gbp",
        "relevancy_comment",
        "other_comments",
        "relevant",
        "investible",
    ]
]
company_list_funding = company_list_funding.fillna(
    {
        "cb_url": "n/a",
        "relevant": "not evaluated",
    }
)

# %% [markdown]
# Edge cases:
# - Funding exists, but no new funding since 2020
# -

# %%
company_list_funding = company_list_funding.sort_values(
    ["category", "relevant", "investible", "funding_since_2020_gbp"],
    ascending=[True, False, False, False],
).reset_index(drop=True)

# %%
company_list_funding.to_csv(
    utils.PROJECT_INPUTS_DIR / "init_company_list_funding.csv", index=False
)

# %%
company_list_funding_ = company_list_funding.copy()

# %%
company_list_funding_.to_csv("test.csv", index=False)
company_list_funding_ = pd.read_csv("test.csv")

# %%
# Save to Google Sheet
upload_to_google_sheet(
    company_list_funding_,
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="initial_list",
)

# %%
