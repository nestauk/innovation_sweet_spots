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

# %% [markdown]
# # Scan Crunchbase for childcare companies + add details from Holon IQ

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

CB = CrunchbaseWrangler()


# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.google_sheets import upload_to_google_sheet
import utils
import pandas as pd


# %% [markdown]
# ## Match HolonIQ companies with Crunchbase companies

# %%
holoniq_df = pd.read_csv(
    PROJECT_DIR / "inputs/data/misc/2023_childcare/holoniq_taxonomy.csv"
)


# %%
holoniq_df_unique = holoniq_df.drop_duplicates(
    subset=["company_name", "websites"]
).drop(columns=["category", "image_links"])


# %%
import re


def normalise_company_name(company_name: str) -> str:
    """
    Normalises company name by removing special characters (using regex) and lowercasing
    """
    if type(company_name) is not str:
        return ""
    else:
        return re.sub(r"[^a-zA-Z0-9]", "", company_name).lower()


def normalise_website(website: str) -> str:
    """
    Normalises website by removing "https://", "http://" and "www." and lowercasing
    """
    if type(website) is not str:
        return ""
    else:
        return (
            website.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            # Remove trailing slash
            .rstrip("/")
            .lower()
        )


def add_normalised_names_and_websites(
    df: pd.DataFrame, name_column: str = "company_name", website_column: str = "website"
) -> pd.DataFrame:
    """
    Adds normalised company names and websites to df
    """
    return df.assign(
        company_name_norm=df[name_column].apply(normalise_company_name),
        websites_norm=df[website_column].apply(normalise_website),
    )


def find_companies_in_crunchbase(
    companies_df: pd.DataFrame, cb_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds companies from companies_df in cb_df (specified by name and website)

    Parameters
    ----------
    companies_df : pd.DataFrame
        Dataframe containing company names and websites (should contain company_name and websites columns)
    cb_df : pd.DataFrame
        Dataframe containing Crunchbase companies (should contain name and homepage_url columns)

    Returns
    -------
    pd.DataFrame
        Dataframe containing companies from companies_df that were found in cb_df with their Crunchbase data

    """
    return add_normalised_names_and_websites(
        companies_df, "company_name", "websites"
    ).merge(
        add_normalised_names_and_websites(cb_df, "name", "homepage_url"),
        how="left",
        on=["websites_norm"],
    )


# %%
holoniq_cb_df = find_companies_in_crunchbase(holoniq_df_unique, CB.cb_organisations)

# %%
holoniq_cb_df = holoniq_cb_df.query("websites != 'https://'")

# %%
# Check how many companies were found in Crunchbase
holoniq_cb_df["name"].notnull().sum()

# %%
len(holoniq_cb_df)

# %%
# Add crunchbase IDs and URLs to holoniq_df
holoniq_df_with_cb = holoniq_df.merge(
    holoniq_cb_df[["company_name", "websites", "id", "cb_url"]].rename(
        columns={"id": "cb_id"}
    ),
    on=["company_name", "websites"],
    how="left",
)

# Save to CSV
holoniq_df_with_cb.to_csv(
    PROJECT_DIR / "inputs/data/misc/2023_childcare/holoniq_taxonomy_with_cb.csv",
    index=False,
)

# Save to Google Sheet
upload_to_google_sheet(
    holoniq_df_with_cb,
    google_sheet_id=utils.GOOGLE_SHEET_ID,
    wks_name=utils.GOOGLE_SHEET_TAB,
)


# %%
