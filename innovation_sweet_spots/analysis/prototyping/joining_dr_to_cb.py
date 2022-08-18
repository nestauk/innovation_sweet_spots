# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.getters.dealroom import get_foodtech_companies
from innovation_sweet_spots.getters.crunchbase import get_crunchbase_orgs
import pandas as pd

pd.set_option("mode.chained_assignment", None)
from jacc_hammer.name_clean import preproc_names


# %%
def cols_replace_space_and_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces with underscores and make lowercase
    for column names in dataframe"""
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


def remove_http_https(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove http:// and https:// from the start of all rows in
    specified df and col"""
    df[col] = df[col].str.replace("https://", "").str.replace("http://", "")
    return df


def remove_fw_slash(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove '/' from the end of all rows in specified
    df and col"""
    df[col] = df[col].str.rstrip("/")
    return df


def add_www(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add www. to start of all rows in specified
    df and col"""
    df[col] = df[col].map(lambda x: x if str(x).startswith("www.") else f"www.{str(x)}")
    return df


def clean_website_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Clean website text to make it easier to join by:
        - removing / from end of website
        - remove http/https from start of website
        - add www. to start of website

    Args:
        df: Dataframe with column containing website text
        col: Column containing website text

    Returns:
        Dataframe with column with cleaned website text
    """
    return df.pipe(remove_fw_slash, col).pipe(remove_http_https, col).pipe(add_www, col)


def find_dr_cb_matches(
    dr_companies: pd.DataFrame, cb_companies: pd.DataFrame, dr_on: str, cb_on: str
) -> pd.DataFrame:
    """Match crunchbase companies to dealroom companies

    Args:
        dr_companies: Dealroom dataframe of companies
            containing id and column to join on
        cb_companies: Crunchbase dataframe of companies
            containing id and column to join on
        dr_on: Column to match the dealroom dataframe on
        cb_on: Column to match the crunchbase dataframe on

    Returns:
        Dataframe with a column for dealroom ids
            and related crunchbase id

    """
    return dr_companies.merge(
        right=cb_companies,
        left_on=dr_on,
        right_on=cb_on,
        how="left",
        suffixes=("_dr", "_cb"),
    )[["id_dr", "id_cb"]].dropna()


def find_url_matches(
    dr_companies: pd.DataFrame, cb_companies: pd.DataFrame, dr_url: str, cb_url: str
) -> pd.DataFrame:
    """Return dealroom to crunchbase lookup dataframe.
    Matches are found using specified urls."""
    return (
        dr_companies.dropna(subset=[dr_url])
        .pipe(clean_website_col, col=dr_url)
        .pipe(
            find_dr_cb_matches,
            cb_companies=cb_companies.dropna(subset=[cb_url]).pipe(
                clean_website_col, col=cb_url
            ),
            dr_on=dr_url,
            cb_on=cb_url,
        )
    )


def make_combined_dr_cb_lookup(lookups: list) -> pd.DataFrame:
    """Make combined dealroom to crunchbase lookup
    from a list of lookups and drop duplicates"""
    return (
        pd.concat(lookups)
        .reset_index(drop=True)
        .drop_duplicates(subset=["id_dr"], keep="first")
    )


def add_clean_name_col(companies_dataset: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned name column to specified company dataset"""
    return companies_dataset.assign(
        clean_name=lambda x: preproc_names(x.name, stopwords=STOPWORDS)
    )


STOPWORDS = [
    "ltd",
    "llp",
    "limited",
    "holdings",
    "group",
    "cic",
    "uk",
    "plc",
    "inc",
    "gmbh",
    "srl",
    "sa",
    "nz",
    "co",
    "se",
    "sarl",
    "sl",
    "bv",
    "doo",
    "fmcg",
    "llc",
    "zao",
    "kg",
    "kft",
    "aps",
    "ab",
    "as",
    "oy",
    "sro",
    "sas",
    "ccl",
    "sdn",
    "bhd",
    "ug",
    "ek",
    "kk",
    "kc",
    "sp",
    "website",
    "nbsp",
    "hk",
    "int",
    "rl",
    "usa",
    "nc",
    "sac",
    "sac",
    "pvt",
    "intl",
    "gbr",
]

DR_URL_COLS = ["crunchbase", "website", "linkedin", "facebook", "twitter"]

CB_URL_COLS = ["cb_url", "domain", "linkedin_url", "facebook_url", "twitter_url"]

# %%
# Load datasets
dr = (
    get_foodtech_companies()
    .pipe(cols_replace_space_and_lowercase)
    .drop_duplicates()
    .dropna(subset=["id"])
    .pipe(add_clean_name_col)
)
cb = get_crunchbase_orgs().pipe(add_clean_name_col)

# %%
"""
filter columns only for those that are being used
"""

# %%
# Find dealroom to crunchbase matches using urls
combined_url_matches = []
dr_companies_left_to_match = dr.copy()
for dr_url, cb_url in zip(DR_URL_COLS, CB_URL_COLS):
    url_matches = find_url_matches(dr_companies_left_to_match, cb, dr_url, cb_url)
    # Add current matches to the combined dr to cb lookup
    combined_url_matches.append(url_matches)
    combined_dr_cb_lookup = make_combined_dr_cb_lookup(combined_url_matches)
    # Update list of dr ids that have been matched to cb ids
    dr_matched_ids = list(combined_dr_cb_lookup.id_dr.values)
    # Update dataframe of dealroom companies that have not been matched yet
    dr_companies_left_to_match = dr.query(f"id not in {dr_matched_ids}")

# %%
dr_companies_left_to_match

# %%
# Find exact matches of companies with the same country and name
matches_from_country_and_name = dr_companies_left_to_match.dropna(
    subset=["clean_name"]
).pipe(
    find_dr_cb_matches,
    cb_companies=cb,
    dr_on=["clean_name", "hq_country"],
    cb_on=["clean_name", "country"],
)
# Add exact matches of same country and name to the combined dr to cb lookup
combined_dr_cb_lookup = make_combined_dr_cb_lookup(
    [combined_dr_cb_lookup, matches_from_country_and_name]
)
# Update list of dr ids that have been matched to cb ids
dr_matched_ids = list(combined_dr_cb_lookup.id_dr.values)
# Update dataframe of dealroom companies that have not been matched yet
dr_companies_left_to_match = dr.query(f"id not in {dr_matched_ids}")

# %%
dr_companies_left_to_match

# %%
