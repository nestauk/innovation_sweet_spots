# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.getters.dealroom import get_foodtech_companies
from innovation_sweet_spots.getters.crunchbase import get_crunchbase_orgs
import pandas as pd
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


def find_dr_cb_matches(dr_companies, cb_companies, dr_on, cb_on):
    return dr_companies.merge(
        right=cb_companies,
        left_on=dr_on,
        right_on=cb_on,
        how="left",
        suffixes=("_dr", "_cb"),
    )[["id_dr", "id_cb"]].dropna()


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
        Dataframe with cleaned website text
    """
    return df.pipe(remove_fw_slash, col).pipe(remove_http_https, col).pipe(add_www, col)


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

# %%
dr = cols_replace_space_and_lowercase(
    get_foodtech_companies()
).drop_duplicates()  # [["id", "NAME", "PROFILE URL", "WEBSITE", "HQ COUNTRY", ]]
cb = get_crunchbase_orgs()  # .query("country_code == 'GBR'")

# %%
dr_clean_cb_url = clean_website_col(dr.dropna(subset=["crunchbase"]), "crunchbase")
cb_clean_cb_url = clean_website_col(cb.dropna(subset=["cb_url"]), "cb_url")

# %%
# find matches on cb url
id_dr_to_id_cb_from_cb_url = find_dr_cb_matches(
    dr_clean_cb_url, cb_clean_cb_url, "crunchbase", "cb_url"
)

# %%
dr_already_matches_ids = list(id_dr_to_id_cb_from_cb_url.id_dr.values)

# %%
dr_left_to_match = dr.query(f"id not in {dr_already_matches_ids}")

# %%
dr_clean_website = clean_website_col(
    dr_left_to_match.dropna(subset=["website"]), "website"
)
cb_clean_website = clean_website_col(cb.dropna(subset=["domain"]), "domain")

# %%
id_dr_to_id_cb_from_website = find_dr_cb_matches(
    dr_clean_website, cb_clean_website, "website", "domain"
)

# %%
id_dr_to_id_cb_from_cb_url_and_website = pd.concat(
    [id_dr_to_id_cb_from_cb_url, id_dr_to_id_cb_from_website]
).reset_index(drop=True)

# %%
dr_already_matches_ids2 = list(id_dr_to_id_cb_from_cb_url_and_website.id_dr.values)

# %%
dr_left_to_match2 = dr.query(f"id not in {dr_already_matches_ids2}")

# %%
dr_left_to_match2["clean_name"] = preproc_names(
    dr_left_to_match2["name"], stopwords=STOPWORDS
)

# %%
cb["clean_name"] = preproc_names(cb["name"], stopwords=STOPWORDS)

# %%
dr_clean_name = dr_left_to_match2.dropna(subset=["clean_name"])

# %%
cb_clean_name = cb.dropna(subset=["clean_name"])

# %%
id_dr_to_id_cb_from_country_and_name = find_dr_cb_matches(
    dr_clean_name,
    cb_clean_name,
    ["clean_name", "hq_country"],
    ["clean_name", "country"],
)

# %%
id_dr_to_id_cb_from_cb_url_website_country_name = pd.concat(
    [id_dr_to_id_cb_from_cb_url_and_website, id_dr_to_id_cb_from_country_and_name]
)

# %%
id_dr_to_id_cb_from_cb_url_website_country_name

# %%
id_dr_to_id_cb_from_cb_url_website_country_name.to_csv(
    "dr_to_cb_matches.csv", index=False
)

# %%
cb.query("id == '25d07a9f-019e-4965-b2c0-019dbb37ea12'")

# %%
dr.query("id == 1757305")

# %%
19315 / 25166

# %%
25166 - 19278

# %%
dr

# %%
len(dr_already_matches_ids3)

# %%
dr_already_matches_ids3 = list(
    id_dr_to_id_cb_from_cb_url_website_country_name.id_dr.values
)
dr_left_to_match3 = dr.query(f"id not in {dr_already_matches_ids3}")

# %%
dr_left_to_match3

# %%
dr_left_to_match3.to_csv("dr_left_to_match3.csv", index=False)

# %%
for col in cb.columns:
    print(col)

# %%
for col in dr.columns:
    print(col)

# %%
