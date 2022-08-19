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
        Dataframe with cleaned website text column
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


def update_cb_countries_to_match_dr(cb_companies: pd.DataFrame) -> pd.DataFrame:
    """Rename crunchbase country to match dealroom hq country"""
    cb_companies["country"] = cb_companies["country"].replace(
        {
            "Viet Nam": "Vietnam",
            "Macedonia, Republic of": "North Macedonia",
            "American Samoa": "Samoa",
            "Russian Federation": "Russia",
            "Bolivia, Plurinational State of": "Bolivia",
            "Taiwan, Province of China": "Taiwan",
            "Congo, The Democratic Republic of the": "Democratic Republic of the Congo",
            "Congo": "Democratic Republic of the Congo",
            "Korea, Republic of": "South Korea",
            "Moldova, Republic of": "Moldova",
            "Czechia": "Czech Republic",
            "Brunei Darussalam": "Brunei",
            "Lao People's Democratic Republic": "Laos",
            "Myanmar": "Burma (Myanmar)",
            "Iran, Islamic Republic of": "Iran",
            "Côte d'Ivoire": "Côte dIvoire",
        }
    )
    return cb_companies


def update_dr_countries_to_match_cb(dr_companies: pd.DataFrame) -> pd.DataFrame:
    "Rename dealroom hq country to match crunchbase country"
    dr_companies["hq_country"] = dr_companies["hq_country"].replace(
        {
            "Hong Kong SAR": "Hong Kong",
            "Hong Kong-China": "Hong Kong",
            "Côte d'Ivoire": "Côte dIvoire",
        }
    )
    return dr_companies


STOPWORDS = [
    "partnership",
    "control",
    "japan",
    "packaging",
    "investments",
    "publishing",
    "entertainment",
    "luxembourg",
    "scientific",
    "portugal",
    "denmark",
    "danmark",
    "italia",
    "media",
    "brokers",
    "capital",
    "oy",
    "innovate",
    "innovation",
    "ventures",
    "biotech",
    "beverages",
    "hospitality",
    "new zealand",
    "australia",
    "technologies",
    "enterprises",
    "payments",
    "technology",
    "medical",
    "foundation",
    "costa rica",
    "pacific",
    "designs",
    "china",
    "india",
    "green",
    "therapeutics",
    "malaysia",
    "industries",
    "sciences",
    "organics",
    "controls",
    "environmental",
    "toronto",
    "distribuidora",
    "tecnologia",
    "associates",
    "drinks",
    "network",
    "market",
    "health",
    "analytics",
    "delivery",
    "brewing",
    "pest",
    "agro",
    "logistics",
    "restaurants",
    "restaurant",
    "singapore",
    "innovations",
    "brasil",
    "solutions",
    "foods",
    "global",
    "thailand",
    "nigeria",
    "international",
    "national",
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
    .pipe(update_dr_countries_to_match_cb)
)
cb = (
    get_crunchbase_orgs().pipe(add_clean_name_col).pipe(update_cb_countries_to_match_dr)
)

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
dr_companies_left_to_match = dr.query(
    f"id not in {dr_matched_ids}"
)  # .reset_index(drop=True)

# %%
dr_countries = dr_companies_left_to_match.hq_country.unique()
cb_countries = cb.country.unique()
list(set(dr_countries).difference(cb_countries))

# %%
from tempfile import TemporaryDirectory
from jacc_hammer.fuzzy_hash import Cos_config, Fuzzy_config, match_names_stream
from pathlib import Path
from innovation_sweet_spots import PROJECT_DIR

# Load configs
cos_config = Cos_config()
fuzzy_config = Fuzzy_config()
# Save fuzzy matches to dir
DR_TO_CB_FUZZY_MATCHES_DIR = PROJECT_DIR / "outputs/dr_to_cb_fuzzy_matches/"
DR_TO_CB_FUZZY_MATCHES_DIR.mkdir(exist_ok=True)
# Create list of countries in dealroom dataset
dr_countries = [
    country
    for country in dr_companies_left_to_match.hq_country.unique()
    if pd.isnull(country) is False
]
# Set settings
sim_mean_min = 70
chunksize = 100_000
for country in dr_countries:
    # Create temp directory
    tmp_dir = Path(TemporaryDirectory().name)
    tmp_dir.mkdir()
    dr_country_subset = dr_companies_left_to_match.query(
        f"hq_country == '{country}'"
    ).reset_index(drop=True)
    cb_country_subset = cb.query(f"country == '{country}'").reset_index(drop=True)
    if len(cb_country_subset) == 0:
        pass
    else:
        cb_names = cb_country_subset.clean_name.to_list()
        dr_names = dr_country_subset.clean_name.to_list()
        fuzzy_name_matches = pd.concat(
            match_names_stream(
                [cb_names, dr_names],
                chunksize=chunksize,
                tmp_dir=tmp_dir,
                cos_config=cos_config,
                fuzzy_config=fuzzy_config,
            )
        ).query(f"sim_mean >= {sim_mean_min}")
        fuzzy_name_matches_with_info = (
            fuzzy_name_matches.merge(
                right=cb_country_subset, right_index=True, left_on="x", how="left"
            )
            .rename(
                columns={
                    "name": "cb_name",
                    "clean_name": "cb_clean_name",
                    "address": "cb_address",
                    "id": "cb_id",
                    "short_description": "cb_description",
                }
            )
            .merge(right=dr_country_subset, right_index=True, left_on="y", how="left")
            .rename(
                columns={
                    "name": "dr_name",
                    "clean_name": "dr_clean_name",
                    "address": "dr_address",
                    "id": "dr_id",
                    "tagline": "dr_description",
                }
            )
            .sort_values(by=["sim_mean"], ascending=False)
            .reset_index(drop=True)[
                [
                    "x",
                    "y",
                    "cb_name",
                    "dr_name",
                    "cb_clean_name",
                    "dr_clean_name",
                    "cb_description",
                    "dr_description",
                    "cb_address",
                    "dr_address",
                    "cb_id",
                    "dr_id",
                    "sim_ratio",
                    "sim_jacc",
                    "sim_cos",
                    "sim_mean",
                ]
            ]
        )
        country_fuzzy_match_save_path = (
            DR_TO_CB_FUZZY_MATCHES_DIR
            / f"{country.lower().replace(' ', '_')}_fuzzy_name_matches.csv"
        )
        if len(fuzzy_name_matches_with_info) > 0:
            fuzzy_name_matches_with_info.to_csv(country_fuzzy_match_save_path)

# %%
