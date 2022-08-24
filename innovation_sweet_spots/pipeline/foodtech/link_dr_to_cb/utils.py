import pandas as pd

STOPWORDS = [
    "global",
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
    "oy",
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

CB_COLS_RENAME = {
    "name": "cb_name",
    "clean_name": "cb_clean_name",
    "address": "cb_address",
    "id": "cb_id",
    "short_description": "cb_description",
}

DR_COLS_RENAME = {
    "name": "dr_name",
    "clean_name": "dr_clean_name",
    "address": "dr_address",
    "id": "dr_id",
    "tagline": "dr_description",
}

EVALUATION_COLS = [
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

DR_URL_COLS = ["crunchbase", "website", "linkedin", "facebook", "twitter"]

CB_URL_COLS = ["cb_url", "domain", "linkedin_url", "facebook_url", "twitter_url"]


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
        clean_name=lambda x: preproc_company_names(x.name, stopwords=STOPWORDS)
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
    """Rename dealroom hq country to match crunchbase country"""
    dr_companies["hq_country"] = dr_companies["hq_country"].replace(
        {
            "Hong Kong SAR": "Hong Kong",
            "Hong Kong-China": "Hong Kong",
            "Côte d'Ivoire": "Côte dIvoire",
        }
    )
    return dr_companies


def preproc_company_names(company_names: pd.Series, stopwords=STOPWORDS) -> pd.Series:
    """Clean company names by lowercasing, removing spaces, removing
    website terms, removing non alphanumeric characters and removing
    stopwords
    """
    return (
        company_names.astype(str)
        .str.strip()
        .str.replace(
            r"([a-z])([A-Z][a-z])", lambda x: x[1] + "_" + x[2].lower(), regex=True
        )
        .str.lower()
        .str.replace(r".*www\.(.*?)\..*", r"\1", regex=True)
        .str.replace(r"https?://(.*)\..*", r"\1", regex=True)
        .str.replace(r"\.com|\.co\.uk|\.ac\.uk|\.org\.uk", "", regex=True)
        .str.replace("\.|\s+", " ", regex=True)
        .str.replace("[^a-z0-9 ]", "", regex=True)  # Alphanum only
        .str.strip()  # Remove added spaces
        .apply(
            lambda x: " ".join([word for word in x.split() if word not in (stopwords)])
        )
    )
