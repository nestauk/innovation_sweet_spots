"""
This file contains the utils and constants used in the notebooks
"""
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
from typing import Iterable

PROJECT_INPUTS_DIR = PROJECT_DIR / "inputs/data/misc/2023_childcare"
PARENT_TECH_DIR = PROJECT_INPUTS_DIR / "parenting_tech_proj"

# Working document
AFS_GOOGLE_SHEET_ID = "14LOSu8QurLH9kwEmWM-_TDCAbew4_nncR7BQn04CP38"

# Worksheet for the HolonIQ data
GOOGLE_SHEET_TAB = "HolonIQ_taxonomy"

# Parenting tech project data, including rejected companies
CHILDREN_COMPANIES_CSV = (
    "cb_companies_child_ed_v2022_04_27 - cb_companies_child_ed_v2022_04_27.csv"
)
PARENTING_COMPANIES_CSV = (
    "cb_companies_parenting_v2022_04_27 - cb_companies_parenting_v2022_04_27.csv"
)
# Parenting tech project data, with only accepted companies and 'Children' or 'Parent' tags
GOOGLE_SHEET_ID = "1bZjEL13FerZU9GUD0bHCnsYjOUgIYaPMWx7Qb2WLZCU"
GOOGLE_WORKSHEET_NAME = "startups"

# Another custom list with FamTech companies
FAMTECH = "fam-tech-startups-18-01-2023.csv"

# Text path
COMPANY_DESCRIPTIONS_PATH = (
    PROJECT_DIR / "outputs/preprocessed/texts/cb_descriptions_formatted.csv"
)

## Keyword search parameters
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.analysis.query_categories import query_cb_categories
from typing import Iterator


def query_categories(keywords: Iterator[str], CbWrangler: CrunchbaseWrangler) -> set:
    """Helper wrapper function for querying companies by keywords"""
    return set(
        query_cb_categories(
            keywords, CbWrangler, return_only_matches=True, verbose=False
        ).id.to_list()
    )


def get_taxonomy_dict(taxonomy_df: pd.DataFrame, level_1: str, level_2: str) -> dict:
    """
    Extracts a dictionary of taxonomy categories and subcategories from a dataframe

    Args:
        taxonomy_df (pd.DataFrame): Dataframe containing the taxonomy, with columns level_1 and level_2
        level_1 (str): Higher level taxonomy category
        level_2 (str): Lower level taxonomy category

    Returns:
        dict: Dictionary of taxonomy categories and subcategories in the format {level_1: [level_2]}
    """
    taxonomy_dict = dict()
    df = taxonomy_df.drop_duplicates([level_1, level_2]).copy()
    for category in df[level_1].unique():
        taxonomy_dict[category] = df.query(f"`{level_1}` == @category")[
            level_2
        ].to_list()
    return taxonomy_dict


import innovation_sweet_spots.utils.text_processing_utils as tpu
from typing import Iterable

# nlp = tpu.setup_spacy_model()


def process_keyword_string(keyword_string: str) -> str:
    """
    Process a keyword string to remove punctuation and make it lowercase
    """
    return keyword_string.lower().replace("[^\w\s]", " ").replace("\s+", " ").strip()


def add_spaces(keyword_string: str) -> str:
    """Add spaces to the beginning and end of a keyword string"""
    return " " + keyword_string + " "


def create_fake_plural(keyword_string: str) -> str:
    """Add an 's' to the end of a keyword string to create a fake plural"""
    return keyword_string + "s"


def process_keywords(
    terms_df: pd.DataFrame, keywords_column: str, add_plurals: bool = False
) -> pd.DataFrame:
    """
    Process a list of keywords to prepare them for searching in Crunchbase

    Args:
        terms_df (pd.DataFrame): Dataframe containing the keywords
        keywords_column (str): Column name containing the keywords
        add_plurals (bool, optional): Add fake plurals to the keywords. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe containing the processed keywords
    """
    terms_df = terms_df.reset_index(drop=True).copy()
    # Separate multiple keyword search queries into list of terms
    terms = [s.split(",") for s in terms_df[keywords_column].to_list()]
    # Process the terms
    terms_processed = []
    for term in terms:
        terms_processed.append([process_keyword_string(t) for t in term])
    assert len(terms_processed) == len(terms_df)
    terms_df["keywords_processed"] = terms_processed
    # Add plurals
    if add_plurals:
        terms_df_plurals = terms_df.assign(
            keywords_processed=lambda df: df.keywords_processed.apply(
                lambda x: [create_fake_plural(t) for t in x]
            )
        )
        terms_df = pd.concat([terms_df, terms_df_plurals], axis=0)
    # Add spaces to the beginning and end of each term
    terms_df["keywords_processed"] = terms_df["keywords_processed"].apply(
        lambda x: [add_spaces(t) for t in x]
    )
    return terms_df


# Function that outputs a list of European countries
def list_of_countries_in_europe() -> Iterable[str]:
    return [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czechia",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "United Kingdom",
    ]


def list_of_countries_in_north_america() -> Iterable[str]:
    return [
        "United States",
        "Canada",
    ]


list_of_select_countries = (
    list_of_countries_in_europe()
    + list_of_countries_in_north_america()
    + ["Australia", "New Zealand"]
)


def investibility_indicator(
    df: pd.DataFrame, funding_threshold_gbp: float = 1
) -> pd.DataFrame:
    """Add an investibility indicator to the dataframe"""
    df["investible"] = funding_threshold_gbp / df["funding_since_2020_gbp"]
    return df


# Get last rounds for each org_id
def get_last_rounds(funding_rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Get the last round for each org_id

    Args:
        funding_rounds (pd.DataFrame): Dataframe containing the funding rounds

    Returns:
        pd.DataFrame: Dataframe containing the last round for each org_id
    """
    return (
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
def get_last_rounds_since_2020(funding_rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Get the the money raised since 2020 for each org_id

    Args:
        funding_rounds (pd.DataFrame): Dataframe containing the funding rounds

    Returns:
        pd.DataFrame: Dataframe containing the money raised since 2020 for each org_id

    """
    return (
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
def get_total_funding(funding_rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Get the total funding for each org_id

    Args:
        funding_rounds (pd.DataFrame): Dataframe containing the funding rounds

    Returns:
        pd.DataFrame: Dataframe containing the total funding for each org_id

    """
    return (
        funding_rounds.groupby("org_id")
        .agg({"raised_amount_gbp": "sum"})
        .reset_index()
        .rename(columns={"org_id": "cb_id", "raised_amount_gbp": "total_funding_gbp"})
        # convert funding to millions
        .assign(total_funding_gbp=lambda x: x.total_funding_gbp / 1e3)
    )
