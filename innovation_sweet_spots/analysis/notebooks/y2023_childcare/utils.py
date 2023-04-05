# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

"""
This file contains the utils and constants used in the notebooks
"""
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
import numpy as np
from typing import Iterable
import innovation_sweet_spots.getters.google_sheets as gs
from innovation_sweet_spots.utils.io import save_pickle, load_pickle
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import df_to_hf_ds
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)
from sklearn.model_selection import train_test_split

PROJECT_INPUTS_DIR = PROJECT_DIR / "inputs/data/misc/2023_childcare"
PARENT_TECH_DIR = PROJECT_INPUTS_DIR / "parenting_tech_proj"

# Training dataset path
SAVE_DS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/dataset_v2023_03_03"

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


# Supervised ML utils
def prepare_training_data():
    # Get longlist of companies
    longlist_df = gs.download_google_sheet(
        google_sheet_id=AFS_GOOGLE_SHEET_ID,
        wks_name="list_v2",
    )
    # Load a table with processed company descriptions
    processed_texts = get_preprocessed_crunchbase_descriptions()

    # Process data for supervised ML
    data = longlist_df[["cb_id", "relevant", "cluster_relevance"]].copy()

    # Select a random 50% sample of 'noise' clusters and set their relevance to 0
    # This is somewhat ad-hoc, and we could also not use the noise clusters at all here
    companies_in_noise_clusters = (
        data.query('cluster_relevance == "noise"')
        .query('relevant != "1"')
        .sample(frac=0.5, random_state=42)
    )
    data.loc[companies_in_noise_clusters.index, "relevant"] = "0"

    # Add processed company descriptions
    data = (
        data.rename({"cb_id": "id"}, axis=1)
        .merge(processed_texts, on="id")
        .drop(["name", "cluster_relevance"], axis=1)
        .rename({"description": "text"}, axis=1)
    )[["id", "text", "relevant"]]

    # Create a training set and a test set out from the labelled data,
    # with 80% of the relevant and not-relevant labels in the training set
    data_labelled = data.query('relevant != "not evaluated"').astype(
        {"relevant": "int"}
    )

    # Unlabelled data, to be labelled by the model
    data_unlabelled = (
        data.query('relevant == "not evaluated"')
        # make values relevant or not randomly
        .assign(relevant=lambda x: np.random.randint(2, size=len(x)))
    )

    train_df, test_df = train_test_split(
        data_labelled,
        test_size=0.2,
        stratify=data_labelled["relevant"],
        random_state=42,
    )

    # Split column relevant into relevant and not_relevant
    train_df = train_df.assign(not_relevant=lambda x: 1 - x["relevant"])
    test_df = test_df.assign(not_relevant=lambda x: 1 - x["relevant"])
    data_unlabelled = data_unlabelled.assign(not_relevant=0)

    # Make datasets
    train_ds = df_to_hf_ds(train_df)
    test_ds = df_to_hf_ds(test_df)
    to_review_ds = df_to_hf_ds(data_unlabelled)

    # Save dataset tables
    save_pickle(train_df, SAVE_DS_PATH / "train_df.pickle")
    save_pickle(test_df, SAVE_DS_PATH / "test_df.pickle")
    save_pickle(data_unlabelled, SAVE_DS_PATH / "to_review_df.pickle")

    # Save datasets
    save_pickle(train_ds, SAVE_DS_PATH / "train_ds.pickle")
    save_pickle(test_ds, SAVE_DS_PATH / "test_ds.pickle")
    save_pickle(to_review_ds, SAVE_DS_PATH / "to_review_ds.pickle")
