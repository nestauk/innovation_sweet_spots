"""
innovation_sweet_spots.analysis.query_categories

Module for selecting Crunchbase and GtR data using the existing data labels
"""
from innovation_sweet_spots import logging
from innovation_sweet_spots.analysis.wrangling_utils import (
    GtrWrangler,
    CrunchbaseWrangler,
)
import pandas as pd
from typing import Iterator, Callable

# Initialise data wrangler instances
GTR = GtrWrangler()
CB = CrunchbaseWrangler()


def get_items_in_categories(
    matches_init: pd.DataFrame,
    categories: Iterator[str],
    search_function: Callable,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Helper function to produce a data frame with boolean indicators indicating
    if the data items in matches_init table are in the specified categories.

    This helper function can be used for either GtR or Crunchbase data.

    Args:
        matches_init: A dataframe with a single 'id' column
        categories: Item categories
        search_function: Function that is used to determine the ids of the
            items in a specific category
        return_only_matches: If True, will only return the documents that are matches
        verbose: If True, will show logging info

    Returns:
        A dataframe with the following columns:
            - a column 'id' for item identifiers
            - a boolean column for each of the categories, where True indicates that item belongs to it
            - a column 'any_category' which indicates if any categories where matched
    """
    # Initial dataframe with all item ids
    matches = matches_init.copy()
    # Variable holding items which are in any of the provided categories
    items_in_any_category = set()
    for category in categories:
        # Check if items are in the category
        items_in_category = search_function(category)
        # Logging output
        if verbose:
            logging.info(
                f"Found {len(items_in_category)} projects in the category '{category}'"
            )
        # Mark which data items are in the category
        matches[category] = matches["id"].isin(items_in_category)
        # Keep track of items in any of the provided categories
        items_in_any_category = items_in_any_category | set(items_in_category)
    # Add a column to mark items which are in any category
    matches["any_category"] = matches["id"].isin(items_in_any_category)
    # Return all items or only items which are in the category
    if return_only_matches:
        return matches.query("any_category == True").reset_index(drop=True)
    else:
        return matches
    return matches


def is_gtr_project_in_category(category: str, GtR: GtrWrangler = GTR) -> Iterator[str]:
    """
    Helper function that returns project ids that belong to a GtR research topic

    Args:
        category: GtR research topic category
        GtR: GtrWrangler instance

    Returns:
        List of project ids that correspond to the GtR research topic category
    """
    # Find the id corresponding to the category
    topic_id_df = GtR.gtr_topics.query("topic == @category")
    if len(topic_id_df) == 0:
        logging.warning(
            f"Research topic '{category}' does not exist in the GtR database!"
        )
        return []
    else:
        # Make sure that there is only one unique GtR research topic id
        assert (
            len(topic_id_df) == 1
        ), "There is more than one topic id with the same name: Something wrong with GtrWrangler.gtr_topics"
        topic_id = topic_id_df.id.iloc[0]
        # Find the project ids in the specified category
        return GtR.link_gtr_topics.query("id == @topic_id").project_id.to_list()


def initialise_gtr_id_table(GtR: GtrWrangler = GTR) -> pd.DataFrame:
    """Initialises a dataframe with a column for project ids"""
    return GtR.gtr_projects[["project_id"]].rename(columns={"project_id": "id"})


def query_gtr_categories(
    categories: Iterator[str],
    GtR: GtrWrangler = GTR,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Produces a dataframe with GtR projects and boolean indicators whether
    the projects are in the specified categories. This function wraps
    get_items_in_categories() for GtR projects.

    Note: The research topics in `categories` must be present in `GtR.gtr_topics`

    Args:
        categories: List of GtR research topic categories
        GtR: GtrWrangler instance
        return_only_matches: If True, will only return the projects that are matches
        verbose: If True, will output logging info

    Returns:
        A dataframe with the following columns:
            - a column for project identifiers
            - a boolean column for each of the categories, where True indicates that project belongs to it
            - a column 'any_category' which indicates if the project was in any of the specified categories
    """
    return get_items_in_categories(
        # Initialise the output dataframe
        matches_init=initialise_gtr_id_table(),
        # GtR research topics (categories)
        categories=categories,
        # Function that returns a list of projects in a category
        search_function=lambda x: is_gtr_project_in_category(x, GtR),
        # Specify whether to return the full table (with all items)
        return_only_matches=return_only_matches,
        # Logging outputs
        verbose=verbose,
    )


def is_cb_organisation_in_category(
    category: str, CbWrangler: CrunchbaseWrangler = CB
) -> Iterator[str]:
    """
    Helper function that returns Crunchbase organisation ids that belong to an industry

    Args:
        category: Crunchbase industry name
        Cb: CrunchbaseWrangler instance

    Returns:
        List of organisation ids
    """
    if category not in CbWrangler.industries:
        logging.warning(
            f"Industry '{category}' does not exist in the Crunchbase database!"
        )
    return CbWrangler.get_companies_in_industries([category]).id.to_list()


def initialise_cb_id_table(CbWrangler: CrunchbaseWrangler = CB) -> pd.DataFrame:
    """Initialises a dataframe with a column for company ids"""
    return CbWrangler.cb_organisations[["id"]]


def query_cb_categories(
    categories: Iterator[str],
    CbWrangler: CrunchbaseWrangler = CB,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Produces a dataframe with Crunchbase organisations and boolean indicators whether
    the organisations are in the specified categories (industries). This function wraps
    get_items_in_categories() for Crunchbase data.

    Note: The industry names in `categories` must be present in `CbWrangler.industries`

    Args:
        categories: List of Crunchbase organisations
        GtR: CrunchbaseWrangler instance
        return_only_matches: If True, will only return the organisations that are matches
        verbose: If True, will output logging info

    Returns:
        A dataframe with the following columns:
            - a column for organisation identifiers
            - a boolean column for each of the industries, where True indicates that organisation belongs to it
            - a column 'any_category' which indicates if the organisation was in any of the specified industries
    """
    return get_items_in_categories(
        # Initialise the output dataframe
        matches_init=initialise_cb_id_table(),
        # Crunchbase industries (categories)
        categories=categories,
        # Function that returns a list of projects in a category
        search_function=lambda x: is_cb_organisation_in_category(x, CbWrangler),
        # Specify whether to return the full table (with all items)
        return_only_matches=return_only_matches,
        # Logging outputs
        verbose=verbose,
    )
