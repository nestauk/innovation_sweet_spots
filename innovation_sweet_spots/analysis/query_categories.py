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
from typing import Iterator

# Initialise data wrangler instances
GTR = GtrWrangler()
CB = CrunchbaseWrangler()

# To-do -- generalise/optimise the query_categories methods
def query_cb_categories(
    categories: Iterator[str],
    CbWrangler: CrunchbaseWrangler = CB,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """ """
    # Initialise the output dataframe
    matches = CbWrangler.cb_organisations[["id"]]
    orgs_in_any_category = set()
    for category in categories:
        orgs_in_category = CbWrangler.get_companies_in_industries(
            [category]
        ).id.to_list()
        if verbose:
            logging.info(
                f"Found {len(orgs_in_category)} organisations in the category '{category}'"
            )
        matches[category] = matches["id"].isin(orgs_in_category)
        orgs_in_any_category = orgs_in_any_category | set(orgs_in_category)
    matches["any_category"] = matches["id"].isin(orgs_in_any_category)
    if return_only_matches:
        return matches.query("any_category == True")
    else:
        return matches


def is_gtr_project_in_category(category: str, GtR: GtrWrangler = GTR) -> Iterator[str]:
    """
    Returns project ids that belong to a GtR research topic

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
        # Check that there is one unique GtR research topic id
        assert (
            len(topic_id_df) == 1
        ), "There is more than one topic id with the same name: Something wrong with GtrWrangler.gtr_topics"
        topic_id = topic_id_df.id.iloc[0]
        # Find the project ids in the specified category
        return GtR.link_gtr_topics.query("id == @topic_id").project_id.to_list()


def query_gtr_categories(
    categories: Iterator[str],
    GtR: GtrWrangler = GTR,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Indicates if a project is the provided research topics.
    The research topic categories must be present in GtR.gtr_topics

    Args:
        category: GtR research topic category
        GtR: GtrWrangler instance
        return_only_matches: If True, will only return the documents that are matches
        verbose: If True, will show logging info

    Returns:
        A dataframe with the following columns:
            - a column for project identifiers
            - a boolean column for each of the categories, where True indicates that project belongs to it
            - a column 'any_category' which indicates if any categories where matched
    """
    # Initialise the output dataframe
    matches = GtR.gtr_projects[["project_id"]].rename(columns={"project_id": "id"})
    projects_in_any_category = set()
    for category in categories:
        # Check if projects are in the category
        projects_in_category = is_gtr_project_in_category(category, GtR)
        if verbose:
            logging.info(
                f"Found {len(projects_in_category)} projects in the category '{category}'"
            )
        # Store the result
        matches[category] = matches["id"].isin(projects_in_category)
        projects_in_any_category = projects_in_any_category | set(projects_in_category)
    matches["any_category"] = matches["id"].isin(projects_in_any_category)
    if return_only_matches:
        return matches.query("any_category == True")
    else:
        return matches
