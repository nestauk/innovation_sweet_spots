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
    topic_id = GtR.gtr_topics.query("topic == @category").id.iloc[0]
    # Find the project ids in the specified category
    return GtR.link_gtr_topics.query("id == @topic_id").project_id.to_list()


def query_gtr_categories(
    categories: Iterator[str],
    GtR: GtrWrangler = GTR,
    return_only_matches: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Indicates if a project is in a research topic category

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
