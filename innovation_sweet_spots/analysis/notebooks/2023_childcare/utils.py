"""
This file contains the utils and constants used in the notebooks
"""
from innovation_sweet_spots import PROJECT_DIR

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


CHILDCARE_INDUSTRIES = [
    "child care",
    "children",
    "underserved children",
    "family",
    "baby",
]
