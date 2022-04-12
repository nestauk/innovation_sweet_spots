"""
innovation_sweet_spots.analysis.notebooks.parenting.utils

Helper module for analysing parenting companies
"""
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
import innovation_sweet_spots.analysis.analysis_utils as au
from typing import Iterator
import pandas as pd

CB = CrunchbaseWrangler()


def add_space_in_front(list_of_term_lists: Iterator[Iterator[str]]):
    """Adds a space in front of each search term"""
    return [f" {term}" for term_list in list_of_term_lists for term in term_list]


def get_digital_companies(cb_orgs: pd.DataFrame, cb_wrangler: CrunchbaseWrangler):
    return cb_wrangler.select_companies_by_industries(cb_orgs, DIGITAL_INDUSTRIES)


def digital_proportion(cb_orgs, digital_orgs, since: int = None):
    """Fraction of all companies in the digital industries"""
    if since is None:
        return len(digital_orgs) / len(cb_orgs)
    else:
        return len(digital_orgs.query(f"founded_on >= '{since}'")) / len(
            cb_orgs.query(f"founded_on >= '{since}'")
        )


def digital_proportion_ts(
    cb_orgs: pd.DataFrame, digital_orgs, min_year: int, max_year: int
):
    """Time series of the fraction of newly founded companies in digital industries"""
    return (
        au.cb_orgs_founded_per_period(
            cb_orgs.query(f"founded_on >= '{min_year}'"),
            period="Y",
            min_year=min_year,
            max_year=max_year,
        )
        .set_index("time_period")
        .assign(
            no_of_digital_orgs=au.cb_orgs_founded_per_period(
                digital_orgs, period="Y", min_year=min_year, max_year=max_year
            ).set_index("time_period")["no_of_orgs_founded"]
        )
        .assign(digital_fraction=lambda x: x.no_of_digital_orgs / x.no_of_orgs_founded)
        .reset_index()
    )


#### Crunchbase industry parameters ####

# Industry labels related to parents or child care
USER_INDUSTRIES = [
    "parenting",
    "child care",
    "children",
    "underserved children",
    "family",
    "baby",
]

# Industry labels related to education
EDUCATION_INDUSTRIES = [
    "education",
    "edtech",
    "e-learning",
    "edutainment",
    "language learning",
    "mooc",
    "music education",
    "personal development",
    "skill assessment",
    "stem education",
    "tutoring",
    "training",
    "primary education",
    "continuing education",
    "charter schools",
]

# Industries that are definitely unrelated to preschool education
INDUSTRIES_TO_REMOVE = [
    "secondary education",
    "higher education",
    "universities",
    "vocational education",
    "corporate training",
    "college recruiting",
]

# Digital industry groups
DIGITAL_INDUSTRY_GROUPS = [
    "information technology",
    "hardware",
    "software",
    "mobile",
    "consumer electronics",
    "music and audio",
    "gaming",
    "design",
    "privacy and security",
    "messaging and telecommunications",
    "internet services",
    "artificial intelligence",
    "media and entertainment",
    "platforms",
    "data and analytics",
    "apps",
    "video",
    "content and publishing",
    "advertising",
]

EXCLUDE_FROM_DIGITAL = [
    "consumer research",
    "fashion",
    "industrial design",
    "interior design",
    "mechanical design",
    "product design",
    "product research",
    "usability testing",
]

# Additional industries to add to the digital industry list
# EXTRA_INDUSTRIES = [
#     "toys",
# ]

# Digital industries
DIGITAL_INDUSTRIES = [
    i
    for i in sorted(CB.get_all_industries_from_groups(DIGITAL_INDUSTRY_GROUPS))
    if i not in EXCLUDE_FROM_DIGITAL
]

#### Search term parameters ####

USER_TERMS = [
    ["parent"],
    ["mother"],
    ["mom "],
    ["moms "],
    ["father"],
    ["dad "],
    ["dads "],
    ["baby"],
    ["babies"],
    ["infant"],
    ["child"],
    ["toddler"],
    ["kid "],
    ["kids "],
    ["son "],
    ["sons "],
    ["daughter"],
    ["boy"],
    ["girl"],
]

LEARNING_TERMS = [
    ["learn"],
    ["educat"],
    ["develop"],
    ["study"],
]

PRESCHOOL_TERMS = [
    ["preschool"],
    ["pre school"],
    ["kindergarten"],
    ["pre k "],
    ["montessori"],
    ["literacy"],
    ["numeracy"],
    ["math"],
    ["phonics"],
    ["early year"],
]

USER_TERMS = add_space_in_front(USER_TERMS)
LEARNING_TERMS = add_space_in_front(LEARNING_TERMS)
PRESCHOOL_TERMS = add_space_in_front(PRESCHOOL_TERMS)
ALL_LEARNING_TERMS = LEARNING_TERMS + PRESCHOOL_TERMS
