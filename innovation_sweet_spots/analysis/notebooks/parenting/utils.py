"""
innovation_sweet_spots.analysis.notebooks.parenting.utils

Helper module for analysing parenting companies
"""
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from typing import Iterator

CB = CrunchbaseWrangler()


def add_space_in_front(list_of_term_lists: Iterator[Iterator[str]]):
    """Adds a space in front of each search term"""
    return [f" {term}" for term_list in list_of_term_lists for term in term_list]


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
    "consumer electronics",
]

# Additional industries to add to the digital industry list
EXTRA_INDUSTRIES = [
    "toys",
]

# Digital industries
DIGITAL_INDUSTRIES = sorted(
    CB.get_all_industries_from_groups(DIGITAL_INDUSTRY_GROUPS) + EXTRA_INDUSTRIES
)

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
