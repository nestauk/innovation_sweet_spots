# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Finding companies related to early years and parenting
#
# This script queries Crunchbase data by industry categories and keywords, to find a set of companies that are related to (1) early years education, and (2) parenting.
#
# - Inputs: Crunchbase data on companies (global)
# - Outputs: Two csv tables with potentially relevant companies and their data
#
# The exported lists of companies were then manually reviewed based on their descriptions and information on their website, to select only relevant companies. For example, we aimed to exclude companies specialising in services for children older than 5 years.
#
# During the manual review, each company was also assigned either a 'Parents' or 'Children' tag to indicate the primary user of the products or services developed by the companies.
#
# Note: For this analysis, we used a data snapshot last updated in March 2022. If you repeat this process with more recently updated data then you might get a different (larger) set of companies.

# %%
from innovation_sweet_spots.analysis.notebooks.parenting import utils
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots import PROJECT_DIR

CB = CrunchbaseWrangler()

OUTPUTS_DIR = PROJECT_DIR / "outputs/parenting/cb_companies"
COLUMNS_TO_EXPORT = [
    "id",
    "name",
    "short_description",
    "long_description",
    "country",
    "cb_url",
    "homepage_url",
    "industry",
]


# %%
if __name__ == "__main__":

    ### COMPANIES RELATED TO EARLY YEARS EDUCATION

    ## Approach 1, using company industry categories
    # Fetch companies (identifiers) in categories related to children
    children_industry_ids = utils.query_keywords(utils.CHILDREN_INDUSTRIES, CB)
    # Fetch companies in education categories
    education_industry_ids = utils.query_keywords(utils.EDUCATION_INDUSTRIES, CB)
    # Fetch companies in 'stopword' industries, which we wish to remove
    remove_industry_ids = utils.query_keywords(utils.INDUSTRIES_TO_REMOVE, CB)
    # Companies that are in industries related to children AND education AND NOT in the 'stopword' industries
    children_education_ids = children_industry_ids.intersection(
        education_industry_ids
    ).difference(remove_industry_ids)

    ## Approach 2, using keywords
    # Fetch tokenised Crunchbase corpus
    corpus_full = get_full_crunchbase_corpus()
    Query = QueryTerms(corpus=corpus_full)
    # Query by keywords related to children AND education
    children_term_ids = set(
        Query.find_matches(utils.CHILDREN_TERMS, return_only_matches=True).id.to_list()
    )
    education_term_ids = set(
        Query.find_matches(
            utils.ALL_LEARNING_TERMS, return_only_matches=True
        ).id.to_list()
    )
    children_education_term_ids = children_term_ids.intersection(education_term_ids)

    ## Combine selections from both approaches
    children_education_ids_all = children_education_ids.union(
        children_education_term_ids
    )
    # Keep companies that have data on investment deals
    child_ed_companies_with_funds = utils.select_companies_with_funds(
        children_education_ids_all, CB
    )
    # Export
    (
        child_ed_companies_with_funds.merge(
            CB.get_company_industries(child_ed_companies_with_funds, return_lists=True),
            on=["id", "name"],
        )[COLUMNS_TO_EXPORT].to_csv(
            OUTPUTS_DIR / "cb_companies_child_ed_v2022_04_27.csv", index=False
        )
    )

    ### COMPANIES RELATED TO PARENTING

    ## Simpler approach: Select companies that are in the parenting category
    cb_orgs_parenting = (
        CB.get_companies_in_industries(utils.PARENT_INDUSTRIES)
        .pipe(utils.select_by_role, "company")
        .pipe(au.get_companies_with_funds)
    )
    # Export
    (
        cb_orgs_parenting.merge(
            CB.get_company_industries(child_ed_companies_with_funds, return_lists=True),
            on=["id", "name"],
        )[COLUMNS_TO_EXPORT].to_csv(
            OUTPUTS_DIR / "cb_companies_parenting_v2022_04_27.csv", index=False
        )
    )
