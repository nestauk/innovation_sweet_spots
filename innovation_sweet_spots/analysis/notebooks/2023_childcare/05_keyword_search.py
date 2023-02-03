# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %%
import utils
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
import pandas as pd
company_list_funding = pd.read_csv(utils.PROJECT_INPUTS_DIR / 'init_company_list_funding.csv')

# %%
already_known_ids = set(company_list_funding.cb_id.to_list())
len(already_known_ids)

# %%
import importlib
importlib.reload(utils)

# %%
# Fetch companies (identifiers) in categories related to children
# childcare_industry_ids = utils.query_categories(utils.CHILDCARE_INDUSTRIES, CB)
childcare_industry_ids = utils.query_categories(['child care'], CB)

# %%
new_ids = childcare_industry_ids.difference(already_known_ids)
len(new_ids)

# %%
df = CB.cb_organisations.query("id in @new_ids")

# %%
df.to_csv(utils.PROJECT_INPUTS_DIR / "test_childcare_industry_hits.csv", index=False)

# %%
