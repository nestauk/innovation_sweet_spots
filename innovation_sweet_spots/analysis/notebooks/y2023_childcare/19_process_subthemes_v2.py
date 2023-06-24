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
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Descriptive analysis
#
# - [done] Download data
# - [done] Remove countries outside of scope
# - Descriptive analysis (number of companies per category etc)
# - Trends per category
# - Visualise trends
#
# - Discuss improvements to the taxonomy and categorisation
# - Approach to validation

# +
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.utils.google_sheets as gs
from innovation_sweet_spots.analysis import wrangling_utils as wu
import re
import utils

import importlib

importlib.reload(utils)
# -

CB = wu.CrunchbaseWrangler()

# ## Load and process data

# +
# Download data with partially corrected subthemes
subthemes_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID_APRIL,
    wks_name="list_v3",
)

# # Download company data
# longlist_df = gs.download_google_sheet(
#     google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
#     wks_name="list_v2",
# )
# -

subthemes_df[subthemes_df.company_name.str.contains("Age of Learning")]

# Download data with partially corrected subthemes
taxonomy_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID_APRIL,
    wks_name="taxonomy_final",
)

# +
# Companies in the countries in scope
companies_in_countries = subthemes_df.query(
    "country in @utils.list_of_select_countries"
).cb_id.to_list()

# Final list of included companies
childcare_df = (
    subthemes_df.copy()
    .assign(
        keep=lambda df: df.keep.str.lower().apply(
            lambda x: True if x == "true" else False
        )
    )
    .query("cb_id in @companies_in_countries")
    .query("keep == True")
)
# -

childcare_categories_df = (
    childcare_df[["cb_id", "company_name", "model_subthemes"]]
    .rename(columns={"model_subthemes": "subtheme_tag"})
    .merge(taxonomy_df, on="subtheme_tag", how="left")
    .query("subtheme_tag != '<other>'")
    .drop_duplicates(["cb_id", "subtheme_tag"])
)

len(childcare_categories_df)

childcare_categories_df

childcare_categories_df[
    childcare_categories_df.cb_id.str.contains("79e378a1-2e03-b378-3e33-70f8e123944e")
]

childcare_categories_df.to_csv(
    PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_06_06.csv",
    index=False,
)

# +
# childcare_categories_df.to_csv(
#     PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_05_16.csv",
#     index=False,
# )
# -

# ## Descriptive analysis
#

#
