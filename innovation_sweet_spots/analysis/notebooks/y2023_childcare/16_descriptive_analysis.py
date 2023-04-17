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

# - Download data
# - Remove countries outside of scope
# - Descriptive analysis (number of companies per category etc)
# - Trends per category
# - Visualise trends
#
# - Discuss improvements to the taxonomy and categorisation
# - Approach to validation

import innovation_sweet_spots.utils.google_sheets as gs

df_final = gs.download_google_sheet(
    google_sheet_id="141iLNJ5e4NHlsxf73L0GX3LMxmZk-VxIDxoYug5Aglg",
    wks_name="list_v3",
)

df_final
