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
# ---

# # Consolidate results and develop final model
#
# - Review disagreements between model v1 and ChatGPT
#   - eg: Where only one model classifies as non-relevant
# - Load in labelled data into notebook
# - Load in ChatGPT results and inspect under-represented labels
#   - For under-represented labels, consider labelling more of those
# - Produce a consolidated dataset with labels
# - Train a final model on the consolidated dataset
# - Check which countries are we including, and which we should still add
# - Write up all the analysis steps again
