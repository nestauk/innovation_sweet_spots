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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Collect Guardian articles, subset relevant documents and extract text
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html

# ## 1. Import dependencies

from innovation_sweet_spots.utils.pd import pd_analysis_utils as au


# +
CATEGORIES = [
    # "Environment",
    # "Guardian Sustainable Business",
    # "Technology",
    # "Science",
    # "Business",
    # "Money",
    # "Cities",
    # "Politics",
    # "Opinion",
    # "Global Cleantech 100",
    # "The big energy debate",
    # "UK news",
    # "Life and style",
]

REQUIRED_TERMS = [
    "UK",
    "Britain",
    "Scotland",
    "Wales",
    "England",
    "Northern Ireland",
    "Britons",
    "London",
]
# -


# # Trying out search terms

# +
import importlib

importlib.reload(au)
# -

query_id = "foodtech_plants"
search_terms = [
    "plant based meat",
    "plant-based meat",
    "plant based burger",
    "plant-based burger",
]
banned_terms = []


articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

importlib.reload(au)
g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=False,
    query_identifier=query_id,
)

len(g.document_text)

g.plot_document_mentions()

g.save_analysis_results()

g.sentence_mentions

g.view_mentions("plant based burger")
