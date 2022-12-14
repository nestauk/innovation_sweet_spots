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
# # Additional Guardian keyword analysis

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as au

# %%
CATEGORIES = []

REQUIRED_TERMS = [
    # "UK",
    # "Britain",
    # "Scotland",
    # "Wales",
    # "England",
    # "Northern Ireland",
    # "Britons",
    # "London",
]


# %%
query_id = "reformulation"
search_terms = ["food,reformulation", "food reformulation", "reformulated food"]

# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
query_id = "obesity"
search_terms = ["obesity", "obese"]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
query_id = "food_environment"
search_terms = ["food environment", "food environments"]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
query_id = "overweight"
search_terms = ["overweight"]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
query_id = "healthy_eating"
search_terms = [
    "healthy food",
    "healthy foods",
    "healthy eating",
    "healthy meal",
    "healthy meals",
]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
query_id = "obesity_healthy_eating"
search_terms = [
    "obesity",
    "obese",
    "overweight",
    "healthy food",
    "healthy foods",
    "healthy eating",
    "healthy meal",
    "healthy meals",
    "healthy diet",
    "healthy diets",
]

# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=False,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
# query_id = "health"
# search_terms = [
#     "health",
#     "healthy",
#     "healthier",
# ]

# %%
# articles, metadata = au.get_guardian_articles(
#     search_terms=search_terms,
#     use_cached=False,
#     allowed_categories=CATEGORIES,
#     query_identifier=query_id,
#     save_outputs=True,
# )

# %% [markdown]
# # Time series

# %%
query_id = "reformulation"
search_terms = ["food,reformulation"]
REQUIRED_TERMS = ["food"]
banned_terms = ["Australia"]

g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
g.document_text.sort_values("year")

# %%
query_id = "food_environment"
search_terms = ["food environment", "food environments"]
REQUIRED_TERMS = ["food"]
banned_terms = ["Australia"]

g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
query_id = "obesity"
search_terms = ["obesity", "obese"]
REQUIRED_TERMS = search_terms
banned_terms = ["Australia"]

g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
query_id = "overweight"
search_terms = ["overweight"]
REQUIRED_TERMS = search_terms
banned_terms = ["Australia"]

g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
query_id = "healthy_eating"
search_terms = [
    "healthy food",
    "healthy foods",
    "healthy eating",
    "healthy meal",
    "healthy meals",
]
REQUIRED_TERMS = search_terms
banned_terms = ["Australia"]

g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
