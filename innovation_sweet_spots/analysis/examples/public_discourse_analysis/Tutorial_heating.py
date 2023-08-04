# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Collect Guardian articles, subset relevant documents and extract text
#
# - Fetch articles using Guardian API
# - Extract metadata and text from html

# %% [markdown]
# ## 1. Import dependencies

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as au

# import innovation_sweet_spots.utils.plotting_utils as pu
# import altair as alt


# %%
CATEGORIES = [
    "Environment",
    "Guardian Sustainable Business",
    "Technology",
    "Science",
    "Business",
    "Money",
    "Cities",
    "Politics",
    "Opinion",
    "Global Cleantech 100",
    "The big energy debate",
    "UK news",
    "Life and style",
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


# %% [markdown]
# # Heat pumps

# %%
import importlib

importlib.reload(au)

# %%
query_id = "hp"
search_terms = ["heat pump", "heat pumps"]
banned_terms = ["Australia"]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=True,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
importlib.reload(au)
g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=REQUIRED_TERMS,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
len(g.document_text)

# %%
g.plot_document_mentions()

# %%
# x = g.view_collocations('government')
x

# %%
x = g.view_mentions(["government"])


# %%
len(x)

# %%
g.view_collocations("restaurant")

# %%
g.save_analysis_results()

# %% [markdown]
# # Hydrogen

# %%
importlib.reload(au)

# %%
disambiguation_terms = [
    "home",
    "homes",
    "heating, cooling",
    "hot water",
    "electricity",
    "boiler",
    "boilers",
    "house",
    "houses",
    "building",
    "buildings",
    "radiators",
    "low carbon",
    "carbon emissions",
    "domestic",
    "heating and cooling",
    "heating fuel",
    "heating systems",
    "heating schemes",
    "hydrogen for heating",
    "electric heating",
    "electrification of heating",
    "electrifying heating",
    "heating and lighting",
    "district heating",
    "clean burning hydrogen",
    "hydrogen gas heating",
    "gas fired hydrogen",
    "gas grid",
    "climate targets",
    "climate goals",
    "households",
    "energy grid",
    "energy grids",
    "central heating",
    "heating homes",
    "net zero",
    "net-zero",
    "appliances",
    "hobs",
]


# %%
query_id = "hydrogen"
search_terms = ["hydrogen"]

filtering_terms = [REQUIRED_TERMS, disambiguation_terms]
banned_terms = ["peroxide", "Australia"]

# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=True,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=filtering_terms,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.plot_document_mentions()

# %%
importlib.reload(au.dpu)

# %%
heating_terms = [
    "heat",
    "heating",
    "boiler",
    "boilers",
]

# %%
g.view_collocations_terms(heating_terms)

# %% [markdown]
# # Hydrogen heating

# %%
importlib.reload(au)

# %%
query_id = "hydrogen_heat"
# search_terms = ["hydrogen", "hydrogen ready", "hydrogen-ready"]

search_terms = [
    "hydrogen boiler",
    "hydrogen boilers",
    "hydrogen-ready boiler",
    "hydrogen-ready boilers",
    "hydrogen ready boiler",
    "hydrogen ready boilers",
    "hydrogen heating",
    "hydrogen heat",
    "hydrogen for heat",
    "hydrogen for heating",
    "heat with hydrogen",
    "heating with hydrogen",
]

filtering_terms = [REQUIRED_TERMS]
banned_terms = ["peroxide", "Australia"]


# %%
articles, metadata = au.get_guardian_articles(
    search_terms=search_terms,
    use_cached=True,
    allowed_categories=CATEGORIES,
    query_identifier=query_id,
    save_outputs=True,
)

# %%
g = au.DiscourseAnalysis(
    search_terms=search_terms,
    required_terms=filtering_terms,
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
g.save_preprocessed_data()

# %%
g.plot_document_mentions()
