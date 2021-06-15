# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import json
from innovation_sweet_spots import PROJECT_DIR

GUARDIAN_PATH = PROJECT_DIR / "inputs/data/guardian"


# %%
def get_guardian_json(filename="sections.json"):
    with open(GUARDIAN_PATH / filename, "r") as infile:
        sections = json.load(infile)
    return sections


# %%
sections = get_guardian_json()

# %%
len(sections["response"]["results"])

# %%
sections_webTitle = [sect["webTitle"] for sect in sections["response"]["results"]]


# %%
sections_webTitle

# %%
tags = get_guardian_json("tags_test_page1_pagesize100.json")

# %%
641

# %%
# tags['response']

# %%
