# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR, config
from data_getters.inspector import get_schemas
from data_getters.core import get_engine
import pandas as pd

schemas = get_schemas(config["database_config_path"])

# %%
# List all databases
for key in schemas.keys():
    print(key)

# %%
# GTR tables
for key in schemas["gtr"].keys():
    print(key)

# %%
# Crunchbase tables
for key in schemas["crunchbase"].keys():
    print(key)

# %%
for x in list(schemas["crunchbase"]["crunchbase_organizations"].columns):
    print(x)

# %%
con = get_engine(config["database_config_path"])
chunks = pd.read_sql_table(
    "crunchbase_organizations", con, columns=["long_description"], chunksize=1000
)

# %% [markdown]
# There are around 1.15M companies in the Crunchbase database

# %%
