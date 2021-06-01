"""
Notebook for exploring Nesta database schemas. It can also be used to manually
specify the Crunchbase tables and columns of interest that ought to be downloaded.

"""

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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook for exploring database schemas

# %%
from innovation_sweet_spots import PROJECT_DIR, config
from data_getters.inspector import get_schemas
from data_getters.core import get_engine
import pandas as pd

schemas = get_schemas(config["database_config_path"])


# %%
def print_column_names(table):
    for x in list(table.columns):
        print(x.name)


# %%
# List all databases
for key in schemas.keys():
    print(key)

# %% [markdown]
# # GTR schemas

# %%
# GTR tables
for key in schemas["gtr"].keys():
    print(key)

# %% [markdown]
# # Crunchbase schemas

# %%
# Crunchbase tables
for key in schemas["crunchbase"].keys():
    print(key)

# %%
print_column_names(schemas["crunchbase"]["crunchbase_funds"])

# %%
print_column_names(schemas["crunchbase"]["crunchbase_organizations_categories"])

# %%
# Use this get the tables from the database
con = get_engine(config["database_config_path"])
chunks = pd.read_sql_table("crunchbase_organizations_categories", con, chunksize=1000)
chunks = [c for c in chunks]
df = pd.concat(chunks, axis=0)

# %% [markdown]
# ### Specify the relevant fields for analysis

# %%
# Relevant tables and their fields, follows the format: [{'table_name': ['field_1', 'field_2'...]}]
cb_data_spec = {
    "crunchbase_organizations": [
        "name",
        "city",
        "country",
        "country_code",
        "employee_count",
        "founded_on",
        "closed_on",
        "short_description",
        "long_description",
        "primary_role",
        "roles",
        "status",
        "updated_at",
        "total_funding",
        "total_funding_currency_code",
        "total_funding_usd",
        "num_exits",
        "num_funding_rounds",
    ],
    "crunchbase_organizations_categories": ["category_name", "organization_id"],
}

# %%
# Export the spec as yaml
fpath = f"{PROJECT_DIR}/innovation_sweet_spots/config/cb_data_spec.yaml"
with open(fpath, "w", encoding="utf-8") as yaml_file:
    dump = yaml.dump(
        cb_data_spec, default_flow_style=False, allow_unicode=True, encoding=None
    )
    yaml_file.write(dump)

# %%
# Check
with open(fpath, "r", encoding="utf-8") as yaml_file:
    spec = yaml.safe_load(yaml_file)

# %%
spec

# %%
list(spec.keys())

# %%
