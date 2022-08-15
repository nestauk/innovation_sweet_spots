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
# # Checking the updated GtR data
# August 2022

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_json
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler
import os
import pandas as pd

# %%
GTR_FOLDER = PROJECT_DIR / "inputs/data/gtr_2022_august"
OLD_FOLDER = PROJECT_DIR / "inputs/data/gtr"

# %%
# Check the files in the data folder
os.listdir(GTR_FOLDER)

# %%
# Check the files in the data folder
os.listdir(OLD_FOLDER)

# %%
# Initialise GtR wrangler
GTR = GtrWrangler()

# %% [markdown]
# ## Research topics

# %%
topics = load_json(GTR_FOLDER / "gtr_projects-topic.json")
len(topics)

# %%
topics[0]

# %%
pd.read_csv(OLD_FOLDER / "gtr_topic.csv").head(2)

# %% [markdown]
# ## Research projects

# %%
projects = load_json(GTR_FOLDER / "gtr_projects-projects.json")
len(projects)

# %%
sorted(list(projects[0].keys()))

# %%
projects[21]

# %%
pd.read_csv(OLD_FOLDER / "gtr_projects.csv").head(2)

# %% [markdown]
# ## Research funding

# %%
funds = load_json(GTR_FOLDER / "gtr_projects-funds.json")
len(funds)

# %%
df = pd.DataFrame(funds)
df.project_id.duplicated().sum()

# %%
df.project_id.duplicated().sum()

# %%
df[df.project_id.duplicated(keep=False)].sort_values("project_id").head(20)

# %%
