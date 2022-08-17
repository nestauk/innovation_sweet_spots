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

pd.set_option("max_colwidth", 300)

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
import importlib
import innovation_sweet_spots.getters.path_utils

importlib.reload(innovation_sweet_spots.getters.path_utils)
from innovation_sweet_spots.getters import gtr_2022 as gtr

importlib.reload(gtr)

# %%
projects = load_json(GTR_FOLDER / "gtr_projects-projects.json")

# %%
projects_df = gtr.get_gtr_projects()

# %%
from pandas_profiling import ProfileReport

profile = ProfileReport(projects_df, title="GtR projects report")
profile.to_file(GTR_FOLDER / "gtr_projects-projects.html")

# %%
projects_df.info()

# %%
import innovation_sweet_spots.utils.text_processing_utils as tpu

importlib.reload(tpu)

COLUMNS = ["abstractText", "techAbstractText"]
text_documents = tpu.create_documents_from_dataframe(projects_df, columns=COLUMNS)

# %%
from matplotlib import pyplot as plt

# %matplotlib inline
plt.hist([len(s) for s in text_documents], bins=50)

# %%
projects_df[
    ["title", "abstractText", "start", "end", "techAbstractText", "id", "project_id"]
]

# %%
projects_df

# %%
df_old = pd.read_csv(OLD_FOLDER / "gtr_projects.csv")

# %%
df_old.info()

# %% [markdown]
# ## Research funding
#
# - Duplicated entries of funding (different `id` but same `project_id`)

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

# %% [markdown]
# ### Quick keyword check

# %%
