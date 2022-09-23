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
# ## Cross-reference with Innovate UK data

# %%
from innovation_sweet_spots.getters import google_sheets
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

from innovation_sweet_spots.getters import gtr_2022 as gtr

# %%
gtr_projects = gtr.get_wrangled_projects()

# %%
gtr_projects = gtr_projects[gtr_projects.title.apply(lambda x: type(x) is str) == True]

# %%
gtr_projects_ = gtr.get_gtr_projects()

# %%
ukri_df_reviewed = google_sheets.get_foodtech_reviewed_gtr(from_local=False)

# %%
ukri_df_reviewed.info()

# %%
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR

innovate_uk_df = pd.read_excel(
    PROJECT_DIR / "inputs/data/misc/innovate_uk/Masterfile_Food.xlsx"
)

# %%
import re


def replace_stopchars(text):
    text = re.sub(" amp ", " ", text)
    text = re.sub(" quot ", " ", text)
    return text


# %%
from innovation_sweet_spots.utils import text_processing_utils

# %%
innovate_projects = innovate_uk_df["Project Title"].unique()

# %%
innovate_uk_projects_df = innovate_uk_df.drop_duplicates(
    ["Project Title"], keep="first"
).reset_index(drop=True)

# %%
# GtR all titles
gtr_projects["title_processed"] = gtr_projects.title.apply(
    text_processing_utils.preprocess_clean_text
)

# %%
gtr_projects["title_processed_"] = gtr_projects["title_processed"].apply(
    replace_stopchars
)

# %%
# Innovate UK table, all titles
innovate_uk_projects_df["title_processed"] = innovate_uk_projects_df[
    "Project Title"
].apply(text_processing_utils.preprocess_clean_text)

# %%
innovate_uk_projects_df["title_processed_"] = innovate_uk_projects_df[
    "title_processed"
].apply(replace_stopchars)

# %%
# Innovate UK projects that are not in the GtR projects
df_not_included = innovate_uk_projects_df[
    innovate_uk_projects_df.title_processed_.isin(
        gtr_projects.title_processed_.to_list()
    )
    == False
]

# %%
len(df_not_included)

# %%
gtr_projects[gtr_projects.title_processed_.str.contains("mycoprotein")].title_processed_

# %%
gtr_projects[gtr_projects.title.str.contains("ptibix")].title

# %%
for p in df_not_included["title_processed"]:
    print(p)
    print("\n---")

# %%
innovate_uk_projects_df_ = innovate_uk_projects_df[
    innovate_uk_projects_df["Project Title"].isin(
        df_not_included["Project Title"].to_list()
    )
    == False
]
len(innovate_uk_projects_df_)


# %% [markdown]
# ## Check reviewed projects

# %%
ukri_df_reviewed["title_processed"] = ukri_df_reviewed.title.apply(
    text_processing_utils.preprocess_clean_text
)
ukri_df_reviewed["title_processed_"] = ukri_df_reviewed.title_processed.apply(
    replace_stopchars
)


# %%
ukri_df_reviewed.title_processed_.isin(
    innovate_uk_projects_df.title_processed_.to_list()
).sum()

# %%
gtr_projects.title.isin(innovate_projects).sum()

# %%
# Innovate UK projects that are not in the reviewed projects
df_not_included_ = innovate_uk_projects_df[
    innovate_uk_projects_df.title_processed_.isin(
        ukri_df_reviewed.title_processed_.to_list()
    )
    == False
]


# %%
gtr_columns_to_export = [
    "id",
    "title",
    "abstractText",
    "techAbstractText",
    "grantCategory",
    "leadFunder",
    "leadOrganisationDepartment",
    "found_terms",
    "tech_area",
]

# %%
ids_to_check = gtr_projects[
    gtr_projects.title_processed_.isin(df_not_included_.title_processed_.to_list())
].project_id.to_list()


# %%
gtr_projects_to_check = (
    gtr_projects_.query("id in @ids_to_check")
    .assign(found_terms="[]")
    .assign(tech_area="-")[gtr_columns_to_export]
)


# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/foodtech/interim/research_funding/"
gtr_projects_to_check.to_csv(
    OUTPUTS_DIR / "gtr_projects_v2022_09_14_innovateUK.csv", index=False
)

# %%
for p in df_not_included_["Project Title"].iloc[10:20]:
    print(p)

# %% [markdown]
# - Projects in the Innovate UK table that are not in the GtR dataset
# - Projects in the Innovate UK table that are not in our reviewed set

# %%
