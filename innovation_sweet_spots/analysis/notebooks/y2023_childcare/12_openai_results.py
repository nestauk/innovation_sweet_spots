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
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_pickle
import innovation_sweet_spots.getters.google_sheets as gs
import utils
import pandas as pd

# +
from ast import literal_eval
import re
from typing import List


def extract_answer(text: str) -> str:
    """Extract the answer from the OpenAI response"""
    text = text.split('"content": ')[1].split(",")[0]
    return text


def split_category_string(text: str) -> List[str]:
    # remove double quotes
    text = text.replace('"', "")
    if " and " in text:
        return text.split(" and ")
    else:
        return [text]


# -

# Get longlist of companies
longlist_df = utils.gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)

# check country counts
pd.DataFrame(
    longlist_df.astype({"model_relevant": float})
    .query("model_relevant > 0.5")
    .country.value_counts()
).head(20)

longlist_df.relevant.value_counts()

len(longlist_df)

utils.list_of_select_countries

longlist_df.columns

# +
FIELDS = [
    "id",
    "object",
    "created",
    "model",
    "usage",
    "choices",
    "cb_id",
]

openai_df = pd.read_csv(
    PROJECT_DIR
    / "outputs/2023_childcare/interim/openai/chatgpt_labels_v2023_03_14.csv",
    header=None,
    names=FIELDS,
)
# -

openai_df["chatgpt_subthemes"] = openai_df["choices"].apply(lambda x: extract_answer(x))

longlist_df_chatgpt = longlist_df.merge(
    openai_df[["cb_id", "chatgpt_subthemes"]], on="cb_id", how="left"
)

# +
# # Reupload to google sheets
# import innovation_sweet_spots.utils.google_sheets as gs
# gs.upload_to_google_sheet(
#     df=longlist_df_chatgpt,
#     google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
#     wks_name="list_v2_chatgpt",
# )
# -

longlist_df.columns

# +
# Fixing dict
fixing_dict = {
    "<Category: Tech>": "<Tech>",
    "<Category: Family support: General>": "<Family support: General>",
    "<Category: Play>": "<Content: Play>",
    "<Content: Tech>": "<Tech>",
    "<Creative>": "<Content: Creative>",
    "<Category: Traditional models: Child care>": "<Traditional models: Child care>",
    "<Category: Content: General>": "<Content: General>",
    "<Category: Creative>": "<Content: Creative>",
    "<Category: Not relevant>": "<Not relevant>",
    "<Category not provided>": "<Not relevant>",
    "<Category: <Tech>>": "<Tech>",
    "<Category: Traditional models: Preschool>": "<Traditional models: Preschool>",
    "<Content: General>>": "<Content: General>",
    "<Category: Content: Literacy>": "<Content: Literacy>",
    "<Not relevant> (as the company does not provide childcare": "<Not relevant>",
    "<Category: <Traditional models: Child care>>": "<Traditional models: Child care>",
    "<Category: Creative> or <Content: Literacy>": "<Content: Creative>",
    "<Traditional models: Child care> (assuming that the scientific experiment": "<Traditional models: Child care>",
    "entertainment activities are provided in a child care setting)": "<Content: Play>",
    "<Category not provided> - The description is not clear enough to match any of the predefined categories.": "<Not relevant>",
    "<Category: <Content: General>>": "<Content: General>",
    "<Category: <Tech>": "<Tech>",
    "<Category: Finances>": "<Family support: Finances>",
    "<Traditional models: Early Childhood Education>": "<Traditional models: Preschool>",
    "<Category: <Content: Play>>": "<Content: Play>",
    "<Category: Content: Creative>": "<Content: Creative>",
    "<Category: Content: Play>": "<Content: Play>",
    "It is not clear from the description which specific educational services the company provides. Therefore": "<Not relevant>",
    "<Family support: Special needs>": "<Traditional models: Special needs>",
    "<Not relevant> (as the company provides education beyond early years education)": "<Not relevant>",
    "<Workforce: Optimisation>": "<Workforce: Optimisation>",
    "<Traditional models: Online education>": "<Not relevant>",
    "<Not relevant> (for the given industries)": "<Not relevant>",
    "<Family support: General> (specifically <Family support: General - travel": "<Family support: General>",
    "<Workforce: Recruitment>": "<Workforce: Recruitment>",
    "<Finances>": "<Family support: Finances>",
    "<Play>": "<Content: Play>",
    "<Category: Management>": "<Management>",
    "<Category: <Play>>": "<Content: Play>",
    "<Not relevant> (as it is not related to childcare": "<Not relevant>",
    "<Category: Family support: Peers>": "<Family support: Peers>",
    "<Not relevant> (as the company is not working on improving childcare": "<Not relevant>",
    "<Category: Content: Numeracy>": "<Content: Numeracy>",
    "<Traditional model: Child care>": "<Traditional models: Child care>",
}


def fix_categories(text: str) -> str:
    if text in fixing_dict.keys():
        return fixing_dict[text.strip()]
    else:
        return text


# +
relevant_labels = ["1", "0"]
id_to_subtheme_df = (
    longlist_df
    # For all rows where relevant=0, make columnn chatgpt_subthemes equal to "<Not relevant>"
    .assign(
        chatgpt_subthemes=lambda df: df.apply(
            lambda x: "<Not relevant>" if x.relevant == "0" else x.chatgpt_subthemes,
            axis=1,
        )
    )
    .query("relevant in @relevant_labels")
    .assign(
        subthemes_list=lambda df: df.chatgpt_subthemes.apply(
            lambda x: split_category_string(x)
        )
    )
    .explode("subthemes_list")
    .assign(
        subthemes_list=lambda df: df.subthemes_list.apply(lambda x: fix_categories(x))
    )
)

# # distribution of subthemes
# df = pd.DataFrame(id_to_subtheme_df.subthemes_list.value_counts()).tail(30).reset_index().rename(columns={'index': 'subtheme'})
# for i, row in df.iterrows():
#     print(f'"{row["subtheme"]}": ')
id_to_subtheme_df.subthemes_list.value_counts()
# -

id_to_subtheme_df.to_csv(
    PROJECT_DIR
    / "outputs/2023_childcare/interim/openai/subtheme_labels_v2023_04_06.csv",
    index=False,
)
