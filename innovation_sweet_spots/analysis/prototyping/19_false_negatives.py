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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Checking for false negatives
#
# Have we excluded important GTR projects from the analysis

# %%
from innovation_sweet_spots.utils.io import load_pickle, save_pickle
from innovation_sweet_spots import PROJECT_DIR, logging

RESULTS_DIR = PROJECT_DIR / "outputs/data/results_july"

import innovation_sweet_spots.getters.gtr as gtr
from innovation_sweet_spots.analysis.analysis_utils import (
    create_documents_from_dataframe,
)
from innovation_sweet_spots.utils.text_pre_processing import ngrammer, process_text
from innovation_sweet_spots.analysis.text_analysis import (
    setup_spacy_model,
    DEF_LANGUAGE_MODEL,
)

nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)

from innovation_sweet_spots.analysis.green_document_utils import (
    find_green_gtr_projects,
)

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# %%
# Import guided topics
guided_topic_dict = load_pickle(
    RESULTS_DIR / "guidedLDA_July2021_projects_orgs_stopwords_e400.p"
)
model = guided_topic_dict["model"]

# %%
# Import all GTR projects
gtr_projects = gtr.get_gtr_projects()

# %%
# Green projects
green_projects = find_green_gtr_projects()

# %%
projects_to_check = gtr_projects[
    -gtr_projects.project_id.isin(green_projects.project_id.to_list())
]

# %%
# Create documents
project_texts = create_documents_from_dataframe(
    gtr_projects,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)

# %%
# Create documents
project_to_check_texts = create_documents_from_dataframe(
    projects_to_check,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)

# %%
# Import text processor
ngram_phraser = load_pickle(PROJECT_DIR / "outputs/models/bigram_phraser_gtr_cb.p")


# %%
# [process_text(doc) for doc in nlp.pipe(corpus)]

# %%
def check_project_topic(tokenised_text, return_dataframe=True):
    txt = ngrammer(tokenised_text, ngram_phraser, nlp)
    x = guided_topic_dict["vectorizer"].transform([txt])
    probs = model.transform(x)[0]
    if return_dataframe:
        topic_df = pd.DataFrame(
            data={"topic": guided_topic_dict["topics"], "probability": probs}
        ).sort_values("probability", ascending=False)
        return topic_df
    else:
        return probs


# %%
proj = gtr.get_cleaned_project_texts()

# %%
# proj.project_text[24838]

# %% [markdown]
# ## Quick checks

# %%
titles_to_check = [
    "Cryogenic-temperature Cold Storage using Micro-encapsulated Phase Change Materials in Slurries",
    "Sustainable Electric Heating System",
    "Algal biofuels: novel approaches to strain improvement",
    "A system to process and store waste heat in WWTPs and supply to heating networks",
    "Metropolitan Integrated Cooling and Heating",
    "CODES: The Control of District-heating Efficiency through Smart data-driven models",
    "District Heating Digital Canopy",
    "Advanced Burner Flame Monitoring through Digital Imaging",
    "Green Hydrogen for Humber",
    "Machine learning with anti-hydrogen",
    "Hydrogen gas inject",
    "TwinGen",
    "Remote Boiler Management System",  #
    "Remote CO monitoring for Domestic Boilers",  #
    "Island Hydrogen",  #
    "Boiler Fault Finder",  #
    "New Biomass Boiler Technology",
    "Domestic Boiler Management",  #
    "Hydrogen's value in the energy system",
    "To a 100 % hydrogen domestic boiler",  #
]

# %%
j = -1
title = titles_to_check[j]
green_projects[green_projects.title.str.contains(title)]

# %%
df = gtr_projects[gtr_projects.title.str.contains(title)]
df

# %%
# ngrammer(project_texts[df.index[0]], ngram_phraser, nlp)

# %% [markdown]
# ## Topic model checks

# %%
# import warnings


# %%
logging.disable()
docs = project_to_check_texts[0:100]
project_probs = []
for doc in tqdm(docs, total=len(docs)):
    project_probs.append(check_project_topic(doc, return_dataframe=False))
logging.disable(0)

# %%
probs_dict = {
    "projects": projects_to_check[["project_id", "title"]],
    "probs": np.array(project_probs),
    "topics": guided_topic_dict["topics"],
}
# save_pickle(probs_dict, RESULTS_DIR / 'false_positive_check.p')

# %% [markdown]
# ### Get most prob topics

# %%
guided_topic_dict["topics"]

# %%
probs_dict = load_pickle(RESULTS_DIR / "false_positive_check.p")

# %%
probs_dict["probs"].shape

# %%
# guided_topic_dict['topics']

# %%
# Select categories
relevant_categories = sorted([0, 2, 3, 5, 6, 7, 8, 9, 19, 24, 25, 29, 30])
relevant_secondary_categories = sorted(
    [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 26, 33, 34, 35]
)
# energy_labels = [
#     "Buildings",
#     "Batteries",
#     "Biofuels",
#     "Fuel cells",
#     "Demand management",
#     "Nuclear",
#     "Solar",
#     "Heating",
#     "Biomass",
#     "Greenhouse Gase Removal",
#     "Electric vehicles & charging",
#     "Wind & Offshore",
#     "Wind & Offshore",
# ]
# energy_category_id = list(range(len(energy_categories)))
[guided_topic_dict["topics"][s] for s in relevant_categories]


# %%
# probs_dict['probs'].argsort(axis=1).shape

# %%
# i = 0
# # Check if relevant category is top 1
# np
# # Check if relevant category has at least 0.2 prob
# np.where(probs_dict['probs'][i,:] > 0.2)

# %%
def add_topic_label(df, topic_id_column, topic_list):
    label_column = f"{topic_id_column}_label"
    not_nulls = df[topic_id_column].isnull() == False
    df.loc[not_nulls, label_column] = df.loc[not_nulls, topic_id_column].apply(
        lambda x: topic_list[int(x)]
    )
    return df


def probs_topics(probs):
    topic_df = pd.DataFrame(
        data={"topic": guided_topic_dict["topics"], "probability": probs}
    ).sort_values("probability", ascending=False)
    return topic_df


def check_top_topics(probs_dict, remove_green=True):
    #     df = pd.DataFrame(
    #         data={
    #             'topic_1': probs_dict['probs'].argsort(axis=1)[:,-1],
    #             'topic_2': probs_dict['probs'].argsort(axis=1)[:,-2]
    #         }
    #     )
    dff = probs_dict["projects"].copy()
    dff["topic_1"] = probs_dict["probs"].argsort(axis=1)[:, -1]
    dff["topic_2"] = probs_dict["probs"].argsort(axis=1)[:, -2]
    dff = add_topic_label(dff, "topic_1", probs_dict["topics"])
    dff = add_topic_label(dff, "topic_2", probs_dict["topics"])
    if remove_green:
        dff_ = dff[-dff.project_id.isin(green_projects.project_id.to_list())]
        return dff_
    else:
        return dff


def check_relevant(dff_, secondary=True):
    keep = dff_["topic_1"].isin(relevant_categories) | (
        dff_["topic_1"].isin(relevant_secondary_categories)
        & dff_["topic_2"].isin(relevant_categories)
    )
    return dff_[keep]


# %%
# probs_dict['projects']

# %% [markdown]
# - How many FN are potentially relevan to primary categories relevant?
#   - Is it feasible to check all of them?
# - How many FN candidates are there for each category?
#   - How many are relevant to heating or buildings? (primary and secondary)
#     - Is it feasible to check all of them?
#     - Do they make sense?
#

# %%
df_all = check_top_topics(probs_dict, remove_green=False)
df_relevant = check_relevant(df_all)

# %%
t = 0
df = df_relevant[(df_relevant.topic_1.isin([9])) | (df_relevant.topic_2.isin([9]))]

# %%
# t = 0
# df = df_relevant[
#     (df_relevant.topic_1.isin([9, 33]))
#     & (df_relevant.topic_2.isin([9, 33]))
# ]

# %%
len(df)

# %%
pd.set_option("max_colwidth", 200)

# %%
# df.sample(5)

# %%
import innovation_sweet_spots.analysis.analysis_utils as iss

gtr_topics = gtr.get_gtr_topics()
link_gtr_topics = gtr.get_link_table("gtr_topic")
df_relevant_with_topics = iss.link_gtr_projects_and_topics(
    df_relevant, gtr_topics, link_gtr_topics
)

# %%
proj_ids = df_relevant_with_topics[
    df_relevant_with_topics.text == "Unclassified"
].project_id.to_list()

# %%
df_relevant[df_relevant.project_id.isin(proj_ids)].sample(5)

# %%
gtr_project_topics.groupby("text").count().sort_values(
    "project_id", ascending=False
).head(20)

# %%
# 'Innovation in control systems for zero emission refuse collection vehicles'

# %%
gtr_project_topics_all = iss.link_gtr_projects_and_topics(
    gtr_projects, gtr_topics, link_gtr_topics
)

# %%
# gtr_project_topics_all.groupby('text').count().sort_values('project_id')

# %%
green_candidates = df_relevant.project_id.to_list()

# %%
from innovation_sweet_spots.utils.io import read_list_of_terms, save_list_of_terms

# %%
green_ids = read_list_of_terms(
    PROJECT_DIR / "outputs/data/gtr/green_gtr_project_ids.txt"
)
green_ids += green_candidates
save_list_of_terms(
    green_ids, PROJECT_DIR / "outputs/data/gtr/green_gtr_project_ids_v2.txt"
)

# %%
len(np.unique(green_ids))

# %% [markdown]
# # Check Crunchbase FN

# %%
from innovation_sweet_spots.getters import crunchbase

# %%
cb = crunchbase.get_crunchbase_orgs()

# %%
cb = cb.drop_duplicates("id")

# %%
# Create documents
cb_to_check_texts = create_documents_from_dataframe(
    cb,
    columns=["name", "short_description", "long_description"],
    preprocessor=(lambda x: x),
)

logging.disable()
docs = cb_to_check_texts
project_probs = []
for doc in tqdm(docs, total=len(docs)):
    project_probs.append(check_project_topic(doc, return_dataframe=False))
logging.disable(0)

# %%
cb_ = cb[["id", "name"]].copy()
cb_["description"] = cb_to_check_texts

# %%
cb_probs_dict = {
    "projects": cb_[["id", "name", "description"]],
    "probs": np.array(project_probs),
    "topics": guided_topic_dict["topics"],
}
save_pickle(probs_dict, RESULTS_DIR / "false_positive_check_cb.p")

# %%
from innovation_sweet_spots.getters.green_docs import (
    get_green_gtr_docs,
    get_green_cb_docs_by_country,
)

cb_uk_corpus, green_orgs_uk = get_green_cb_docs_by_country()

# %%
df_all = check_top_topics(cb_probs_dict, False)
df_relevant = check_relevant(df_all)
df_relevant = df_relevant[-df_relevant.id.isin(green_orgs_uk.id.to_list())]

# %%
df_relevant_ = df_relevant[df_relevant.topic_1 == 0]

# %%
# cb_categories = crunchbase.get_crunchbase_category_groups()
# cb_org_categories = crunchbase.get_crunchbase_organizations_categories()
# # green_tags = cb_categories_for_group(cb_categories, "Sustainability")
# # green_orgs = cb_org_categories[cb_org_categories.category_name.isin(green_tags)]

# %%
df = df_relevant_.merge(
    cb_org_categories, left_on="id", right_on="organization_id", how="left"
)
df.head(1)

# %%
df.groupby("category_name").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(25)

# %%
df[df.category_name == "architecture"].sample(10)

# %%
green_docs = get_green_gtr_docs()

# %%
x = list(green_docs.values())

# %%
import innovation_sweet_spots.utils.text_pre_processing as iss_preproc

# %%
gtr_projects

# %%
t = [project_texts[i] for i in df.index]

# %%
tok_ngram, ngram_phr = iss_preproc.pre_process_corpus(t, n_gram=4, min_count=5)

# %%
k = []
for j in range(len(tok_ngram)):
    k.append(
        [
            tok_ngram[j][x]
            for x in np.where([len(t.split("_")) == 4 for t in tok_ngram[j]])[0]
        ]
    )
sorted(np.unique([kk for kks in k for kk in kks]))

# %%
cbx = crunchbase.get_crunchbase_orgs()

# %%
len(cbx)

# %%
len(cbx.drop_duplicates())

# %%
