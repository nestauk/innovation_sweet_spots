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
# # Checking specific technologies
#
# - heat pump
# - biomass boiler
# - hydrogen boiler
# - district heating
# - solar thermal
# - geothermal

# %%
from innovation_sweet_spots import logging, PROJECT_DIR
import pandas as pd

# import innovation_sweet_spots.analysis.guided_topics as guided_topics
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
from innovation_sweet_spots.utils.io import save_pickle
import innovation_sweet_spots.analysis.analysis_utils as iss

import numpy as np

# %%
import innovation_sweet_spots.getters.gtr as gtr


# %%
def similar_words(words, model):
    words, word_scores = model.similar_words(
        keywords=words, keywords_neg=[], num_words=20
    )
    for word, score in zip(words, word_scores):
        print(f"{word} {score}")


# %%
RESULTS_DIR = PROJECT_DIR / "outputs/data/results_july"
RUN = "July2021_projects_orgs_stopwords_e400"

# %%
# Load top2vec model, load projects

# Import top2vec model
top2vec_model = iss_topics.get_top2vec_model(RUN)

# %%
# Import all GTR projects
gtr_projects = gtr.get_gtr_projects()
gtr_projects_clean = gtr.get_cleaned_project_texts()

# %%
# Create documents
project_texts = iss.create_documents_from_dataframe(
    gtr_projects,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)

# %%
min_mentions = 1

# %%
keywords_list = [
    [["heat pump"], ["heat pumps"]],
    [["district heating"], ["district heat"], ["heat network"]],
    [["hydrogen boiler"], ["hydrogen", "boiler"], ["hydrogen", "heating"]],
    [["geothermal", "heating"]],
    [["heat store"], ["heat storage"], ["thermal storage"], ["thermal store"]],
    [
        ["insulation", "heat"],
        ["insulat", "heat"],
        ["insulat", "heating"],
        ["thermal", "insulat"],
    ],
]


# %%
def find_projects_with_keywords(keywords):
    f = np.array([True] * len(gtr_projects_clean))
    for keyword in keywords:
        f = f & np.array(
            iss.is_term_present_in_sentences(
                keyword, gtr_projects_clean.project_text.to_list(), min_mentions
            )
        )
    return f


# %%
# keywords = ['hydrogen', 'heating']
# gtr_projects[find_projects_with_keywords(keywords)]

# %%
big_f = np.array([False] * len(gtr_projects_clean))
for keywords in keywords_list[0]:
    f = np.array([True] * len(gtr_projects_clean))
    for keyword in keywords:
        f = f & np.array(
            iss.is_term_present_in_sentences(
                keyword, gtr_projects_clean.project_text.to_list(), min_mentions
            )
        )
    big_f = big_f | f

# %%
# gtr_projects[big_f]

# %%
# top2vec_model.

# %%
# similar_words(['thermal', 'storage'], top2vec_model)

# %%

# %%

# %%
