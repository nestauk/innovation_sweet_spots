# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Checking top2vec model clusters

# +
from innovation_sweet_spots.analysis import top2vec_utils
from innovation_sweet_spots.getters.models import (
    get_pilot_top2vec_document_clusters,
    get_pilot_top2vec_model,
)
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler

GTR = GtrWrangler()
# -

document_cluster_dict = get_pilot_top2vec_document_cluster_data()

QueryClusters = top2vec_utils.QueryClusters(document_cluster_dict)

QueryClusters.get_cluster_description(109)

sorted_docs = QueryClusters.get_cluster_documents(109)

GTR.add_project_data(sorted_docs, id_column="id", columns=["title", "start"]).head(20)[
    ["title", "start"]
]

# ## Load a model

tp = m.get_pilot_top2vec_model()

tp["model"]

tp["model"].search_topics(["battery"], num_topics=3)
