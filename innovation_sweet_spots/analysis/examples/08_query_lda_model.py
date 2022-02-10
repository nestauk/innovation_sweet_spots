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
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selecting documents by using LDA model

# %%
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.getters.models
from innovation_sweet_spots.analysis import lda_modelling_utils as lmu

# %%
MODEL_DIR = innovation_sweet_spots.getters.models.PILOT_MODELS_DIR
MODEL_NAME = "lda_full_corpus"

# %%
model_data = innovation_sweet_spots.getters.models.get_pilot_lda_model(MODEL_NAME)

# %%
lmu.print_model_info(model_data["model"])

# %%
doc_prob_df = lmu.create_document_topic_probability_table(
    model_data["document_ids"], model_data["model"]
)

# %%
lmu.query_topics([0, 1], doc_prob_df)

# %%
lmu.make_pyLDAvis(model_data["model"], MODEL_DIR / f"{MODEL_NAME}_pyLDAvis.html")

# %%
# Takes a long time
# coh = tp.coherence.Coherence(model_data["model"])

# %%
# coh.get_score()

# %%
