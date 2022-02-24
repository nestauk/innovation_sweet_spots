#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
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
# # Define patterns for phrase matching with spacy
#
# - Search terms: heat pumps and hydrogen
# - Types: noun phrases, adjective phrases, phrases with verbs 'is'/'have'/'can', verbs that precede and follow, SVOs

# %% [markdown]
# ## 1. Import dependencies

# %%
import os
import json

# %%
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_lookups"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Generating patterns with provided search terms

# %%
search_terms = [
    "unhealthy food advertisements",
    "hfss",
    "junk food",
    "childhood obesity",
    "ad ban",
]
# Bigrams and trigrams will not be picked up when phrase matching
pattern_terms = search_terms + [
    "obesity",
    "health",
    "child",
    "poverty",
    "food",
    "foods",
    "diet",
    "school",
    "online",
    "social",
    "media",
    "advertisements",
    "ads",
    "adverts",
    "portion",
    "retailers",
    "retailer",
    "supermarkets",
    "supermarket",
    "exposure",
    "salt",
    "family",
    "environment",
    "calories",
    "calorie",
    "ban",
    "meals",
    "policy",
    "risk",
    "benefit",
    "healhty",
    "unhealthy",
    "fresh food",
    "excess weight",
    "sugar",
    "fat",
    "nutrition",
    "nutritional",
    "fast food",
    "purchase",
    "purchases",
    "revenue",
    "revenues",
    "marketing",
    "promotion",
    "lobby",
    "burden",
    "action",
    "legislation",
    "consumption",
    "healthier",
    "early years",
    "intake",
    "intakes",
    "intervention",
    "snacks",
    "evidence",
    "obese",
    "health conditions",
    "disease",
    "ill",
    "illness",
]

# %%
generic_phrases = {
    "noun_phrase": [
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"POS": "NOUN", "OP": "?"},
        {"TEXT": {"IN": pattern_terms}},
    ],
    "adj_phrase": [
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADJ"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "?"},
        {"TEXT": {"IN": pattern_terms}},
    ],
    "term_is": [
        {"TEXT": {"IN": pattern_terms}},
        {"LEMMA": "be"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_has": [
        {"TEXT": {"IN": pattern_terms}},
        {"LEMMA": "have"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_can": [
        {"TEXT": {"IN": pattern_terms}},
        {"LEMMA": "have"},
        {"LEMMA": "can"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_is_a": [
        {"TEXT": {"IN": pattern_terms}},
        {"LEMMA": "be"},
        {"DEP": "prep"},
        {"POS": "DET"},
        {"POS": "NOUN"},
    ],
    "verb_obj": [
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": "VERB"},
        {"OP": "?"},
        {"TEXT": {"IN": pattern_terms}},
    ],
    "verb_subj": [
        {"TEXT": {"IN": pattern_terms}},
        {"POS": "VERB"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
    ],
}

# %%
with open(os.path.join(DISC_OUTPUTS_DIR, "phrases_foods.json"), "w") as outfile:
    json.dump(generic_phrases, outfile, indent=4)

# %% [markdown]
# ## 3. Patterns used for hydrogen energy

# %%
hydrogen_phrases = {
    "noun_phrase_hydrogen": [
        {"POS": "ADJ", "OP": "*"},
        # {'POS': 'NOUN'},
        {"POS": "NOUN", "OP": "*"},
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
    ],
    "adj_phrase_hydrogen": [
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADJ"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "?"},
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
    ],
    "term_is_hydrogen": [
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
        {"LEMMA": "be"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_can_hydrogen": [
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
        {"LEMMA": "can"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_have_hydrogen": [
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
        {"LEMMA": "have"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_is_at_hydrogen": [
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "?"},
        {"LEMMA": "be"},
        {"DEP": "prep"},
        {"POS": "DET"},
        {"POS": "NOUN"},
    ],
    "verb_obj_hydrogen": [
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": "VERB"},
        {"OP": "?"},
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "?"},
    ],
    "verb_subj_hydrogen": [
        {"TEXT": "hydrogen"},
        {"POS": "NOUN", "OP": "*"},
        {"POS": "VERB"},
        # {'POS': 'VERB', 'OP': '?'},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV"]}, "OP": "?"},
    ],
}


# %%
with open(os.path.join(DISC_OUTPUTS_DIR, "hydrogen_phrases.json"), "w") as outfile:
    json.dump(hydrogen_phrases, outfile, indent=4)

# %% [markdown]
# ## 4. Patterns used for heat pumps

# %%
heat_pump_phrases = {
    "noun_phrase_hp": [
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"POS": "NOUN", "OP": "?"},
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
    ],
    "adj_phrase_hp": [
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADJ"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "?"},
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
    ],
    "term_is_hp": [
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
        {"LEMMA": "be"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_have_hp": [
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
        {"LEMMA": "have"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_can_hp": [
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
        {"LEMMA": "have"},
        {"LEMMA": "can"},
        {"DEP": "neg", "OP": "?"},
        {"POS": {"IN": ["ADV", "DET"]}, "OP": "*"},
        {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "*"},
    ],
    "term_is_at_hp": [
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
        {"LEMMA": "be"},
        {"DEP": "prep"},
        {"POS": "DET"},
        {"POS": "NOUN"},
    ],
    "verb_obj_hp": [
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": "VERB"},
        {"OP": "?"},
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
    ],
    "verb_subj_hp": [
        {"TEXT": "heat"},
        {"TEXT": {"IN": ["pump", "pumps"]}},
        {"POS": "VERB"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV", "VERB"]}, "OP": "?"},
    ],
}


# %%
with open(os.path.join(DISC_OUTPUTS_DIR, "heat_pump_phrases.json"), "w") as outfile:
    json.dump(heat_pump_phrases, outfile, indent=4)

# %%
