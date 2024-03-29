#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analyse mentions and language used to describe a set of terms
#
# Analyse language used to describe search terms:
# - Adjective and noun phrases
# - Verb phrases
# - Subject-verb-object triples

# ## 1. Import dependencies

import spacy
import collections
import pickle
import os
import pandas as pd
import json

from innovation_sweet_spots.utils.pd import pd_pos_utils as pos
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_LOOKUPS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_lookups"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

search_terms = ["heat pump", "heat pumps"]

# ## 2. Read in preprocessed data

# +
# Read in outputs.
with open(
    os.path.join(DISC_OUTPUTS_DIR, "sentence_record_dict_hp.pkl"), "rb"
) as infile:
    sentence_record_dict = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, "metadata_dict_hp.pkl"), "rb") as infile:
    metadata_dict = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, "noun_chunks_hp.pkl"), "rb") as infile:
    noun_chunks_all_years = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, "combined_sentences_hp.pkl"), "rb") as infile:
    combined_term_sentences = pickle.load(infile)

# +
# Read in patterns used for phrase matching
# with open (os.path.join(DISC_LOOKUPS_DIR, 'hydrogen_phrases.json'), "r") as f:
#    hydrogen_phrases = json.load(f)

with open(os.path.join(DISC_LOOKUPS_DIR, "heat_pump_phrases.json"), "r") as f:
    heat_pump_phrases = json.load(f)
# -

flat_sentences = pd.concat(
    [combined_term_sentences[y] for y in combined_term_sentences]
)

# ## 3. Analyse language used to describe search terms

# ### 3.1. Noun phrases

# +
# All spacy identified noun chunks that contain search terms.
term_phrases = pos.noun_chunks_w_term(
    noun_chunks_all_years, search_terms
)  # invididual years

# To view noun chunks in a given year:
# term_phrases['2020']

# +
# Define time period to investigate
# For heat pumps, using 3 year periods results in 5 periods:
# ['2007, 2008, 2009', '2010, 2011, 2012', '2013, 2014, 2015', '2016, 2017, 2018', '2019, 2020, 2021']
time_period = "2019, 2020, 2021"

# Noun phrases that describe heat pumps.
# In the example below these are aggregated over a 3 year time period
# NB: patterns are defined separately and read in as a json file
nouns = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["noun_phrase"], 1
    )
)
# -

# To view full sentences that use the noun phrases including links to original articles:
time_period = ["2019", "2020", "2021"]
for y in time_period:
    pos.view_phrase_sentences(
        y, nouns, flat_sentences, metadata_dict, sentence_record_dict, output_data=False
    )

# Sometimes it's helpful to aggregate results for several years. The size of a time period is controlled
# by the argument n_years in the function match_patterns_across_years.
# For heat pumps, using 3 year periods results in 5 periods:
# ['2007, 2008, 2009', '2010, 2011, 2012', '2013, 2014, 2015', '2016, 2017, 2018', '2019, 2020, 2021']
# Check the keys to phrase dictionary for exact definitions of time periods.
# The 3 year period are generated by the function above.
nouns_3y = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["noun_phrase"], 3
    )
)
time_period = "2020, 2021, 2022"
pos.view_phrase_sentences_period(
    time_period, nouns_3y, flat_sentences, metadata_dict, sentence_record_dict
)

# ### 3.2. Adjective phrases

# Adjectives used to describe heat pumps
adjectives = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["adj_phrase"], 1
    )
)


# To view full sentences that use the adjective phrases including links to original articles:
time_period = ["2020", "2021", "2022"]
for y in time_period:
    pos.view_phrase_sentences(
        y, adjectives, flat_sentences, metadata_dict, sentence_record_dict
    )

# ### 3.3. Verb phrases

# Phrases that match the pattern 'heat pumps are ...'
hp_are = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["term_is"], 3
    )
)

time_period = "2020, 2021, 2022"
pos.view_phrase_sentences_period(
    time_period, hp_are, flat_sentences, metadata_dict, sentence_record_dict
)

# Phrases that match the pattern 'heat pumps can ...'
hp_can = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["term_can"], 3
    )
)

# To view full sentences that use the verb phrases including links to original articles:
pos.view_phrase_sentences_period(
    time_period, hp_can, flat_sentences, metadata_dict, sentence_record_dict
)

# Phrases that match the pattern 'heat pumps have ...'
hp_have = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["term_have"], 3
    )
)

# To view full sentences that use the verb phrases including links to original articles:
pos.view_phrase_sentences_period(
    time_period, hp_have, flat_sentences, metadata_dict, sentence_record_dict
)

# Verbs that follow heat pumps.
verbs_follow = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["verb_subj_hp"], 3
    )
)

# To view full sentences that use the verb phrases including links to original articles:
pos.view_phrase_sentences_period(
    time_period, verbs_follow, flat_sentences, metadata_dict, sentence_record_dict
)

# Phrases where verbs precede heat pumps.
verbs_precede = pos.aggregate_matches(
    pos.match_patterns_across_years(
        combined_term_sentences, nlp, heat_pump_phrases["verb_obj"], 3
    )
)

# To view full sentences that use the verb phrases including links to original articles:
pos.view_phrase_sentences_period(
    time_period, verbs_precede, flat_sentences, metadata_dict, sentence_record_dict
)

# ### 3.4. Subject-verb-object triples

# +
# Subject verb object triples with search term acting both as subject and object.
subject_phrase_dict = collections.defaultdict(list)
object_phrase_dict = collections.defaultdict(list)

for term in search_terms:
    for year in combined_term_sentences:
        given_term_sentences = combined_term_sentences[year]["sentence"]
        subj_triples, obj_triples = pos.get_svo_triples(given_term_sentences, term, nlp)
        subject_phrases, object_phrases = pos.get_svo_phrases(subj_triples, obj_triples)
        subject_phrase_dict[year] = subject_phrases
        object_phrase_dict[year] = object_phrases
# -

for period in nouns:
    for year in period.split(", "):
        print(year, subject_phrase_dict[year])

# +
# Save outputs.

# pos.save_phrases([nouns, adjectives, verbs_follow, verbs_precede]).to_csv(
#     DISC_OUTPUTS_DIR / "pos_phrases.csv", index=False
# )

# with open(os.path.join(DISC_OUTPUTS_DIR, 'term_phrases_hp.pkl'), "wb") as outfile:
#        pickle.dump(term_phrases, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, 'adj_hp.pkl'), "wb") as outfile:
#        pickle.dump(adjectives, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, 'nouns_heat_pumps.pkl'), "wb") as outfile:
#        pickle.dump(nouns, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, 'hp_are.pkl'), "wb") as outfile:
#        pickle.dump(hp_are, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, 'verbs_follow_hp.pkl'), "wb") as outfile:
#        pickle.dump(verbs_follow, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, 'verbs_precede_hp.pkl'), "wb") as outfile:
#        pickle.dump(verbs_precede, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, "subj_hp.pkl"), "wb") as outfile:
#     pickle.dump(subject_phrase_dict, outfile)

# with open(os.path.join(DISC_OUTPUTS_DIR, "obj_hp.pkl"), "wb") as outfile:
#     pickle.dump(object_phrase_dict, outfile)
# -
