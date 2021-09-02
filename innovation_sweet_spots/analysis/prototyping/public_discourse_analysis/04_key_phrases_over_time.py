#!/usr/bin/env python3
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
# # Analyse mentions and language used to describe a set of terms
#
# - Identify most relevant words and phrases used together with search terms
# - Aggregate rank and pmi measures across search terms and years
# - Analyse language used to describe search terms: adjective and noun phrases, verbs, subject-verb-object patterns

# %% [markdown]
# ## 1. Import dependencies

# %%
import spacy
import collections
import pickle
import os
import pandas as pd

# %%
# Change first element to location of project folder.
os.chdir(os.path.join('/Users/jdjumalieva/Documents/Analysis/', 'innovation_sweet_spots'))

# %%
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.analysis import analysis_utils as iss
from innovation_sweet_spots.analysis import discourse_utils as disc
from innovation_sweet_spots.utils import text_cleaning_utils as tcu
from innovation_sweet_spots.utils import text_pre_processing as tpu
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Patterns for phrase matching:

noun_phrase_hydrogen = [{"POS": "ADJ", "OP": "*"}, 
                        #{'POS': 'NOUN'},
                        {'POS': 'NOUN', 'OP': '*'},
                        {'TEXT': 'hydrogen'},
                        {'POS': 'NOUN', 'OP': '*'}, 
                       ]

adj_phrase_hydrogen = [{"POS": "ADV", "OP": "*"}, 
                       {'POS': 'ADJ'},
                       {"POS": "ADJ", "OP": "*"}, 
                       {'POS': 'NOUN', 'OP': '?'},
                       {'TEXT': 'hydrogen'},
                       {'POS': 'NOUN', 'OP': '*'}, 
                      ]


term_is_hydrogen = [{'TEXT': 'hydrogen'},
                    {'POS': 'NOUN', 'OP': '*'}, 
                    {"LEMMA": "be"}, 
                    {"DEP": "neg", "OP": '?'},           
                    {"POS": {'IN': ['ADV', 'DET']}, 'OP': '*'},
                    {"POS": {'IN': ['NOUN', 'ADJ']}, 'OP': '*'}]

term_can_hydrogen = [{'TEXT': 'hydrogen'},
                    {'POS': 'NOUN', 'OP': '*'}, 
                    {"LEMMA": "can"}, 
                    {"DEP": "neg", "OP": '?'},           
                    {"POS": {'IN': ['ADV', 'DET']}, 'OP': '*'},
                    {"POS": {'IN': ['NOUN', 'ADJ']}, 'OP': '*'}]


term_is_at_hydrogen = [{'TEXT': 'hydrogen'},
                       {'POS': 'NOUN', 'OP': '?'}, 
                       {"LEMMA": "be"}, 
                       {"DEP": "prep"}, 
                       {"POS": "DET"},
                       {"POS": "NOUN"}]


verb_obj_hydrogen = [{'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
                     {'POS': {'IN': ['NOUN', 'ADJ', 'ADV', 'VERB']}, 'OP': '?'},
                     {'POS': 'VERB'},
                     {'OP': '?'},
                     {'TEXT': 'hydrogen'},
                     {'POS': 'NOUN', 'OP': '?'},
                    ]

verb_subj_hydrogen = [{'TEXT': 'hydrogen'},
                      {'POS': 'NOUN', 'OP': '*'},
                      {'POS': 'VERB'},
                      {'POS': 'VERB', 'OP': '?'},
                      {'POS': {'IN': ['NOUN', 'ADJ', 'ADV']}, 'OP': '?'},
                      {'POS': {'IN': ['NOUN', 'ADJ', 'ADV']}, 'OP': '?'},
                     ]   

# %%
nlp = spacy.load("en_core_web_sm")

# %%
search_terms = ['hydrogen boiler', 'hydrogen boilers', 
                 'hydrogen-ready boiler', 'hydrogen-ready boilers', 'hydrogen ready boiler', 'hydrogen ready boilers',
                 'hydrogen heating', 'hydrogen heat','hydrogen']

# %% [markdown]
# ## 2. Read in preprocessed data

# %%
# Read in outputs.

article_text = pd.read_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_hansard_hydro.csv'))

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentences_by_year_hansard_hydro.pkl'), "rb") as infile:
        sentences_by_year = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'processed_articles_by_year_hansard_hydro.pkl'), "rb") as infile:
        processed_articles_by_year = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_records_hansard_hydro.pkl'), "rb") as infile:
        sentence_records = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'sentence_record_dict_hansard_hydro.pkl'), "rb") as infile:
        sentence_record_dict = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'metadata_dict_hansard_hydro.pkl'), "rb") as infile:
        metadata_dict = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'noun_chunks_hansard_hydro.pkl'), "rb") as infile:
        noun_chunks_all_years = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'term_sentences_hansard_hydrogen.pkl'), "rb") as infile:
        term_sentences = pickle.load(infile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'combined_sentences_hansard_hydrogen.pkl'), "rb") as infile:
        combined_term_sentences = pickle.load(infile)

# %% [markdown]
# ## 3. Identify most relevant terms
# %%
related_terms = collections.defaultdict(dict)
normalised_ranks = collections.defaultdict(dict)

for year in sentences_by_year:
    print(year)
    year_articles = [elem for elem in sentences_by_year[year]] # nested list of sentences within each article
    year_sentences = [sent for art in year_articles for sent in art]
    noun_chunks = noun_chunks_all_years[str(year)]
    for term in search_terms:
        key_terms, normalised_rank = disc.get_key_terms(term, year_sentences, nlp, noun_chunks,
                                                            mentions_threshold = 2, token_range = (1,3))

        related_terms[year][term] = list(key_terms.items())
        normalised_ranks[year][term] = list(normalised_rank.items())

# %%
# Write to disk.
with open(os.path.join(DISC_OUTPUTS_DIR, 'related_terms_hansard_hydrogen.pkl'), "wb") as outfile:
        pickle.dump(related_terms, outfile)
        
with open(os.path.join(DISC_OUTPUTS_DIR, 'normalised_ranks_hansard_hydrogen.pkl'), "wb") as outfile:
        pickle.dump(normalised_ranks, outfile)     

# %%
# Read in previously generated outputs
#with open(os.path.join(DISC_OUTPUTS_DIR, 'related_terms_hydrogen.pkl'), "rb") as infile:
#        related_terms = pickle.load(infile)
        
#with open(os.path.join(DISC_OUTPUTS_DIR, 'normalised_ranks_hydrogen.pkl'), "rb") as infile:
#        normalised_ranks = pickle.load(infile) 

# %% [markdown]
# ### 3.1 Combine related terms and normalised ranks for a set of terms

# %%
combined_pmi= disc.combine_pmi(related_terms, search_terms)

# %%
combined_pmi['2020']

# %%
combined_ranks = disc.combine_ranks(normalised_ranks, search_terms)

# %%
combined_ranks['2020']

# %%
combined_pmi_dict = collections.defaultdict(dict)
for year in combined_pmi:
    for term in combined_pmi[year]:
        combined_pmi_dict[year][term[0]] = term[1]

# %%
# Dictionary: for each year return frequency of mentions, normalised rank and pmi for a given term.
pmi_inters_ranks = collections.defaultdict(dict)
for year in combined_ranks:
    for term in combined_ranks[year]:
        if term[0] in combined_pmi_dict[year]:
            pmi_inters_ranks[year][term[0]] = (term[1], combined_ranks[year][term], combined_pmi_dict[year][term[0]])

# %%
# Aggregate into one long dataframe.
agg_pmi = disc.agg_combined_pmi_rank(pmi_inters_ranks)

# %%
# Preprocess for further analysis of changes over time.
agg_terms = disc.analyse_rank_pmi_over_time(agg_pmi)

# %%
# Save to disc to explore separately.
agg_pmi.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'combined_pmi_rank_hansard_hydrogen.csv'), index = False)
agg_terms.to_csv(os.path.join(DISC_OUTPUTS_DIR, 'agg_pmi_rank_hansard_hydrogen.csv'), index = False)

# %% [markdown]
# ### 3.2 Compare and plot term ranks over time

# %%
comparative_df = disc.compare_term_rank(pmi_inters_ranks, ['boilers', 'homes'], measure = 1)

# %%
disc.plot_ranks(comparative_df, 'Comparison of term ranks over time')

# %% [markdown]
# ### 3.3 Quick exploration of collocations

# %%
flat_sentences = pd.concat([combined_term_sentences[y] for y in combined_term_sentences])

# %%
# Retrieve sentences where a given term was used together with any of the search terms
grouped_sentences = disc.check_collocations(flat_sentences, 'green')
disc.collocation_summary(grouped_sentences)

# %%
disc.view_collocations(grouped_sentences, metadata_dict, sentence_record_dict)

# %% [markdown]
# ## 4. Analyse language used to describe search terms

# %%
# All spacy identified noun chunks that contain search terms.
term_phrases = disc.noun_chunks_w_term(noun_chunks_all_years, search_terms)

# %%
term_phrases_agg = disc.aggregate_patterns(group_noun_chunks(term_phrases))

# %%
term_phrases_agg.keys()

# %%
term_phrases_agg['2019, 2020, 2021']

# %%
# Adjectives used to describe heat pumps.
# NB: patterns are defined at the start of the file.
adjectives = disc.match_patterns_across_years(combined_term_sentences, nlp, adj_phrase_hydrogen)

# %%
adj_aggregated = disc.aggregate_patterns(adjectives)

# %%
adj_aggregated['2019, 2020, 2021']

# %%
# Noun phrases that describe heat pumps.
nouns = disc.match_patterns_across_years(combined_term_sentences, nlp, noun_phrase_hydrogen)

# %%
nouns_aggregated = disc.aggregate_patterns(nouns)

# %%
nouns_aggregated['2019, 2020, 2021']

# %%
# Phrases that match the pattern 'heat pumps are ...'
hp_are = disc.match_patterns_across_years(combined_term_sentences, nlp, term_is_hydrogen)

# %%
hp_aggregated = disc.aggregate_patterns(hp_are)

# %%
hp_aggregated['2019, 2020, 2021']

# %%
# Phrases that match the pattern 'heat pumps can ...'
hp_can = disc.match_patterns_across_years(combined_term_sentences, nlp, term_can_hydrogen)

# %%
hp_can_aggregated = disc.aggregate_patterns(hp_can)

# %%
hp_can_aggregated['2019, 2020, 2021']

# %%
# Verbs that follow heat pumps.
verbs_follow = disc.match_patterns_across_years(combined_term_sentences, nlp, verb_subj_hydrogen)

# %%
verbs_aggregated = disc.aggregate_patterns(verbs_follow)

# %%
verbs_aggregated['2019, 2020, 2021']

# %%
# Phrases where verbs precede heat pumps.
verbs_precede = disc.match_patterns_across_years(combined_term_sentences, nlp, verb_obj_hydrogen)

# %%
verbs_p_aggregated = disc.aggregate_patterns(verbs_precede)

# %%
verbs_p_aggregated['2019, 2020, 2021']

# %%
# Subject verb object triplets with search term acting both as subject and object.
subject_phrase_dict = collections.defaultdict(list)
object_phrase_dict = collections.defaultdict(list)
terms = ['hydrogen boiler', 'hydrogen']
for given_term in terms:
    for year in term_sentences[given_term]:
        given_term_sentences = term_sentences[given_term][year]['sentence']
        subj_triples, obj_tripels = disc.get_svo_triples(given_term_sentences, given_term, nlp)
        subject_phrases, object_phrases = disc.get_svo_phrases(subj_triples, obj_tripels)
        subject_phrase_dict[year] = subject_phrases
        object_phrase_dict[year] = object_phrases

# %%
for chunk in nouns:
    for year in chunk.split(', '):
        print(year, subject_phrase_dict[year])

# %%
# Save outputs.

with open(os.path.join(DISC_OUTPUTS_DIR, 'term_phrases_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(term_phrases, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'adj_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(adj_aggregated, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'nouns_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(nouns_aggregated, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'heat_pumps_are.pkl'), "wb") as outfile:
        pickle.dump(hp_aggregated, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'verbs_f_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(verbs_aggregated, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'verbs_p_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(verbs_p_aggregated, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'subj_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(subject_phrase_dict, outfile)

with open(os.path.join(DISC_OUTPUTS_DIR, 'obj_heat_pumps.pkl'), "wb") as outfile:
        pickle.dump(object_phrase_dict, outfile)


# %% [markdown]
# ### Appendix

# %%
def group_noun_chunks(noun_chunk_dict, n_years = 3):
    """
    Collect noun chunks into bundles corresponding to n-year periods.
    """
    period = list(noun_chunk_dict.keys())
    year_chunks = []
    for i in range(0, len(period), n_years):
        year_chunks.append(period[i:i + n_years])
    phrases = collections.defaultdict(list)
    for chunk in year_chunks:
        chunk_name = ', '.join(chunk)
        for year in chunk:
            year_phrases = noun_chunk_dict.get(year,{})
            if len(year_phrases):
                phrases[chunk_name].append(year_phrases)
    return phrases

# %%
