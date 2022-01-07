#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:18:01 2021

@author: jdjumalieva
"""

import pytest
import spacy
import textacy
import pandas as pd
import pickle

from innovation_sweet_spots.utils import pd_pos_utils as pos
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"


nlp = spacy.load("en_core_web_sm")
test_data = pd.read_csv(DISC_OUTPUTS_DIR / 'sentence_sample.pkl')
sentences_by_year = {str(y): v for y, v in test_data.groupby('year')}

with open(DISC_OUTPUTS_DIR / 'sentence_record_dict_hp.pkl', "rb") as infile:
        sentence_record_dict = pickle.load(infile)

with open(DISC_OUTPUTS_DIR / 'metadata_dict_hp.pkl', "rb") as infile:
        metadata_dict = pickle.load(infile)

def test_noun_chunks_w_term():
    search_terms = ['heat pumps', 'heat pump']    
    noun_chunks = {'2020': ['gas and heat pumps',
                   'only a start',
                   'smart meters',
                   'these solutions',
                   'supply chain contracts',
                   'less affluent communities',
                   'carbon pricing',
                   'electric powered heat pumps',
                   'these heat pumps',
                   'air source heat pumps',
                   '2bn green homes grant',
                   'cheaper option',
                   'northern gas networks',
                   'greener future',
                   'heat pump installations']}   
    expected_output = list({'2020': ['air source heat pumps',
                                'gas and heat pumps',
                                'heat pump installations',
                                'electric powered heat pumps',
                                'these heat pumps']}.values())
    # the function outputs a nested list with elements corresponding to years in
    # noun_chunks
    function_output = list(pos.noun_chunks_w_term(noun_chunks, search_terms).values())
    assert(set(expected_output[0]) == set(function_output[0]))


def test_find_pattern():
    innovation_is = [{'TEXT': 'heat'},
               {'TEXT': {'IN': ['pump', 'pumps']}},
               {'LEMMA': 'be'},
               {'DEP': 'neg', 'OP': '?'},
               {'POS': {'IN': ['ADV', 'DET']}, 'OP': '*'},
               {'POS': {'IN': ['NOUN', 'ADJ']}, 'OP': '*'}]
    sentences = ['heat pumps are crucial for meeting climate goals.',
                 'many home owners are considering installing a heat pump.',
                 'this particular heat pump is a breakthrough.',
                 'heat pumps are considered to be an answer.']
    expected_output = ['heat pumps are crucial', 
                       'heat pump is a breakthrough', 
                       'heat pumps are']
    function_output = pos.find_pattern(sentences, nlp, innovation_is)
    assert(set(expected_output) == set(function_output))


def test_get_phrase_sentences():
    noun_phrase = [{'POS': 'ADJ', 'OP': '*'},
                   {'POS': 'NOUN'},
                   {'POS': 'NOUN', 'OP': '?'},
                   {'TEXT': 'heat'},
                   {'TEXT': {'IN': ['pump', 'pumps']}}]
    time_period = '2019, 2020, 2021'
    nouns = pos.aggregate_matches(pos.match_patterns_across_years(sentences_by_year, 
                                                               nlp, 
                                                               noun_phrase, 
                                                               3))
    assert(list(nouns.keys()) == ['2007, 2008, 2009', 
                                  '2010, 2011, 2012', 
                                  '2013, 2014, 2015', 
                                  '2016, 2017, 2018', 
                                  '2019, 2020, 2021'])
    assert(list(nouns.values()) == [[('ground source heat pump', 1), 
                                     ('ground source heat pumps', 1)],
                                    [('air source heat pumps', 1), 
                                     ('ground source heat pumps', 1)],
                                    [('ground source heat pumps', 1), 
                                     ('own ground source heat pumps', 1)],
                                    [('ground source heat pumps', 1)],
                                    [('air source heat pump', 2), 
                                     ('ground source heat pump', 1)]])
    

def test_get_svo_phrases():
    subj_svo, obj_svo = pos.get_svo_triples(test_data['sentence'], 'heat pump', nlp)
    function_output = pos.get_svo_phrases(subj_svo, obj_svo)
    expected_output = ([],
                       ['who...is using...source heat pump',
                        'innovations...include...heat pump',
                        'oslo...has added...heat pump',
                        "we...'ve had...source heat pump"])
    assert(set(function_output[1]) == set(expected_output[1]))
    

    