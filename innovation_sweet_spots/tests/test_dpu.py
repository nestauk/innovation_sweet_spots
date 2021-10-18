#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:38:37 2021

@author: jdjumalieva
"""

import pytest
from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import \
    pd_data_processing_utils as dpu


def test_clean_articles():
    # Convert to lower case
    assert(dpu.clean_articles(['Upper case', 'Another upper case']) == \
           ['upper case', 'another upper case'])
    # Split camel-case: imperfect and will miss cases such as 1.First and camel.Case
    assert(dpu.clean_articles(['camelCase', 'another camelCase']) == \
           ['camel. case', 'another camel. case'])
    # Normalise punctuation
    assert(dpu.clean_articles(['• Detect a bullet point',
                               'Convert :\/-',
                               "Standardise ’`",
                               "Keep ,.;'",
                               'Replace (inside parentheses) ',
                               'Replace ...',
                               '1.First', #!should catch such instances
        ]) == \
           [', detect a bullet point',
            'convert ,',
            "standardise '",
            "keep ,.;'",
            'replace, inside parentheses,',
            'replace .',
            '1.first'])
    # Remove empty spaces
    assert(dpu.clean_articles([' empty spaces', 'clean up  spaces '])) == \
        ['empty spaces', 'clean up spaces']
    # Check that no empty docs are included
    assert(dpu.clean_articles(['sentence is blank', 'this is a sentence'])) == \
        ['this is a sentence']


def test_generate_sentence_corpus():
    mock_articles = ['This is the first sentence in a sample article. The second sentence follows.',
                     'And here is another example. Has three sentences. Very short ones.'
        ]
    nlp = spacy.load("en_core_web_sm")
    # Test type of outputs
    sents, docs = dpu.generate_sentence_corpus(mock_articles, nlp)
    assert isinstance(sents, list)
    assert isinstance(docs, list)
    assert type(docs[0]) == spacy.tokens.doc.Doc
    # Length of outputs
    assert len(sents[1]) == 3 # Check that we are extracting all sentences
    assert len(docs) ==2 # Check that we are extracting all documents
        

def test_generate_sentence_corpus_by_year():
    mock_data = pd.DataFrame({'id': [1,2,3,4],
                              'year': ['2018', '2019', '2020', '2020'],
                              'text': ['First sentence. Second sentence.',
                                       'Just one sentence.',
                                       'Adding more sentences. One. Two. Three',
                                       'Another mock article. Just two sentences']})
    nlp = spacy.load("en_core_web_sm")
    spacy_docs, all_sentences = dpu.generate_sentence_corpus_by_year(mock_data, 
                                                                     nlp, 
                                                                     'year',
                                                                     'text',
                                                                     'id')
    assert isinstance(spacy_docs, dict)
    assert isinstance(all_sentences, list)
    assert len(spacy_docs) == 3 # Articles are grouped by year
    assert len(all_sentences) == 9 # Individual sentences are extracted
    assert spacy_docs['2019'][0].text ==  'just one sentence.'
    assert all_sentences[-1] == ('just two sentences', 4, '2020')


def test_get_noun_chunks():
    nlp = spacy.load("en_core_web_sm")
    docs = ['SiCEDS is an agent-based model which will enable the holistic design of a city’s future energy architecture.',
            'thermal rating and life assessment of a transformer are dependent on the hotspot temperature inside the transformer.'
        ]
    spacy_docs = [nlp(doc) for doc in docs]
    func_output = dpu.get_noun_chunks(spacy_docs, remove_det_articles = True)
    expected_output = ['agent-based model',
                       'thermal rating',
                       'life assessment',
                       'transformer',
                       'holistic design',
                       'hotspot temperature',
                       'city’s future energy architecture',
                       'SiCEDS']
    assert(func_output == expected_output)


def test_remove_determiner_articles():
    assert(dpu.remove_determiner_articles('thermal rating') == 'thermal rating')
    assert(dpu.remove_determiner_articles('a good idea') == 'good idea')
    assert(dpu.remove_determiner_articles('an option') == 'option')
    assert(dpu.remove_determiner_articles('another in a way') == 'another in a way')


def test_get_flat_mentions():
    mock_data  = [('this sentence mentions heat pumps.', 4, '2020'),
                  ('no such mentions in this', 4, '2020'),
                  ('heat pump operates like a fridge in reverse', 4, '2020'),
                  ('heat pumps are referenced here',5, '2021'),
                  ('heat pumps, solar panels and hydrogen are often mentioned together', 6, '2021'),
                  ('these represent low carbon heating technologies', 6, '2021'),
                  ('many would consider installing a heat pump.', 7, '2021')]
    func_output = dpu.get_flat_sentence_mentions('heat pumps', mock_data)
    assert(sum([len(v) for v in func_output.values()]) == 3)


def test_combine_term_sentences():
    search_terms = ['heat pump', 'heat pumps']
    mock_data  = [('this sentence mentions heat pumps.', 4, '2020'),
                  ('no such mentions in this', 4, '2020'),
                  ('heat pump operates like a fridge in reverse', 4, '2020'),
                  ('heat pumps are referenced here',5, '2021'),
                  ('heat pumps, solar panels and hydrogen are often mentioned together', 6, '2021'),
                  ('these represent low carbon heating technologies', 6, '2021'),
                  ('many would consider installing a heat pump.', 7, '2021')]
    term_sentences = {term: dpu.get_flat_sentence_mentions(term, mock_data) \
                      for term in search_terms}
    func_output = dpu.combine_term_sentences(term_sentences, search_terms)
    assert(sum([len(v) for v in func_output.values()]) == 5)


