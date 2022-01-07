#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:08:57 2021

@author: jdjumalieva
"""
import itertools
import pandas as pd
from collections import defaultdict, Counter
from textacy import extract
from spacy.matcher import Matcher
from spacy.util import filter_spans

from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import \
    pd_data_processing_utils as dpu
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH


def noun_chunks_w_term(noun_chunks_dict, search_terms):
    """Identifies noun chunks that contain any of the search terms.
    
    Args:
        noun_chunks_dict: A dict with year as key and list of noun chunks as values.
        search_terms: A list of search terms.
    
    Returns:
        A dict with year as key and list of noun chunks as values.
    """
    chunks_with_term = defaultdict(list)
    for year, chunks in noun_chunks_dict.items():
        contain_term = [[elem for elem in chunks if term in elem] for term in search_terms]
        contain_term = list(itertools.chain(*contain_term))
        chunks_with_term[year] = list(set(contain_term))
    return chunks_with_term


def find_pattern(sentences, nlp_model, pattern):
    """Identifies matches to specified pattern in a collection of sentences.
    
    Args:
        sentences: A list of sentences.
        nlp_model: A spacy model used.
        pattern: A pattern in the form of list of dicts. These are defined separately.
    
    Returns:
        A list with matches.
    """
    matcher = Matcher(nlp_model.vocab) 
    matcher.add("pattern", [pattern])
    all_matches = []
    for sentence in sentences:
        doc = nlp_model(sentence)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        longest_span = filter_spans(spans)
        if longest_span:
            all_matches.append(longest_span)
    all_matches = [item.text for sublist in all_matches for item in sublist]
    return all_matches


def match_patterns_across_years(sentence_dict, nlp_model,
                                pattern, n_years = 3, field = 'sentence'):
    """Identifies pattern matches in a collection of sentences over several years.
    
    Args:
        sentence_dict: A dict with year as key and list of sentences as values.
        nlp_model: A spacy model used.
        pattern: A pattern in the form of list of dicts.
        n_years: An int specifying the size of time period (e.g. number of years),
            the default is 3.
        field: A str referring to the name of the dataframe field that contains sentences. 
            The default is 'sentence'.
    
    Returns:
        A dict with period name as key and list of phrases as values.
    """
    period = list(sentence_dict.keys())
    year_chunks = []
    for i in range(0, len(period), n_years):
        year_chunks.append(period[i:i + n_years])
    phrases = defaultdict(list)
    for chunk in year_chunks:
        chunk_name = ', '.join(chunk)
        for year in chunk:
            year_sentences = sentence_dict.get(year,{})
            if len(year_sentences):
                year_phrases = find_pattern(year_sentences[field], nlp_model, pattern)
                phrases[chunk_name].append(year_phrases)
    return phrases


def aggregate_matches(phrase_dict, sort_phrases = True):
    """Combines phrase matches over a time period and and counts frequency.
    
    Args:
        phrase_dict: A dict with period name as key and list of phrases as values.
        sort_phrases: If true sorts phrases alphabetically, the default is True.
    
    Returns:
        A dict with period name as key and list of (phrase, count) tuples as values.
    """
    agg_results = defaultdict(list)
    for year_period, phrases in phrase_dict.items():
       flat_results = [elem for sublist in phrases for elem in sublist]
       sorted_results = sorted(Counter(flat_results).items())
       agg_results[year_period] = sorted_results
    return agg_results
    return agg_results


def view_phrase_sentences(year, agg_phrases, sentence_collection_df, 
                          metadata_dict, sentence_record_dict, year_field = 'year',
                          output_data = False, output_path = OUTPUT_DATA_PATH):
    """Prints out the original sentences in which phrases were were used.
    

    Args:
        time_period: A list of years included in a given time_period.
        agg_phrases: A dict with period name as key and list of (phrase, count) 
            tuples as values.
        sentence_collection_df: A pandas dataframe with sentences.
        metadata_dict: A dict mapping article IDs to original article metadata.
        sentence_record_dict: A dict mapping sentences to article IDs.
        year_field: A string referring to the dataframe field with year of the article.
            The default is 'year'.

    Returns:
        None.
    """
    print(year)
    if len(agg_phrases[year]) == 0:
        print('No phrases in this time period')
    else:
        for p in agg_phrases[year]:
            grouped_sentences = dpu.check_collocations(sentence_collection_df, p[0])
            if year in grouped_sentences.groups:
                dpu.view_collocations_given_year(grouped_sentences.get_group(year), 
                                                 metadata_dict, 
                                                 sentence_record_dict)
            else:
                continue
        

def view_phrase_sentences_period(time_period, agg_phrases, sentence_collection_df, 
                          metadata_dict, sentence_record_dict, year_field = 'year'):
    """Prints out the original sentences in which phrases were were used.
    

    Args:
        time_period: A string referring to years included in a given time_period.
        agg_phrases: A dict with period name as key and list of (phrase, count) 
            tuples as values.
        sentence_collection_df: A pandas dataframe with sentences.
        metadata_dict: A dict mapping article IDs to original article metadata.
        sentence_record_dict: A dict mapping sentences to article IDs.
        year_field: A string referring to the dataframe field with year of the article.
            The default is 'year'.

    Returns:
        None.
    """
    if len(agg_phrases[time_period]) == 0:
        print('No phrases in this time period')
    else:
        for p in agg_phrases[time_period]:
            year_subset = []
            grouped_sentences = dpu.check_collocations(sentence_collection_df, p[0])
            for some_year in time_period.split(', '):
                if some_year in grouped_sentences.groups:
                    year_subset.append(grouped_sentences.get_group(some_year))
            print(p)
            if len(year_subset) > 0:
                year_subset_df = pd.concat(year_subset)
                dpu.view_collocations(year_subset_df.groupby('year'), metadata_dict, 
                                      sentence_record_dict)
            else:
                print('No results')
                print('')
        print('********')
        
##########
# Below are functions used to extract Subject-Verb-Object triples

def get_svo_triples(sentence_collection, search_term, nlp_model):
    """Extracts subject verb object (SVO) triples from sentences using textacy.
    
    Only returns SVO triples that contain a search_term.
    
    Args:
        sentence_collection: A list of sentences.
        search_term: A string referring to the term of interest.
        nlp_model: A spacy model used.
    
    Returns:
        A list of phrases where search term acts as subject.
        A list of phrases where search term acts as object.
    """
    svo_subject = []
    svo_object = []
    
    for sent in sentence_collection:
        sent_svos = list(extract.triples.subject_verb_object_triples(nlp_model(sent)))
        for svo in sent_svos:                
            if set(search_term.split()).issubset(set([str(elem) for elem in svo.subject])):
                svo_subject.append(svo)
            if set(search_term.split()).issubset(set([str(elem) for elem in svo.object])):
                svo_object.append(svo)
                
    return svo_subject, svo_object


def get_svo_phrases(svo_subject, svo_object):
    """Concatenates textacy SVO triples.
    
    Args:
        svo_subject: A list of SVO triples where search term is a subject.
        svo_object: A list of SVO triples where search term is an object.
   
    Returns:
    -------
    A concatenated string with SVO elements where search term is a subject.
    A concatenated string with SVO elements where search term is an object.
    """
    subject_phrases = []
    for triple in svo_subject:
        subj = ' '.join([str(elem) for elem in triple.subject])
        verb = ' '.join([str(elem) for elem in triple.verb])
        obj = ' '.join([str(elem) for elem in triple.object])
        subject_phrase = '...'.join([subj, verb, obj])
        subject_phrases.append(subject_phrase)
        
    subject_phrases = list(set(subject_phrases))
    
    object_phrases = []
    for triple in svo_object:
        subj = ' '.join([str(elem) for elem in triple.subject])
        verb = ' '.join([str(elem) for elem in triple.verb])
        obj = ' '.join([str(elem) for elem in triple.object])
        object_phrase = '...'.join([subj, verb, obj])
        object_phrases.append(object_phrase)
    
    object_phrases = list(set(object_phrases))
    return subject_phrases, object_phrases


def save_phrases(phrase_objects):
    results = []
    for obj in phrase_objects:
        for time_period, phrases in obj.items():
            for p in phrases:
                results.append([time_period, p[0], p[1]])
    result_df = pd.DataFrame.from_records(results)
    result_df.columns = ["year", "phrase", "number_of_mentions"]
    return result_df
            
            