#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:04:02 2021

@author: jdjumalieva
"""


import re
from bs4 import BeautifulSoup
import math


def extract_text_from_html(text, tags):
    """
    Extract text from specified html tags.

        Parameters
        ----------
        text : str
            
        tags: list
    
        Returns
        -------
        list of strings

    """
    segments = BeautifulSoup(text, 'html.parser').find_all(tags)
    no_html = [seg.get_text() for seg in segments]
    return no_html


# Placeholder for function to generate simple variations of the search term
# A possible starting point is a trained language model and items with shortest
# Levenstein distance.

def get_text_segments(articles, tags):
    """
    

    Parameters
    ----------
    articles : list
        
    tags : list

    Returns
    -------
    list

    """
    article_html = [article['fields']['body'] for article in articles]
    clean_segments = [extract_text_from_html(article, tags) for article in article_html]
    return clean_segments


def tokenize_asis(some_list):
    """
    Takes list as input.
    Returns the same list unchanged. The function is 
    used as an argument for CountVectorizer or TfidfVectorizer.

    """
    tokens = [elem for elem in some_list]
    return tokens

def identity_tokenizer(some_list):
    token_str = [str(elem) for elem in some_list]
    return token_str

def pmi(word1, word2, both, all_freq, ngram_freq):
    if word1 == 0 or word2 == 0:
        res = 0
    if both == 0:
        res = 0
    else:
        prob_word1 = word1 / float(all_freq)
        prob_word2 = math.pow(word2, 0.75) / math.pow(float(all_freq), 0.75)
        prob_word1_word2 = both / float(ngram_freq)
#        res = (math.log(float(prob_word1*prob_word2),2)/math.log(prob_word1_word2,2))-1
        res = math.log(prob_word1_word2/float(prob_word1*prob_word2),2)
    return(res)