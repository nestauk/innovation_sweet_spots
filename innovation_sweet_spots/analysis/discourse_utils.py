#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:04:02 2021

@author: jdjumalieva
"""


import re
from bs4 import BeautifulSoup
import math
import pandas as pd
from collections import Counter


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
        res = math.log(prob_word1_word2/float(prob_word1*prob_word2),2)
    return(res)

def prop_pos(sentiment_score_list):
    pos_sent = len([elem for elem in sentiment_score_list if elem >0])
    neutral_sent = len([elem for elem in sentiment_score_list if elem ==0])
    total_sent = len(sentiment_score_list)
    neg_sent = total_sent - (pos_sent + neutral_sent)
    prop_pos = round(pos_sent/total_sent, 3)
    prop_neg = round(neg_sent/total_sent, 3)
    prop_neut = round(neutral_sent/total_sent, 3)
    return (prop_pos, prop_neut, prop_neg)


def define_context(sentence, options = 'sentence', n_words = None, search_term = None):
    """
    Specify context for sentiment analysis. 
    sentence is default option and will just return sentence itself.
    n_words window will return subset of the sentence with n words to the left 
    and right of the search term.
    phrase will return sentence subtree that contains the search term.

    """

    if options == 'sentence':
        result = sentence
        
    elif options == 'n_words':
        result = get_window(sentence, n_words, search_term)
        
    elif options == 'phrase':
        result = get_phrase(sentence, search_term)
    
    return result


def get_window(sentence, n_words, search_term):
    sentence = sentence.replace(' a ', ' ').replace(' the ',' ')
    parts = sentence.partition(search_term)
    parts = [elem.replace(', ', ' ') for elem in parts]
    parts = [elem.strip() for elem in parts]
    left = ' '.join(parts[0].split()[-n_words:])
    right = ' '.join(parts[2].split()[:n_words])
    combined = ' '.join([left, search_term, right])
    return combined

def get_phrase(sentence, search_term):
    matches = []
    for token in nlp(sentence):
        if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            # to test for only verb forms 'is' and 'was' use token.head.lower_ in ['is', 'was']
            matches.append([t.text for t in token.subtree])
        
    subtrees = []
    for match in matches:
        if set(search_term.split()).issubset(set(match)):
            subtrees.append(match)
    
    # Here we need the second longest match, because the longest one is the whole sentence
    subtrees = [elem for elem in subtrees if ' '.join(elem) != sentence]
    lens = {ix: len(elem) for ix, elem in enumerate(subtrees)}
    target_ix = max(lens.items(), key = lambda x: x[1])
    phrase = ' '.join(subtrees[target_ix[0]])
    phrase = tcu.remove_punctuation(phrase)
    phrase = tcu.clean_up(phrase)
    return phrase
        

def get_num_articles(grouped_articles):
    n_per_year = {k: len(v) for k,v in grouped_articles.items()}
    return n_per_year

def show_num_articles(n_per_year):
    num_articles_df = pd.Series(n_per_year).to_frame()
    num_articles_df.columns = ['Number of articles']
    print(num_articles_df)
    
def extract_categories(articles):
    article_categories = [article['sectionName'] for article in articles]
    return article_categories    

def agg_categories(article_categories, top_n = 10):
    category_count = sorted(Counter(article_categories).items(), 
                        key = lambda x: x[1], 
                        reverse=True)
    return category_count[:top_n]
    
def get_top_n_categories(grouped_articles, top_n =10):
    cat_dict = dict()
    for y, articles in grouped_articles.items():
        article_categories = extract_categories(articles)
        aggregated_categories = agg_categories(article_categories)
        cat_dict[y] = aggregated_categories
    return cat_dict