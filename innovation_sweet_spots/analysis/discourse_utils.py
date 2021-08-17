#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:04:02 2021

@author: jdjumalieva
"""

import os
import re
from bs4 import BeautifulSoup
import math
import pandas as pd
import numpy as np
import pickle
from typing import Iterator
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from textacy import extract

from innovation_sweet_spots.utils import text_cleaning_utils as tcu
from innovation_sweet_spots.utils import text_pre_processing as tpu
from innovation_sweet_spots.analysis import analysis_utils as iss
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH



DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["tok2vec"]}
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"

# THis contains a list of terms that should have neutral sentimen
# for the purposes of green tech analysis (e.g. 'energy demand', 'lower emissions')
with open(os.path.join(DISC_OUTPUTS_DIR, 'vader_exceptions.pkl'), "rb") as infile:
        vader_exceptions = pickle.load(infile)

vader_replacements = {elem: ' ' for elem in vader_exceptions}


def extract_text_from_html(html, tags: Iterator[str]) -> Iterator[str]:
    """
    Extract text from specified html tags.
    
    Parameters
    ----------
    html (str): content of an article in html format.
    tags (list): list of tags to extract content from.

    Returns
    -------
    no_html (list): list of chunks (e.g. paragraphs) of text extracted from the tags.

    """
    segments = BeautifulSoup(html, 'html.parser').find_all(tags)
    no_html = [seg.get_text() for seg in segments]
    return no_html


# Placeholder for function to generate simple variations of the search term
# A possible starting point is a trained language model and items with shortest
# Levenstein distance.

def get_text_segments(articles: Iterator[dict], tags: Iterator[str])\
    -> Iterator[str]:
    """
    Extract segments of article text from Guardian articles in html format.

    Parameters
    ----------
    articles (list): list of dicts containing result of the Guardian 
    search_content function.
    tags (list): list of tags to extract content from.

    Returns
    -------
    clean_segments (list): nested list of article segments.

    """
    article_html = [article['fields']['body'] for article in articles]
    clean_segments = [extract_text_from_html(article, tags) for article in article_html]
    return clean_segments

def get_article_text(articles: Iterator[dict], tags: Iterator[str]):
    """
    Extract full text of an article from Guardian articles in html format.
    
    Parameters
    ----------
    articles (list): list of dicts containing result of the Guardian 
    search_content function.
    tags (list): list of tags to extract content from.

    Returns
    -------
    article_text (list): list of article text.

    """
    article_segments = get_text_segments(articles, tags)
    article_text = [' '.join(segment) for segment in article_segments]
    return article_text


# def tokenize_asis(some_list: Iterator[str]):
#     """
#     Takes list as input.
#     Returns the same list unchanged. The function is 
#     used as an argument for CountVectorizer or TfidfVectorizer.

#     """
#     tokens = [elem for elem in some_list]
#     return tokens


def identity_tokenizer(some_list):
    """
    Convert list elements to string and return resulting list. This is passed
    as an argument to CountVectorizer or TfidfVectorizer so that we can use existing
    list of elements as tokens.
    
    Parameters
    ----------
    some_list (list): list of spacy tokens.

    Returns
    -------
    token_str (list): list of strings.

    """
    token_str = [str(elem) for elem in some_list]
    return token_str


def pmi(word1, word2, both, all_freq, ngram_freq):
    """
    Calculate pointwise mutual information for two words/ngrams.
    Depending on the definition of context, different denominator will be used.
    In our particular case all_freq and ngram_freq would be the same - 
    total number of sentences. But if the whole document is used, then number of ngrams
    should be used as denominator for prob_word1_word2.
    See for further info:
        https://courses.engr.illinois.edu/cs447/fa2018/Slides/Lecture17HO.pdf
        https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

    Parameters
    ----------
    word1 (int): count of occurrence of word1.
    word2 (int): count of occurrence of word2.
    both (int): number of cooccurrences in the defined context.
    all_freq (int): total number of words.
    ngram_freq (int): total number of possible cooccurrences.

    Returns
    -------
    res (float): pointwise mutual information for given two words.

    """
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
    """
    Currently unused function to estimate proportion of positive, neutral and
    negative sentences for a given term

    """
    pos_sent = len([elem for elem in sentiment_score_list if elem >0])
    neutral_sent = len([elem for elem in sentiment_score_list if elem ==0])
    total_sent = len(sentiment_score_list)
    neg_sent = total_sent - (pos_sent + neutral_sent)
    prop_pos = round(pos_sent/total_sent, 3)
    prop_neg = round(neg_sent/total_sent, 3)
    prop_neut = round(neutral_sent/total_sent, 3)
    return (prop_pos, prop_neut, prop_neg)


def define_context(sentence, options, n_words = None, 
                   search_term = None, nlp_model = None):
    """
    Specify context for sentiment analysis. 
    The default option is 'sentence', which will just return sentence itself.
    Specifying 'n_words' window will return subset of the sentence with n_words 
    to the left and right of the search term.
    Option 'phrase' will return sentence subtree that contains the search term.
    
    Parameters
    ----------
    sentence (str): individual sentence.
    options (str): definition of context, possible values are 'sentence', 
    'n_words' and 'phrase'.
    n_words (int): size of n_word window.
    search_term (str): central term within the context.
    nlp_model (spacy.lang.en.English): spacy model used for analysis.

    Returns
    -------
    res (str): context which will vary depending on options specified in args.
    
    """    
    if options == 'n_words':
        return get_window(sentence, n_words, search_term)
        
    elif options == 'phrase':
        return get_phrase(sentence, search_term, nlp_model)
        
    elif options == 'sentence':
        return sentence
    

def get_window(sentence, n_words, search_term):
    """
    Return subset of the sentence with n_words to the left and right of the 
    search term. Includes cleaning of determiner articles and commas.
    This function is used in define_context.

    Parameters
    ----------
    sentence (str): input sentence.
    n_words (int): size of n_word window.
    search_term (str): central term within the context.

    Returns
    -------
    combined (str): n_word phrase.

    """
    sentence = sentence.replace(' a ', ' ').replace(' the ',' ')
    parts = sentence.partition(search_term)
    parts = [elem.replace(', ', ' ') for elem in parts]
    parts = [elem.strip() for elem in parts]
    left = ' '.join(parts[0].split()[-n_words:])
    right = ' '.join(parts[2].split()[:n_words])
    combined = ' '.join([left, search_term, right])
    return combined


def get_phrase(sentence, search_term, nlp_model):
    """
    Return sentence subtree that contains the search term. Need to refine
    patterns used in spacy phrase matching. Currently it extracts subtrees for 
    any noun, adjective or verb. These are then filtered to include search term.
    The longest such subtree is usually the whole sentence itself, so we go
    for the second longest subtree. But these may be unnecessarily long
    themselves.

    Parameters
    ----------
    sentence (str): input sentence.
    search_term (str): central term within the context.
    nlp_model (spacy.lang.en.English): spacy model used for analysis.

    Returns
    -------
    corresponding_sent (str): subset of original sentence derived from obtained
    spacy subtree (which is a list of tokens).

    """
    sentence = tcu.clean_up(tcu.remove_punctuation(sentence))
    if nlp_model is None:
        nlp_model = tpu.setup_spacy_model(DEF_LANGUAGE_MODEL)
    matches = []
    for token in nlp_model(sentence):
        if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            # to test for only verb forms 'is' and 'was' use 
            # token.head.lower_ in ['is', 'was']
            matches.append([t.text for t in token.subtree])
        
    subtrees = []
    for match in matches:
        if set(search_term.split()).issubset(set(match)):
            subtrees.append(match)
    
    # Here we need the second longest match, because the longest one is the whole sentence
    lens = {ix: len(elem) for ix, elem in enumerate(subtrees)}
    if len(lens) >1:
        target_ix = sorted(lens.items(), key = lambda x: x[1], reverse = True)[1]
        # phrase = ' '.join(subtrees[target_ix[0]])
        phrase_start = subtrees[target_ix[0]][0]
        phrase_end = subtrees[target_ix[0]][-1]
        start_ix = sentence.split().index(phrase_start)
        end_ix = sentence.split().index(phrase_end)
        corresponding_sent = ' '.join(sentence.split()[start_ix:end_ix+1])

        # phrase = tcu.remove_punctuation(phrase)
        # phrase = tcu.clean_up(phrase)
        # phrase_start = phrase[0]
        return corresponding_sent
    else:
        return sentence # this means that there isn't a proper subtree for search term
    
    
def extract_categories(articles: Iterator[dict]):
    """
    Extract article categories from article data returned by the Guardian API.

    Parameters
    ----------
    articles (list): list of dicts containing result of the Guardian 
    search_content function.

    Returns
    -------
    article_categories (list): list of categories.

    """
    article_categories = [article['sectionName'] for article in articles]
    return article_categories    


def agg_categories(article_categories: Iterator[str], top_n = 10):
    """
    Aggregate categories across articles and identify 10 most popular ones.

    Parameters
    ----------
    article_categories (list): list of article categories
    top_n (int, optional): the number of most popular categories.
    the default is 10.

    Returns
    -------
    category_count (list): list of tuples showing category and number of articles.

    """
    category_count = sorted(Counter(article_categories).items(), 
                        key = lambda x: x[1], 
                        reverse=True)
    return category_count[:top_n]

    
def get_top_n_categories(grouped_articles, top_n =10):
    """
    Aggregate categories for articles and extract most popular categories 
    for each year.

    Parameters
    ----------
    grouped_articles (dict): articles grouped by year using itertools.
    top_n (int, optional): the number of most popular categories.
    the default is 10.

    Returns
    -------
    cat_dict (dict): most popular categories in each year.

    """
    cat_dict = dict()
    for y, articles in grouped_articles.items():
        article_categories = extract_categories(articles)
        aggregated_categories = agg_categories(article_categories)
        cat_dict[y] = aggregated_categories
    return cat_dict


def generate_sentence_corpus(articles: Iterator[str], nlp = None):
    """
    Clean article text, process with spacy and break up into sentences.

    Parameters
    ----------
    articles (list): text of articles.
    nlp (spacy.lang.en.English, optional): spacy model used. the default is None.

    Returns
    -------
    article_sentences (list): list of sentences
    spacy_docs (list[]): list of processed spacy docs (spacy.tokens.doc.Doc)

    """
    if nlp is None:
        nlp = tpu.setup_spacy_model(DEF_LANGUAGE_MODEL)
    clean_article_text = [tcu.clean_text_minimal(article) for article in articles]
    clean_article_text = [elem for elem in clean_article_text if elem != 'sentence is blank']
    spacy_docs = [nlp(article) for article in clean_article_text]
    article_sentences = [[sent.text for sent in article.sents] for article in spacy_docs]
    return article_sentences, spacy_docs


def remove_determiner_articles(text):
    """
    Remove determiner articles 'a', 'an', 'the'. Used to clean up noun phrases.
    A regex pattern would make this more elegant.

    Parameters
    ----------
    text (str): some text

    Returns
    -------
    text (str): text with determiner articles removed

    """
    # replacements = {' a ': ' ', 
    #                 ' an ': ' ',
    #                 ' the ': ' '}
    # for art, rep in replacements.items():
    #     text = text.replace(art, rep)
    #     return text
    if text[:2] == 'a ':
        text = text.lstrip('a').lstrip()
    elif text[:2] == 'an':
        text = text.lstrip('an').lstrip()
    elif text[:2] == 'th':
        text = text.lstrip('the').lstrip()
    return text



def calculate_sentiment(sentence_collection, search_term, context, n_words = None, 
                        nlp_model = None):
    """
    Return sentiments for a given collection of contexts. Sentiment is calculated
    using VADER and functionality from iss.get_sentece_sentiment_modified.

    Parameters
    ----------
    sentence_collection (list): list of sentences
    search_term (str): central term in a context.
    context (str): definition of context, possible values are 'sentence', 
    'n_words' and 'phrase'.
    n_words (int): size of n_word window. the default is None.
    nlp_model (spacy.lang.en.English): spacy model. the default is None.

    Returns
    -------
    sentiment_df (pandas.core.frame.DataFrame): dataframe with VADER scores for
    each sentence/context.

    """
    target_context = [define_context(sentence, context, n_words, search_term, nlp_model) for\
                       sentence in sentence_collection]
    # modified function is used to measure sentiment to address biases in VADER
    # related to use of green tech terms (e.g. 'energy demand', 'lower emissions')
    context_sentiment = [iss.get_sentence_sentiment_modified(elem, vader_replacements) for\
                         elem in target_context]
    sentiment_df = pd.DataFrame(context_sentiment)
    sentiment_df["context"] = target_context
    return sentiment_df


# Addressing determiner articles, but not stopwords
def get_noun_chunks(spacy_corpus, remove_det_articles = False):
    """
    Extract noun phrases from articles using spacy's inbuit methods.

    Parameters
    ----------
    spacy_corpus (): spacy processed documents.
    remove_det_articles (boolean): option to remove determiner articles.
    the default is False.

    Returns
    -------
    dedup_noun_chunks (list): deduplicated list of noun chunks as strings.

    """
    noun_chunks = []
    for article in spacy_corpus:
        for chunk in article.noun_chunks:
            noun_chunks.append(chunk) #could change to chunk.lemma_ if needed 

    #convert spacy tokens to string
    noun_chunks_str = [str(elem) for elem in noun_chunks]
    if remove_det_articles:
        noun_chunks_str = [remove_determiner_articles(elem) for elem in noun_chunks_str]
    dedup_noun_chunks = list(set(noun_chunks_str))
    return dedup_noun_chunks


def get_spacy_tokens(sentence_collection, nlp_model):
    """
    Tokenise the corpus of sentences using spacy. Preprocessing includes:
    filtering of stopwords, punctuation and certain entities.

    Parameters
    ----------
    sentence_collection (list): list of sentences.
    nlp_model (spacy.lang.en.English): spacy model used.

    Returns
    -------
    tokenised (list): list of spacy tokens.

    """
    tokenised = [tpu.process_text_disc(doc) for doc in nlp_model.pipe(sentence_collection)]
    return tokenised
    

def get_ngrams(spacy_tokens, token_range, min_mentions):
    """
    Generate cooccurrence matrix and document term matrix that will be then
    used to calcualte PMI.
    
    Parameters
    ----------
    spacy_tokens (list): tokenised sentences.
    token_range (tuple): range of ngrams.
    min_mentions (int): threshold for minimum frequency.

    Returns
    -------
    cooccurrence_m (scipy.sparse.csc.csc_matrix): sparse cooccurrence matrix.
    doc_term_m (scipy.sparse.csc.csc_matrix): sparse document-term matrix.
    names (list): ngram labels.
    count_dict (dict): token frequency counts in the corpus.

    """
    count_model = CountVectorizer(tokenizer = identity_tokenizer,
                              lowercase = False, 
                              ngram_range = token_range,
                              min_df = min_mentions) # default unigram model
    doc_term_m = count_model.fit_transform(spacy_tokens)
    cooccurrence_m = (doc_term_m.T * doc_term_m) # this is co-occurrence matrix in sparse csr format
    cooccurrence_m.setdiag(0)
    
    vocab = count_model.vocabulary_
    names = count_model.get_feature_names()
    count_list = doc_term_m.toarray().sum(axis=0)
    count_dict = dict(zip(names,count_list))
    
    return cooccurrence_m, doc_term_m, names, count_dict


def calculate_positive_pmi(cooccurrence_matrix, doc_term_m, token_names, 
                           token_counts, search_term, coocc_threshold = 2):
    """
    Calculate positive PMI for a given search term. For this analysis context
    is defined as cooccurrence in the same sentence.

    Parameters
    ----------
    cooccurrence_matrix (scipy.sparse.csc.csc_matrix): sparse cooccurrence matrix.
    doc_term_m (scipy.sparse.csc.csc_matrix): sparse document-term matrix.
    token_names (list): ngram labels.
    token_counts (dict): token frequency counts in the corpus.
    search_term (str): term of interest.
    coocc_threshold (int): minimum numer of cooccurrences, the default is 2.

    Returns
    -------
    pruned_pmis (dict): positive PMI value for a given token in relation to search_term.

    """
    try: #if a term is rare (e.g. 'waste heat'), it may not be in the cooccurrence matrix
        search_index = token_names.index(search_term)
        pmis = {}
        for ix, name in enumerate(token_names):
            cooccurrence_freq = cooccurrence_matrix[search_index, ix]
            if cooccurrence_freq < coocc_threshold:
                continue
            else:
                association = pmi(np.sum(doc_term_m[:, search_index]),
                              np.sum(doc_term_m[:, ix]),
                              cooccurrence_freq,
                              doc_term_m.shape[0],
                              doc_term_m.shape[0])
            pmis[name] = association
            
        pruned_pmis = {k:v for k,v in pmis.items() if v >0}
    except:
        print('term not in vocabulary')
        pruned_pmis = {}
    return pruned_pmis
    

def get_related_terms(noun_chunks, pmi, token_names, token_counts, min_mentions):
    """
    Identify noun chunks that have a positive PMI with the search term.

    Parameters
    ----------
    noun_chunks (list): list of spacy noun chunks converted to string.
    pmi (dict): mapping of terms to PMI values in relation to search term.
    token_names (list): ngram labels.
    token_counts (dict): token frequency counts in the corpus.
    min_mentions (int): threshold for minimum frequency.

    Returns
    -------
    chunk_pmi (dict): PMIs for noun chunks.

    """
    pruned_noun_chunks = [elem for elem in noun_chunks if elem in token_names]
    pruned_noun_chunks = [elem for elem in pruned_noun_chunks if \
                          token_counts.get(elem, 0) > min_mentions]


    chunk_pmi = {chunk: pmi.get(chunk, 0) for chunk in pruned_noun_chunks}
    chunk_pmi = {k:v for k,v in chunk_pmi.items() if v >0}
    
    return chunk_pmi


def get_normalised_rank(cooccurrence_m, token_names, token_counts, search_term, 
                        freq = 3, threshold = 2):
    """
    Calculate normalised rank for related terms by dividing term rank in frequency
    of cooccurrences with the search_term by the total number of terms that have
    been mentioned together with the search_term. Normalisaiton is performed to
    enable comparisons over years.

    Parameters
    ----------
    cooccurrence_matrix (scipy.sparse.csc.csc_matrix): sparse cooccurrence matrix.
    token_names (list): ngram labels.
    token_counts (dict): token frequency counts in the corpus.
    search_term (str): term of interest.
    freq (int): minimum frequency of token, the default is 3.
    threshold (int): minimum frequency of cooccurrences, the default is 2.

    Returns
    -------
    normalised_rank (dict): term: normalised rank

    """
    count_rank = dict()
    search_index = token_names.index(search_term)
    total_word_set = np.count_nonzero(cooccurrence_m[search_index,:].toarray())
    for ix, name in enumerate(token_names):
        if token_counts[name] > freq:
            cooccurence_frequency = cooccurrence_m[search_index, ix]
            if cooccurence_frequency < threshold:
                continue
            else:
                count_rank[name] = cooccurence_frequency
    count_rank_items = sorted(count_rank.items(), key = lambda x: x[1], reverse = True)
    normalised_rank = {name: ix/total_word_set for ix, name in enumerate(count_rank_items)}
    return normalised_rank


def get_article_metadata(grouped_articles, fields_to_extract = ['id']):
    """
    Extract useful article fields from raw data returned by the Guardian API.

    Parameters
    ----------
    grouped_articles (dict): articles grouped by year using itertools.
    fields_to_extract (list): list of fields.

    Returns
    -------
    year_article_df_combined (pandas.core.frame.DataFrame): dataframe with 
    content of article fields.

    """
    year_article_dfs = []
    for year, articles in grouped_articles.items():
        year_data = dict()
        for field in fields_to_extract:
            article_field = [article[field] for article in articles]
            year_data[field] = article_field
            year_data['year'] = year
        year_article_df = pd.DataFrame(year_data)
        year_article_dfs.append(year_article_df)
    year_article_df_combined = pd.concat(year_article_dfs)
    year_article_df_combined = year_article_df_combined.drop_duplicates()
    return year_article_df_combined


def get_article_text_df(grouped_articles, tags, article_metadata):
    """
    Extract article text from articles grouped by year. Combine with metadata
    to keep track in the subsequent analysis.

    Parameters
    ----------
    grouped_articles (dict): articles grouped by year using itertools.
    tags (list): specified html tags.
    article_metadata (pandas.core.frame.DataFrame): dataframe with article content.

    Returns
    -------
    year_articles (pandas.core.frame.DataFrame): dataframe with article text, id and year.

    """
    year_articles_across_years = []
    for year, articles in grouped_articles.items():
        article_text = get_article_text(articles, tags)
        article_id = [article['id'] for article in articles]
        year_article_df = pd.DataFrame({'id': article_id,
                                       'text': article_text})
        year_articles_across_years.append(year_article_df)
    year_articles = pd.concat(year_articles_across_years)
    year_articles = year_articles.merge(article_metadata, left_on = 'id',
                                        right_on = 'id',
                                        how = 'left')
    year_articles = year_articles.drop_duplicates()

    return year_articles
        
        
def get_sentence_corpus(article_text_df, nlp_model):
    """
    Generate corpus of sentences, spacy processed docs and sentence records
    for each year in the dataset.

    Parameters
    ----------
    article_text_df (pandas.core.frame.DataFrame): dataframe with article text, id and year.
    nlp_model (spacy.lang.en.English): spacy model used.

    Returns
    -------
    sentences_by_year (dict): sentence corpus for each year.
    processed_articles_by_year (dict): spacy docs for each year.
    sentence_records (list): list of tuples with sentence, id, year.

    """
    sentence_records = []
    sentences_by_year = defaultdict(dict)
    processed_articles_by_year = defaultdict(dict)
    for year, group in article_text_df.groupby('year'):
        sentences, processed_articles = generate_sentence_corpus\
            (group['text'], nlp_model)
        ids = group['id']
        for sentence_bunch in zip(sentences, ids):
            article_id = sentence_bunch[1]
            for sentence in sentence_bunch[0]:
                sentence_records.append((sentence, article_id, year))
        sentences_by_year[year] = sentences
        processed_articles_by_year[year] = processed_articles
    return(sentences_by_year, processed_articles_by_year, sentence_records)


def get_flat_sentence_mentions(search_term, sentence_collection):
    """
    Identify sentences that contain search_term using regex.
    Convert list of lists (i.e. lists of sentences in a given article) into
    a flat list of sentences.

    Parameters
    ----------
    search_term (str): term of interest.
    sentence_collection (list): list of tuples with sentence, id, year.

    Returns
    -------
    year_flat_sentences (dict): sentences with term in each year.

    """
    base = r'{}'
    expr = '(?:\s|^){}(?:,?\s|$)'
    combined_expr = base.format(''.join(expr.format(search_term)))
    year_flat_sentences = dict()
    sentence_collection_df = pd.DataFrame(sentence_collection)
    sentence_collection_df.columns = ['sentence', 'id', 'year']
    for year, sentences in sentence_collection_df.groupby('year'):
        sentences_with_term = sentences[sentences['sentence'].\
                                        str.contains(combined_expr, regex = True)]
        year_flat_sentences[year] = sentences_with_term
    return year_flat_sentences
    

def get_term_mentions(year_flat_sentences):
    """
    Calculate number of mentions.

    Parameters
    ----------
    year_flat_sentences (dict): sentences with term in each year.

    Returns
    -------
    num_mentions (dict): number of sentences with term in each year.

    """
    num_mentions = {k: len(v) for k,v in year_flat_sentences.items()}
    return num_mentions

        
def combine_flat_sentence_mentions(search_terms, sentence_collection):
    """
    Extend get_term_mentions to work with set of terms.

    Parameters
    ----------
    search_terms (list): list of terms.
    sentence_collection (list): list of tuples with sentence, id, year.

    Returns
    -------
    flat_sentences_all_terms (dict): sentences with term in each year.

    """
    flat_sentences_all_terms = defaultdict(dict)
    for term in search_terms:
        term_sentences = get_flat_sentence_mentions(term, sentence_collection)
        flat_sentences_all_terms[term] = term_sentences
    return flat_sentences_all_terms


def collate_mentions(search_terms, flat_sentences_dict):
    """
    Aggregate mentions across terms and years.

    Parameters
    ----------
    search_terms (list): list of terms.
    flat_sentences_dict (dict): sentences with term in each year.

    Returns
    -------
    mentions_all_terms (list): list of dicts.

    """
    mentions_all_terms = []
    for term in search_terms:
        mentions = get_term_mentions(flat_sentences_dict[term])
        mentions_all_terms.append(mentions)
    # mentions_combined = pd.DataFrame.from_records(mentions_all_terms)
    # mentions_combined = mentions_combined.T
    # mentions_combined.columns = search_terms
    return mentions_all_terms
        
        
def total_docs(article_text_df):
    """
    Get total count of articles in each year.

    Parameters
    ----------
    article_text_df (pandas.core.frame.DataFrame): dataframe with article text, 
    id and year.

    Returns
    -------
    total_docs (dict): number of articles in each year.

    """
    total_docs = {year: len(articles) for year, articles in \
                  article_text_df.groupby('year')}
    return total_docs


def agg_term_sentiments(term, term_sentences, context = 'n_words', n_words = 5, 
                        nlp_model = None):
    """
    Calculate sentiment for each sentence with term, then take average of those.

    Parameters
    ----------
    term (str): term of interest
    term_sentences (dict): 
    context (str): definition of context, possible values are 'sentence', 
    'n_words' and 'phrase'.
    n_words (int): size of n_word window. the default is 5.
    nlp_model (spacy.lang.en.English): spacy model. the default is None.
    
    Returns
    -------
    agg_sentiments (pandas.core.frame.DataFrame): dataframe with average scores.
    sentiments_across_years_df (pandas.core.frame.DataFrame): all sentence seniments.

    """
    # Sentiment for different types of context
    sentiments_all_years = []
    for year in term_sentences[term]:
        sentences = term_sentences[term][year]['sentence']
        sentiments = calculate_sentiment(sentences,
                                                    term, 
                                                    context, 
                                                    n_words)
        sentiments['year'] = year
        sentiments_all_years.append(sentiments)

    sentiments_across_years_df = pd.concat(sentiments_all_years)
    agg_sentiments = sentiments_across_years_df.groupby('year').agg({'neg': 'mean', 
                                                                 'neu': 'mean', 
                                                                 'pos': 'mean', 
                                                                 'compound': 'mean'})
    return agg_sentiments, sentiments_across_years_df


def average_sentiment_across_terms(aggregated_sentiments):
    """
    Average agg_sentiments dataframes across set of terms.

    Parameters
    ----------
    aggregated_sentiments (dict): dataframes with average sentiment scores

    Returns
    -------
    averaged_sentiment (pandas.core.frame.DataFrame): average sentiment for all terms.

    """
    combined_df = pd.concat(aggregated_sentiments.values())
    averaged_sentiment = combined_df.groupby(combined_df.index).mean()
    return averaged_sentiment



def tokenise_and_count(individual_sentences, 
                       nlp_model, 
                       mentions_threshold = 3,
                       token_range = (1,3)):
    """
    Produce spacy tokens and cooccurrence and document-term matrices.

    Parameters
    ----------
    individual_sentences (list): sentences
    nlp_model (spacy.lang.en.English): spacy model used.
    mentions_threshold (int): min frequency, the default is 3.
    token_range (tuple): ngram range, the default is (1,3).

    Returns
    -------
    cooccurrence_matrix (scipy.sparse.csc.csc_matrix): sparse cooccurrence matrix.
    doc_term_matrix (scipy.sparse.csc.csc_matrix): sparse document-term matrix.
    token_names (list): ngram labels.
    token_counts (dict): token frequency counts in the corpus.

    """
            
    tokenised_sentences = get_spacy_tokens(individual_sentences, nlp_model)
    cooccurrence_matrix, doc_term_matrix, token_names, token_counts = get_ngrams(\
                                                    tokenised_sentences,
                                                    token_range, 
                                                    min_mentions = mentions_threshold)
    return  cooccurrence_matrix, doc_term_matrix, token_names, token_counts


def identify_related_terms(search_term,
                           cooccurrence_matrix, 
                           doc_term_matrix, 
                           token_names, 
                           token_counts,
                           noun_chunks,
                           mentions_threshold = 3,
                           coocc_threshold = 2):
    """
    Identify terms that have positive PMI and calculate their normalised frequency
    rank.

    Parameters
    ----------
    search_term (str): term of interest
    cooccurrence_matrix (scipy.sparse.csc.csc_matrix): sparse cooccurrence matrix.
    doc_term_matrix (scipy.sparse.csc.csc_matrix): sparse document-term matrix.
    token_names (list): ngram labels.
    token_counts (dict): token frequency counts in the corpus.    
    noun_chunks (list): spacy noun chunks converted to string.
    mentions_threshold (int): min frequency, the default is 3.
    coocc_threshold (int): min cooccurrence, the default is 2.

    Returns
    -------
    key_related_terms (dict): terms and their associated PMI.
    normalised_rank (dict): terms and their normalised rank.

    """

    # Calculate PMI defining sentence as a context
    pmis = calculate_positive_pmi(cooccurrence_matrix, 
                                              doc_term_matrix, 
                                              token_names, 
                                              token_counts,
                                              search_term)
    
    if len(pmis):

        # Identify most relevant noun phrases
        key_related_terms = get_related_terms(noun_chunks, 
                                            pmis, 
                                            token_names, 
                                            token_counts, 
                                            min_mentions = mentions_threshold)
    
        # Add normalised cooccurrence rank
        normalised_rank = get_normalised_rank(cooccurrence_matrix, token_names, 
                                                     token_counts, search_term, 
                                                     freq = mentions_threshold,
                                                     threshold = coocc_threshold)
    else:
        key_related_terms = {}
        normalised_rank = {}
    
    return key_related_terms, normalised_rank  


def get_key_terms(search_term, sentence_collection, nlp_model, noun_chunks, 
                      mentions_threshold = 3, coocc_threshold = 2, 
                      token_range = (1,3)):
    """
    Identify most relevant terms.
    This function chains outputs from tokenise_and_count and identify_related_terms.

    Parameters
    ----------
    search_term (str): term of interest.
    sentence_collection (list): sentences.
    nlp_model (spacy.lang.en.English): spacy model used.
    noun_chunks (list): spacy noun chunks converted to string.
    mentions_threshold (int): min frequency, the default is 3.
    coocc_threshold (int): cooccurrence threshold, the default is 2.
    token_range (tuple): ngram range, the default is (1,3).


    Returns
    -------
    key_terms (dict): terms and their associated PMI.
    norm_rank (dict): terms and their normalised rank.

    """
    cm, dtm, tn, tc = tokenise_and_count(sentence_collection, nlp_model, mentions_threshold,
                       token_range)
    
    key_terms, norm_rank = identify_related_terms(search_term, cm, dtm, tn, tc, noun_chunks)
    return key_terms, norm_rank
    

def agg_rank_dfs(term, norm_ranks, freq_threshold = 50):
    """
    Collect all normalised ranks across years into a dataframe.
    This will be used to assess language shift over time.

    Parameters
    ----------
    term (str): term of interest.
    norm_ranks (dict): normalised ranks in each year
    freq_threshold (int): min frequency. the default is 50.

    Returns
    -------
    all_ranks : TYPE
        DESCRIPTION.

    """
    rank_dfs = []
    for y in norm_ranks:
        rank_df = pd.DataFrame(norm_ranks[y][term])
        rank_df['year'] = y
        rank_df.columns = ['term_count', 'normalised_rank', 'year']
        rank_df['term'] = rank_df['term_count'].apply(lambda x: x[0])
        rank_df['frequency'] = rank_df['term_count'].apply(lambda x: x[1])
        rank_dfs.append(rank_df)
    all_ranks = pd.concat(rank_dfs)
    all_ranks = all_ranks[all_ranks['frequency']>freq_threshold]
    all_ranks= all_ranks.sort_values(by = ['term', 'year'])
    return all_ranks


def get_svo_triples(sentence_collection, search_term, nlp_model):
    """
    Extract subject verb object triples from sentences using textacy.

    Parameters
    ----------
    sentence_collection (list): list of sentences.
    search_term (str): term of interest.
    nlp_model (spacy.lang.en.English): spacy model used.

    Returns
    -------
    svo_subject (list): phrases where search term acts as subject.
    svo_object (list): phrases where search term acts as object.

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
    """
    Convert textacy SVO triplets into phrases.

    Parameters
    ----------
    svo_subject (list): list of SVO triplets
    svo_object (list): list of SVO triplets

    Returns
    -------
    subject_phrases (list): list of whole phrases
    object_phrases (list): list of whole phrases

    """
    subject_phrases = []
    for triple in svo_subject:
        subj = ' '.join([str(elem) for elem in triple.subject])
        verb = ' '.join([str(elem) for elem in triple.verb])
        obj = ' '.join([str(elem) for elem in triple.object])
        subject_phrase = ' '.join([subj, verb, obj])
        subject_phrases.append(subject_phrase)
        
    subject_phrases = list(set(subject_phrases))
    
    object_phrases = []
    for triple in svo_object:
        subj = ' '.join([str(elem) for elem in triple.subject])
        verb = ' '.join([str(elem) for elem in triple.verb])
        obj = ' '.join([str(elem) for elem in triple.object])
        object_phrase = ' '.join([subj, verb, obj])
        object_phrases.append(object_phrase)
    
    object_phrases = list(set(object_phrases))
    return subject_phrases, object_phrases


def get_verbs(svo_list):
    """
    Obtain verbs from subject-verb-object triplets.

    Parameters
    ----------
    svo_list (list): list of SVO triplets.

    Returns
    -------
    flat_verbs (list): list of verbs

    """
    verbs = [svo.verb for svo in svo_list]
    flat_verbs = list(set([elem for sublist in verbs for elem in sublist]))
    flat_verbs = list(set([str(elem) for elem in flat_verbs]))
    return flat_verbs
    

