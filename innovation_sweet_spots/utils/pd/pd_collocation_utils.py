#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:04:48 2021

@author: jdjumalieva
"""
from collections import defaultdict
import pandas as pd
import math
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from innovation_sweet_spots.utils import text_processing_utils as tpu

###
# The following functions relate to tokenising the corpus and producing
# cooccurrence and document-term matrices.
def get_spacy_tokens(sentence_collection, nlp_model):
    """Tokenises the corpus of sentences using spacy.

    Preprocessing includes filtering of stopwords, punctuation and certain entities.

    Args:
        sentence_collection: A list of sentences.
        nlp_model: A spacy model used.

    Returns:
        A list of spacy tokens.
    """
    tokenised = [
        tpu.process_text_disc(doc) for doc in nlp_model.pipe(sentence_collection)
    ]
    return tokenised


def identity_tokenizer(some_list):
    """Converts list elements to string and returns resulting list.

    This is passed as an argument to CountVectorizer or TfidfVectorizer so that
    we can use existing list of elements as tokens.

    Args:
        some_list: A list of spacy tokens.

    Returns:
        A list of strings.
    """
    token_str = [str(elem) for elem in some_list]
    return token_str


def get_ngrams(spacy_tokens, token_range, min_mentions):
    """
    Generates cooccurrence matrix and document term matrices.

    These are later used to calcualte Pointwise Mutual Information.

    Args:
        spacy_tokens: A list of tokenised sentences.
        token_range: A tuple indicating range of ngrams.
        min_mentions: An integer threshold for minimum frequency.

    Returns:
        A sparse cooccurrence matrix.
        A sparse document-term matrix.
        A list of ngram labels.
        A dict mapping token labels to frequency counts in the corpus.
    """
    count_model = CountVectorizer(
        tokenizer=identity_tokenizer,
        lowercase=False,
        ngram_range=token_range,
        min_df=min_mentions,
    )  # default unigram model
    doc_term_m = count_model.fit_transform(spacy_tokens)
    cooccurrence_m = (
        doc_term_m.T * doc_term_m
    )  # this is co-occurrence matrix in sparse csr format
    cooccurrence_m.setdiag(0)

    vocab = count_model.vocabulary_
    names = count_model.get_feature_names()
    count_list = doc_term_m.toarray().sum(axis=0)
    count_dict = dict(zip(names, count_list))

    return cooccurrence_m, doc_term_m, names, count_dict


def tokenise_and_count(
    individual_sentences, nlp_model, mentions_threshold, token_range
):
    """Produces spacy tokens as well as cooccurrence and document-term matrices.

    These outputs are used to calculate Pointwise Mutual Information (PMI) and
    rank terms by frequency of cooccurrences (also referred to as collocations).

    Args:
        individual_sentences: A list of sentences
        nlp_model: A spacy model used.
        mentions_threshold: An integer value referring to the min frequency of mentions.
        token_range: A tuple indicating ngram range.

    Returns:
        A sparse cooccurrence matrix.
        A sparse document-term matrix.
        A list of ngram labels.
        A dict mapping token labels to frequency counts in the corpus.
    """

    tokenised_sentences = get_spacy_tokens(individual_sentences, nlp_model)
    cooccurrence_matrix, doc_term_matrix, token_names, token_counts = get_ngrams(
        tokenised_sentences, token_range, min_mentions=mentions_threshold
    )
    return cooccurrence_matrix, doc_term_matrix, token_names, token_counts


###
# The following functions relate to calculating Pointwise Mutual Information (PMI), which
# measures a strength of association and normalised rank, which captures relative frequency
# of collocated terms (e.g. the lower the rank, the greater the proportion of all collocations)
def calculate_positive_pmi(
    cooccurrence_matrix,
    doc_term_m,
    token_names,
    token_counts,
    search_term,
    coocc_threshold=1,
):
    """Calculates positive Pointwise Mutual Information (PMI) for a given search term.

    For specifics on PMI calculation see docstring for pmi function further below.
    For this analysis context is defined as cooccurrence in the same sentence.

    Args:
        cooccurrence_matrix: A sparse cooccurrence matrix.
        doc_term_m: A sparse document-term matrix.
        token_names: A list of ngram labels.
        token_counts: A dict mapping token labels to frequency counts in the corpus.
        search_term: A string referring to the term of interest.
        coocc_threshold: An intiger value for minimum numer of cooccurrences,
            the default is 1.

    Returns:
        A dict mapping tokens to the value of their positive PMI with the search_term.
    """
    try:  # if a term is rare (e.g. 'waste heat'), it may not be in the cooccurrence matrix
        search_index = token_names.index(search_term)
        pmis = {}
        for ix, name in enumerate(token_names):
            cooccurrence_freq = cooccurrence_matrix[search_index, ix]
            if cooccurrence_freq < coocc_threshold:
                continue
            else:
                association = pmi(
                    np.sum(doc_term_m[:, search_index]),
                    np.sum(doc_term_m[:, ix]),
                    cooccurrence_freq,
                    doc_term_m.shape[0],
                    doc_term_m.shape[0],
                )
            pmis[name] = association

        pruned_pmis = {k: v for k, v in pmis.items() if v > 0}
    except:
        # print(f"Term {search_term} not in vocabulary")
        pruned_pmis = {}
    return pruned_pmis


def pmi(word1, word2, both, all_freq, ngram_freq):
    """Calculates Pointwise Mutual Information (PMI) for two words/ngrams.

    This indicator measures the strength of associations between terms.
    Depending on the definition of context, different values should be used
    to calculate probability of terms co-occurring together (e.g. mentions in the
    same document vs mentions in the same sentence).
    In our particular case all_freq and ngram_freq would be the same -
    total number of sentences.

    See for further info:
        https://courses.engr.illinois.edu/cs447/fa2018/Slides/Lecture17HO.pdf
        https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

    Args:
        word1: An integer count of occurrence of word1.
        word2: An  integer count of occurrence of word2.
        both: An integer number of cooccurrences of both words in the defined context.
        all_freq: A total number of words.
        ngram_freq: A total number of possible cooccurrences.

    Returns:
        A float score for PMI between the given two words/ngrams.
    """
    if word1 == 0 or word2 == 0:
        res = 0
    if both == 0:
        res = 0
    else:
        prob_word1 = word1 / float(all_freq)
        prob_word2 = math.pow(word2, 0.75) / math.pow(
            float(all_freq), 0.75
        )  # taking to the power is used to reduce impact of rare terms
        prob_word1_word2 = both / float(ngram_freq)
        res = math.log(prob_word1_word2 / float(prob_word1 * prob_word2), 2)
    return res


def get_normalised_rank(
    cooccurrence_m, token_names, token_counts, search_term, freq, threshold
):
    """Calculates normalised rank for related terms.

    The measure is derived by dividing term rank in frequency of cooccurrences
    with the search_term by the total number of terms that have been mentioned
    together with the search_term. The lower the rank, the more frequent the
    collocations between a given term and the search_term.

    Normalisaiton is performed to enable comparisons over years.

    Args:
        cooccurrence_matrix: A sparse cooccurrence matrix.
        token_names: A list of ngram labels.
        token_counts: A dict mapping token labels to frequency counts in the corpus.
        search_term: A string referring to the term of interest.
        freq: A minimum frequency of token.
        threshold: A minimum frequency of cooccurrences.

    Returns:
        A dict mapping term to its normalised rank in cooccurrences with a given search_term
    """
    count_rank = dict()
    search_index = token_names.index(search_term)
    total_word_set = np.count_nonzero(cooccurrence_m[search_index, :].toarray())
    for ix, name in enumerate(token_names):
        if token_counts[name] >= freq:
            cooccurence_frequency = cooccurrence_m[search_index, ix]
            if cooccurence_frequency < threshold:
                continue
            else:
                count_rank[name] = cooccurence_frequency
    count_rank_items = sorted(count_rank.items(), key=lambda x: x[1], reverse=True)
    normalised_rank = {
        name: ix + 1 / total_word_set for ix, name in enumerate(count_rank_items)
    }
    return normalised_rank


def get_related_terms(pmi, token_names, token_counts, min_mentions):
    """Identifies terms that have a positive PMI with the search term.

    Args:
        pmi: A dict mapping tokens to the value of their positive PMI with the search_term.
        token_names: A list of ngram labels.
        token_counts: A dict mapping token labels to frequency counts in the corpus.
        min_mentions: A minimum frequency of mentions for a term.

    Returns:
        A dict mapping noun chunks to associated PMIs with the search term.
    """
    terms_above_thresh = [
        elem for elem in token_names if token_counts.get(elem, 0) >= min_mentions
    ]
    term_pmi = {term: pmi.get(term, 0) for term in terms_above_thresh}
    term_pmi_pos = {k: v for k, v in term_pmi.items() if v > 0}
    return term_pmi_pos


###
# The functions below build on previously defined functions and combine their outputs


def identify_related_terms(
    search_term,
    cooccurrence_matrix,
    doc_term_matrix,
    token_names,
    token_counts,
    mentions_threshold,
    coocc_threshold,
):
    """Detects terms that have positive PMI and calculates their normalised frequency rank.

    This function wraps up calculate_positive_pmi, get_related_terms and
    get_normalised_rank.

    Args:
        search_term: A string referring to the term of interest.
        cooccurrence_matrix: A sparse cooccurrence matrix.
        doc_term_matrix: A sparse document-term matrix.
        token_counts: A dict mapping token labels to frequency counts in the corpus.
        mentions_threshold: An integer value referring to the min frequency of mentions.
        coocc_threshold: A minimum frequency of cooccurrences.

    Returns:
        A dict mapping terms to their associated PMI.
        A dict mapping terms to their normalised rank.
    """
    pmis = calculate_positive_pmi(
        cooccurrence_matrix, doc_term_matrix, token_names, token_counts, search_term
    )

    if len(pmis):

        key_related_terms = get_related_terms(
            pmis, token_names, token_counts, min_mentions=mentions_threshold
        )

        normalised_rank = get_normalised_rank(
            cooccurrence_matrix,
            token_names,
            token_counts,
            search_term,
            freq=mentions_threshold,
            threshold=coocc_threshold,
        )
    else:
        key_related_terms = {}
        normalised_rank = {}

    return key_related_terms, normalised_rank


def get_key_terms(
    search_term,
    sentence_collection,
    nlp_model,
    mentions_threshold=2,
    coocc_threshold=1,
    token_range=(1, 3),
):
    """Identifies most relevant terms.

    This function chains outputs from tokenise_and_count and identify_related_terms.
    Given that the code was written to analyse fairly niche technologies, the
    thresholds for mentions and co-locations are very low. This means that there
    would be spurious results included. If you are working with a larger corpus and
    more established topic, you should consider raising the thresholds.

    Args:
        search_term: A string referring to the term of interest.
        sentence_collection: A list of sentences.
        nlp_model: A spacy model used.
        mentions_threshold (int): An integer value referring to the min frequency
            of mentions. The default is 2.
        coocc_threshold: A minimum frequency of cooccurrences. The default is 1.
        token_range: A tuple indicating ngram range. The default is (1,3).

    Returns:
        A dict mapping terms to their associated PMI.
        A dict mapping terms to their normalised rank.
    """
    cm, dtm, tn, tc = tokenise_and_count(
        sentence_collection, nlp_model, mentions_threshold, token_range
    )

    key_terms, norm_rank = identify_related_terms(
        search_term, cm, dtm, tn, tc, mentions_threshold, coocc_threshold
    )
    return key_terms, norm_rank


###
# Functions below serve to aggregate results for a set of terms
def combine_pmi_given_year(related_term_dict, year, search_terms):
    """Aggregates pmi values across the set of terms for a given year.

    If a word/phrase has been mentioned with more than one search term, we take the max pmi.

    Args:
        related_term_dict: A dict mapping year and search term to a corresponding
            list of (word, pmi) items.
        year: A string denoting a year.
        search_terms: A list of search terms.

    Returns:
        A sorted list of (word, pmi) items.
    """
    term_lists = []
    for term in search_terms:
        term_list = related_term_dict[year][term]
        term_lists.append(term_list)
    flat_term_list = sorted(list(itertools.chain(*term_lists)))
    flat_term_dict = dict()
    for term in flat_term_list:
        this_pmi = term[1]
        existing_pmi = flat_term_dict.get(term[0], 0)
        if existing_pmi > this_pmi:
            continue
        else:
            flat_term_dict[term[0]] = this_pmi
    return sorted(flat_term_dict.items(), key=lambda x: x[1], reverse=True)


def combine_pmi(related_term_dict, search_terms):
    """Aggregates pmi values over several years.

    Args:
        related_term_dict: A dict mapping year and search term to a corresponding
            list of (word, pmi) items.
        search_terms: A list of search terms.

    Returns:
        A dict mapping year to the corresponding list of (word, pmi) items.
    """
    combined_related_terms = defaultdict(list)
    for year in related_term_dict:
        given_year_pmi = combine_pmi_given_year(related_term_dict, year, search_terms)
        combined_related_terms[year] = given_year_pmi
    return combined_related_terms


def combine_ranks_given_year(normalised_rank_dict, year, search_terms):
    """Aggregates normalised rank values across the set of terms for a given year.

    Recalculates rank using updated value of total frequency of collocated terms.

    Args:
        normalised_rank_dict: A dict mapping year and term as keys to the list
            of ((word, freq), rank)) items.
        year: A string denoting a year.
        search_terms: A list of search terms.

    Returns:
        A dict mapping terms to the list of ((word, freq), new_rank)) items.
    """
    rank_lists = []
    for term in search_terms:
        rank_list = normalised_rank_dict[year][term]
        if rank_list:
            rank_lists.append(rank_list)
    total_freqs = sum([1 / rank_list[0][1] for rank_list in rank_lists])
    flat_rank_list = sorted(list(itertools.chain(*rank_lists)))
    flat_freq_dict = dict()
    for term in flat_rank_list:
        this_freq = term[0][1]
        existing_freq = flat_freq_dict.get(term[0][0], 0)
        new_freq = existing_freq + this_freq
        flat_freq_dict[term[0][0]] = new_freq
    count_rank_items = sorted(flat_freq_dict.items(), key=lambda x: x[1], reverse=True)
    new_normalised_rank = {
        name: ix + 1 / total_freqs for ix, name in enumerate(count_rank_items)
    }
    return new_normalised_rank


def combine_ranks(normalised_rank_dict, search_terms):
    """Aggregates normalised rank values over several years.

    Args:
        normalised_rank_dict: A dict mapping year and term as keys to the list
            of ((word, freq), rank)) items.
        search_terms: A list of search terms.

    Returns:
        A dict mapping year and (word, freq) to normalised rank.
    """
    combined_ranks = defaultdict(list)
    for year in normalised_rank_dict:
        given_year_rank = combine_ranks_given_year(
            normalised_rank_dict, year, search_terms
        )
        combined_ranks[year] = given_year_rank
    return combined_ranks


def agg_combined_pmi_rank(combined_dict, freq_threshold=2):
    """Converts dict with collocation stats to one dataframe.

    Args:
        combined_dict: A dict mapping search terms to frequency of mentions,
            normalised rank and pmi in a given year.
        freq_threshold: An integer threshold for frequency of mentions. The default is 2.

    Returns:
        A pandas dataframe with stats for each collocation.

    """
    pmi_dfs = []
    for y in combined_dict:
        pmi_df = pd.Series(combined_dict[y]).to_frame().reset_index()
        pmi_df["year"] = y
        pmi_df.columns = ["term", "indicators", "year"]
        pmi_df["freq"] = pmi_df["indicators"].apply(lambda x: x[0])
        pmi_df["rank"] = pmi_df["indicators"].apply(lambda x: x[1])
        pmi_df["pmi"] = pmi_df["indicators"].apply(lambda x: x[2])
        pmi_df = pmi_df[["term", "year", "freq", "rank", "pmi"]]
        pmi_dfs.append(pmi_df)
    all_pmis = pd.concat(pmi_dfs)
    if freq_threshold is not None:
        all_pmis = all_pmis[all_pmis["freq"] > freq_threshold]
    all_pmis = all_pmis.sort_values(by=["term", "year", "freq"])
    return all_pmis


def analyse_rank_pmi_over_time(agg_pmi_df, group_field="term"):
    """Calculates subtotals for dataframe with freq, rank and pmi values for terms over years.

    THe resulting dataframe will then be used to analyse changes over time.

    Args:
        agg_pmi_df: A pandas dataframe with freq, rank, pmi values for terms over years.
        group_field: Avdataframe field name to group on, the default is 'term'.

    Returns:
        A pandas dataframe which shows for each term: year of first mention,
            total number of years with mentions, standard deviation of the rank
            and mean pmi.
    """
    agg_pmi_df.sort_values(by=group_field, inplace=True)
    grouped_terms = agg_pmi_df.groupby(group_field)
    agg_terms = grouped_terms.agg(
        {
            "term": "first",
            "year": lambda x: np.min([int(elem) for elem in x]),
            "freq": "count",
            # below we invert rank to make it more intuitive
            # so higher values would mean higher importance
            "rank": lambda x: np.std([1 / elem for elem in x]),
            "pmi": lambda x: np.mean(x),
        }
    )
    agg_terms.columns = [
        "term",
        "year_first_mention",
        "num_years",
        "st_dev_rank",
        "mean_pmi",
    ]
    agg_terms = agg_terms.round(
        {"year_first_mention": 0, "num_years": 0, "st_dev_rank": 3, "mean_pmi": 3}
    )
    return agg_terms
