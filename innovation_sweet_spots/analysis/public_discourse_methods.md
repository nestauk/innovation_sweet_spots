# Public discourse analysis

- [] Collecting articles from the Guardian
- [] Preprocessing articles
- [] Analysing collocations
- [] POS analysis and phrase matching

## Introduction
This document provides an overview of the methodology used to analyse the coverage
of `heat pumps` and `hydrogen heating` in the Guardian articles. The broad motivation
for the analysis was to measure the level of media attention that `heat pumps` and
`hydrogen heating` have been receiving and to study the discourse around these technologies.

Nesta's Discovery Hub team combined the signals derived from the public discourse analysis
with the insights from the analysis of research and investment activity to perform
data-driven horizon scan of technologies for decarbonising heating. You can read
about the key findings and recommendations in the [**Innovation Sweet Spots**](https://www.nesta.org.uk/data-visualisation-and-interactive/innovation-sweet-spots/) report.

In the following sections we describe how the articles were collected and preprocessed.
We then show how we applied NLP techniques to analyse the sentences and phrases that
mention the terms that we are interested in. Figure 1 shows the key stages of the analysis.
![key_stages](pd_key_stages.png)


## Data collection
A dataset of over 2 million Guardian articles is publicly accessible via the Guardian
API. We extracted a subset of relevant articles using a set of pre-defined search terms and
the API's in-built search functionality (Figure 2).

![data_collection_steps](pd_data_collection.png)

In addition to article text we also extracted information related to
the article title, publication date, web url, contributors and keywords. Retaining
the article metadata made it possible to quickly retrieve the original articles and
was useful for interpreting results.

For the purpose of our analysis, we restricted the search to articles published
between 1st of January 2017 and 1st of June 2021 (specified in config settings for guardian_api).
We were also particularly interested in the coverage of `heat pumps` and `hydrogen`
in the context of low carbon heating in the UK. This is why we only included articles
from select categories aligned with our research focus. These categories were
`'Environment', 'Guardian Sustainable Business', 'Technology', 'Science', 'Business',
'Money', 'Cities', 'Politics', 'Opinion', 'Global Cleantech 100', 'The big energy debate',
'UK news', 'Life and style'`. The resulting list of articles was further refined to
remove any articles that did not mention UK, Britain or any nation states.

In the case of `hydrogen heating` additional filtering was necessary because a broad
`hydrogen` term was often used in the articles to refer to hydrogen energy and its
applications in heating and transport. However the same term was used in articles
related to astronomy and chemistry. We defined a set of disambiguation terms (e.g.
heating, homes, boilers, etc.) and used it to further refine the article selection.

Once the dataset of articles was formed, as a next step we extracted article text
from the `p` (paragraph) and `h2` (second level heading) tags in the original html.
The sections corresponding to the tags were subsequently joined to form the article text,
but it's possible to adapt current functionality to retrieve the original nested
text segments if necessary.

`Example_data_collection.py` notebook illustrates the steps described above.
The notebook outputs a dict capturing article metadata and a dataframe with
extracted article text and article ID. The utility functions used in the notebook
are defined in `pd_data_collection_utils.py` with corresponding tests in `test_dcu.py`.

## Data preprocessing
To prepare the article text for subsequent analysis, we performed several operations (see Figure 3).

![data_preprocessing_steps](pd_data_preprocessing.png)

`Example_data_preprocessing.py` notebook illustrates the steps described above.
The notebook generates several outputs that we describe below. The utility functions used in the notebook
are defined in `pd_data_processing_utils.py` with corresponding tests in `test_dpu.py`.

First, we cleaned the article text. This involved splitting camel-case, removing extra whitespaces
and converting strings to lower case. We also normalised punctuation and retained
it to aid sentence detection. To better understand the language used to describe
terms of interest (e.g. search terms), we also made a decision to minimise initial text
cleaning to avoid removing terms that might be informative.
Which is why we don't perform lemmatisation or stemming. Stopwords, punctuation,
digits, urls and spaces are only removed at a later stage (as part of tokenisation
to calculate association measures), but no named entities are filtered out.

As a next step, we used `spacy` to process clean articles in order to generate a
corpus of `spacy` documents and a collection of sentences. The former output
`processed_articles_by_year` stores `spacy` representation of articles mapped
to each year covered in the dataset.  This output was used to apply `spacy` methods
for extracting `noun_chunks`, which are phrases that contain a noun and some words
describing it. The collection of sentences, `sentence_records`,
maps individual sentences to a unique ID and year of publication. This information
was used to group sentences by year and to retrieve metadata, such as original article
url and title.

After generating sentence records, we extracted from them the sentences, which
contained individual search terms. We used the resulting `term_sentences` to
analyse the frequency of mentions of terms across the dataset. We created an
aggregated version of this output (`combined_term_sentences`), which contained
all sentences that mentioned any of the search terms.

Term mentions are summarised in `mentions_df` dataframe. In this file we show
the number of sentences that mentioned individual search terms across the dataset.
We also show the total number of documents that contained any of the search terms.
This figure is useful as it accounts for instances when a document contains multiple
references to search terms.

We found that reviewing the resulting time series of mentions was very informative.
Intuitively, just looking at the mentions allows us to gauge the level of discourse
around a technology of interest. So, we can see whether it is receiving more or less  
attention from the media or in the parliament. We can then look at the content of
the articles that mention our terms of interest to understand what is driving
the increase or decline in interest. This is where analysis of collocations and phrase
matching come into play.

## Analysis of collocations
Collocations help us understand what terms are often used together with the terms of interest.
The definition of a collocation depends on what we consider to be a context. By context, we refer
to the text surrounding search terms. It can vary depending on the analytical question at hand.
In some instances the whole document can be considered as context. In this case, we narrow the
definition of the context to a sentence. In subsequent analyses it would be useful to investigate
whether using paragraphs as context improves results.

Of all collocations, we are most interested in the ones that are mentioned most frequently or consistently
with our search terms. There are several approaches for measuring importance of collocations.
The first approach is to calculate *Pointwise Mutual Information (PMI)*. This measure reflects
the strength of association between terms and it is higher when terms are more likely to coincide
mutually rather than independently.

The second measure is purely descriptive and involves calculating a *proportion of co-occurrences*
for a given collocated term. Unlike PMI, where a rare term might have a high PMI
because it was only ever mentioned with a given search term, the proportion of co-occurrences will be
directly influenced by the absolute frequency of terms coinciding in the same context (i.e. sentence).
The measure is still relative, so we are able to compare it over time.

The third measure is a *normalised rank*, which is similar to the proportion of co-occurrences,
but designed to reflect relative importance of terms. As the name suggests, to calculate
this measure we rank collocations by frequency of co-occurrences with the search term in a
descending order. We then normalise the rank by the total number of unique collocations in
a given time period. Doing so makes it possible to compare the values in different time periods.

While studying the measures in a given year is useful, it is more informative to analyse how the
importance of terms varies over time. The findings generated by this analysis can point to shifts in
language used to describe technologies of interest. The shifts might be caused by the emergence
of new entities and applications of the technology in question.

One way to detect shifts in language used to describe terms of interest is to calculate
a standard deviation of association measures of collocated terms in different time periods.
The results can then be compared to identify terms that changed the most in their
mentions. If the standard deviation is low, this means that the frequency and consistency
with which a given term was collocated with the search term did not change much over time.
In contrast, a large standard deviation might indicate that importance of terms rose and fell
over time. We can also compare changes in collocation measures of different terms. This can be useful if
we want to evaluate the relative prominence of different types of a particular technology
over time.  For instance, comparing `air heat pumps` and `ground heat pumps`
showed that more recently the former have featured more prominently in the news articles.

It is important to note that some search terms are only mentioned in a few articles each year
and as a result some of their collocations are spurious. It is possible to control this
to some extent by increasing the threshold for a minimum number of occurrences of a given
term in the dataset. By default this value is set to 1, but can be adjusted in `get_key_terms`
function.

There is a trade off, however, to increasing the mentions threshold. It might be that
terms with one a few mentions actually refer to novel applications of the
technologies of interest. By only looking at more common terms, we might fail to detect
emergence of novel contexts in which technology is used.

The process we used to analyse collocations for `heat pumps` and `hydrogen` is shown in Figure 4.

![analysis_of_collocations](pd_collocations.png)

The relevant code is in `Example_collocations.py` and corresponding utilities in `pd_collocation_utils.py`.
At the moment, indicators for collocations and calculations of variation over time are outputted
to `agg_terms_stats.cvs` to be reviewed manually. In future work, we would explore best
approaches to automatically flag terms that demonstrate a large shift in their mentions.

## Part of speech analysis and phrase matching
We use Part of speech (POS) analysis as a semi-supervised way to gauge sentiment and
as an alternative to fully automated sentiment analysis. We initially trialed
using VADER to automatically measure sentiment of sentences that mention any of
the search terms. What we found is that results had to be investigated manually.
There were various reasons why sentiment scores were spurious in some instances.
One of the reasons was that some noun phrases and verbs have inherent non-neutral sentiment  
in VADER. For example `energy` has a positive score and `lower emissions` has a negative
score. Another reason is that the sentiment was often driven by phrases that were
mentioned alongside search terms, but did not related to them. As it was not trivial
to extract only the phrase immediately relevant to the search term, the results
of the automated sentiment scores were not as useful for our project. Because of this
we used automated methods for extracting phrases containing terms of interest and
then reviewed them manually.

Different types of phrases can help us understand discourse around technologies of interest.
* Adjective phrases tend to describe attributes of a given technology.
* Noun phrases can also capture attributes, but also describe types/ applications and as
well as entities.
* Verb phrases are useful for identifying actions and effects *by* innovation of interest and
 *on* it.
* Among verb phrases some patterns are particularly useful as they tend to summarise
arguments in favour or against a particular tech. Examples include `innovation is` and `innovation has`.

The process is outlined in Figure 5.

![part_of_speech_analysis](pd_pos_analysis.png)

We defined and matched the phrases of interest using spacy `Matcher` method. Given that
both Guardian and Hansard datasets span multiple years, we first matched phrases in
articles for a given year and aggregated those across different search terms (e.g. `hydrogen
energy` and `hydrogen boiler`). The results were then reviewed manually. For additional insights,
we linked phrases back to original articles and provided a corresponding url.

Currently, the POS analysis outputs are designed to aid an analyst in rapidly building an understanding
of the key features of the discourse in a particular area. More work is needed to automate
the process for generating insights and ensure the methods are also generalisable to other areas of interest.
In case of search terms that generate a large number of results an intermediate step
will be necessary, where documents or paragraphs that mention terms of interest
are pre-grouped using unsupervised machine learning techniques such as LDA, top2vec and others.
Some of these were used in other analytical streams of this project.

## Links to results of the public discourse analysis
As mentioned earlier, the key findings from the public discourse analysis can be accessed in the
[**Innovation Sweet Spots**](https://www.nesta.org.uk/data-visualisation-and-interactive/innovation-sweet-spots/)
report. The report also covers analysis of research funding and investment activity and provides policy
recommendations.
