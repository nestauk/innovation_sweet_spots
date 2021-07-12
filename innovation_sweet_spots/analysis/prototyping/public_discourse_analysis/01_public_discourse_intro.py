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
# # Introduction to public discourse data analysis
#
# - Accessing Hansard data
# - Accessing Guardian news articles
# - Fetching speeches or articles with certain keywords
# - Calculating sentiment

# %% [markdown]
# ## 1. Import dependencies

# %%
from innovation_sweet_spots.getters import hansard, guardian
from innovation_sweet_spots.analysis import analysis_utils as iss

# %% [markdown]
# ## 2. Hansard
#
# ### 2.1 Preparing the data
# First, you need to load in the data. By default, it will load speeches starting around 2006; the full data goes back to 1979.

# %%
hans = hansard.get_hansard_data()

# %%
hans.head(1)

# %% [markdown]
# Next, you can create a text "document" for each row by combining texts from a set of specified columns and preprocessing the text as per your needs (using the `preprocessor` variable). Eventually, you might want to create your own, dataset-specific preprocessing function.

# %%
hans_docs = iss.create_documents_from_dataframe(
    hans, columns=["speech"], preprocessor=iss.preprocess_text
)

# %%
hans_docs[0]

# %% [markdown]
# Note that there is another, a bit more sophisticated text preprocessing module in development that also allows extracting phrases and tokenising the text. You can find it via `innovation_sweet_spots.utils.text_pre_processing`. At the moment, there are bits of text preprocessing functions across different parts of the codebase - this will require some refactoring in the near future. For the time being, feel free to use them as is, or you can also create a self-contained module (e.g. `discourse_analysis.py` in `innovation_sweet_spots.analysis_utils`)

# %% [markdown]
# ### 2.2 Early analyses of Hansard

# %%
# Defiene a term to look for in the speeches
search_term = "heat pump"

# %% [markdown]
# ### Number of speeches mentioning a term
#
# Presently, the search for the term is super simple and doesn't involve any synonyms

# %%
# Get all speeches that mention the search term
speeches = iss.search_via_docs(search_term, hans_docs, hans)
# Get stats about the mentions across years
mentions = iss.get_hansard_mentions_per_year(speeches, max_year=2021)

# %%
iss.show_time_series_fancier(mentions, y="mentions", show_trend=False)

# %%
# Percentage growth in the last 5 years, compared to the previous 5 years
iss.estimate_growth_level(mentions, column="mentions")

# %% [markdown]
# ### Sentiment

# %%
# Get all preprocessed documents with the search term
docs_with_term = iss.get_docs_with_term(search_term, hans_docs)
# Extract the specific sentences that mention the term
sentences = iss.get_sentences_with_term(search_term, docs_with_term)
# Calculate sentence sentiment
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# Most negative sentences (not that some of them are spurious)
for i, row in sentiment_df.iloc[0:5].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %%
# Most positive sentences
for i, row in sentiment_df.sort_values("compound").iloc[-5:].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %% [markdown]
# ## 3. Guardian
#
# In case of the news dataset, we're not downloading any large data beforehand. Instead, you can search for a specific term via Guardian API. The code automatically stores the query results on your disk, and next time will fetch the cached query so that you don't exceed the API rate limits. The limits to the best of my knowledge are 5000 calls per day - this is quite enough, as I think each results page is one call, and you can get 100 results per page.
#
# To use this, you'll need to
# - Request an API key from Guardian website ([very easy](https://open-platform.theguardian.com/documentation/))
# - Store it somewhere safe on your local machine (outside the repo) in a `.txt` file
# - Specify the path to this file in `.env` file, by adding a new line with `export GUARDIAN_API_KEY=path/to/file`
#
# See the parameters of the `guardian.search_content` function for more info.

# %% [markdown]
# ### 3.1 Fetching articles
#
# Note that small variations in the search term (e.g. plural vs singular) might change the number of articles you find. As a first step, it would probably be good to have a process which considers the simplest variations of a term, fetches the articles and then deduplcates them.

# %%
search_term = "heat pumps"

# %%
# Fetch articles
articles = guardian.search_content(search_term)

# %%
# Refetch articles from the online source
articles = guardian.search_content(search_term, use_cached=False)

# %%
# You can also fetch only the first page of results, to see how many results there might be in total
test = guardian.search_content(
    search_term="climate crisis", save_to_cache=False, only_first_page=True
)

# %%
# Check the articles
iss.articles_table(articles)

# %% [markdown]
# ### 3.2 Mentions in the news

# %%
mentions = iss.get_guardian_mentions_per_year(articles)
iss.nicer_axis(iss.show_time_series_fancier(mentions, y="articles", show_trend=False))

# %% [markdown]
# ### 3.3 News sentiment

# %%
# Overal sentiment across years (very basic analysis)
yearly_sentiment = iss.news_sentiment_over_years(search_term, articles)
iss.show_time_series_fancier(yearly_sentiment, y="mean_sentiment")

# %%
# Detailed check of sentences
sentences = iss.get_guardian_sentences_with_term(search_term, articles, field="body")
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# Most negative sentences
# Note that there might be quite a few spurious cases; note also that sentence separation is suboptimal,
# and should be changed to spacy sentence detector
for i, row in (
    sentiment_df.sort_values("compound", ascending=True).iloc[0:10].iterrows()
):
    print(row.compound, row.sentences, end="\n\n")

# %%
# Most positive sentences
for i, row in sentiment_df.sort_values("compound").iloc[-10:].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %%
