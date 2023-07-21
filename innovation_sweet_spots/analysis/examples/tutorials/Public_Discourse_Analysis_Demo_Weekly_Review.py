# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Public discourse analysis demo
#
# **February 2023**
#
# This notebook is a demonstration on using Innovation Sweet Spot's public discourse analysis module.
#
# It can be used to fetch and analyse *The Guardian* news articles, but the analysis can also be applied to any other text data.
#
# We provide examples for either news articles or the UK parliament speeches, to:
#
# *   See mentions of search terms over time
# *   Understand the vocabulary used around these terms
# *   Group the text into topics
#

# %% [markdown]
# ## Step 1: Download and setup dependencies
#
# Running the following cells will install the Innovation Sweet Spots code and other necessary python packages.
#
# Skip this step if running locally instead not on Colab.

# %%
# !git clone https://github.com/nestauk/innovation_sweet_spots.git

# %%
import sys

sys.path.insert(0, "/content/innovation_sweet_spots")

# %%
# !cd innovation_sweet_spots && \
# pip install -r requirements.txt

# %% [markdown]
# ## Step 2: Import packages

# %%
import altair as alt

# Import gdown to download the Hansard data
import gdown

# Import pandas and set a long column width
import pandas as pd

pd.set_option("max_colwidth", 1000)
# Import innovation_sweet_spots utils
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.pd import pd_analysis_utils as au
import innovation_sweet_spots.analysis.analysis_utils as analysis_utils

# %% [markdown]
# ## Step 3: Download Guardian data
#
# This step shows how to fetch news articles from the Guardian mentioning "heat pumps"
#
# Alternatively, you can skip and go to **step 3-B** to use already prepared Hansard dataset

# %% [markdown]
# First you should define your Guardian API key.
#
# Setting it to `"test"` might work, but you should set up your own key here: https://open-platform.theguardian.com/access/
#
# When using these utils locally, you can also store the key in a safe place, and then provide the path to the key in the .env, which is located in the repository's root folder.

# %%
API_KEY = "test"

# %% [markdown]
# You can take a peek at the results by setting `using only_first_page=True`

# %%
g = au.guardian.search_content(
    "heat pump",
    api_key=API_KEY,
    only_first_page=True,
    use_cached=False,
    save_to_cache=False,
)

# %% [markdown]
# It should say that 100 articles is about 37% of the total number of results, so you can work it out that there are a little bit less than 300 on the Guardian mentioning heat pumps

# %% [markdown]
# You can check that the most recent article should be from this year:

# %%
# Get the first (most recent) result
g[0]["id"]

# %% [markdown]
# Now let's get all articles mentioning heat pumps.
#
# In my experience, best to use both singular and plural forms to get catch relevant results.

# %%
guardian_search_terms = ["heat pump", "heat pumps"]
text_df, text_metadata = au.get_guardian_articles(
    # Specify the search terms
    search_terms=guardian_search_terms,
    # To fetch the most recent articles, you can also set use_cached to False
    use_cached=True,
    # A query identifier is used to name the output folders and files
    query_identifier="heat_pump_tutorial",
    # Specify the API key
    api_key=API_KEY,
)

# %%
# Article texts
text_df

# %%
# Article metadata
text_metadata[text_df.iloc[0].id]

# %% [markdown]
# ## Step 4: Public discourse analysis
#
# You will use the `DiscourseAnalysis` class.
#
# You will need to define an `outputs_path` and a `query_id` for loading and saving data.
#
# With our variables below we will be saving an dloading discourse analysis hansard data to and from `innovation_sweet_spots/outputs/data/discourse_analysis_outputs/{query_id}`.

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/data/discourse_analysis_outputs/"

# %% [markdown]
# In the following, run the relevant cell for either *Guardian* or Hansard analysis

# %%
# Run the following lines when doing Guardian analysis
QUERY_ID = "guardian_heat_pumps_tutorial"
SEARCH_TERMS = guardian_search_terms

# %%
pda = au.DiscourseAnalysis(
    search_terms=SEARCH_TERMS,
    outputs_path=OUTPUTS_DIR,
    query_identifier=QUERY_ID,
)
# Note that you can pass required_terms and banned_terms to filter your results (see documentation of DiscourseAnalysis)

# %% [markdown]
# The warning message above says we are missing document text and metadata.
#
# Metadata is optional and can be used when using *Guardian* articles. The next steps will add document text to the class using the `load_documents` function.
#
#
# This function has an argument `document_text` which can take a dataframe variable or if left blank will search for a file `document_text_{query_id}.csv` in `outputs/data/discourse_analysis_outputs/{query_id}/`.
#
# Note that you can use `load_documents` to input any text data, as long as it has columns for `text`, `date`, `year` and `id`.

# %%
pda.load_documents(document_text=text_df)

# %% [markdown]
# ### Step 4-1: Mentions of the search terms in the documents and sentences.

# %% [markdown]
# All the documents that contain the search terms.

# %%
pda.document_text

# %% [markdown]
# The number of documents per year that contain the search terms.
#
# (The results for each search term are combined and deduplicated)

# %%
pda.document_mentions

# %% [markdown]
# Plot of the number of documents per year that contain the search terms.

# %%
pda.plot_mentions(use_documents=True)

# %% [markdown]
# You can also plot the number of sentences, and disaggregate the number of sentences per each search term.
#
# (This might take a minute, as the text is processed into sentences using spacy)

# %%
pda.plot_mentions(use_documents=False)

# %% [markdown]
# You can then get all sentences with the search terms for a specific year, using the dictionary `combined_term_sentences`

# %%
pda.combined_term_sentences["2022"]

# %% [markdown]
# ### Step 4-2 Collocation mentions
#
# The `view_collocations` function can be used to find sentences where the search term appears with another specified term.

# %%
pda.view_collocations("air source")

# %% [markdown]
# #### Deeper dive in co-locations

# %% [markdown]
# You can also check `term_rank` table for all co-located terms for each year (using pointwise mutual information, PMI)
#
# Frequency and rank indicates how often the terms have been used together with the search terms.
#
# Note that this might take a minute, and sometimes on Colab it surprisingly runs out of memory on this. As an alternative, you can also ran a simpler query `analyse_colocated_unigrams`, which does not use the PMI measure, to find frequently mentioned single-word terms.

# %%
pda.term_rank

# %%
# Check most often co-located terms
(
    pda.term_rank.groupby("term")
    .agg(freq=("freq", "sum"))
    .sort_values("freq", ascending=False)
    .head(25)
)

# %%
# Run this example if analysing Guardian data on heat pumps
check_terms = ["air source", "ground source"]
fig = (
    alt.Chart(pda.term_rank.query("term in @check_terms"))
    .mark_line()
    .encode(
        x=alt.X("year:O", title=""),
        y=alt.Y("freq:Q", title="Frequency"),
        color=alt.Color("term:N", title="Term"),
    )
)
fig


# %%
# Run this example if analysing Guardian data on heat pumps
check_terms = ["install", "installation"]

fig = (
    alt.Chart(pda.term_rank.query("term in @check_terms"))
    .mark_line()
    .encode(
        x=alt.X("year:O", title=""),
        y=alt.Y("freq:Q", title="Frequency"),
        color=alt.Color("term:N", title="Term"),
    )
)
fig


# %%
pda.term_rank.query("term in @check_terms")

# %% [markdown]
# The `term_temporal_rank` can be used to potentially highlight interesting terms whose rank has changed significantly (ie, has a high variation across years)

# %%
pda.term_temporal_rank.sort_values("st_dev_rank", ascending=False).head(15)

# %% [markdown]
# Try out also this simpler approach using only unigrams

# %%
# Simpler measure using only unigrams
pda.analyse_colocated_unigrams().sort_values("counts", ascending=False).head(20)

# %% [markdown]
# ### Step 4-3: Part of Speech (POS) phrase matching
#
# We can also analyse specific types of phrase patterns containing the search term. For example, phrases that contain adjectives or verbs together with the search term.
#
# More information of POS phrase matching can be found [here](https://spacy.io/usage/rule-based-matching).
#
# The phrase patterns can either be loaded from a json file  theor default phrase patterns can be generated on the fly using `make_patterns=True`.

# %% [markdown]
# First we need to set the phrases patterns. Here we are making patterns based on the search terms.

# %%
pda.set_phrase_patterns(load_patterns=False, make_patterns=True)

# %% [markdown]
# Then we can find matches in the documents that match the phrase patterns.
#
# This might take a minute.

# %%
pda.pos_phrases

# %%
sorted(pda.pos_phrases.pattern.unique())

# %%
# query the dataframe for rows where pattern column has the word 'verb' in it
pos = "is"
(
    pda.pos_phrases.groupby(["phrase", "pattern"], as_index=False)
    .agg(number_of_mentions=("number_of_mentions", "sum"))
    .query(f"pattern.str.contains('{pos}')", engine="python")
    .sort_values("number_of_mentions", ascending=False)
    # .head(10)
)

# %% [markdown]
# ## Step 4-4: Topic modelling
#
# We can use BERTopic to find topics within our documents. More info on BERTopic can be found [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To create a topic model, use function `fit_topic_model`. If you want to use sentences found from phrase matching set the variable `use_phrases` to `True` (note, if using phrases, `set_phrase_patterns` will need to be run first.) If set to `False` it will use the `sentence_mentions`.

# %% [markdown]
# Creating a topic model.

# %%
topic_model, docs = pda.fit_topic_model(use_phrases=False)

# %% [markdown]
# Visualising the topics.

# %%
topic_model.visualize_topics()

# %% [markdown]
# Visualising the words with the highest TF-IDF in each topic.

# %%
topic_model.visualize_barchart(top_n_topics=len(set(topic_model.topics_)))

# %% [markdown]
# Visualising the documents and topics in 2d space (mouseover for the text). Note that the `visualize_documents` function is somewhat random. You can make the output deterministic by following the steps [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To make the plot larger (to be able to view more of the mouse over text, increase the values for the parameters `width` and `height`)

# %%
topic_model.visualize_documents(docs, width=1400, height=750)

# %% [markdown]
# ## Save the analysis outputs
#
# Speeds up the analysis. Next time, if you specify the same query id, it should load the results and you won't have to compute the phrases and patterns again.

# %%
pda.save_analysis_results()

# %%
