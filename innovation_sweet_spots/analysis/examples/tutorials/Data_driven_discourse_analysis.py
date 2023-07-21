# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Tutorial: Data-driven news discourse analysis with Python
#
# **July 2023**
#
# This notebook follows the Medium tutorial article, and uses Innovation Sweet Spots' public discourse analysis modules.
#
# We will fetch and analyse *The Guardian* news articles, but the analysis can also be applied to any other text data. We will provide exampels for:
#
# *   Checking mentions of search terms over time
# *   Exploring the news topics using BERTopic
# *   Understanding the language used around these terms using spaCy
#

# ## Setting up
#
# Running the following cells will install the Innovation Sweet Spots code and other necessary python packages.
#
# Skip this step if running locally instead not on Colab.

# !git clone https://github.com/nestauk/innovation_sweet_spots.git

import sys

sys.path.insert(0, "/content/innovation_sweet_spots")

# !cd innovation_sweet_spots && \
# pip install -r requirements.txt

# ## Importing requirements

# +
# Import packages
import altair as alt
import pandas as pd
from innovation_sweet_spots.utils.pd import pd_analysis_utils as au

# Specify the location for analysis outputs
from innovation_sweet_spots import PROJECT_DIR

OUTPUTS_DIR = PROJECT_DIR / "outputs/data/discourse_analysis_outputs"
# -

# ## Getting the data: Using the Guardian Open Platform
#
# This step shows how to fetch news articles from the Guardian mentioning "heat pumps".

# First you should define your Guardian API key.
#
# Setting it to `"test"` might work, but you should set up your own key here: https://open-platform.theguardian.com/access/

API_KEY = "test"

# You can take a peek at the results by setting `using only_first_page=True`

test_articles = au.guardian.search_content(
    "heat pumps",
    api_key=API_KEY,
    only_first_page=True,
    use_cached=True,
    save_to_cache=False,
)

# At the time of writing this tutorial, tt should say that 100 articles is about 14% of the total number of results, so you can work it out that there are around 700 articles on the Guardian mentioning heat pumps

# You can check that the most recent article

# Get the first (most recent) result
test_articles[0]

# Now let's get all articles mentioning heat pumps.
#
# In my experience, best to use both singular and plural forms to get catch relevant results.
#
# We will also specify the following article categories to reduce the possibility of irrelevant articles.

# Define allowed article categories
CATEGORIES = [
    "Environment",
    "Technology",
    "Science",
    "Business",
    "Money",
    "Cities",
    "Politics",
    "Opinion",
    "UK news",
    "Life and style",
]


# +
# List of search terms
SEARCH_TERMS = ["heat pump", "heat pumps"]

articles_df, articles_metadata = au.get_guardian_articles(
    # Specify the search terms
    search_terms=SEARCH_TERMS,
    # To fetch the most recent articles, set use_cached to False
    use_cached=True,
    # Specify the API key
    api_key=API_KEY,
    # Specify which news article categories we'll consider
    allowed_categories=CATEGORIES,
)

# -

# Article texts
articles_df.head(3)

# Article metadata
articles_metadata[articles_df.iloc[0].id]

# ## Initialising the `DiscourseAnalysis` class
#
# First, we can specify the name for this analysis session, which will allow us to save the results of this analysis in a folder of the same name and load them in a future session.

QUERY_ID = "guardian_heat_pumps_tutorial"

# We will be saving and loading our analysis results to and from `innovation_sweet_spots/outputs/data/discourse_analysis_outputs/{QUERY_ID}`.
#
# We will then define a couple of additional filtering criteria to keep the most relevant results to our context, by specifying a (non-exhaustive) list of UK-related geographic terms and excluding any article that mentions Australia.

# +
REQUIRED_TERMS = [
    "UK",
    "Britain",
    "Scotland",
    "Wales",
    "England",
    "Northern Ireland",
    "Britons",
    "London",
]

BANNED_TERMS = [
    "Australia",
]

# +
pda = au.DiscourseAnalysis(
    search_terms=SEARCH_TERMS,
    outputs_path=OUTPUTS_DIR,
    query_identifier=QUERY_ID,
    required_terms=REQUIRED_TERMS,
    banned_terms=BANNED_TERMS,
)

pda.load_documents(document_text=articles_df)
# -

# The warning message above says we are missing document text and metadata. Metadata is optional and can be used when using *Guardian* articles.
#
# The `load_documents` step adds document text to the class. This function has an argument `document_text` which can take a dataframe variable or if left blank will search for a file `document_text_{query_id}.csv` in `outputs/data/discourse_analysis_outputs/{query_id}/`.
#
# Note that you can use `load_documents` to input any text data, as long as it has columns for `text`, `date`, `year` and `id`.

# ## Number of news articles across years

# The number of documents per year that contain the search terms.
#
# (The results for each search term are combined and deduplicated)

pda.document_mentions

# Plot of the number of documents per year that contain the search terms.

pda.plot_mentions(use_documents=True)

# You can also plot the number of sentences, and disaggregate the number of sentences per each search term.
#
# (This might take a minute, as the text is processed into sentences using spacy)

pda.plot_mentions(use_documents=False)

# You can then get all sentences with the search terms for a specific year, using the dictionary `combined_term_sentences`

pd.set_option("max_colwidth", 500)
pda.combined_term_sentences["2022"].head(5)

# ### Characterising discourse topics using BERTopic
#
# We can use BERTopic to find topics within our documents. More info on BERTopic can be found [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To create a topic model, use function `fit_topic_model`. If you want to use sentences found from phrase matching set the variable `use_phrases` to `True` (note, if using phrases, `set_phrase_patterns` will need to be run first.) If set to `False` it will use the `sentence_mentions`.

topic_model, docs = pda.fit_topic_model()

topic_model.visualize_barchart(top_n_topics=len(set(topic_model.topics_)))

# Visualising the documents and topics in 2d space (mouseover for the text). Note that the `visualize_documents` function is not determinstic. You can make the output deterministic by following the steps [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To make the plot larger (to be able to view more of the mouse over text, increase the values for the parameters `width` and `height`)

topic_model.visualize_documents(docs, width=1400, height=750)

# ## What we talk about when we talk about X: Co-location analysis
#
# The `view_collocations` function can be used to find sentences where the search term appears with another specified term.

pda.view_collocations("source")

# #### Deeper dive in co-locations

# You can also check `term_rank` table for all co-located terms for each year (using pointwise mutual information, PMI)
#
# Frequency and rank indicates how often the terms have been used together with the search terms.
#
# Note that this might take a minute, and sometimes on Colab it surprisingly runs out of memory on this. As an alternative, you can also ran a simpler query `analyse_colocated_unigrams`, which does not use the PMI measure, to find frequently mentioned single-word terms.

pda.term_rank

# Check most often co-located terms
(
    pda.term_rank.groupby("term")
    .agg(freq=("freq", "sum"))
    .sort_values("freq", ascending=False)
    .head(20)
)

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


# +
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

# -

pda.term_rank.query("term in @check_terms")

# The `term_temporal_rank` can be used to potentially highlight interesting terms whose rank has changed significantly (ie, has a high variation across years)

pda.term_temporal_rank.sort_values("st_dev_rank", ascending=False).head(15)

# Try out also this simpler approach using only unigrams

# Simpler measure using only unigrams
pda.analyse_colocated_unigrams().sort_values("counts", ascending=False).head(20)

# ## Extracting patterns using spaCy
#
# We can also analyse specific types of phrase patterns containing the search term. For example, phrases that contain adjectives or verbs together with the search term.
#
# More information of POS phrase matching can be found [here](https://spacy.io/usage/rule-based-matching).
#
# The phrase patterns can either be loaded from a json file  theor default phrase patterns can be generated on the fly using `make_patterns=True`.

# First we need to set the phrases patterns. Here we are making patterns based on the search terms.

pda.set_phrase_patterns(load_patterns=False, make_patterns=True)

pda.set_phrase_patterns(load_patterns=False, make_patterns=True).keys()

# Then we can find matches in the documents that match the phrase patterns.
#
# This might take a minute.

pda.pos_phrases

sorted(pda.pos_phrases.pattern.unique())

# query the dataframe for rows where pattern column has the word 'verb' in it
pos = "is"
(
    pda.pos_phrases.groupby(["phrase", "pattern"], as_index=False)
    .agg(number_of_mentions=("number_of_mentions", "sum"))
    .query(f"pattern.str.contains('{pos}')", engine="python")
    .sort_values("number_of_mentions", ascending=False)
    # .head(10)
)

# Creating a topic model.

# Visualising the topics.

#

topic_model.visualize_documents(docs, width=1400, height=750)

# ## Save the analysis outputs
#
# Speeds up the analysis. Next time, if you specify the same query id, it should load the results and you won't have to compute the phrases and patterns again.

pda.save_analysis_results()

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant who is labelling companies by using predefined categories.",
    },
    {
        "role": "user",
        "content": "'You are a helpful assistant who is labelling companies by using predefined categories. This is for a project to map companies working on improving childcare, parental support and early years education solutions, focussed on children between 0-5 years old. Your are given keywords for each category, and the company description. You will output one or maximum two categories that best match the company description. You can also label the company as “Not relevant”. For example, we are not interested in solutions for middle or high schoolers; universities; healthcare; or companies not related to families or education.\n\nHere are the categories and their keywords provided in the format Category name - keywords.\nContent: General - curriculum, education content, resource\nContent: Numeracy - numeracy, mathematics, coding\nContent: Literacy - phonics, literacy, reading, ebook\nContent: Play - games, play, toys\nContent: Creative - singing, song, songs, art, arts, drawing, painting\nTraditional models: Preschool - pre school, kindergarten, montessori\nTraditional models: Child care - child care, nursery, child minder, babysitting\nTraditional models: Special needs - special needs, autism, mental health\nManagement - management, classroom technology, monitoring technology, analytics, waitlists\nTech - robotics, artificial intelligence, machine learning, simulation\nWorkforce: Recruitment - recruitment, talent acquisition, hiring\nWorkforce: Training - teacher training, skills\nWorkforce: Optimisation - retention, wellness, shift work\nFamily support: General - parents, parenting advice, nutrition, feeding, sleep, travel, transport\nFamily support: Peers - social network, peer to peer\nFamily support: Finances - finances, cash, budgeting.\n\nHere are examples of company descriptions and categories.\n\nExample 1: Description: privacy- first speech recognition software delivers voice- enabled experiences for kids of all ages, accents, and dialects. has developed child- specific speech technology that creates highly accurate, age- appropriate and safe voice- enabled experiences for children. technology is integrated across a range of application areas including toys, gaming, robotics, as well as reading and English Language Learning . Technology is fully and GDPR compliant- offering deep learning speech recognition based online and offline embedded solutions in multiple languages. Industries: audio, digital media, events\nCategory: <Tech>\n\nExample 2: Description: is a personalized learning application to improve math skills. is a personalized learning application to improve math skills. It works by identifying a child’s level, strengths and weaknesses, and gradually progressing them at the rate that’s right for them. The application is available for download on the App Store and Google Play. Industries: accounting, finance, financial services.\nCategory: <Content: Numeracy>\n\nNow categorise this company: Description: The company helps over 1.8M middle-school, high-school and college students worldwide, to understand and solve their math problems step-by-step.",
    },
    {"role": "assistant", "content": "<Not relevant>"},
    {
        "role": "user",
        "content": "Description: The company  is an EdTech startup company providing game-based math and reading courses to students in pre-kindergarten to grade five.",
    },
    {"role": "assistant", "content": "<Content: Numeracy> and <Content: Literacy>"},
    {
        "role": "user",
        "content": "Description: The company is a global digital- first entertainment company for kids. The company is a global entertainment company that creates and distributes inspiring and engaging stories to expand kids’ worlds and minds. Founded in 2018, with offices in and, The company creates, produces and publishes thousands of minutes of video and audio content every month with the goal of teaching compassion, empathy and resilience to kids around the world.",
    },
    {"role": "assistant", "content": "<Content: General>"},
]
PROMPT
