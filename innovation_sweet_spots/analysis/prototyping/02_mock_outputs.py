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

# %%
"""
Notebook for creating 'mock' outputs with examples of the type of insights we could obtain

"""

# %% [markdown]
# # Notebook for creating mock outputs

# %%
from innovation_sweet_spots.getters import gtr, crunchbase
import pandas as pd
import time
import numpy as np
from innovation_sweet_spots.getters.path_utils import GTR_PATH, HANSARD_PATH
from datetime import datetime

# %%
import altair as alt
import seaborn as sns
from matplotlib import pyplot as plt
import umap

# %%
import importlib

importlib.reload(gtr)


# %%
def timeit(method):
    """
    Ref: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def create_documents(lists_of_texts):
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']
    Args:
        lists_of_skill_texts (iterable of list of str): Contains lists of text
            features (e.g. label or description) to be joined up and processed to
            create the "documents"; i-th element of each list corresponds
            to the i-th entity/document
    Yields:
        (str): Created documents
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            " ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def create_documents_from_dataframe(df, columns):
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy()
    # Preprocess project text
    text_lists = [df_[col].str.lower().to_list() for col in columns]
    # Create project documents
    docs = list(create_documents(text_lists))
    return docs


# %%
def is_term_present(search_term, docs):
    """ """
    return [search_term in doc for doc in docs]


def search_in_projects(search_term):
    bool_mask = is_term_present(search_term, gtr_docs)
    return gtr_projects[bool_mask].copy()


def get_project_funding(project_df):
    fund_cols = ["project_id", "id", "rel", "category", "amount", "currencyCode"]
    df = (
        project_df.merge(gtr_funded_projects[fund_cols], on="project_id", how="left")
        .rename(columns={"rel": "rel_funds"})
        .drop("id", axis=1)
    )
    return df


def convert_date_to_year(str_date):
    """String date to integer year"""
    if type(str_date) is str:
        return int(str_date[0:4])
    else:
        return str_date


def get_search_term_funding(search_term):
    projects = search_in_projects(search_term)
    projects_with_funding = get_project_funding(projects)
    projects_with_funding["year"] = projects_with_funding.start.apply(
        lambda date: convert_date_to_year(date)
    )
    return projects_with_funding


def get_breakdown_by_year(search_term):
    projects_with_funding = get_search_term_funding(search_term)
    df = (
        projects_with_funding.groupby("year")
        .agg({"project_id": "count", "amount": "sum"})
        .rename(columns={"project_id": "no_of_projects"})
        .reset_index()
    )
    df = df[df.year >= 2006]
    # To do - add missing years as 0s
    return df


def show_projects_by_year(search_term):
    df = get_breakdown_by_year(search_term)
    alt.Chart(df).mark_line(point=True).encode(x="year:T", y="no_of_projects:Q")


def show_funding_amount_by_year(search_term):
    df = get_breakdown_by_year(search_term)
    alt.Chart(df).mark_line(point=True).encode(x="year:T", y="amount:Q")


def get_project_orgs(project_df):
    projects_orgs = (
        project_df.merge(
            link_gtr_organisations[["project_id", "id", "rel"]], how="left"
        )
        .merge(gtr_organisations[["id", "name"]], how="left")
        .drop("id", axis=1)
        .rename(columns={"rel": "rel_organisations"})
    )
    return projects_orgs


def get_org_stats(project_df):
    projects_orgs = get_project_orgs(project_df)
    project_orgs_funds = get_project_funding(projects_orgs)
    org_stats = (
        project_orgs_funds.groupby("name")
        .agg({"project_id": "count", "amount": "sum"})
        .rename(columns={"project_id": "counts"})
        .sort_values("counts", ascending=False)
    )
    return org_stats


# %%
importlib.reload(au)

# %%
gtr_project_funds = au.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = au.get_gtr_project_funds(gtr_projects, gtr_project_funds)

# %%
search_term = "heat pump"
proj = au.search_via_docs(search_term, gtr_docs, funded_projects)
search_term_funding = au.gtr_funding_per_year(proj)

# %%
search_term_funding

# %%
au.show_time_series(search_term_funding, y="amount_median")

# %%
au.estimate_funding_level(search_term_funding)

# %%
au.estimate_growth_level(search_term_funding, column="amount_median")

# %%
importlib.reload(au)

# %%
project_to_org = au.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
project_orgs = au.get_gtr_project_orgs(proj, project_to_org)

# %%
au.get_org_stats(project_orgs).head(15)


# %%
# au.search_via_docs(search_term, hans_docs, hans)

# %%
### SPEECHES
def search_in_speeches(search_term):
    bool_mask = is_term_present(search_term, hans_docs)
    return hans[bool_mask].copy()


def get_speech_breakdown_by_year(search_term):
    speeches = search_in_speeches(search_term)
    df = (
        speeches.groupby("year")
        .agg({"id": "count"})
        .rename(columns={"id": "counts"})
        .reset_index()
    )
    # To do - add missing years as 0s
    return df


def get_speech_breakdown_by_party(search_term):
    speeches = search_in_speeches(search_term)
    df = (
        speeches.groupby("party")
        .agg({"id": "count"})
        .rename(columns={"id": "counts"})
        .reset_index()
    )
    return df


def get_speech_breakdown_by_person(search_term):
    speeches = search_in_speeches(search_term)
    df = (
        speeches.groupby("speakername")
        .agg({"id": "count"})
        .rename(columns={"id": "counts"})
        .reset_index()
    )
    return df


# %% [markdown]
# ## 1. Setup the analysis

# %% [markdown]
# ### 1.1 Load in the data

# %%
# %%capture
# Crunchbase
cb_df = crunchbase.get_crunchbase_orgs()
# GTR
gtr_projects = gtr.get_gtr_projects()
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# %%
gtr_topics = gtr.get_gtr_topics()

# %%
gtr_projects.info()

# %%
# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")

# %%
link_gtr_topics = gtr.get_link_table("gtr_topic")

# %% [markdown]
# ### 1.2 Hansard

# %%
filename = HANSARD_PATH / "hansard-speeches-v310.csv"

# %%
df = pd.read_csv(filename, nrows=1000, skiprows=range(1, 1750000))

# %%
# 2008 starts around 1.75M rows

# %%
hans = pd.read_csv(filename, skiprows=range(1, 2000000))

# %%
hans.info()

# %%
hans = hans.query("speech_class=='Speech'")

# %%
len(hans)

# %%
hans.head(1)

# %%
hans.head(1)

# %%

# %% [markdown]
# ### 1.3 Create documents

# %%
cb_columns = ["name", "short_description", "long_description"]
cb_docs = create_documents_from_dataframe(cb_df, cb_columns)

# %%
len(cb_docs)

# %%
gtr_columns = ["title", "abstractText", "techAbstractText"]
gtr_docs = create_documents_from_dataframe(gtr_projects, gtr_columns)

# %%
hans_columns = ["speech"]
hans_docs = create_documents_from_dataframe(hans, hans_columns)

# %% [markdown]
# ### Link funds to projects

# %%
gtr_funded_projects = gtr_funds.merge(link_gtr_funds)

# %%
gtr_funded_projects.groupby("category").agg({"id": "count"})

# %%
gtr_funded_projects.info()

# %%
gtr_funded_projects = (
    gtr_funded_projects[gtr_funded_projects.category == "INCOME_ACTUAL"]
    .sort_values("amount", ascending=False)
    .drop_duplicates("project_id", keep="first")
)


# %%
gtr_funded_projects.head(2)

# %%
link_gtr_organisations.info()

# %%
# gtr_organisations.sample(25)

# %% [markdown]
# ## 2. Characterisation

# %%
gtr_df.info()

# %%
gtr_funded_projects.info()

# %% [markdown]
# ## 3. Analysis

# %% [markdown]
# ## Research projects

# %%
search_term = "heat pump"

df = get_breakdown_by_year(search_term)

alt.Chart(df).mark_line(point=True).encode(x="year:O", y="no_of_projects:Q")

# %%
alt.Chart(df).mark_line(point=True).encode(x="year:O", y="amount:Q")

# %% [markdown]
# What next?
# - Estimate trends
#     - Best way to do it?
#     - Don't take 2021 into account
#
# - Perhaps need to tease out if the project is tech-specific or there are other techs also mentioned (for that need an existing list of tech)

# %%
check_columns = [
    "title",
    "grantCategory",
    "leadFunder",
    "potentialImpact",
    "year",
    "amount",
]
(
    get_search_term_funding(search_term)[check_columns].sort_values(
        ["year", "amount"], ascending=False
    )
).head(20)


# %%
# proj = search_in_projects(search_term)
org_stats = get_org_stats(proj)

# %%
#
org_stats.sort_values("amount", ascending=False)

# %%
project_orgs_funds[project_orgs_funds["name"] == "Doosan Babcock Limited"]

# %% [markdown]
# - What next - we can see build a network and visualise which organisations are "central" (i.e. collaborating with many other orgs and/or hubs in the network)
# - NB: There might be local hubs (not many connections, but connecting other e.g. small uni)
# - Should think about a standardised insights from this
#    - Central orgs
#    - Orgs with many projects
#    - Orgs involved in high value projects
#    - Is it possible to get how much each org got?
#

# %% [markdown]
# ## 3. Research project "landscape"

# %% [markdown]
# ### 3.1 Identifying green projects

# %%
gtr_project_topics = (
    gtr_projects.merge(link_gtr_topics, how="left")
    .merge(gtr_topics, how="left")
    .drop(["id", "table_name", "rel"], axis=1)
)

# %%
gtr_project_topics.info()

# %%
gtr_project_topics.topic_type.unique()

# %%
gtr_topic_counts = (
    gtr_project_topics.groupby("text")
    .agg({"project_id": "count"})
    .reset_index()
    .rename(columns={"project_id": "counts", "text": "topics"})
    .sort_values("counts", ascending=False)
)
gtr_topic_counts

# %%
tags_per_project = (
    gtr_project_topics.groupby(["project_id"])
    .agg({"title": "count"})
    .reset_index()
    .rename(columns={"title": "counts"})
)

# %%
tags_per_project.counts.median()

# %%
sns.displot(tags_per_project.counts)

# %%
gtr_project_topics[
    gtr_project_topics.project_id.isin(
        tags_per_project.query("counts==1").project_id.to_list()
    )
].groupby("text").agg({"project_id": "count"})

# %%
gtr_project_topics

# %%
from gensim.models import Word2Vec


def token_2_vec(lists_of_tokens):
    """Surface form to vector model (sf2vec)"""

    # Get skills per job, and determine the max window size
    n_tokens_per_list = [len(x) for x in lists_of_tokens]
    max_window = max(n_tokens_per_list)

    # Build the model
    model = Word2Vec(
        sentences=lists_of_tokens,
        size=200,
        window=max_window,
        min_count=1,
        workers=4,
        sg=1,
        seed=123,
        iter=30,
    )

    # filepath = f"models/sf2vec_{clustering_params['session_name']}.model"
    # model.save(f'{DATA_PATH.parent / filepath}')
    return model


def get_token_vectors(model, unique_tokens):
    """extract sf2vec vectors from model"""
    token2vec_emb = [model.wv[token] for token in unique_tokens]
    token2vec_emb = np.array(token2vec_emb)
    return token2vec_emb


# def get_sf2vec_embeddings(detected_surface_forms, sf_to_cluster):
#     cluster_forms = sf_to_cluster.surface_form.to_list()
#     sf2vec_model = token_2_vec(detected_surface_forms)
#     sf2vec_embeddings = get_sf2vec_vectors(sf2vec_model, cluster_forms)
#     return sf2vec_embeddings


# %%
def get_lists_of_tokens():
    pass


# %%
project_topics = (
    gtr_project_topics.query("text!='Unclassified'")
    .groupby(["project_id", "text"])
    .count()
)
projs = np.unique(project_topics.index.get_level_values("project_id").to_list())
topic_lists = [project_topics.loc[proj].index.to_list() for proj in projs]


# %%
len(gtr_topics.text.unique())

# %%
n_topics = [len(topics) for topics in topic_lists]

# %%
topic_lists_ = [topics for topics in topic_lists if len(topics) > 1]

# %%
unique_topics = sorted(
    np.unique([topic for topics in topic_lists_ for topic in topics])
)

# %%
len(unique_topics)

# %%
topics_word2vec_model = token_2_vec(topic_lists_)
topics_embeddings = get_token_vectors(topics_word2vec_model, unique_topics)

# %%
topics_embeddings.shape

# %%
# Find topics associated with keyword
proj = (
    search_in_projects("heating")
    .merge(gtr_project_topics, how="left")
    .groupby("text")
    .agg({"project_id": "count"})
    .sort_values("project_id", ascending=False)
    .reset_index()
    .rename(columns={"text": "topics", "project_id": "counts"})
)

# %%
reducer = umap.UMAP(random_state=111, n_neighbors=20, min_dist=0.01, n_components=2)
embedding = reducer.fit_transform(topics_embeddings)
embedding.shape

# %%
# Colours for visualisations
colour_pal = [
    "#000075",
    "#e6194b",
    "#3cb44b",
    "#f58231",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#DCDCDC",
    "#9a6324",
    "#800000",
    "#808000",
    "#ffe119",
    "#46f0f0",
    "#4363d8",
    "#911eb4",
    "#aaffc3",
    "#000000",
    "#ffd8b1",
    "#808000",
    "#000075",
    "#DCDCDC",
]


df_viz = pd.DataFrame(
    data={"topics": unique_topics, "vect_id": range(0, len(unique_topics))}
)
df_viz["x"] = embedding[:, 0]
df_viz["y"] = embedding[:, 1]
df_viz = df_viz.merge(proj, how="left").fillna(0)
# df_viz['cluster'] = partition_1.cluster.to_list()
# df_viz = df_viz[df_viz.cluster==0]

# %%
# alt.data_transformers.disable_max_rows()
def alt_scatter(df_viz):
    return (
        alt.Chart(df_viz)
        .mark_circle(size=50)
        .encode(
            x=alt.X("x", axis=alt.Axis(grid=False)),
            y=alt.Y("y", axis=alt.Axis(grid=False)),
            #     color=alt.Color('cluster', scale=alt.Scale(scheme='category20')),
            #     color=alt.Color('cluster', scale=alt.Scale(domain=list(range(len(colour_pal))), range=colour_pal)),
            color=alt.Color("counts"),
            tooltip=["topics", "counts", "vect_id"],
        )
        .interactive()
    )


from scipy.spatial.distance import cdist


# def check_most_similar_skills(vect_id, vects, n=5):
#   return skills.loc[check_most_similar(vect_id, vects)].preferred_label.iloc[0:n].to_list()

# %%
alt_scatter(df_viz)

# %%
dfdf = df_viz.sort_values("counts", ascending=False)
dfdf.head(20)


# %%
def check_most_similar(vect_id, vects):
    sims = cdist(vects[vect_id, :].reshape(1, -1), vects, "cosine")
    return list(np.argsort(sims[0]))


def find_most_similar_topics(topic):
    i = df_viz[df_viz.topics == topic].iloc[0].vect_id
    return df_viz.loc[check_most_similar(i, topics_embeddings)]


# %%
find_most_similar_topics(dfdf.iloc[3].topics).head(10)

# %% [markdown]
# ### Quick clustering

# %%
from sklearn.cluster import KMeans

# %%
kmeans = KMeans(n_clusters=20, random_state=0).fit(topics_embeddings)

# %%
clust_labels = kmeans.predict(topics_embeddings)
df_viz["cluster"] = [str(c) for c in clust_labels]

# %%
alt.Chart(df_viz).mark_circle(size=50).encode(
    x=alt.X("x", axis=alt.Axis(grid=False)),
    y=alt.Y("y", axis=alt.Axis(grid=False)),
    #     color=alt.Color('cluster', scale=alt.Scale(scheme='category20')),
    color=alt.Color(
        "cluster:Q",
        scale=alt.Scale(domain=list(range(len(colour_pal))), range=colour_pal),
    ),
    #         color=alt.Color('counts'),
    tooltip=["topics", "counts", "vect_id", "cluster"],
).interactive()

# %%
green_clust = [2, 17, 10]

# %%
green_tags = df_viz[df_viz.cluster.isin([str(g) for g in green_clust])].copy()

# %%
green_tags


# %%
def get_neighbors():
    neighbors = set()
    for i, row in green_tags.iterrows():
        neighbors = neighbors.union(
            set(find_most_similar_topics(row.topics).head(5).topics.to_list())
        )
    return neighbors


# %%
neighbors = get_neighbors()

# %%
all_tags = set(green_tags.topics.to_list()).union(neighbors)

# %%
len(all_tags)

# %%
from innovation_sweet_spots import PROJECT_DIR

# %%
dd = (
    df_viz[df_viz.topics.isin(all_tags)]
    .drop("counts", axis=1)
    .merge(gtr_topic_counts)
    .sort_values(["cluster", "counts"])
)
dd.to_csv(PROJECT_DIR / "outputs/data/aux/green_topics_check.csv", index=False)

# %%
len(neighbors)

# %% [markdown]
# ### Quick model

# %%

# %% [markdown]
# ### 3.2 Thematic composition of projects

# %%
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss

# %%
import innovation_sweet_spots.analysis.analysis_utils as au
import importlib

importlib.reload(au)

# %%
gtr_docs_dict = dict(zip(gtr_projects.project_id.to_list(), gtr_docs))

# %%
proj_heating = au.search_in_items("heating", gtr_docs, gtr_projects)

# %%
proj_climate_change = gtr_project_topics[
    gtr_project_topics.text == "Climate & Climate Change"
]

# %%
proj_docs = [gtr_docs_dict[proj] for proj in proj_heating.project_id.to_list()]
len(proj_docs)

# %%
# Transform data into a sparse matrix
vectorizer = CountVectorizer(
    stop_words="english", max_features=20000, binary=True, ngram_range=(1, 2)
)
doc_word = vectorizer.fit_transform(proj_docs)
doc_word = ss.csr_matrix(doc_word)
doc_word.shape

# %%
words = list(np.asarray(vectorizer.get_feature_names()))

# %%
not_digit_inds = [ind for ind, word in enumerate(words) if not word.isdigit()]
doc_word = doc_word[:, not_digit_inds]
words = [word for ind, word in enumerate(words) if not word.isdigit()]

doc_word.shape

# %%
# Train the CorEx topic model with 50 topics
topic_model = ct.Corex(n_hidden=50, words=words, max_iter=200, verbose=False, seed=1)
topic_model.fit(
    doc_word,
    words=words,
    anchors=[
        ["heat pump", "heat pumps"],
        ["boiler"],
        ["district heating"],
        ["solar panel"],
    ],
)

# %%
# Print a single topic from CorEx topic model
topic_model.get_topics(topic=1, n_words=20)

# %%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n, topic in enumerate(topics):
    topic_words, _, _ = zip(*topic)
    print("{}: ".format(n) + ", ".join(topic_words))

# %%
topic_model.get_top_docs(topic=0, n_docs=10, sort_by="log_prob")

# %%
proj_docs[727]

# %% [markdown]
# ## 4. Speeches

# %%
df = search_in_speeches(search_term)

# %%
len(df)

# %%
df.tail(15)

# %%
len(df.url.unique())

# %%
len(hans.major_heading.unique())

# %%
len(hans.minor_heading.unique())

# %%
get_speech_breakdown_by_party(search_term)

# %%
get_speech_breakdown_by_person(search_term)

# %%
sents = []
for j, row in speeches.iterrows():
    for sent in row.speech.split("."):
        if search_term in sent:
            sents.append(sent)

# %%
for s in sents:
    print(f"{s}\n")

# %%

# %% [markdown]
#

# %% [markdown]
# ## 5. Crunchbase

# %%
# Check if the text contains the search term
contains_term = is_term_present(search_term, cb_docs)

# %%
len(cb_df[contains_term])

# %%
cb_df_with_term = cb_df[contains_term].copy()
cb_df_with_term = cb_df_with_term[-cb_df_with_term.founded_on.isnull()]
cb_df_with_term["year"] = cb_df_with_term.founded_on.apply(
    lambda x: extract_year_from_cb(x)
)

# %%
cb_df_with_term.name


# %%
def extract_year_from_cb(str_date):
    if type(str_date) is str:
        return int(str_date[0:4])
    else:
        return str_date


# %%
cb_dff = (
    cb_df_with_term.groupby("year")
    .agg({"id": "count", "total_funding": "sum"})
    .reset_index()
)

# %%
cb_dff.year = pd.datetime(cb_dff.year)

# %%
cb_dff = cb_dff[cb_dff.year > 2008]

# %%
alt.Chart(cb_dff).mark_line().encode(x="year:Q", y="id:Q")

# %%
alt.Chart(cb_dff).mark_line().encode(x="year:T", y="total_funding:Q")

# %%
