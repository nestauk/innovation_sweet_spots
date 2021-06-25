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
# # Green project analysis

# %%
from innovation_sweet_spots.getters import gtr, misc
import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
import importlib

importlib.reload(iss)

# %%
# %%capture
import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import cdist
import altair as alt


# %%
def check_most_similar(vect_id, vects):
    sims = cdist(vects[vect_id, :].reshape(1, -1), vects, "cosine")
    return list(np.argsort(sims[0]))


def find_most_similar_topics(topic):
    i = df_viz[df_viz.topics == topic].iloc[0].vect_id
    return df_viz.loc[check_most_similar(i, topics_embeddings)]


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

# %% [markdown]
# ## Overview of GTR project topics

# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()
gtr_topics = gtr.get_gtr_topics()
link_gtr_topics = gtr.get_link_table("gtr_topic")


# %%
gtr_columns = ["title", "abstractText", "techAbstractText"]
gtr_docs = iss.create_documents_from_dataframe(gtr_projects, gtr_columns)

# %%
gtr_project_topics = iss.link_gtr_projects_and_topics(
    gtr_projects, gtr_topics, link_gtr_topics
)

# %%
project_topics = (
    gtr_project_topics.query("text!='Unclassified'")
    .groupby(["project_id", "text"])
    .count()
)
projs = np.unique(project_topics.index.get_level_values("project_id").to_list())
topic_lists = [project_topics.loc[proj].index.to_list() for proj in projs]
topic_lists_ = [topics for topics in topic_lists if len(topics) > 1]

unique_topics = sorted(
    np.unique([topic for topics in topic_lists_ for topic in topics])
)
print(len(unique_topics))

# %%
# topics_word2vec_model = iss.token_2_vec(topic_lists_);
# topics_embeddings = iss.get_token_vectors(topics_word2vec_model, unique_topics);

# %%
reducer = umap.UMAP(random_state=111, n_neighbors=20, min_dist=0.01, n_components=2)
embedding = reducer.fit_transform(topics_embeddings)
embedding.shape

# %%


df_viz = pd.DataFrame(
    data={"topics": unique_topics, "vect_id": range(0, len(unique_topics))}
)
df_viz["x"] = embedding[:, 0]
df_viz["y"] = embedding[:, 1]
# df_viz = df_viz.merge(proj, how="left").fillna(0)
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
            #             color=alt.Color("counts"),
            tooltip=["topics"],  # , "counts", "vect_id"],
        )
        .interactive()
    )


# def check_most_similar_skills(vect_id, vects, n=5):
#   return skills.loc[check_most_similar(vect_id, vects)].preferred_label.iloc[0:n].to_list()


# %%
alt_scatter(df_viz)

# %%
# find_most_similar_topics('Sustainable Energy Vectors')
# find_most_similar_topics('Sustainable Energy Vectors').head(50).topics.to_list()

# %% [markdown]
# ## Preliminary definition of green projects

# %%
# Find topics associated with keyword
proj = (
    iss.search_via_docs("heat pump", gtr_docs, gtr_projects)
    .merge(gtr_project_topics, how="left")
    .groupby("text")
    .agg({"project_id": "count"})
    .sort_values("project_id", ascending=False)
    .reset_index()
    .rename(columns={"text": "topics", "project_id": "counts"})
)

# %%
proj

# %%
green_topics = [
    "Energy Efficiency",
    "Energy Storage",
    "Sustainable Energy Networks" "Solar Technology",
    "Waste Minimisation",
    "Waste Management",
    "Sustainable Energy Vectors",
]

# %%
import innovation_sweet_spots.analysis.text_analysis as iss_text

importlib.reload(iss_text)
nlp = iss_text.setup_spacy_model()

# %%
tech_nav = misc.get_tech_navigator()

# %%
techs = tech_nav["Technology Name"].to_list()

# %%
tech_chunks = list(iss_text.chunk_forms(techs, nlp))

# %%
tech_chunks_ = [t for ts in tech_chunks for t in ts]

# %%
tech_chunks_rev = [
    "adaptive energy system",
    "smart heating thermostat",
    "energy recovery",
    "positive input ventilation system",
    "multifoil insulation",
    "solid wall insulation",
    "cavity wall insulation",
    "loft insulation",
    "solar thermal heat storage",
    "storage tank",
    "passive solar gain",
    "home heating",
    "concentrated solar power",
    "solar power" "solar photovoltaics",
    "ground source heat pump",
    "gshp",
    "air source heat pump",
    "ashp",
    "heat pump",
    "district heating",
    "geothermal energy system",
    "biomass wood fuel heating",
    "boiler",
    "gas boiler",
    "oil boiler",
    "combined heat",
    "batch water heater",
    "thermodynamic solar panel",
    "photovoltaic thermal collector",
    "pvt",
    "micro wind turbine",
    "micro hydropower system",
    "ridgeblade",
    "tubular wind turbine",
    "wind turbine",
    "water source heat pump",
    "wshp",
    "induction heater",
]

# %%
# from tqdm.notebook import tqdm
# clean_gtr_docs =[]
# for i, doc in tqdm(enumerate(gtr_docs), total=len(gtr_docs)):
#     clean_gtr_docs.append(iss_text.clean_text(doc))

# %%
# gtr_docs

# %%
ids = []
for tech_chunk in tech_chunks_rev:
    ids.append(
        iss.search_via_docs(tech_chunk, gtr_docs, gtr_projects).project_id.to_list()
    )

# %%
ids_ = [i for i_list in ids for i in i_list]

# %%
green_topic_ids = gtr_project_topics[
    gtr_project_topics.text.isin(green_topics)
].project_id.to_list()

# %%
# green_topic_ids

# %%
green_projects = gtr_projects[gtr_projects.project_id.isin(ids_ + green_topic_ids)]

# %%
len(green_projects)

# %%
# ≈ = (tech_chunk_projects
#     .merge(gtr_project_topics, how="left")
#     .groupby("text")
#     .agg({"project_id": "count"})
#     .sort_values("project_id", ascending=False)
#     .reset_index()
#     .rename(columns={"text": "topics", "project_id": "counts"})
# )

# %%
# tech_chunk_project_topics.head(20)

# %%
# import gensim, spacy, logging, warnings
# import gensim.corpora as corpora
# from gensim.utils import lemmatize, simple_preprocess
# from gensim.models import CoherenceModel
# import re

# # NLTK Stop words
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


# %%
from tqdm.notebook import tqdm

green_project_docs = iss.create_documents_from_dataframe(green_projects, gtr_columns)
# clean_green_docs = [iss_text.clean_text(s) for s in green_project_docs]
clean_green_docs = []
for s in tqdm(green_project_docs, total=len(green_projects)):
    clean_green_docs.append(iss_text.clean_text(s))


# %% [markdown]
# ## Overview of all green projects

# %%
def calculate_embeddings(list_of_sentences, save_name=None, download=False):

    # Calculate the sentence embeddings
    t = time()
    print(f"Calculating {len(list_of_sentences)} embeddings...", end=" ")
    sentence_embeddings = np.array(bert_transformer.encode(list_of_sentences))
    print(f"Done in {time()-t:.2f} seconds")

    # Save the embeddings
    if save_name is not None:
        save_and_download(sentence_embeddings, save_name, download)

    return sentence_embeddings


# %%
bert_model = "paraphrase-distilroberta-base-v1"

# %%
from time import time

# %%
from sentence_transformers import SentenceTransformer

bert_transformer = SentenceTransformer(bert_model)

# %%
green_project_embeddings = calculate_embeddings(green_project_docs)

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=0).fit(green_project_embeddings)
clust_labels = kmeans.predict(green_project_embeddings)

# %%
# clust_labels

# %%
reducer = umap.UMAP(random_state=111, n_neighbors=20, min_dist=0.01, n_components=2)
embedding = reducer.fit_transform(green_project_embeddings)
embedding.shape

# %%
# gtr_project_funding

# %%
project_topics = (
    green_projects.merge(
        gtr_project_topics[["project_id", "text", "topic_type"]], how="left"
    )
    .groupby(["project_id", "text"])
    .count()
)
projs = np.unique(project_topics.index.get_level_values("project_id").to_list())
topic_lists = [project_topics.loc[proj].index.to_list() for proj in projs]
topic_lists = [str(t) for t in topic_lists]

# %%
df_viz = green_projects.copy()
df_viz["x"] = embedding[:, 0]
df_viz["y"] = embedding[:, 1]
df_viz["cluster"] = [str(c) for c in clust_labels]
df_viz["research_areas"] = topic_lists

# %%
df_viz

# %%
fig = (
    alt.Chart(df_viz, width=1000, height=1000)
    .mark_circle(size=50)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        color=alt.Color(
            "cluster",
            scale=alt.Scale(domain=list(range(len(colour_pal))), range=colour_pal),
        ),
        tooltip=["title", "research_areas", "cluster"],  # , "counts", "vect_id"],
    )
).interactive()
fig

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

# %%
importlib.reload(alt_save)

# %%
driver = alt_save.google_chrome_driver_setup()

# %%
alt_save.save_altair(fig, "GTR_test_Green_projects", driver)

# %% [markdown]
# ## Stats for different clusters

# %%
gtr_funds = gtr.get_gtr_funds()
# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")

gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(gtr_projects, gtr_project_funds)

# %%
green_project_funds = df_viz.merge(funded_projects[["project_id", "amount"]])

# %%
CLUSTER = 5
cluster_projects = green_project_funds[green_project_funds.cluster == str(CLUSTER)]
cluster_funding = iss.gtr_funding_per_year(cluster_projects.copy())
iss.show_time_series(cluster_funding, y="no_of_projects")

# %%
iss.nicer_axis(iss.show_time_series_fancier(cluster_funding, y="no_of_projects"))

# %%
# iss.show_time_series(cluster_funding, y="amount_total")

# %%
iss.show_time_series(cluster_funding, y="amount_median")

# %% [markdown]
# ## Topic modelling using corex

# %%
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss
from tqdm.notebook import tqdm

# %%
green_project_docs = iss.create_documents_from_dataframe(green_projects, gtr_columns)
# clean_green_docs = [iss_text.clean_text(s) for s in green_project_docs]
clean_green_docs = []
for s in tqdm(green_project_docs, total=len(green_projects)):
    clean_green_docs.append(iss_text.clean_text(s))

# %%
# Transform data into a sparse matrix
vectorizer = CountVectorizer(
    stop_words="english", max_features=20000, binary=True, ngram_range=(1, 2)
)
doc_word = vectorizer.fit_transform(clean_green_docs)
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
topic_model = ct.Corex(n_hidden=25, words=words, max_iter=200, verbose=False, seed=1)
topic_model.fit(doc_word, words=words, anchors=[["cutting edge", ""]])

# %%
# Print a single topic from CorEx topic model
topic_model.get_topics(topic=1, n_words=20)

# %%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n, topic in enumerate(topics):
    topic_words, _ = zip(*topic)
    print("{}: ".format(n) + ", ".join(topic_words))

# %%
topics[22]

# %%
topic_model.get_top_docs(topic=22, n_docs=10, sort_by="log_prob")

# %%
j = 200

# %%
green_projects.iloc[j].title

# %%
green_projects.iloc[j].abstractText

# %%
print(topic_model.p_y_given_x.shape)

# %%
np.sort(np.round(topic_model.p_y_given_x[j], 2))

# %%
np.flip(np.argsort(np.round(topic_model.p_y_given_x[j], 2)))

# %%
topics[12]

# %%
