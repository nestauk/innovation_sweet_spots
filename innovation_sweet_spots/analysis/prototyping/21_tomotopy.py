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
# # Whole dataset analysis with tomotopy

# %%
from innovation_sweet_spots import PROJECT_DIR, logging, config
import pandas as pd
import numpy as np
import innovation_sweet_spots.utils.io as iss_io
import pyLDAvis
import tomotopy as tp

print(tp.isa)

# %%
import warnings

warnings.filterwarnings("ignore")

# %%
# Number of topics
N_TOPICS = 150
# Default topic keyword prior probabilities
DEF_TOPIC_PROB = 1
DEF_OTHER_PROB = 0
SET_PRIORS = False


# %%
def print_model_info(mdl):
    print(
        "Num docs:{}, Num Vocabs:{}, Total Words:{}".format(
            len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
        )
    )
    print("Removed Top words: ", *mdl.removed_top_words)


def train_model(mdl, iterations=1000, step=20):
    """Let's train the model"""
    for i in range(0, iterations, step):
        logging.info("Iteration: {:04}, LL per word: {:.4}".format(i, mdl.ll_per_word))
        mdl.train(step)
    logging.info(
        "Iteration: {:04}, LL per word: {:.4}".format(iterations, mdl.ll_per_word)
    )
    mdl.summary()
    return mdl


def get_topic_term_dists(mdl):
    return np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])


def get_doc_topic_dists(mdl):
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    return doc_topic_dists


def make_pyLDAvis(mdl, fpath=PROJECT_DIR / "outputs/data/ldavis_tomotopy.html"):
    topic_term_dists = get_topic_term_dists(mdl)
    doc_topic_dists = get_doc_topic_dists(mdl)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
        sort_topics=False,  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    )
    pyLDAvis.save_html(prepared_data, str(fpath))


# def topic_prior_probs(topic, topic_prob=1, other_prob=0.1):
#     return [topic_prob if k == topic else other_prob for k in range(N_TOPICS)]


def create_prior_prob_matrix(topic_seeds):
    unique_seeds = sorted(np.unique([t for seeds in topic_seeds for t in seeds]))
    seed_to_index = dict(zip(unique_seeds, range(len(unique_seeds))))
    prior_prob_matrix = np.ones((len(unique_seeds), N_TOPICS)) * DEF_OTHER_PROB
    for t, topic in enumerate(topic_seeds):
        for seed in topic:
            prior_prob_matrix[seed_to_index[seed], t] = DEF_TOPIC_PROB
    return prior_prob_matrix, seed_to_index


# %%
# Import datasets
corpus_gtr = iss_io.load_pickle(
    PROJECT_DIR / "outputs/data/gtr/gtr_docs_tokenised_full.p"
)
corpus_cb = iss_io.load_pickle(PROJECT_DIR / "outputs/data/cb/cb_docs_tokenised_full.p")

# %%
# Import topic seeds
topic_seeds_df = pd.read_excel(PROJECT_DIR / "outputs/data/aux/topic_seeds.xlsx")
topic_seeds = [s.strip().split() for s in topic_seeds_df.topic_words.to_list()]

# %%
# Prepare inputs
full_corpus = list(corpus_gtr.values()) + list(corpus_cb.values())
full_corpus_ids = list(corpus_gtr.keys()) + list(corpus_cb.keys())
full_corpus_sources = ["gtr"] * len(corpus_gtr) + ["cb"] * len(corpus_cb)

# Remove documents without any text
lens = np.array([len(doc) for doc in full_corpus])
empty_docs = np.where(lens == 0)[0]
full_corpus = [doc for i, doc in enumerate(full_corpus) if i not in empty_docs]
full_corpus_ids = [doc for i, doc in enumerate(full_corpus_ids) if i not in empty_docs]
logging.info(f"Found {len(empty_docs)} without content")

# Create a corpus instance
corpus = tp.utils.Corpus()
for doc in full_corpus:
    corpus.add_doc(doc)

# %%
original_indexes = list(range(len(corpus_gtr) + len(corpus_cb)))
original_indexes = [i for i in original_indexes if i not in empty_docs]

# %%

# %%
len(original_indexes)

# %%
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.topic_analysis as iss_topics

doc_df = pd.DataFrame(data={"doc_id": full_corpus_ids})
doc_df = iss_topics.get_doc_details(doc_df)

# %%
doc_df["top2vec_index"] = original_indexes

# %%
doc_df[["doc_id", "title", "source", "top2vec_index"]].to_csv(
    PROJECT_DIR / "outputs/models/topic_models/topic_model_docs.csv", index=False
)

# %% [markdown]
# # Global topic model

# %%
# Initialise model
mdl = tp.LDAModel(min_df=5, rm_top=40, k=N_TOPICS, corpus=corpus, seed=1111)

if SET_PRIORS:
    logging.info("Setting prior probabilities")
    prior_prob_matrix, seed_to_index = create_prior_prob_matrix(topic_seeds)
    for t, topics in enumerate(topic_seeds):
        for seed in topics:
            mdl.set_word_prior(seed, list(prior_prob_matrix[seed_to_index[seed], :]))

mdl.train(0)
print_model_info(mdl)

# %%
train_model(mdl, iterations=1000, step=20)

# %%
make_pyLDAvis(mdl)

# %%
mdl.save(str(PROJECT_DIR / "outputs/models/full_corpus_lda_model.bin"))

# %%
mdl = tp.LDAModel.load(str(PROJECT_DIR / "outputs/models/full_corpus_lda_model.bin"))

# %% [markdown]
# ## Inspect tomotopy results

# %%

# %%
import importlib

importlib.reload(iss_topics)

# %%
doc_df = pd.DataFrame(data={"doc_id": full_corpus_ids})
doc_df = iss_topics.get_doc_details(doc_df)

# %%
# Document topic probabilities
doc_topic_dists = get_doc_topic_dists(mdl)
doc_topic_dists.shape

# %%
# Topic terms
vocab = list(mdl.used_vocabs)
topic_term_dists = get_topic_term_dists(mdl)


# %% [markdown]
# ### Extract topic keywords according to relevance

# %%
def get_topic_terms(mdl, n=10, _lambda=0.5, n_docs=len(full_corpus)):
    p = mdl.used_vocab_df / n_docs
    topic_term_dists = get_topic_term_dists(mdl)
    topic_terms = []
    for t in range(topic_term_dists.shape[0]):
        #         top_keyterms = np.flip(np.argsort(topic_term_dists[t,:]))[0:n]
        #         topic_terms.append([vocab[i] for i in top_keyterms])
        rel = _lambda * np.log(topic_term_dists[t, :]) + (1 - _lambda) * np.log(
            topic_term_dists[t, :] / p
        )
        terms = [mdl.used_vocabs[i] for i in np.flip(np.argsort(rel))[0:n]]
        topic_terms.append(terms)
    return topic_terms


# %%
topic_keyterms = get_topic_terms(mdl, n=50)

# %%
topic_keyterms_str = [" ".join(s) for s in topic_keyterms]

# %%
tomotopy_topics = pd.DataFrame(
    data={"topic": range(len(topic_keyterms_str)), "keywords": topic_keyterms_str}
)
# tomotopy_topics.to_csv(PROJECT_DIR / 'outputs/data/results_august/tomotopy_topics.csv', index=False)

# %%
# np.argsort(topic_term_dists[:, 1812])[-3:]

# %%
# np.where(np.array(mdl.used_vocabs)=='biomass')

# %%
# for i, terms in enumerate(topic_keyterms):
#     print(f'Topic {i}: {terms}')

# %% [markdown]
# ### Add top2vec results

# %%
import innovation_sweet_spots.analysis.topic_analysis as iss_topics

# %%
# Add info about top2vec clusters
run = "August2021_gtr_cb_stopwords_e100"
clustering = iss_topics.get_clustering(run)
#
doc_df_ = doc_df.merge(clustering, how="left")
doc_df_["original_index"] = original_indexes

# %%
# Add info about top2vec clusters from previous run
run = "July2021_projects_orgs_stopwords_e400"
clustering_old = iss_topics.get_clustering(run)

# %%
clustering.head(1)

# %% [markdown]
# ### Check document-topic distributions

# %%
import seaborn as sns
from matplotlib import pyplot as plt
import altair as alt

alt.data_transformers.disable_max_rows()


# %%
def cumulative_plot(data):
    data_sorted = np.sort(data)
    p = 1.0 * np.arange(len(data)) / (len(data) - 1)
    fig = plt.plot(data_sorted, p)
    return fig


def prob_histogram(topic_probs):
    df = pd.DataFrame(data={"probs": topic_probs})
    fig = (
        alt.Chart(df)
        .mark_bar()
        .encode(alt.X("probs:Q", bin=alt.Bin(step=0.01)), y="count()")
    )
    return fig


# %%
topic = 68

# %%
topic_probs = doc_topic_dists[:, topic]
fig = cumulative_plot(topic_probs)
plt.xscale("log")

# %%
p_thresh = np.percentile(topic_probs, 99)
p_thresh

# %%
# doc_df[doc_df.title.str.contains('domestic boiler')]

# %%
# doc_df_['prob'] = topic_probs
# doc_df_[doc_df_['prob'] > p_thresh].sort_values('prob')

# %% [markdown]
# ### Calibrate topic probabilities?

# %%
import importlib
import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
importlib.reload(iss)

# %%
doc_df_gtr = doc_df[doc_df.source == "gtr"]
doc_df_gtr = iss.get_gtr_project_topics(
    doc_df_gtr.rename(columns={"doc_id": "project_id"})
)

# %%
df = doc_df_gtr[doc_df_gtr.text == "Energy Storage"]

# %%
# df_probs = doc_df_[doc_df_.doc_id.isin(df.project_id.to_list())].prob.to_list()

# %%
# doc_df_[doc_df_.doc_id.isin(df.project_id.to_list())].prob.mean()

# %%
# prob_histogram(df_probs)

# %%
# fig = cumulative_plot(df_probs)
# plt.xscale('log')

# %%
# doc_df_[doc_df_.doc_id.isin(df.project_id.to_list())].sort_values('prob')

# %% [markdown]
# ### Select all projects that fall within 99-th percentile of relevant topics

# %%
checked_topics = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/tomotopy_topics_checked.csv"
)
stop_topics = checked_topics[checked_topics.stop_topic == 1].topic.to_list()
primary_topics = checked_topics[checked_topics.primary == 1].topic.to_list()

# %%
doc_topic_dists_adjusted = doc_topic_dists.copy()
doc_topic_dists_adjusted[:, stop_topics] = 0
for i in range(doc_topic_dists_adjusted.shape[0]):
    doc_topic_dists_adjusted[i, :] /= doc_topic_dists_adjusted[i, :].sum()

# %%
topic_keyterms_short = get_topic_terms(mdl, n=10, _lambda=0.5)


def topic_doc_df(topic_docs, topic_probabilities):
    df = pd.DataFrame(data={"doc_id": topic_docs, "prob": topic_probabilities})
    return df.merge(doc_df, how="left")


def check_document_topics(doc_id):
    index = doc_df[doc_df.doc_id == doc_id].index[0]
    return pd.DataFrame(
        data={
            "topic_id": range(len(topic_keyterms_short)),
            "topic": topic_keyterms_short,
            "prob": doc_topic_dists_adjusted[index, :],
        }
    ).sort_values("prob", ascending=False)


# %%
PRCTILE = 99
SELECT_TOPICS = primary_topics
topic_docs = []
topic_probabilities = []
for topic in SELECT_TOPICS:
    topic_probs = doc_topic_dists_adjusted[:, topic]
    p_thresh = np.percentile(topic_probs, PRCTILE)
    selected_docs = np.where(topic_probs >= p_thresh)[0]
    docs = doc_df.iloc[selected_docs].doc_id.to_list()
    topic_docs.append(docs)
    topic_probabilities.append([topic_probs[i] for i in selected_docs])

# %%
np.percentile(doc_topic_dists_adjusted[:, 68], 98)

# %%
data = doc_topic_dists[:, 68]
fig = cumulative_plot(data)
# plt.plot([0.01, 1], [0.99, 0.99])
plt.xscale("log")
plt.ylim([0.9, 1])

# %%
# p_thresh = np.percentile(doc_topic_dists_adjusted[:, 68], 99)
# p_thresh

# %%
# j = 3
# df = topic_doc_df(topic_docs[j], topic_probabilities[j]).sort_values('prob')
# df[df.prob>p_thresh].head(10)
# # df[df.prob>.1].head(10)

# %%
# data_sorted = np.sort(data)
# # p = 1. * np.arange(len(data)) / (len(data) - 1)
# plt.plot(
#     data_sorted[2:],
#     np.diff(np.diff(data_sorted))
# )
# plt.xscale('log')

# %%
# topic_keyterms[SELECT_TOPICS[3]][0:10]

# %%
all_docs = set([doc for docs in topic_docs for doc in docs])
final_set = all_docs.union(set(clustering_old.doc_id.to_list()))
len(final_set)

# %%
doc_df_ = doc_df.reset_index().rename(columns={"index": "i"}).set_index("doc_id")
doc_df_ = doc_df_.loc[final_set]
doc_df_.head(1)

# %%
new_corpus = [full_corpus[i] for i in doc_df_.i.to_list()]

# %%
# doc_df_.to_csv(PROJECT_DIR / 'outputs/data/results_august/narrow_set_docs.csv')

# %%
# check_document_topics('4F101486-7A7B-4A34-8FDD-73D33A3881D6')

# %%
# doc_df.iloc[151]

# %%
# '00610688-F021-4190-B1F6-5AC5CB981077' in list(final_set)

# %% [markdown]
# # Re-run tomotopy model on the smaller corpus

# %%
# Create a corpus instance
new_tp_corpus = tp.utils.Corpus()
for doc in new_corpus:
    new_tp_corpus.add_doc(doc)

# %%
# Initialise model
mdl_new = tp.LDAModel(min_df=5, rm_top=40, k=150, corpus=new_tp_corpus, seed=1111)

# if SET_PRIORS:
#     logging.info('Setting prior probabilities')
#     prior_prob_matrix, seed_to_index = create_prior_prob_matrix(topic_seeds)
#     for t, topics in enumerate(topic_seeds):
#         for seed in topics:
#             mdl.set_word_prior(seed, list(prior_prob_matrix[seed_to_index[seed],:]))

mdl_new.train(0)
print_model_info(mdl_new)

# %%
train_model(mdl_new, iterations=1000, step=20)

# %%
# mdl_new.save(str(PROJECT_DIR / 'outputs/models/narrow_corpus_lda_model.bin'))

# %%
mdl_new = tp.LDAModel.load(
    str(PROJECT_DIR / "outputs/models/narrow_corpus_lda_model.bin")
)

# %%
topic_keyterms_new = get_topic_terms(mdl_new, n=15, n_docs=len(new_corpus))
topic_keyterms_new_str = [" ".join(s) for s in topic_keyterms_new]

# %% [markdown]
# ## Check top2vec model

# %%
run_full = "August2021_gtr_cb_stopwords_e100"

# %%
# Import top2vec model
top2vec_model = iss_topics.get_top2vec_model(run_full)

# %%
# (clustering[clustering.doc_id.isin(final_set)]
#  .groupby('cluster_keywords')
#  .agg(counts=('doc_id', 'count'))
#  .sort_values('counts', ascending=False)
#  .reset_index()
#  .merge(clustering[['cluster_keywords','cluster_id']].drop_duplicates('cluster_id'), how='left')
#  .to_csv(PROJECT_DIR / 'outputs/data/results_august/narrow_set_top2vec_cluster_counts.csv')
# )

# %%
len(top2vec_model.documents)

# %%
clustering_ = clustering.copy()
clustering_["cluster_probs"] = top2vec_model.cluster.probabilities_

# %%
len(full_corpus)

# %%
# document_vectors

# %% [markdown]
# # Calculate detailed topic probabilities
# For each document:
# - Check if it has a global topic probability (for any relevant topics) larger than 0.001
# - Check if it has a local topic probability larger than 0.1
# - Add documents that are in the relevant CB or GTR categories
# - Add doucments that are in the relevant top2vec clusters (high prob being relevant)
# - For CB only: Might need to check according to keywords...

# %%
import innovation_sweet_spots.getters.crunchbase as crunchbase

# %%
# Doc dataframe
narrow_set_df = doc_df_.copy()

# %%
doc_df_gtr = doc_df[doc_df.source == "gtr"]
doc_df_gtr = iss.get_gtr_project_topics(
    doc_df_gtr.rename(columns={"doc_id": "project_id"})
)
doc_df_gtr = doc_df_gtr.rename(columns={"project_id": "doc_id"})

df = crunchbase.get_crunchbase_organizations_categories()
doc_df_cb = doc_df[doc_df.source == "cb"]
doc_df_cb = doc_df_cb.rename(columns={"doc_id": "organization_id"}).merge(
    df[["organization_id", "category_name"]], how="left"
)
doc_df_cb = doc_df_cb.rename(columns={"organization_id": "doc_id"})
del df


# %%
def check_document_topics(
    doc_id, df=narrow_set_df, doc_topic_dists_adjusted=doc_topic_dists_adjusted
):
    dff = df.reset_index().copy()
    index = dff[dff.doc_id == doc_id].index[0]
    return pd.DataFrame(
        data={
            "topic_id": range(len(topic_keyterms_new)),
            "topic": topic_keyterms_new,
            "prob": doc_topic_dists_adjusted[index, :],
        }
    ).sort_values("prob", ascending=False)


def check_topic_probs(top_n, doc_topic_dists_adjusted):
    # Best topics and topic probabilities for each topic
    max_topic = []
    max_topic_prob = []
    for i in range(doc_topic_dists_adjusted.shape[0]):
        max_topic.append(np.argsort(doc_topic_dists_adjusted[i, :])[-top_n])
        max_topic_prob.append(np.sort(doc_topic_dists_adjusted[i, :])[-top_n])
    max_topic = np.array(max_topic)
    max_topic_prob = np.array(max_topic_prob)
    return max_topic_prob


def get_topic_documents(
    topic_id, cb_tags=None, gtr_tags=None, custom_keywords=None, compile_text=True
):
    """ """
    # Get top documents
    doc_ids = []
    tags = []

    topic_docs = np.where(max_topic == topic_id)[0]
    topic_doc_probs = max_topic_prob[topic_docs]
    top_min_probability = np.min(topic_doc_probs)

    d = narrow_set_df.iloc[topic_docs].index.to_list()
    doc_ids += d
    tags += ["best_topic"] * len(d)

    # Get top documents (excluding 'enabling docs')
    topic_docs = np.where(max_topic_wout_enabling == topic_id)[0]
    d = narrow_set_df.iloc[topic_docs].index.to_list()
    doc_ids += d
    tags += ["best_topic_enabler"] * len(d)

    # Get documents above the minimum probability
    topic_docs = np.where(doc_topic_dists_adjusted[:, topic_id] > top_min_probability)[
        0
    ]
    d = narrow_set_df.iloc[topic_docs].index.to_list()
    doc_ids += d
    tags += ["topic_above_threshold"] * len(d)

    # Get GTR projects with special categories
    if gtr_tags is not None:
        d = doc_df_gtr[doc_df_gtr.text.isin(gtr_tags)].doc_id.to_list()
        doc_ids += d
        tags += ["gtr_category"] * len(d)

    # Get CB companies with special tags
    if cb_tags is not None:
        d = doc_df_cb[doc_df_cb.category_name.isin(cb_tags)].doc_id.to_list()
        doc_ids += d
        tags += ["crunchbase_category"] * len(d)

    df_docs = narrow_set_df.reset_index().copy()
    df_docs["probs"] = doc_topic_dists_adjusted[:, topic_id]
    df_docs["top_topic"] = max_topic
    df_docs = df_docs.merge(clustering[["doc_id", "cluster_keywords"]], how="left")
    # Make the dataframe
    df = (
        pd.DataFrame(data={"doc_id": doc_ids, "selection": tags})
        .drop_duplicates("doc_id", keep="first")
        .merge(df_docs)
    )

    category_tags = []
    for i, row in df.iterrows():
        if row.source == "gtr":
            df_cats = doc_df_gtr[doc_df_gtr.doc_id == row.doc_id]
            df_cats = df_cats[-df_cats.text.isnull()]
            cats = ", ".join(sorted(df_cats.text.to_list()))
        else:
            df_cats = doc_df_cb[doc_df_cb.doc_id == row.doc_id]
            df_cats = df_cats[-df_cats.category_name.isnull()]
            cats = ", ".join(sorted(df_cats.category_name.to_list()))
        category_tags.append(cats)
    df["source_categories"] = category_tags

    df = df[
        [
            "i",
            "doc_id",
            "title",
            "description",
            "top_topic",
            "cluster_keywords",
            "source",
            "source_categories",
            "selection",
            "probs",
        ]
    ]
    df = df.rename(columns={"cluster_keywords": "top2vec_keywords"})

    df.selection = pd.Categorical(
        df.selection,
        categories=list(
            reversed(
                [
                    "best_topic",
                    "best_topic_enabler",
                    "topic_above_threshold",
                    "gtr_category",
                    "crunchbase_category",
                ]
            )
        ),
        ordered=True,
    )
    #     df = df.sort_values('probs', ascending=False).sort_values(['selection'])
    df = df.sort_values(["selection", "probs"], ascending=False)

    # Add text to use in labelling
    if compile_text:
        data_for_labelling = []
        for i, row in df.iterrows():
            s = f"doc_id: {row.doc_id}\nsource: {row.source}\nprobability: {np.round(row.probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
            s += f"\n\ntop_topic: Topic {row.top_topic}: {topic_keyterms_new_str[row.top_topic]}"
            s += f"\n\ntop2vec keywords: {row.top2vec_keywords}\n\nsource_categories: {row.source_categories}"
            data_for_labelling.append(s)
        df["text"] = data_for_labelling

    return df


# %%
clusters_categories = pd.read_excel(
    PROJECT_DIR / "outputs/data/results_august/top2vec_clusters_categories.xlsx"
)
narrow_set_categories = pd.read_excel(
    PROJECT_DIR
    / "outputs/data/results_august/narrow_set_top2vec_cluster_counts_checked.xlsx"
)

# %%
clusters_categories = clusters_categories[-clusters_categories.manual_category.isnull()]
narrow_set_categories = narrow_set_categories[-narrow_set_categories.label.isnull()]

# %%
cats = clusters_categories.manual_category.unique()
for cat in cats:
    print(cat)

# %%
# GTR categories
categories_to_gtr_categories = {
    "Batteries": ["Energy Storage"],
    "Bioenergy": ["Bioenergy"],
    "Carbon Capture & Storage": ["Carbon Capture & Storage"],
    "Demand management": None,
    "EV": None,
    "Heating & Buildings": None,
    "Hydrogen & Fuel Cells": ["Fuel Cell Technologies"],
    "Nuclear": ["Energy - Nuclear"],
    "Solar": ["Solar Technology"],
    "Wind & Offshore": ["Wind Power", "Energy - Marine & Hydropower"],
}


# %%
categories_to_cb_categories = {
    "Batteries": ["battery", "energy storage"],
    "Bioenergy": ["biomass energy", "biofuel"],
    "Carbon Capture & Storage": None,
    "Demand management": [
        "energy management",
        "smart building",
        "smart cities",
        "smart home",
    ],
    "EV": ["electric vehicle"],
    "Heating & Buildings": ["green building"],
    "Hydrogen & Fuel Cells": ["fuel cell"],
    "Nuclear": ["nuclear"],
    "Solar": ["solar"],
    "Wind & Offshore": ["wind energy"],
}


# %%
categories_to_top2vec = {}
for cat in cats:
    categories_to_top2vec[cat] = clusters_categories[
        clusters_categories.manual_category == cat
    ].cluster_id.to_list()

# %%
categories_to_narrow_topics = {}
for cat in cats:
    categories_to_narrow_topics[cat] = narrow_set_categories[
        narrow_set_categories.label == cat
    ].topic.to_list()

# %%
categories_to_narrow_topics

# %%
global_topics = [26, 52, 68, 71, 78, 83, 128, 139, 140, 142]
local_topics = [
    40,
    68,
    92,
    93,
    39,
    137,
    138,
    8,
    149,
    19,
    94,
    130,
    83,
    2,
    67,
]

# %%
# Check global topic probs
global_doc_topic_probs = get_doc_topic_dists(mdl)
global_doc_topic_probs.shape

# %%
np.median(global_doc_topic_probs)

# %%
bool_mat = global_doc_topic_probs[:, global_topics] > 0.1
global_OK = bool_mat[:, 0]
for i in range(len(global_topics)):
    global_OK = global_OK | bool_mat[:, i]

# Documents to bother checking for topics
# check_docs_id = doc_df.loc[x].index.to_list()

# %%
from tqdm.notebook import tqdm

# %%
# infer_doc_topic_dists = []
# for d_id in tqdm(check_docs_id, total=len(check_docs_id)):
#     doc_instance = mdl_new.make_doc(full_corpus[d_id])
#     infer_doc_topic_dist, _ = mdl_new.infer(doc_instance)
#     infer_doc_topic_dists.append(infer_doc_topic_dist)

# %%
# infer_doc_topic_dists = []
# for doc in tqdm(full_corpus, total=len(full_corpus)):
#     doc_instance = mdl_new.make_doc(doc)
#     infer_doc_topic_dist, _ = mdl_new.infer(doc_instance)
#     infer_doc_topic_dists.append(infer_doc_topic_dist)
# infer_doc_topic_dists = np.array(infer_doc_topic_dists)

# %%
# np.save(PROJECT_DIR / 'outputs/data/results_august/narrow_topics_full_corpus.npy', np.array(infer_doc_topic_dists))

# %%
infer_doc_topic_dists = np.load(
    PROJECT_DIR / "outputs/data/results_august/narrow_topics_full_corpus.npy"
)

# %%
infer_doc_topic_dists.shape

# %%
# global_doc_topic_probs.shape

# %%
# check_docs_id_ = check_docs_id
# del infer_doc_topic_dists_

# %%
from collections import defaultdict

# %%
TOPIC_THRESH = 0.10

# %%
from innovation_sweet_spots.utils.io import read_list_of_terms

already_checked_docs = read_list_of_terms(
    PROJECT_DIR / "outputs/data/results_august/check_doc_id_all.txt"
)

# %%
cats

# %%
selected_docs_dfs = []
for cat in cats:
    hits = defaultdict(list)
    # Find all documents that are in CB category
    if categories_to_cb_categories[cat] is not None:
        hits["cb"] = (
            doc_df_cb[doc_df_cb.category_name.isin(categories_to_cb_categories[cat])]
            .drop_duplicates("doc_id")
            .doc_id.to_list()
        )
    # Find all documents that are in GTR category
    if categories_to_gtr_categories[cat] is not None:
        hits["gtr"] = (
            doc_df_gtr[doc_df_gtr.text.isin(categories_to_gtr_categories[cat])]
            .drop_duplicates("doc_id")
            .doc_id.to_list()
        )
    # Documents in the top2vec cluster
    hits["top2vec"] = clustering[
        clustering.cluster_id.isin(categories_to_top2vec[cat])
    ].doc_id.to_list()
    # Documents above the topic prob
    docs_above_topic_thresh = []
    probs = []
    for c in categories_to_narrow_topics[cat]:
        is_OK = (infer_doc_topic_dists[:, c] >= TOPIC_THRESH) & global_OK
        docs_above_topic_thresh += list(np.where(is_OK)[0])
        probs += list(infer_doc_topic_dists[is_OK, c])
    # docs_above_topic_thresh = docs_above_topic_thresh
    # docs_above_topic_thresh = [check_docs_id[i] for i in docs_above_topic_thresh]
    hits["topics"] = doc_df.loc[docs_above_topic_thresh].doc_id.to_list()
    # Combine
    doc_ids_final = []
    doc_ids_source = []
    doc_ids_probs = []
    for key in hits:
        doc_ids_final += hits[key]
        doc_ids_source += [key] * len(hits[key])
        if key != "topics":
            doc_ids_probs += [0] * len(hits[key])
        else:
            doc_ids_probs += probs
    df = pd.DataFrame(
        data={
            "doc_id": doc_ids_final,
            "selection": doc_ids_source,
            "topic_prob": doc_ids_probs,
        }
    )
    df["selection"] = pd.Categorical(
        df.selection,
        categories=list(reversed(["gtr", "cb", "top2vec", "topics"])),
        ordered=True,
    )
    df = df.merge(doc_df, how="left")
    df = df.merge(
        clustering_[["doc_id", "cluster_keywords", "cluster_probs"]], how="left"
    )
    df.loc[df.selection == "top2vec", "topic_prob"] = df.loc[
        df.selection == "top2vec", "cluster_probs"
    ]
    df = df.drop("cluster_probs", axis=1)
    df = df.sort_values(["selection", "topic_prob"], ascending=False).drop_duplicates(
        ["doc_id"]
    )
    df["category"] = cat

    # Add text to use in labelling
    data_for_labelling = []
    for i, row in df.iterrows():
        s = f"CATEGORY: {cat}\n\n"
        s += f"doc_id: {row.doc_id}\nselection_factor: {row.selection}\nprobability: {np.round(row.topic_prob,4)}\n\n{row.title}\n------\n{row.description}\n------"
        s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
        data_for_labelling.append(s)
    df["text"] = data_for_labelling
    df = df[-df.doc_id.isin(already_checked_docs)]
    selected_docs_dfs.append(df)

# %%
print(selected_docs_dfs[0].iloc[0].text)

# %%
selected_docs_df_all = pd.DataFrame()
for df in selected_docs_dfs:
    selected_docs_df_all = selected_docs_df_all.append(df, ignore_index=True)

# %%
# selected_docs_df_all.to_csv(PROJECT_DIR / 'outputs/data/results_august/reference_categories_to_check.csv', index=False)

# %%
selected_docs_df_all = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/reference_categories_to_check.csv"
)

# %%
selected_docs_df_all[
    selected_docs_df_all.doc_id == "34FFD228-2F72-461E-AD54-A92D2D826225"
]

# %%
print(selected_docs_df_all.iloc[1491].description)

# %%
np.sum([len(df) for df in selected_docs_dfs])

# %%
cats

# %%
i = 5
print(cats[i])
selected_docs_dfs[i].groupby("selection").count()

# %%
[len(df) for df in selected_docs_dfs]

# %%
selected_docs_dfs[i]

# %% [markdown]
# # Select categories and documents

# %%
narrow_topics_checked = pd.read_excel(
    PROJECT_DIR / "outputs/data/results_august/narrow_set_clusters_checked.xlsx",
    sheet_name="narrow_set_tomotopy",
)

# %%
narrow_topics_checked[-narrow_topics_checked.report.isnull()]

# %% [markdown]
# - Select topic id number
# - Inspect topic probability distribution
# - Select top X documents
# - Check if category labels/tags can be additionally used
# - Export for manual review

# %%
stop_topics = narrow_topics_checked[
    narrow_topics_checked.stop_topic == 1
].topic.to_list()
primary_topics = narrow_topics_checked[
    (narrow_topics_checked.report == 1) & (narrow_topics_checked.enabling != 1)
].topic.to_list()
enabling_topics = narrow_topics_checked[
    #     (narrow_topics_checked.report==1)
    (narrow_topics_checked.enabling == 1)
].topic.to_list()

# %%
len(primary_topics)

# %%
len(enabling_topics)

# %%
len(stop_topics)

# %%
# topic_keyterms_new = get_topic_terms(mdl_new, n=15, n_docs=len(new_corpus))
# topic_keyterms_new_str = [' '.join(s) for s in topic_keyterms_new]
[f"Topic {i}: {topic_keyterms_new_str[i]}" for i in enabling_topics]

# %%
# topic_keyterms_new = get_topic_terms(mdl_new, n=15, n_docs=len(new_corpus))
# topic_keyterms_new_str = [' '.join(s) for s in topic_keyterms_new]
[f"Topic {i}: {topic_keyterms_new_str[i]}" for i in primary_topics]

# %% [markdown]
# ### Prepare doc dataframes

# %%
# Doc dataframe
narrow_set_df = doc_df_.copy()

# %%

# %%

# %%

# %%

# %%

# %%
# Document topic probabilities
doc_topic_dists = get_doc_topic_dists(mdl_new)
doc_topic_dists.shape
# Remove stop topics from the probabilities
doc_topic_dists_adjusted = doc_topic_dists.copy()
doc_topic_dists_adjusted[:, stop_topics] = 0
for i in range(doc_topic_dists_adjusted.shape[0]):
    doc_topic_dists_adjusted[i, :] /= doc_topic_dists_adjusted[i, :].sum()

# %%
doc_topic_dists_adjusted.shape

# %%
data = doc_topic_dists_adjusted[:, 2]
fig = cumulative_plot(data)
# plt.plot([0.01, 1], [0.99, 0.99])
plt.xscale("log")
plt.ylim([0.9, 1])
plt.show()

# %%
# Probabilities without enabling
doc_topic_dists_adjusted_wout_enabling = doc_topic_dists_adjusted.copy()
doc_topic_dists_adjusted_wout_enabling[:, enabling_topics] = 0

# %%
# Best topics and topic probabilities for each topic
max_topic = []
max_topic_prob = []
for i in range(doc_topic_dists_adjusted.shape[0]):
    max_topic.append(np.argsort(doc_topic_dists_adjusted[i, :])[-1])
    max_topic_prob.append(np.sort(doc_topic_dists_adjusted[i, :])[-1])
max_topic = np.array(max_topic)
max_topic_prob = np.array(max_topic_prob)

# %%
# Best topics and topic probabilities for each topic
max_topic_wout_enabling = []
max_topic_prob_wout_enabling = []
for i in range(doc_topic_dists_adjusted.shape[0]):
    max_topic_wout_enabling.append(
        np.argsort(doc_topic_dists_adjusted_wout_enabling[i, :])[-1]
    )
    max_topic_prob_wout_enabling.append(
        np.sort(doc_topic_dists_adjusted_wout_enabling[i, :])[-1]
    )
max_topic_wout_enabling = np.array(max_topic_wout_enabling)
max_topic_prob_wout_enabling = np.array(max_topic_prob_wout_enabling)

# %%
topic_docs = np.where(max_topic_wout_enabling == 2)[0]
topic_doc_probs = max_topic_prob_wout_enabling[topic_docs]

# %%
doc_topic_dists_adjusted_wout_enabling

# %%
# Check the top probabilities
np.mean(doc_topic_dists_adjusted.copy().max(axis=1))
np.median(doc_topic_dists_adjusted.copy().max(axis=1))

# %% [markdown]
# ### Topic: Solar

# %%
clustering_clusters = (
    clustering.sort_values("cluster_id")
    .drop_duplicates("cluster_id")
    .reset_index(drop=True)
)[["cluster_id", "cluster_keywords"]]
clustering_clusters.to_csv(
    PROJECT_DIR / "outputs/data/results_august/top2vec_clusters.csv", index=False
)

# %%
clustering_clusters_cat = pd.read_excel(
    PROJECT_DIR / "outputs/data/results_august/top2vec_clusters_categories.xlsx"
)

# %%
clustering_clusters_cat[-clustering_clusters_cat.manual_category.isnull()]

# %%
# narrow_set_df['probs']

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/data/results_august/topic_documents"

# %%
topic_data = [
    # Solar
    {"topic_id": 2, "gtr_tags": ["Solar Technology"], "cb_tags": ["solar"]},
    # Heating
    {"topic_id": 19, "gtr_tags": None, "cb_tags": None},
    # Building insulation
    {"topic_id": 149, "gtr_tags": None, "cb_tags": None},
    # Households
    {"topic_id": 138, "gtr_tags": None, "cb_tags": None},
]

# %%
for topic_params in topic_data:
    print(topic_params["topic_id"])
    get_topic_documents(**topic_params).to_csv(
        OUTPUTS_DIR / f'Topic_{topic_params["topic_id"]}.csv'
    )

# %%
get_topic_documents(**topic_data[1]).to_csv(
    OUTPUTS_DIR / f'Topic_{topic_params["topic_id"]}.csv'
)

# %%
df_docs = get_topic_documents(**topic_data[1])

# %%
df_docs = get_topic_documents(**topic_data[2])

# %%
df_docs.groupby("top2vec_keywords").agg(counts=("i", "count")).sort_values("counts")

# %%
df_docs.info()

# %%
df_docs.groupby(["selection", "source"]).agg(counts=("doc_id", "count"))

# %%
topic_docs = np.where(max_topic == topic_id)[0]
topic_doc_probs = max_topic_prob[topic_docs]

# %%
np.min(topic_doc_probs)

# %%
narrow_set_df["probs"] = doc_topic_dists_adjusted[:, topic_id]
# narrow_set_df.sort_values('probs', ascending=False).head(10)

# %%
prob_histogram(topic_doc_probs)

# %%
# Check other solar projects
ids = doc_df_gtr[doc_df_gtr.text == "Solar Technology"].doc_id.to_list()
narrow_set_df.loc[ids].probs.min()

# %% [markdown]
# # Keyword approach

# %%
# Doc dataframe
narrow_set_df = doc_df_.copy()

# %%
make_pyLDAvis(
    mdl_new,
    fpath=PROJECT_DIR / "outputs/data/results_august/ldavis_tomotopy_narrow_set.html",
)

# %%
# Reconstruct document
new_corpus_document_texts = [
    " ".join([" ".join(s.split("_")) for s in doc]) for doc in new_corpus
]
# Where are new document ids?
# narrow_set_df.head(1)

# %%
full_corpus_document_texts = [
    " " + " ".join([" ".join(s.split("_")) + " " for s in doc]) for doc in full_corpus
]

# %%
# full_corpus_document_texts_2 = [' ' + ' '.join([' '.join(s.split('_')) + ' ' for s in doc]) for doc in full_corpus]

# %%
full_corpus_document_texts_2 = []
for tokens_list in full_corpus:
    s_list = [s for s_list in [tok.split("_") for tok in tokens_list] for s in s_list]
    full_corpus_document_texts_2.append(" " + " ".join(s_list) + " ")

# %%
full_corpus_document_texts[0]

# %%
full_corpus_document_texts_2[0]

# %%
global_doc_topic_dists = get_doc_topic_dists(mdl)
local_doc_topic_dists = get_doc_topic_dists(mdl_new)


# %%
# words, word_scores = top2vec_model.similar_words(keywords=["solar_thermal"], keywords_neg=[], num_words=50)
# for word, score in zip(words, word_scores):
#     print(f"{word} {score}")

# %%
# Find documents using substring matching
def find_docs_with_terms(
    terms,
    corpus_texts=new_corpus_document_texts,
    corpus_df=narrow_set_df,
    return_dataframe=True,
):
    x = np.array([False] * len(corpus_texts))
    for term in terms:
        x = x | np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def find_docs_with_all_terms(
    terms,
    corpus_texts=new_corpus_document_texts,
    corpus_df=narrow_set_df,
    return_dataframe=True,
):
    x = np.array([True] * len(corpus_texts))
    for term in terms:
        x = x & np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def get_docs_with_keyphrases(keyphrases):
    x = np.array([False] * len(full_corpus_document_texts))
    for terms in keyphrases:
        print(terms)
        x = x | find_docs_with_all_terms(
            terms,
            corpus_texts=full_corpus_document_texts,
            corpus_df=doc_df,
            return_dataframe=False,
        )
    return doc_df.iloc[x]


def get_docs_with_keyphrases_2(keyphrases):
    x = np.array([False] * len(full_corpus_document_texts_2))
    for terms in keyphrases:
        print(terms)
        x = x | find_docs_with_all_terms(
            terms,
            corpus_texts=full_corpus_document_texts_2,
            corpus_df=doc_df,
            return_dataframe=False,
        )
    return doc_df.iloc[x]


def add_info(df, GLOBAL_TOPIC, LOCAL_TOPIC):
    df = df.merge(clustering, how="left")

    # Add global topic probs
    df_ = doc_df[["doc_id"]]
    df_["global_topic_prob"] = global_doc_topic_dists[:, GLOBAL_TOPIC]
    df = df.merge(df_, how="left")
    # Add local topic probs
    df_ = narrow_set_df.reset_index()[["doc_id"]]
    df_["local_topic_prob"] = local_doc_topic_dists[:, LOCAL_TOPIC]
    df = df.merge(df_, how="left")
    # Determine best cluster
    c = (
        df.groupby("cluster_keywords")
        .agg(counts=("doc_id", "count"))
        #     .sort_values('counts', ascending=False)
        .reset_index()
        .sort_values("cluster_keywords")
        .sort_values("counts", ascending=False)
    )
    df["cluster_keywords"] = pd.Categorical(
        df["cluster_keywords"],
        categories=reversed(c.cluster_keywords.to_list()),
        ordered=True,
    )
    return df.sort_values(["cluster_keywords", "local_topic_prob"], ascending=False)


def print_categories_keywords(d):
    for key in d:
        print(f"{key}: {d[key]}")


# %%

# %%
categories_keyphrases_heat = {
    # Heating subcategories
    "Heat pumps": [["heat pump"]],
    "Geothermal energy": [["geothermal", "energy"], ["geotermal", "heat"]],
    "Solar thermal": [["solar thermal"]],
    "Waste heat": [["waste heat"], ["heat recovery"]],
    "Heat storage": [
        ["heat stor"],
        ["thermal energy stor"],
        ["thermal stor"],
        ["heat batter"],
    ],
    "District heating": [["heat network"], ["district heat"]],
    "Electric boilers": [["electric", "boiler"], ["electric heat"]],
    "Biomass boilers": [["pellet", "boiler"], ["biomass", "boiler"]],
    "Hydrogen boilers": [["hydrogen", "boiler"]],
    "Micro CHP": [["combined heat power", "micro"], ["micro", "chp"], ["mchp"]],
    "Hydrogen heating": [
        ["hydrogen", "heating"],
        ["hydrogen heat"],
        ["green hydrogen"],
    ],
    "Biomass heating": [["biomass", "heating"], ["biomass", " heat"]]
    #     'Hydrogen heating': [['hydrogen', 'heat']],
}

categories_keyphrases_build = {
    # Building energy efficiency subcategories (including construction)
    "Building insulation": [
        ["insulat", "build"],
        ["insulat", "hous"],
        ["insulat", "home"],
        ["insulat", "retrofit"],
        ["cladding", "hous"],
        ["cladding", "build"],
        ["glazing", "window"],
        ["glazed", "window"],
    ],
    "Building insulation (extra)": [
        ["retrofit", "build"],
        ["retrofit", "hous"],
        ["retrofit", "home"],
    ],
    "Radiators": [["radiator"]],
}

categories_keyphrases_energy = {
    "Energy management (old)": [
        ["energy management", "build"],
        ["energy management", "domestic"],
        ["energy management", "hous"],
        ["thermostat"],
        ["smart meter"],
    ],
    "Energy management (extra)": [["load shift", "heat"]],
    "Energy management": [
        ["smart home", "heat"],
        ["demand response", "heat"],
        ["load shift", "heat"],
        ["energy management", "build"],
        ["energy management", "domestic"],
        ["energy management", "hous"],
        ["energy management", "home"],
        ["thermostat"],
        ["smart meter"],
    ],
}
# Not so sure about the categories below
#     'Heat transfer': [['heat transfer'], ['heat exchange']],
#     'Cooling and refrigeration': [['cooling'], ['refriger']],
# 'Other heating' category

# %% [markdown]
# #### Extra checks

# %%
df = get_docs_with_keyphrases(categories_keyphrases_heat["Hydrogen heating"])
df = df[
    df.doc_id.isin(
        get_docs_with_keyphrases(
            categories_keyphrases_heat["Hydrogen boilers"]
        ).doc_id.to_list()
    )
    == False
]
df = add_info(df, GLOBAL_TOPIC=68, LOCAL_TOPIC=19).reset_index(drop=True)
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Hydrogen heating.csv"
)

# %%
df_check = pd.read_excel(
    PROJECT_DIR
    / "outputs/data/results_august/aux/ISS_technologies_to_review_August_10.xlsx",
    sheet_name="Building insulation",
)
df = get_docs_with_keyphrases(
    categories_keyphrases_build["Building insulation (extra)"]
)
df = df[df.doc_id.isin(df_check.doc_id.to_list()) == False]
df = add_info(df, GLOBAL_TOPIC=140, LOCAL_TOPIC=149).reset_index(drop=True)
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Building insulation (extra).csv"
)

# %%
df = get_docs_with_keyphrases(categories_keyphrases_heat["Biomass heating"])
df = df[
    df.doc_id.isin(
        get_docs_with_keyphrases(
            categories_keyphrases_heat["Biomass boilers"]
        ).doc_id.to_list()
    )
    == False
]
df = add_info(df, GLOBAL_TOPIC=68, LOCAL_TOPIC=19).reset_index(drop=True)
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Biomass heating.csv"
)

# %%
df_check = pd.read_excel(
    PROJECT_DIR
    / "outputs/data/results_august/aux/ISS_technologies_to_review_August_10.xlsx",
    sheet_name="Energy management",
)
df = get_docs_with_keyphrases(categories_keyphrases_energy["Energy management (extra)"])
df = df[df.doc_id.isin(df_check.doc_id.to_list()) == False]
df = add_info(df, GLOBAL_TOPIC=140, LOCAL_TOPIC=149).reset_index(drop=True)
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Energy management (extra).csv"
)

# %%
df_check = pd.read_excel(
    PROJECT_DIR
    / "outputs/data/results_august/aux/ISS_technologies_to_review_August_10.xlsx",
    sheet_name="Building insulation",
)
df = get_docs_with_keyphrases(
    categories_keyphrases_build["Building insulation (extra)"]
)
df = df[df.doc_id.isin(df_check.doc_id.to_list()) == False]
df = add_info(df, GLOBAL_TOPIC=140, LOCAL_TOPIC=149).reset_index(drop=True)
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Building insulation (extra).csv"
)

# %%
GTR_DOCS_ALL = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_GTR.csv"
)

# %%
len(df)

# %%
df = df[
    df.doc_id.isin(
        GTR_DOCS_ALL[
            GTR_DOCS_ALL.tech_category == "Building insulation"
        ].doc_id.to_list()
    )
    == False
]
df.to_csv(
    PROJECT_DIR
    / "outputs/data/aux/ISS_technologies_to_review_August_10 - Building insulation (extra).csv"
)

# %%
# df = get_docs_with_keyphrases(categories_keyphrases_energy['Energy management'])
# df = df[df.doc_id.isin(get_docs_with_keyphrases(categories_keyphrases_energy['Energy management']).doc_id.to_list())==False]
# df = add_info(df, GLOBAL_TOPIC=140, LOCAL_TOPIC = 149).reset_index(drop=True)
# df.to_csv(PROJECT_DIR / 'outputs/data/aux/ISS_technologies_to_review_August_10 - Energy management (extra).csv')

# %%
# df = pd.read_csv(OUTPUTS_DIR.parent / 'aux/ISS_technologies_to_review_August_10 - Electric boilers.csv')

# %%
# keyphrases = categories_keyphrases_heat['Electric boilers']
# add_info(get_docs_with_keyphrases([['electric heat'], ['electric', 'boiler']]), 68, 19).merge(df[['doc_id', 'hit_KK']], how='left').to_csv(OUTPUTS_DIR.parent / 'aux/ISS_technologies_to_review_August_10 - Electric boilers_v2.csv')

# %%
df1 = get_docs_with_keyphrases(categories_keyphrases_build["Building insulation"])
df2 = get_docs_with_keyphrases([["insulat", "home"]])

# %%
# df1 = get_docs_with_keyphrases(categories_keyphrases_energy['Energy management'])
# df2 = get_docs_with_keyphrases([['smart home', 'heat']])

# %%
# df1 = get_docs_with_keyphrases(categories_keyphrases_energy['Energy management'])
# df2 = get_docs_with_keyphrases([['demand response', 'heat']])

# %%
# df1 = get_docs_with_keyphrases(categories_keyphrases_heat['Hydrogen boilers'])
# df2 = get_docs_with_keyphrases([['hydrogen', 'heating'], ['hydrogen', 'heat'], ['green hydrogen']])

# %%
df2[df2.doc_id.isin(df1.doc_id.to_list()) == False]

# %%
# add_info(df2[df2.doc_id.isin(df1.doc_id.to_list())==False], 140, 149)

# %%
print_categories_keywords(categories_keyphrases_heat)

# %%
print_categories_keywords(categories_keyphrases_build)

# %%
print_categories_keywords(categories_keyphrases_energy)

# %%
GLOBAL_TOPIC = 68
LOCAL_TOPIC = 19

# %% [markdown]
# #### Initial checks

# %%
# df = get_docs_with_keyphrases(categories_keyphrases['Biomass boilers'])
dfs = {
    key: add_info(
        get_docs_with_keyphrases(categories_keyphrases_heat[key]),
        GLOBAL_TOPIC=68,
        LOCAL_TOPIC=19,
    ).reset_index(drop=True)
    for key in categories_keyphrases_heat
}

# %%
for key in categories_keyphrases_build:
    dfs[key] = add_info(
        get_docs_with_keyphrases(categories_keyphrases_build[key]),
        GLOBAL_TOPIC=140,
        LOCAL_TOPIC=149,
    ).reset_index(drop=True)

# %%
for key in categories_keyphrases_energy:
    dfs[key] = add_info(
        get_docs_with_keyphrases(categories_keyphrases_energy[key]),
        GLOBAL_TOPIC=68,
        LOCAL_TOPIC=138,
    ).reset_index(drop=True)

# %%
# dfs['Building insulation']

# %%
with pd.ExcelWriter(
    PROJECT_DIR
    / "outputs/data/results_august/ISS_technologies_to_review_August_10.xlsx"
) as writer:
    for key in dfs:
        dfs[key].to_excel(writer, sheet_name=key)

# %% [markdown]
# ### Check spacing and keyword search differences

# %%
# df = get_docs_with_keyphrases(categories_keyphrases['Biomass boilers'])
dfs_1 = {
    key: add_info(
        get_docs_with_keyphrases(categories_keyphrases_heat[key]),
        GLOBAL_TOPIC=68,
        LOCAL_TOPIC=19,
    ).reset_index(drop=True)
    for key in categories_keyphrases_heat
}
dfs_2 = {
    key: add_info(
        get_docs_with_keyphrases_2(categories_keyphrases_heat[key]),
        GLOBAL_TOPIC=68,
        LOCAL_TOPIC=19,
    ).reset_index(drop=True)
    for key in categories_keyphrases_heat
}

for key in categories_keyphrases_build:
    dfs_1[key] = add_info(
        get_docs_with_keyphrases(categories_keyphrases_build[key]),
        GLOBAL_TOPIC=140,
        LOCAL_TOPIC=149,
    ).reset_index(drop=True)
    dfs_2[key] = add_info(
        get_docs_with_keyphrases_2(categories_keyphrases_build[key]),
        GLOBAL_TOPIC=140,
        LOCAL_TOPIC=149,
    ).reset_index(drop=True)

for key in categories_keyphrases_energy:
    dfs_1[key] = add_info(
        get_docs_with_keyphrases(categories_keyphrases_energy[key]),
        GLOBAL_TOPIC=68,
        LOCAL_TOPIC=138,
    ).reset_index(drop=True)
    dfs_2[key] = add_info(
        get_docs_with_keyphrases_2(categories_keyphrases_build[key]),
        GLOBAL_TOPIC=140,
        LOCAL_TOPIC=149,
    ).reset_index(drop=True)

# %%
print([len(dfs_1[df]) for df in dfs_1])
print([len(dfs_2[df]) for df in dfs_2])

# %%
dfs_2_removed = {}
for key in dfs_1:
    df = dfs_1[key]
    df_new = dfs_2[key][dfs_2[key].doc_id.isin(df.doc_id.to_list()) == False]
    if len(df_new) != 0:
        dfs_2_removed[key] = df_new

# %%
for key in dfs_2_removed:
    print(key, len(dfs_2_removed[key]))

# %%
with pd.ExcelWriter(
    PROJECT_DIR
    / "outputs/data/results_august/ISS_technologies_to_review_September_1.xlsx"
) as writer:
    for key in dfs_2_removed:
        dfs_2_removed[key].to_excel(writer, sheet_name=key)

# %%
# dfs['Heat pumps']

# %% [markdown]
# ### Keywords and reference categories

# %%
cats

# %%
df_labels = pd.read_csv(
    "/Users/karliskanders/Downloads/project-7-at-2021-08-20-11-17-039f3681.csv"
)

# %%
# category = 'Bioenergy'
# category= 'Nuclear'
category = "EV"
# category = 'Wind & Offshore'
# category = 'Carbon Capture & Storage'
# category = 'Hydrogen & Fuel Cells'
# df = selected_docs_df_all[selected_docs_df_all.category=='Hydrogen & Fuel Cells']
df = selected_docs_df_all[selected_docs_df_all.category == category]
corpus_docs_ids = doc_df[doc_df.doc_id.isin(df.doc_id.to_list())].index.to_list()

# %%
df_labels[df_labels.category == category].groupby(["selection", "sentiment"]).count()

# %%
# full_corpus_document_texts[0]

# %%
# dff = get_docs_with_keyphrases([['solar'], [' pv '], ['perovskite'], ['photovoltaic']])
# dff = get_docs_with_keyphrases([['battery'], ['batteries']])
# dff = get_docs_with_keyphrases([['biofuel'], ['syngas'], ['bioenergy'], ['biogas'], ['biochar'],
#                                 ['biodiesel'], ['bioethanol'], ['biobutanol'], ['biomass', 'energy'], ['biomass', 'power'],
#                                 ['anaerobic digestion'], ['torrefaction'], ['pyrolisis'], ['ethanol'], ['butanol']
#                                ])
# dff = get_docs_with_keyphrases([['hydrogen'], ['fuel cell'], ['sofc']])
dff = get_docs_with_keyphrases([["ccs"], ["carbon storage"], ["carbon capture"]])

# %%
df_labels_cat = df_labels[df_labels.category == category]
df_labels_cat["has_keyword"] = 0
df_labels_cat.loc[df_labels_cat.doc_id.isin(dff.doc_id.to_list()), "has_keyword"] = 1

# %%
df_check = df_labels_cat.groupby(["source", "sentiment", "has_keyword"]).count()
df_check

# %%
# cats_to_numbers = dict(zip(cats, range(len(cats))))

# %%
# selected_docs = [full_corpus[doc_df[doc_df.doc_id==doc_id].index[0]] for doc_id in df.doc_id.to_list()]

# %%
# full_corpus_dict = dict(zip(full_corpus_ids, full_corpus))

# %%
# x=[]
# for doc_id in df.doc_id.to_list():
#     if doc_id in full_corpus_dict:
#         x.append(full_corpus_dict[doc_id])
#     else:
#         x.append([''])
#         print(doc_id)

# %%
# df = selected_docs_df_all.copy()
# df['category_n'] = df.category.apply(lambda x: cats_to_numbers[x])
# iss_topics.topic_keywords(x, df.category_n.to_list(), topic_words='bigrams')

# %%
# print(df_labels_cat[(df_labels_cat.sentiment=='Discard')&(df_labels_cat.has_keyword==1)].sample().text.iloc[0])


# %%
# dff = get_docs_with_keyphrases([[' hydrogen '], ['fuel cell'], ['sofc']])
# dff = get_docs_with_keyphrases([[' hydrogen '], ['fuel cell'], ['sofc']])
# dff = get_docs_with_keyphrases([['ccs'],
#                                 ['carbon storage'],
#                                 ['carbon capture'],
#                                 ['co2 capture'],
#                                 ['post combustion capture'],
#                                 ['carbon abatement'],
#                                 ['flue gas']
#                                ])
# dff = get_docs_with_keyphrases([['wind', 'energy'],
#                                 ['wind', 'renewable'],
#                                 ['wind turbine'],
#                                 ['wind farm'],
#                                 ['wind generator'],
#                                 ['offshore wind'],
#                                 ['onshore wind'],
#                                 ['wind power'],
#                                 ['wave', 'renewable'],
#                                 ['wave', 'energy'],
#                                 ['wave power'],
#                                 ['tidal', 'energy'],
#                                 ['tidal', 'turbine'],
#                                 ['marine', 'renewable'],
#                                 ['marine' , 'energy'],
#                                 ['offshore energy'],
#                                ])
# dff = get_docs_with_keyphrases([['nuclear'],
#                                 ['radioactive', 'waste'],
#                                 ['fusion', 'power'],
#                                 ['fusion', 'reactor'],
#                                 ['tokamak']
#                                ])
# dff = get_docs_with_keyphrases([['electric vehicle'],
# #                                 [' ev '],
# #                                 ['electric', 'vehicle'],
#                                ])

# %%
len(df)

# %%
df_cat = df[df.category == category]
df_cat["has_keyword"] = 0
df_cat.loc[df_cat.doc_id.isin(dff.doc_id.to_list()), "has_keyword"] = 1

# %%
df_cat.has_keyword.sum()

# %%
df_check = df_cat.groupby(
    list(reversed(["selection", "source", "has_keyword"]))
).count()
df_check

# %%
# full_corpus_dict['372CD65D-7E86-4E64-9630-03843144D797']
i = 0

# %%
i += 1
print(
    df_cat[(df_cat.selection == "top2vec") & (df_cat.has_keyword != 1)]
    .sort_values("topic_prob", ascending=False)
    .iloc[i]
    .text
)


# %%
len(dff[dff.doc_id.isin(df_cat.doc_id.to_list()) == False])

# %%
dff[dff.doc_id.isin(df_cat.doc_id.to_list()) == False].sample()[
    ["title", "description"]
].iloc[0]

# %%
# df_cat[df_cat.title=='Lithium Innovations for Future Electric vehicles (LIFE)']

# %%
words, word_scores = top2vec_model.similar_words(
    keywords=["sofc"], keywords_neg=[], num_words=25
)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")

# %% [markdown]
# # Keyword+

# %%
# df_manual_labels = pd.read_csv('/Users/karliskanders/Downloads/project-7-at-2021-08-20-11-17-039f3681.csv')
df_manual_labels = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-7-at-2021-08-20-11-17-039f3681.csv"
)

# %%
ref_keywords = {
    "Carbon Capture & Storage": [
        [" ccs "],
        ["carbon storage"],
        ["carbon capture"],
        ["co2 capture"],
        ["post combustion capture"],
        ["carbon abatement"],
        ["flue gas"],
    ],
    "Bioenergy": [
        ["bioenergy"],
        ["biofuel"],
        ["syngas"],
        ["biogas"],
        ["biochar"],
        ["biodiesel"],
        ["biomass", "energy"],
        ["biomass", "power"],
        ["anaerobic digestion"],
        ["torrefaction"],
        ["pyrolisis"],
        ["ethanol", "fuel"],
        ["butanol", "fuel"],
    ],
    "Nuclear": [
        ["nuclear"],
        ["radioactive", "waste"],
        ["fusion", "power"],
        ["fusion", "reactor"],
        ["tokamak"],
    ],
    "Hydrogen & Fuel Cells": [[" hydrogen "], ["fuel cell"], ["sofc"]],
    "Wind & Offshore": [
        ["wind", "energy"],
        ["wind", "renewable"],
        ["wind turbine"],
        ["wind farm"],
        ["wind generator"],
        ["offshore wind"],
        ["onshore wind"],
        ["wind power"],
        ["wave", "renewable"],
        ["wave", "energy"],
        ["wave power"],
        ["tidal", "energy"],
        ["tidal", "turbine"],
        ["marine", "renewable"],
        ["marine", "energy"],
        ["offshore energy"],
    ],
    "EV": [
        ["electric vehicle"],
        [" ev "],
        ["electric", "vehicle"],
    ],
    "Heating & Buildings": [
        ["infrared", "heating"],
        ["boiler"],
        ["heat"],
    ],
}

# %% [markdown]
# - Check selected documents that don't have the keywords
# - Check unselected documents that have (perhaps a narrower version) of the keywords

# %%
# category = 'Wind & Offshore'
# dff = get_docs_with_keyphrases(ref_keywords[category])
# df_cat = selected_docs_df_all[selected_docs_df_all.category==category]
# df_cat['has_keyword'] = 0
# df_cat.loc[df_cat.doc_id.isin(dff.doc_id.to_list()), 'has_keyword'] = 1

# %%
# len(dff), len(df_cat), df_cat.has_keyword.sum()

# %%
# i+=1
# # print(df_cat[(df_cat.selection=='top2vec') & (df_cat.has_keyword==1)].sort_values('topic_prob', ascending=False).iloc[i].text)


# %% [markdown]
# # Unselected documents that have the keywords

# %%
# df_unselected_key = dff[dff.doc_id.isin(df_cat.doc_id.to_list())==False]

# %%
narrow_ref_keywords = {
    "Batteries": [["battery"], ["batteries"]],
    "Solar": [
        ["solar energy"],
        ["solar", "renewable"],
        ["solar cell"],
        ["solar panel"],
        ["photovoltaic"],
        ["perovskite"],
        ["pv cell"],
    ],
    "Carbon Capture & Storage": [
        [" ccs "],
        ["carbon storage"],
        ["carbon capture"],
        ["co2 capture"],
        ["carbon abatement"],
    ],
    "Bioenergy": [
        ["biofuel"],
        ["ethanol", "fuel"],
        ["butanol", "fuel"],
        ["syngas"],
        ["biogas"],
        ["biochar"],
        ["biodiesel"],
        ["torrefaction", "biomass"],
        ["pyrolisis", "biomass"],
    ],
    "Nuclear": [
        ["nuclear", "energy"],
        ["radioactive", "waste"],
        ["nuclear", "fission"],
        ["fission", "reactor"],
        ["nuclear", "fusion"],
        ["fusion", "reactor"],
        ["tokamak"],
    ],
    "Hydrogen & Fuel Cells": [[" hydrogen "], ["fuel cell"], ["sofc"]],
    "Wind & Offshore": [
        ["wind energy"],
        ["wind", "renewable"],
        ["wind turbine"],
        ["wind farm"],
        ["wind generator"],
        ["wind power"],
        ["wave", "renewable"],
        ["tidal energy"],
        ["tidal turbine"],
        ["offshore energy"],
        ["offshore wind"],
        ["onshore wind"],
    ],
    "EV": [
        ["electric vehicle"],
        [" ev "],
        ["electric", "vehicle"],
    ],
    "Heating (all other)": [
        ["infrared", "heating"],
        ["heat", "build"],
        ["heat", "home"],
        ["heat", "hous"],
        ["heat", "domestic"],
        ["heat", "renewable"],
        ["heating"],
    ],
    "Demand management": [
        [" vpp "],
        ["virtual power plant"],
        ["smart building"],
        ["smart home"],
        ["smart meter"],
        ["smart grid"],
        ["demand response"],
    ],
}

# %%
narrow_topic_probs = np.load(
    PROJECT_DIR / "outputs/data/results_august/narrow_topics_full_corpus.npy"
)

# %%
SMALL_THRESH = 0.01

# %%
categories_to_narrow_topics

# %% [markdown]
# ## Solar: false negative check

# %%
# NB: This is for selecting the new docs to check
manual_cb_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-11-at-2021-08-24-17-28-6e9a63e0.csv"
)

# %%
DF_NEW = {}

# %%
category = "Solar"
categories_to_narrow_topics[category]

# %%
category_reviewed = df_manual_labels[df_manual_labels.category == category]
category_reviewed_hits = category_reviewed[category_reviewed.sentiment == "Keep"]
category_reviewed_misses = category_reviewed[category_reviewed.sentiment == "Discard"]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 2]
# All documents with the keywords
df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))
# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
len(df_selected), df_selected.has_keyword.sum(), len(df_unselected_with_keywords)

# %%
# category_reviewed_hits['doc_id']
pd.set_option("max_colwidth", 140)

# %%
df_ = category_reviewed_hits[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 1
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)

df_new_cb[df_new_cb.doc_id.isin(manual_cb_review.doc_id.to_list()) == False].to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Sep1_cb_{category}.csv",
    index=False,
)

DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]

# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %%
len(df_), len(df_new), len(df_new_cb)

# %% [markdown]
# ## Wind: false positive and negative check

# %%
category = "Wind & Offshore"
categories_to_narrow_topics[category]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 67]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])

df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))

# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = df_selected[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 0
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)

df_new_cb[df_new_cb.doc_id.isin(manual_cb_review.doc_id.to_list()) == False].to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Sep1_cb_{category}.csv",
    index=False,
)

DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]

# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %%
len(DF_NEW[category])

# %%
len(df_new_cb)

# %%
len(df_selected), df_selected.has_keyword.sum(), len(df_unselected_with_keywords)

# %%
len(df_unselected_with_keywords[df_unselected_with_keywords.topic_probs > 0.05])

# %% [markdown]
# ## Batteries: false negative check

# %%
category = "Batteries"
categories_to_narrow_topics[category]

# %%
category_reviewed = df_manual_labels[df_manual_labels.category == category]
category_reviewed_hits = category_reviewed[category_reviewed.sentiment == "Keep"]
category_reviewed_misses = category_reviewed[category_reviewed.sentiment == "Discard"]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 40]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])

df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))

# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = category_reviewed_hits[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 1
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)
DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]

# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %%
len(df_selected), df_selected.has_keyword.sum(), len(df_unselected_with_keywords)

# %%
len(df_unselected_with_keywords[df_unselected_with_keywords.topic_probs > 0.05])

# %%
# df_unselected_with_keywords[df_unselected_with_keywords.topic_probs>0.01]

# %%

# %% [markdown]
# ## Hydrogen: false positive and negative check

# %%
category = "Hydrogen & Fuel Cells"
categories_to_narrow_topics[category]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs_1"] = narrow_topic_probs[:, 94]
doc_df_["topic_probs_2"] = narrow_topic_probs[:, 130]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))

# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = df_selected[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 0
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs_1,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)

df_new_cb[df_new_cb.doc_id.isin(manual_cb_review.doc_id.to_list()) == False].to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Sep1_cb_{category}.csv",
    index=False,
)

DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]


# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %%
len(df_selected), df_selected.has_keyword.sum(), len(df_unselected_with_keywords)

# %%
# len(df_unselected_with_keywords[
#     (df_unselected_with_keywords.topic_probs_1>0.05) |
#     (df_unselected_with_keywords.topic_probs_2>0.05)
# ])

# %%
# DF_NEW.keys()

# %%
# iss_io.save_pickle(DF_NEW, PROJECT_DIR / 'outputs/data/results_august/reference')

# %%
# DF_NEW

# %%
cats

# %% [markdown]
# ## Bioenergy: false positive and negative check

# %%
category = "Bioenergy"
categories_to_narrow_topics[category]

# %%
category_reviewed = df_manual_labels[df_manual_labels.category == category]
category_reviewed_hits = category_reviewed[category_reviewed.sentiment == "Keep"]
category_reviewed_misses = category_reviewed[category_reviewed.sentiment == "Discard"]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs_1"] = narrow_topic_probs[:, 68]
doc_df_["topic_probs_2"] = narrow_topic_probs[:, 92]
doc_df_["topic_probs_3"] = narrow_topic_probs[:, 93]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])

df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))

# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[
    selected_docs_df_all.doc_id.isin(category_reviewed.doc_id.to_list())
]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = category_reviewed_hits[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 1
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs_1,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)


DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]


# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %% [markdown]
# ### Carbon capture

# %%
category = "Carbon Capture & Storage"
categories_to_narrow_topics[category]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 39]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])

df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))
# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = df_selected[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 0
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
df_new_cb = df_[(df_.source == "cb") & (df_.manual_ok == 0)]
df_new_cb.to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Aug23_cb_{category}.csv",
    index=False,
)

df_new_cb[df_new_cb.doc_id.isin(manual_cb_review.doc_id.to_list()) == False].to_csv(
    PROJECT_DIR / f"outputs/data/results_august/aux/review/Sep1_cb_{category}.csv",
    index=False,
)

DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]


# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %% [markdown]
# ### Heating

# %%
CB_DOCS_ALL = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_CB.csv"
)
GTR_DOCS_ALL = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech_GTR.csv"
)

# %%
# CB_DOCS_ALL[CB_DOCS_ALL.title.str.contains('Muovitech')]

# %%
doc_id_checked = GTR_DOCS_ALL.doc_id.to_list() + CB_DOCS_ALL.doc_id.to_list()

# %%
len(doc_id_checked)

# %%
category = "Heating (all other)"

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 19]
# All documents with the keywords
# df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords_old = get_docs_with_keyphrases(narrow_ref_keywords[category])
df_with_keywords = get_docs_with_keyphrases_2(narrow_ref_keywords[category])
print(len(df_with_keywords_old), len(df_with_keywords))
# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.doc_id.isin(doc_id_checked)]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df_ = df_selected[["doc_id", "title", "description", "source"]]
df_ = df_.merge(doc_df_)
df_["manual_ok"] = 1
df_new = df_unselected_with_keywords
df_new["manual_ok"] = 0
df_ = df_.append(df_new, ignore_index=True)
df_ = df_.merge(clustering[["doc_id", "cluster_keywords"]])
df_["category"] = category

data_for_labelling = []
for i, row in df_.iterrows():
    s = f"CATEGORY: {category}\n\n"
    s += f"doc_id: {row.doc_id}\n\nprobability: {np.round(row.topic_probs,4)}\n\n{row.title}\n------\n{row.description}\n------"
    s += f"\n\ntop2vec keywords: {row.cluster_keywords}"
    data_for_labelling.append(s)
df_["text"] = data_for_labelling

# %%
# df_new_cb = df_[(df_.source=='cb') & (df_.manual_ok==0)]
# df_new_cb.to_csv(PROJECT_DIR / f'outputs/data/results_august/aux/review/Aug23_cb_{category}.csv', index=False)
DF_NEW[category] = df_[
    (df_.source == "gtr") | ((df_.source == "cb") & (df_.manual_ok == 1))
]


# %%
print(len(DF_NEW[category]))
DF_NEW[category] = df_[
    (
        (df_.source == "gtr")
        & ((df_.manual_ok == 1) | df_.doc_id.isin(df_with_keywords.doc_id.to_list()))
    )
    | ((df_.source == "cb") & (df_.manual_ok == 1))
]
len(DF_NEW[category])

# %% [markdown]
# ## Combine and save

# %%
DF_REF = DF_NEW.copy()

# %%
DF_REF[cat].head(1)

# %%
manual_cb_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-11-at-2021-08-24-17-28-6e9a63e0.csv"
)
manual_cb_review = pd.read_csv(
    PROJECT_DIR
    / "outputs/data/results_august/aux/project-11-at-2021-09-01-10-01-24e5dd2e.csv"
)

# %%
cat = "Wind & Offshore"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]
DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)

# %%
DF_REF[cat].source.unique()

# %%
cat = "Solar"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]
DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)

# %%
cat = "Batteries"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]
DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)

# %%
DF_REF["Batteries"].head(1)

# %%
cat = "Hydrogen & Fuel Cells"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs_1",
        "topic_probs_2",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]
DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)


# %%
DF_REF[cat]["topic_probs"] = 0
DF_REF[cat] = DF_REF[cat].reset_index(drop=True)
for i, row in DF_REF[cat].iterrows():
    DF_REF[cat].loc[i, "topic_probs"] = max(row["topic_probs_1"], row["topic_probs_2"])

# %%
DF_REF[cat] = DF_REF[cat].drop(["topic_probs_1", "topic_probs_2"], axis=1)

# %%
DF_REF[cat] = DF_REF[cat][
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]

# %%
DF_REF[cat].head(1)

# %%
cat = "Bioenergy"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs_1",
        "topic_probs_2",
        "topic_probs_3",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]

DF_REF[cat]["topic_probs"] = 0
DF_REF[cat] = DF_REF[cat].reset_index(drop=True)
for i, row in DF_REF[cat].iterrows():
    DF_REF[cat].loc[i, "topic_probs"] = max(
        row["topic_probs_1"], row["topic_probs_2"], row["topic_probs_3"]
    )
DF_REF[cat] = DF_REF[cat].drop(
    ["topic_probs_1", "topic_probs_2", "topic_probs_3"], axis=1
)
DF_REF[cat] = DF_REF[cat][
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]

DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)

# %%
cat = "Carbon Capture & Storage"
df = manual_cb_review[
    (manual_cb_review.category == cat) & (manual_cb_review.sentiment == "Keep")
]
df["manual_ok"] = 1
df = df[
    [
        "doc_id",
        "title",
        "description",
        "source",
        "topic_probs",
        "manual_ok",
        "cluster_keywords",
        "text",
    ]
]
DF_REF[cat] = DF_REF[cat].append(df, ignore_index=True)

# %%
iss_io.save_pickle(
    DF_REF, PROJECT_DIR / "outputs/data/results_august/reference_category_data_2.p"
)

# %% [markdown]
# # Fuel poverty & social

# %%
GTR_DOCS = pd.read_csv(PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_GTR.csv")
CB_DOCS = pd.read_csv(PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_CB.csv")

# %% [markdown]
# ## Keywords

# %%
keywords_fuel = [["fuel poverty"], ["fuel poor"]]

keywords_energy = [["energy poverty"], ["energy poor"]]

keywords_social = [["social hous"], ["social landlord"]]

# %%
df_with_keywords = get_docs_with_keyphrases_2(keywords_fuel)
len(df_with_keywords)

# %%
# df_with_keywords[df_with_keywords.doc_id.isin(df_with_keywords_.doc_id.to_list())]

# %%
df_with_keywords = get_docs_with_keyphrases_2(keywords_energy)
len(df_with_keywords)

# %%
# df_with_keywords.to_csv(PROJECT_DIR / 'outputs/data/results_august/fuel_poverty_docs.csv', index=False)

# %%
df_fp = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/fuel_poverty_docs_checked.csv"
)
df_fp = df_fp[df_fp.hit != 0]

# %%
df_lch_eem_gtr = GTR_DOCS[GTR_DOCS.tech_category.isin(["LCH & EEM"])]
fp_gtr = df_lch_eem_gtr[df_lch_eem_gtr.doc_id.isin(df_fp.doc_id.to_list())]

# %%
len(fp_gtr) / len(df_lch_eem_gtr)

# %%
GTR_DOCS[GTR_DOCS.doc_id.isin(df_fp.doc_id.to_list())].groupby(
    "tech_category"
).count().sort_values("doc_id")

# %%
7 / len(fp_gtr)

# %%
5 / len(fp_gtr)

# %%
df_lch_eem_cb = CB_DOCS[CB_DOCS.tech_category.isin(["LCH & EEM"])]
fp_cb = df_lch_eem_cb[df_lch_eem_cb.doc_id.isin(df_fp.doc_id.to_list())]

# %%
len(fp_cb) / len(df_lch_eem_cb)

# %%
fp_cb

# %%
df_with_keywords

# %%
## Topics

# %% [markdown]
# # Enabling technologies

# %%
doc_topic_dists = get_doc_topic_dists(mdl)
doc_topic_dists.shape

# %%
GTR_DOCS_ALL = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/checked_heating_tech.csv"
)

# %%
# df = GTR_DOCS_ALL[GTR_DOCS_ALL.tech_category=='Building insulation']
df = GTR_DOCS_ALL
doc_df_index = doc_df[["doc_id"]].reset_index().rename(columns={"index": "i"})
df = df.merge(doc_df_index, how="left")
df.head(3)

# %%
doc_topic_dists.shape

# %%
df["year"] = df["start"].apply(iss.convert_date_to_year)

# %%

# %%
doc_df_prob = doc_df[["doc_id"]]
doc_df_prob["prob"] = np.max(
    doc_topic_dists[:, [10, 17, 18, 85, 89, 129, 145]], axis=1
)  # doc_topic_dists[:, 17]
df_t = df.merge(doc_df_prob).sort_values("prob")

alt.Chart(df_t[df_t.tech_category == "Geothermal energy"]).mark_circle(size=18).encode(
    x="year:O",
    y=alt.X("prob", scale=alt.Scale(type="log")),
    color="tech_category",
    tooltip=["tech_category", "title", "description"],
).interactive()

# %%
plt.scatter(df_t.year, df_t.prob)
plt.y_scale("log")

# %%
# sorted(df.year.unique())

# %%
year_dists = []
years = sorted(df.year.unique())
for year in years:
    df_y = df[df.year == year]
    doc_topic_dists_y = doc_topic_dists[df_y.i.to_list(), :]
    doc_topic_dists_y_t = doc_topic_dists_y[:, 18]
    year_dists.append(doc_topic_dists_y_t)

# %%
top_topic_pctile = [np.percentile(y, 50) for y in year_dists]

# %%
sns.lineplot(x=years, y=top_topic_pctile)

# %%
# Check topic intensity

# %%
# doc_topic_dists[0,:]

# %% [markdown]
# # Estimate accuracy

# %%
df_manual_labels = pd.read_csv(
    "/Users/karliskanders/Downloads/project-7-at-2021-08-20-11-17-039f3681.csv"
)

# %%
category = "Solar"
categories_to_narrow_topics[category]

# %%
# Narrow topic probability
doc_df_ = doc_df[["doc_id"]].copy()
doc_df_["topic_probs"] = narrow_topic_probs[:, 2]
# All documents with the keywords
df_with_keywords = get_docs_with_keyphrases(narrow_ref_keywords[category])
# Above-threshold, selected documents that have the keyword
df_selected = selected_docs_df_all[selected_docs_df_all.category == category]
df_selected["has_keyword"] = 0
df_selected.loc[
    df_selected.doc_id.isin(df_with_keywords.doc_id.to_list()), "has_keyword"
] = 1
# Documents with keywords that were not selected
df_unselected_with_keywords = df_with_keywords[
    df_with_keywords.doc_id.isin(df_selected.doc_id.to_list()) == False
]
df_unselected_with_keywords = df_unselected_with_keywords.merge(doc_df_, how="left")

# %%
df = df_manual_labels[
    (df_manual_labels.category == category) & (df_manual_labels.source == "gtr")
].merge(doc_df_)
sns.displot(df[df.sentiment == "Keep"].topic_probs.to_list())
sns.displot(df[df.sentiment == "Discard"].topic_probs.to_list())
plt.show()

# %%
# df[df.sentiment=='Keep'].sort_values('topic_probs').head(15)

# %%
