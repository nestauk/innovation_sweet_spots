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
# # Guided LDA with tomotopy

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


def make_pyLDAvis(mdl, fpath=PROJECT_DIR / "outputs/data/"):
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
    pyLDAvis.save_html(prepared_data, str(fpath / "ldavis_tomotopy.html"))


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
# topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
# doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
# doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
# doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
# vocab = list(mdl.used_vocabs)
# term_frequency = mdl.used_vocab_freq

# prepared_data = pyLDAvis.prepare(
#     topic_term_dists,
#     doc_topic_dists,
#     doc_lengths,
#     vocab,
#     term_frequency,
#     start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
#     sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
# )
# pyLDAvis.save_html(prepared_data, 'ldavis_tomotopy.html')

# %% [markdown]
# ## Inspect

# %%
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.topic_analysis as iss_topics

# %%
import importlib

importlib.reload(iss_topics)

# %%
# doc_df = pd.DataFrame(data={'doc_id': full_corpus_ids})
# doc_df = iss_topics.get_doc_details(doc_df)

# %%
doc_topic_dists = get_doc_topic_dists(mdl)
doc_topic_dists.shape

# %%
vocab = list(mdl.used_vocabs)
topic_term_dists = get_topic_term_dists(mdl)

# %%
is_present = iss.is_term_present_in_sentences("hydrogen", full_corpus, 2)
doc_df[is_present]

# %%

# %%
term = "digital_twin"
terms = [(i, v) for i, v in enumerate(vocab) if term in v]

# %%
terms

# %%
# i = 0
# np.flip(np.argsort(topic_term_dists[:, terms[i][0]]))

# %%
# np.flip(np.sort(topic_term_dists[:, terms[i][0]]))

# %%
top_topics = np.argsort(doc_topic_dists[np.where(is_present)[0], :], axis=1)[:, -1]

# %%
# df = doc_df[is_present].copy()
# df['top_topic'] = top_topics
# df.sort_values('top_topic').head(50)

# %%

# np.sum(x > np.mean(x)+5*np.std(x))

# %%
x = doc_topic_dists[:, 59] * doc_topic_dists[:, 120]
pd.set_option("max_colwidth", 200)
doc_df["prob"] = x
doc_df[doc_df.source == "gtr"].sort_values("prob", ascending=False).head(50)

# %%
# Combine docs
# Check top docs
# Check their categories

# %%
doc_topic_dists[:, 63].argmax()

# %%
# full_corpus[doc_topic_dists[:, 24].argmax()]

# %%

# %%
