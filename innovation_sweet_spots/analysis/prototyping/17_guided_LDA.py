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
import numpy as np
import guidedlda

# %%
X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)

# %%
vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)

# %%
word2id = dict((v, idx) for idx, v in enumerate(vocab))

# %%
X.shape

# %%
X.sum()

# %%
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
model.fit(X)

# %%
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][: -(n_top_words + 1) : -1]
    print("Topic {}: {}".format(i, " ".join(topic_words)))

# %%
# Guided LDA with seed topics.
seed_topic_list = [
    ["game", "team", "win", "player", "season", "second", "victory"],
    ["percent", "company", "market", "price", "sell", "business", "stock", "share"],
    ["music", "write", "art", "book", "world", "film"],
    [
        "political",
        "government",
        "leader",
        "official",
        "state",
        "country",
        "american",
        "case",
        "law",
        "police",
        "charge",
        "officer",
        "kill",
        "arrest",
        "lawyer",
    ],
]


# %%
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

# %%
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

# %%
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][: -(n_top_words + 1) : -1]
    print("Topic {}: {}".format(i, " ".join(topic_words)))

# %%
doc_topic = model.transform(X)
for i in range(9):
    print(
        "top topic: {} Document: {}".format(
            doc_topic[i].argmax(),
            ", ".join(np.array(vocab)[list(reversed(X[i, :].argsort()))[0:5]]),
        )
    )

# %%
