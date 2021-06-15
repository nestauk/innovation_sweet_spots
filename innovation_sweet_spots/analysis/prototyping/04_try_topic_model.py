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
import scipy.sparse as ss
from corextopic import corextopic as ct

# Define a matrix where rows are samples (docs) and columns are features (words)
X = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=int)
# Sparse matrices are also supported
X = ss.csr_matrix(X)
# Word labels for each column can be provided to the model
words = ["dog", "cat", "fish", "apple", "orange"]
# Document labels for each row can be provided
docs = ["fruit doc", "animal doc", "mixed doc"]

# Train the CorEx topic model
topic_model = ct.Corex(
    n_hidden=2
)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=words, docs=docs)

# %%
topics = topic_model.get_topics()
for topic_n, topic in enumerate(topics):
    # w: word, mi: mutual information, s: sign
    topic = [(w, mi, s) if s > 0 else ("~" + w, mi, s) for w, mi, s in topic]
    # Unpack the info about the topic
    words, mis, signs = zip(*topic)
    # Print topic
    topic_str = str(topic_n + 1) + ": " + ", ".join(words)
    print(topic_str)

# %%
top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs, probs = zip(*topic_docs)
    topic_str = str(topic_n + 1) + ": " + ", ".join(docs)
    print(topic_str)

# %%
