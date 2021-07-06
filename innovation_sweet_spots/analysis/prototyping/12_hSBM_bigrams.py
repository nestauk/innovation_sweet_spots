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
import pickle
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.utils.io as iss_io
import innovation_sweet_spots.pipeline.topic_modelling as iss_topics

# %%
from innovation_sweet_spots.hSBM_Topicmodel.sbmtm import sbmtm
import graph_tool.all as gt

# %%
corpus_gtr = pickle.load(
    open(PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p", "rb")
)

# %%
len(corpus_gtr)

# %%
# corpus_gtr[0]

# %%
gtr_ids = iss_io.read_list_of_terms(
    PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised_ids.txt"
)

# %%
import importlib

importlib.reload(iss_topics)

# %%
gt.seed_rng(32)
topic_model = iss_topics.train_model(corpus_gtr, gtr_ids)

# %%
import pickle

pickle.dump(topic_model, open("topic_model_bigrams.p", "wb"))

# %%
topic_model.plot()

# %%
topic_df = iss_topics.post_process_model(topic_model, 2)

# %%
len(topic_df)

# %%
for p in topic_df.columns:
    print(p)

# %%

# %%

# %%
p_td_d, p_tw_w = topic_model.group_membership(l=1)

# %%
import matplotlib.pyplot as plt

# %%
# plt.figure(figsize=(15,4))
# plt.subplot(121)
# plt.imshow(p_td_d,origin='lower',aspect='auto',interpolation='none')
# plt.title(r'Document group membership $P(bd | d)$')
# plt.xlabel('Document d (index)')
# plt.ylabel('Document group, bd')
# plt.colorbar()

# plt.subplot(122)
# plt.imshow(p_tw_w,origin='lower',aspect='auto',interpolation='none')
# plt.title(r'Word group membership $P(bw | w)$')
# plt.xlabel('Word w (index)')
# plt.ylabel('Word group, bw')
# plt.colorbar()

# %%
# topic_model.topics(l=1, n=20)
# len(topic_model.topics(l=2))

# %%
topic_model.clusters(l=1, n=5)

# %%
