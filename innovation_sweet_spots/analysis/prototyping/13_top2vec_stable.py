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
# # top2vec approach for categorising research/innovations

# %%
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.Top2Vec.Top2Vec as top2vec
import innovation_sweet_spots.utils.text_pre_processing as iss_preproc
import innovation_sweet_spots.analysis.text_analysis as iss_text_analysis
import innovation_sweet_spots.analysis.green_document_utils as iss_green
import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
import importlib

importlib.reload(iss_io)

# %%
# iss_io.load_pickle('test.p')

# %%
import pickle
import numpy as np
import umap
import pandas as pd
import altair as alt

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %% [markdown]
# ## Fetch 'green' GTR projects and CB companies

# %%
green_keywords = iss_green.get_green_keywords(clean=True)
green_projects = iss_green.find_green_gtr_projects(green_keywords)
green_companies = iss_green.find_green_cb_companies()

# %%
green_project_texts = iss.create_documents_from_dataframe(
    green_projects,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)
green_company_texts = iss.create_documents_from_dataframe(
    green_companies,
    columns=["short_description", "long_description"],
    preprocessor=(lambda x: x),
)
green_texts = green_project_texts + green_company_texts

# %% [markdown]
# ## Preprocessing: Create a bigrammer

# %%
corpus, ngram_phraser = iss_preproc.pre_process_corpus(green_texts)

# %%
corpus_gtr = corpus[0 : len(green_project_texts)]
corpus_cb = corpus[len(green_project_texts) :]

# %%
fpath = PROJECT_DIR / "outputs/models/bigram_phraser_gtr_cb_v1.p"

# %%
pickle.dump(ngram_phraser, open(fpath, "wb"))
pickle.dump(
    corpus_gtr, open(PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p", "wb")
)
pickle.dump(
    corpus_cb, open(PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p", "wb")
)

# %% [markdown]
# ## Testing the tokeniser

# %%
ngram_phraser_load = pickle.load(open(fpath, "rb"))

# %%
nlp = iss_text_analysis.setup_spacy_model(iss_preproc.DEF_LANGUAGE_MODEL)


def bigrammer(text):
    return iss_preproc.ngrammer(text, ngram_phraser_load, nlp)


# %%
example_text = "Climate change is addressed by sustainable energy, heat pumps and behavioural change"
bigrammer(example_text)

# %% [markdown]
# ## Top2Vec
#
# Note: I'm using the top2vec library as a foundation, and making changes to it.

# %%
corpus_cb = pickle.load(
    open(PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p", "rb")
)
corpus_gtr = pickle.load(
    open(PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p", "rb")
)

# %%
# rand_i = np.random.permutation(len(green_project_texts))[0:1000]
# sample_texts = [green_project_texts[i] for i in rand_i]
# sample_texts = [corpus_gtr[i] for i in rand_i]

# %%
sample_texts = corpus_gtr

# %%
umap_args = {
    "n_neighbors": 15,
    "n_components": 5,
    "metric": "cosine",
    "random_state": 42,
}

hdbscan_args = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
}

doc2vec_args = {
    "vector_size": 300,
    "min_count": 10,
    "window": 15,
    "sample": 1e-5,
    "negative": 5,
    "hs": 0,
    "epochs": 1,
    "dm": 0,
    "dbow_words": 1,
    "seed": 123,
}

# %%
import importlib

importlib.reload(top2vec)
Top2Vec = top2vec.Top2Vec

# %%
model = Top2Vec(
    documents=sample_texts,
    speed="test-learn",
    #     tokenizer=bigrammer,
    tokenizer="preprocessed",
    workers=1,
    doc2vec_args=doc2vec_args,
    umap_args=umap_args,
    hdbscan_args=hdbscan_args,
    random_state=111,
    verbose=False,
)


# %%
# pickle.dump(model, open('model.p', 'wb'))

# %%
model_old = pickle.load(open("model.p", "rb"))

# %%
model.model = model_old.model

# %%
model.generate_umap_model()
model.hdbscan_args["min_cluster_size"] = 15
model.hdbscan_args["min_samples"] = None
model.hdbscan_args["cluster_selection_method"] = "eom"
model.cluster_docs()
model.process_topics()

# %%
model.get_num_topics()

# %%
topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())

# %%
topic_words

# %% [markdown]
# ## Inspect the clusters

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# %%
# Create large "cluster documents" for finding best topic words based on tf-idf scores
document_cluster_memberships = model.doc_top
cluster_ids = sorted(np.unique(document_cluster_memberships))
cluster_docs = {i: [] for i in cluster_ids}
for i, clust in enumerate(document_cluster_memberships):
    cluster_docs[clust] += sample_texts[i]


# %%
vectorizer = CountVectorizer(
    analyzer="word", tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None
)
X = vectorizer.fit_transform(list(cluster_docs.values()))

# %%
n = 10

id_to_token = dict(
    zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
)

clust_words = []
# for i in range(X.shape[0]):
#     x = X[i,:].todense()
#     best_i = np.array(np.flip(np.argsort(x)))[0]
#     top_n = best_i[0:n]
#     words = [id_to_token[t] for t in top_n]
#     clust_words.append(words)

reranked_topic_words = []
for i in range(X.shape[0]):
    x = X[i, :].todense()
    topic_word_counts = [
        X[i, vectorizer.vocabulary_[token]] for token in topic_words[i]
    ]
    best_i = np.flip(np.argsort(topic_word_counts))
    top_n = best_i[0:n]
    words = [topic_words[i][t] for t in top_n]
    clust_words.append(words)

# %%
# i=0
# i += 1
i = 6

print(clust_words[i])
print(topic_words[i][0:10])

# %%
clust = i
clust_proj_ids = np.where(model.doc_top == clust)[0]
# Rerank based on clustering
clust_proj_ids = clust_proj_ids[
    np.flip(np.argsort(model.cluster.probabilities_[clust_proj_ids]))
]

# %%
green_projects.iloc[clust_proj_ids].title.to_list()[0:10]

# %%
green_projects.iloc[clust_proj_ids].title.to_list()[-10:]

# %%
# clust_words[0]

# %%
# topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["heat_pump"], num_topics=5)
# for topic in topic_nums:
#     model.generate_topic_wordcloud(topic)

# %% [markdown]
# ## Visualisation

# %%
umap_args_plotting = {
    "n_neighbors": 15,
    "n_components": 2,  # 5 -> 2 for plotting
    "metric": "cosine",
    "random_state": 111,
}
umap_model = umap.UMAP(**umap_args_plotting).fit(
    model._get_document_vectors(norm=False)
)
xy = umap_model.transform(
    model._get_document_vectors(norm=False)[0 : len(green_project_texts)]
)

# %%
xy = umap_model.transform(model._get_document_vectors(norm=False))

# %%
df = green_projects.copy()
df["x"] = xy[:, 0]
df["y"] = xy[:, 1]
df["cluster"] = [str(c) for c in model.doc_top[0 : len(green_project_texts)]]

# %%
topic_labels = ["  ".join(list(t)) for t in clust_words]
clust_labels = pd.DataFrame(
    data={
        "cluster": [str(c) for c in list(range(len(topic_labels)))],
        "topic_label": topic_labels,
    }
)

# %%
# topic_labels

# %%
df_ = df.merge(clust_labels, on="cluster", how="left")

# %%
alt.data_transformers.disable_max_rows()
fig = (
    alt.Chart(
        pd.concat(
            [
                df_,
            ]
        ),
        width=750,
        height=750,
    )
    .mark_circle(size=25)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        #         size='size',
        #         color='cluster',
        color=alt.Color("cluster", scale=alt.Scale(scheme="category20")),
        tooltip=["title", "topic_label", "cluster"],
    )
    .interactive()
)

# %%
# fig

# %%
import json

clust_words_dict = [
    {"id": i, "terms": words[0:6]} for i, words in enumerate(clust_words)
]

# %%
json.dump(
    clust_words_dict,
    open(PROJECT_DIR / "outputs/gtr_green_project_cluster_words.json", "w"),
    indent=4,
)

# %%
# json.dump(clust_words_dict, open(PROJECT_DIR / 'outputs/gtr_green_project_cluster_words.json', 'w'), indent=4)

# %%
# json.load(open(PROJECT_DIR / 'outputs/gtr_green_project_cluster_words.json', 'r'))

# %%
clust_labels = json.load(
    open(PROJECT_DIR / "outputs/gtr_green_project_cluster_words_wiki_labels.json", "r")
)
clust_labels = [d["labels"] for d in clust_labels]

# %%
# clust_labels

# %%
# df_.groupby('cluster').agg(counts=('project_id', 'count')).sort_values('counts')

# %% [markdown]
# ## Check stats

# %%
import innovation_sweet_spots.getters.gtr as gtr

# %%
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
link_gtr_topics = gtr.get_link_table("gtr_topic")

gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(green_projects, gtr_project_funds)
del link_gtr_funds

# %%
project_to_org = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
plots = ["no_of_projects", "amount_total", "amount_median"]

# %%
clust_proj_ids[np.flip(np.argsort(model.cluster.probabilities_[clust_proj_ids]))]


# %%
def get_cluster_funding_level(clust):
    clust_proj_ids = np.where(model.doc_top[0 : len(green_project_texts)] == clust)[0]
    # Rerank based on clustering
    clust_proj_ids = clust_proj_ids[
        np.flip(np.argsort(model.cluster.probabilities_[clust_proj_ids]))
    ]
    df = funded_projects[
        funded_projects.project_id.isin(
            green_projects.iloc[clust_proj_ids].project_id.to_list()
        )
    ].copy()
    cluster_funding = iss.gtr_funding_per_year(df, min_year=2010)
    return cluster_funding


# %%
clust = 86
print(clust, clust_words[clust])
cluster_funding = get_cluster_funding_level(clust)

plt1 = iss.show_time_series_fancier(cluster_funding, y=plots[0], show_trend=False)
iss.nicer_axis(plt1)

# %%
iss.estimate_growth_level(cluster_funding)

# %%
clust_funding_growth = []
for clust in range(len(clust_words)):
    cluster_funding = get_cluster_funding_level(clust)
    clust_funding_growth.append(iss.estimate_growth_level(cluster_funding))

# %%
# iss.estimate_growth_level

# %%
clust_counts = df_.groupby("cluster").agg(counts=("project_id", "count")).reset_index()
clust_counts["cluster"] = clust_counts["cluster"].astype(int)
clust_counts = clust_counts.sort_values("cluster")

# %%
cluster_stats = pd.DataFrame(
    data={
        "clust_id": list(range(len(clust_words))),
        "n_projects": clust_counts["counts"].to_list(),
        "label": clust_labels,
        "keywords": clust_words,
        "funding_growth": clust_funding_growth,
    }
)

# %%
cluster_stats.sort_values("funding_growth")

# %%
cluster_stats.loc[3]

# %%
len(model.topic_vectors)

# %%
xy = umap_model.transform(model.topic_vectors)

# %%
cluster_stats["x"] = xy[:, 0]
cluster_stats["y"] = xy[:, 1]


# %%
def growth_indicator(x):
    indicator = "stable"
    if x > 20:
        indicator = "growth"
    if x > 90:
        indicator = "strong growth"
    if x < -20:
        indicator = "decline"
    return indicator


# %%
cluster_stats["funding_growth_indicator"] = cluster_stats["funding_growth"].copy()
cluster_stats["funding_growth_indicator"] = cluster_stats[
    "funding_growth_indicator"
].apply(lambda x: growth_indicator(x))

# %%
cluster_stats.head(1)

# %%
cluster_stats["top_keywords"] = cluster_stats["keywords"].apply(lambda x: x[0:3])

# %%
# cluster_stats

# %%
# df_amount = df_.merge(gtr_project_funds[['project_id', 'amount']], on='project_id', how='left')
# df_amount['log_amount'] = np.log(df_amount['amount']+1)

# %%
alt.data_transformers.disable_max_rows()
fig_projects = (
    alt.Chart(
        pd.concat(
            [
                df_,
            ]
        ),
        width=750,
        height=750,
    )
    .mark_circle(size=25, color="#bfbfbf", opacity=0.3)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        #         size='size',
        #         color=alt.Color('cluster',  scale=alt.Scale(scheme='category20')),
        tooltip=["title", "topic_label", "cluster"],
    )
    .interactive()
)

# %%
alt.data_transformers.disable_max_rows()
fig = (
    alt.Chart(
        pd.concat(
            [
                cluster_stats,
            ]
        ),
        width=1000,
        height=750,
    )
    .mark_circle(size=70)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        #         size='n_projects',
        text="top_keywords",
        #         color='cluster',
        color=alt.Color(
            "funding_growth_indicator",
            scale=alt.Scale(
                domain=["stable", "decline", "growth", "strong growth"],
                range=["#333333", "#eb6a00", "#b3b536", "#209905"],
            ),
        ),
        tooltip=["clust_id", "label", "keywords", "funding_growth", "n_projects"],
    )
)

text = (
    fig.mark_text(align="left", baseline="middle", dx=7, fontSize=10, fontWeight=300)
    .encode(text="top_keywords")
    .interactive()
)

# fig_projects + fig + text
# (fig_projects + fig+text)

# %%
clust = 22
print(clust, clust_words[clust])
cluster_funding = get_cluster_funding_level(clust)

plt1 = iss.show_time_series_fancier(cluster_funding, y=plots[1], show_trend=False)
iss.nicer_axis(plt1)

# %%
iss.estimate_growth_level(cluster_funding)

# %%
# driver = alt_save.google_chrome_driver_setup()

# %%
# alt_save.save_altair(text, f"GTR_green_landscape_v2", driver)

# %% [markdown]
# ### Guardian

# %%
from innovation_sweet_spots.getters.guardian import search_content

# %%
# search_term = "solar panels"
search_term = "microwave boiler"
articles = search_content(search_term, only_first_page=True, save_to_cache=False)

# %%
search_content

# %%
df = pd.DataFrame(
    data={
        "headline": [a["fields"]["headline"] for a in articles],
        "date": [a["webPublicationDate"] for a in articles],
    }
).head(20)
df.date = [x[0:10] for x in df.date.to_list()]
df.head(5)

# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(
        iss.get_guardian_mentions_per_year(articles), y="articles", show_trend=False
    )
)
plt1

# %% [markdown]
# ### Add Crunchbase

# %%
len(model.doc_top)

# %%
model.add_documents

# %%
# green_companies
# green_company_texts

# %%

# %%
uk_green_corpus = [corpus_cb[i] for i in uk_green_cb.index]
uk_green_company_texts = [green_company_texts[i] for i in uk_green_cb.index]

# %%
len(uk_green_company_texts)

# %%
model.add_documents(uk_green_company_texts, tokenizer=bigrammer)

# %%
len(model.doc_top)

# %%
# [green_company_texts[i] for i in uk_green_cb.index]

# %%
# uk_green_cb

# %%
dv = model._get_document_vectors(norm=False)

# %%
cb_vecs = dv[len(corpus_gtr) :, :]
cb_vecs.shape

# %%
cb_clusters = model.doc_top[len(corpus_gtr) :]

# %%
xy = umap_model.transform(cb_vecs)

# %%
uk_green_cb = green_companies.reset_index(drop=True)
uk_green_cb = uk_green_cb[uk_green_cb.country == "United Kingdom"]

# %%
uk_green_cb["x"] = xy[:, 0]
uk_green_cb["y"] = xy[:, 1]

# %%
uk_green_cb["text"] = uk_green_company_texts

# %%
uk_green_cb["text"] = uk_green_company_texts
uk_green_cb["clust_id"] = cb_clusters

# %%
cb_viz = uk_green_cb.merge(cluster_stats[["clust_id", "top_keywords"]], how="left")

# %%
cb_viz.head(1)

# %%
uk_green_cb["i_id"] = list(range(len(uk_green_cb)))

# %%
import innovation_sweet_spots.analysis.embeddings_utils as iss_emb

importlib.reload(iss_emb)

# %%
cb_clusters[1]

# %%
uk_green_cb[uk_green_cb.name == "Urban Electric Networks"].i_id

# %%
i = 863
print(uk_green_company_texts[i])
best_clust = iss_emb.find_most_similar_vect(
    cb_vecs[i, :], model.topic_vectors, "cityblock"
)
cluster_stats.iloc[best_clust[0:5]].top_keywords

# %%
doc_vecs = model._get_document_vectors()

# %%

# %%
cb_clusts = []
for i in range(len(cb_vecs)):
    closest_docs = iss_emb.find_most_similar_vect(
        cb_vecs[i, :], doc_vecs[0 : len(green_projects)], "cosine"
    )
    best_cluster = int(
        df_.iloc[closest_docs[0:15]]
        .groupby("cluster")
        .agg(counts=("project_id", "count"))
        .sort_values("counts")
        .tail(1)
        .index[0]
    )
    cb_clusts.append(best_cluster)

# %%
cb_viz = uk_green_cb.copy()
cb_viz["clust_id"] = cb_clusts
cb_viz = cb_viz.merge(cluster_stats[["clust_id", "top_keywords"]], how="left")

# %%
alt.data_transformers.disable_max_rows()
fig_cb_companies = (
    alt.Chart(
        pd.concat(
            [
                cb_viz,
            ]
        ),
        width=750,
        height=750,
    )
    .mark_circle(size=25, color="#bd1a1a", opacity=0.7)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        #         size='size',
        color=alt.Color("clust_id", scale=alt.Scale(scheme="category20")),
        tooltip=["name", "text", "clust_id", "top_keywords"],
    )
    .interactive()
)
fig_cb_companies + fig + text

# %%

# %%
# In depth checking of projects

# %% [markdown]
# ## Look at specific projects

# %%
