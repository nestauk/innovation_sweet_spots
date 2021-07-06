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
# %%capture
import numpy as np
import pandas as pd
import umap
import altair as alt
from time import time

# %%
from innovation_sweet_spots.getters import gtr, misc
from innovation_sweet_spots.getters import crunchbase as cb
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.embeddings_utils as iss_emb
import innovation_sweet_spots.utils.visualisation_utils as iss_viz
import innovation_sweet_spots.utils.io as io

# %%
from innovation_sweet_spots.utils.text_cleaning_utils import split_string
import innovation_sweet_spots.analysis.text_analysis as iss_text

# %%
from innovation_sweet_spots import PROJECT_DIR, logging

# %%
DATA_OUTPUTS = PROJECT_DIR / "outputs/data"


# %%
def generate_gtr_topic_lists(gtr_project_topics):
    # Remove unclassified
    project_topics = (
        gtr_project_topics.query("text!='Unclassified'")
        .groupby(["project_id", "text"])
        .count()
    )
    projs = np.unique(project_topics.index.get_level_values("project_id").to_list())
    topic_lists = [project_topics.loc[proj].index.to_list() for proj in projs]
    # Select only the lists with at least two topics
    topic_lists_ = [topics for topics in topic_lists if len(topics) > 1]

    unique_topics = sorted(
        np.unique([topic for topics in topic_lists_ for topic in topics])
    )
    return topic_lists_, unique_topics


def gtr_topic_embedding_pipeline(gtr_project_topics):
    model_name = "gtr_topics_word2vec"
    topic_lists, unique_topics = generate_gtr_topic_lists(gtr_project_topics)
    topics_word2vec_model = iss_emb.token_2_vec(topic_lists, model_name=model_name)
    topics_embeddings = iss_emb.get_token_vectors(topics_word2vec_model, unique_topics)
    return unique_topics, topics_embeddings


# %%
def cb_category_groups(cb_categories):
    """Get unique cateogry group names"""
    split_groups = [
        split_string(s, separator=",")
        for s in cb_categories.category_groups_list.to_list()
    ]
    groups = sorted(set([group for groups in split_groups for group in groups]))
    return groups


def cb_categories_for_group(cb_categories, group_name="Sustainability"):
    """Get unique cateogry group names"""
    df = cb_categories[-cb_categories.category_groups_list.isnull()]
    categories = sorted(
        set(df[df.category_groups_list.str.contains(group_name)].name.to_list())
    )
    return [s.lower() for s in categories]


# %%
def get_initial_chunks_from_tech_navigator(save=False):
    """NB: requires revision"""
    # Relevant tech navigator columns
    cols = ["Technology Name", "Short Descriptor", "Brief description "]
    # Get tech navigator table
    tech_nav = misc.get_tech_navigator()
    # Get all chunks
    nlp = iss_text.setup_spacy_model()
    tech_chunks = []
    for col in cols:
        techs = tech_nav[col].to_list()
        techs = [s for s in techs if type(s) is str]
        tech_chunks += list(iss_text.chunk_forms(techs, nlp))
    tech_chunks_flat = sorted(set([t for ts in tech_chunks for t in ts]))
    if save:
        io.save_list_of_terms(
            keywords, misc.MISC_PATH / "green_keywords_TechNav_before_review.txt"
        )
    return tech_chunks_flat


# %%
def get_all_green_keywords(save=False):
    keyword_types = ["CB", "IK", "TechNav", "KK"]
    keyword_paths = [misc.MISC_PATH / f"green_keywords_{s}.txt" for s in keyword_types]
    keyword_lists = [io.read_list_of_terms(fpath) for fpath in keyword_paths]
    keywords = sorted(
        set([keyword for keyword_list in keyword_lists for keyword in keyword_list])
    )
    if save:
        io.save_list_of_terms(keywords, DATA_OUTPUTS / "aux/green_keywords_all.txt")
    return keywords


# %%
def alt_scatter(topic_table_xy, size_column, color_column, tooltip):
    fig = (
        alt.Chart(topic_table_xy, width=750, height=750)
        .mark_circle(size=50)
        .encode(
            x=alt.X("x", axis=alt.Axis(grid=False)),
            y=alt.Y("y", axis=alt.Axis(grid=False)),
            size=size_column,
            color=color_column,
            tooltip=tooltip,
        )
        .interactive()
    )
    return fig


# %%
keywords = get_all_green_keywords(save=True)

# %% [markdown]
# ## Overview of GTR project topics

# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()
gtr_topics = gtr.get_gtr_topics()
link_gtr_topics = gtr.get_link_table("gtr_topic")
gtr_project_topics = iss.link_gtr_projects_and_topics(
    gtr_projects, gtr_topics, link_gtr_topics
)


# %%
green_topics = [
    "Bioenergy",
    "Carbon Capture & Storage",
    "Climate & Climate Change",
    "Conservation Ecology",
    "Energy Efficiency",
    "Energy Storage",
    "Energy - Marine & Hydropower",
    "Energy - Nuclear",
    "Environmental Engineering",
    "Environmental economics",
    "Fuel Cell Technologies",
    "Solar Technology",
    "Sustainability Management",
    "Sustainable Energy Networks",
    "Sustainable Energy Vectors",
    "Waste Management",
    "Waste Minimisation",
    "Wind Power",
]

# %%
# %%capture
topics_word2vec_model, topics_embeddings = gtr_topic_embedding_pipeline(
    gtr_project_topics
)

# %%
topic_count = (
    gtr_project_topics.groupby("text").agg(counts=("project_id", "count")).reset_index()
)

topic_table = (
    pd.DataFrame(
        data={"topics": unique_topics, "vect_id": range(0, len(unique_topics))}
    )
    .merge(topic_count, left_on="topics", right_on="text", how="left")
    .drop("text", axis=1)
)

topic_table_xy = iss_viz.viz_dataframe(topic_table, topics_embeddings)

# %%
topic_table_xy["green_topic"] = False
topic_table_xy.loc[topic_table_xy.topics.isin(green_topics), "green_topic"] = True
topic_table_xy["counts_scaled"] = np.power(topic_table_xy["counts"], 0.75)

# %%
alt_scatter(
    topic_table_xy,
    size_column="counts_scaled",
    color_column="green_topic",
    tooltip=["topics", "counts"],
)


# %%
green_gtr_projects = gtr_project_topics[
    gtr_project_topics.text.isin(green_topics)
].project_id.unique()
len(green_gtr_projects)

# %%
green_topic_count = (
    gtr_project_topics[gtr_project_topics.text.isin(green_topics)]
    .groupby("text")
    .agg(counts=("project_id", "count"))
    .reset_index()
    .sort_values("counts", ascending=False)
)

# %%
green_topic_count.head(7)

# %%
# gtr_projects[gtr_projects.project_id.isin(green_gtr_projects)].title.to_list()

# %%
len(keywords)

# %% [markdown]
# ## Crunchbase

# %%
cb_orgs = cb.get_crunchbase_orgs_full()
cb_categories = cb.get_crunchbase_category_groups()
cb_org_categories = cb.get_crunchbase_organizations_categories()

# %%
green_tags = cb_categories_for_group(cb_categories, "Sustainability")
io.save_list_of_terms(
    green_tags, misc.MISC_PATH / "green_keywords_CB_before_review.txt"
)

# %%
green_orgs = cb_org_categories[cb_org_categories.category_name.isin(green_tags)]
green_org_ids = list(green_orgs.organization_id.unique())
green_orgs_table = cb_orgs[cb_orgs.id.isin(green_org_ids)]


# %% [markdown]
# ## Green projects and companies

# %%
def document_pipeline(
    df,
    cols=["title", "abstractText", "techAbstractText"],
    id_col="project_id",
    output_path="gtr/gtr_project_clean_text.csv",
):
    """Create and save documents"""
    logging.info(f"Creating {len(df)} documents")
    docs = iss.create_documents_from_dataframe(
        df, cols, iss.preprocess_text_clean_sentences
    )
    # Save the dataframe
    doc_df = pd.DataFrame(data={"project_id": df[id_col], "project_text": docs})
    fpath = DATA_OUTPUTS / f"{output_path}.csv"
    logging.info(f"Saved {len(doc_df)} documents in {fpath}")
    doc_df.to_csv(fpath, index=False)
    return doc_df


# %%
cb_green_docs = document_pipeline(
    green_orgs_table,
    cols=["name", "short_description", "long_description"],
    id_col="id",
    output_path="cb/cb_green_org_clean_text",
)

# %%
gtr_docs = document_pipeline(
    gtr_projects,
    cols=["title", "abstractText", "techAbstractText"],
    id_col="project_id",
    output_path="gtr/gtr_project_clean_text",
)

# %%
gtr_docs.iloc[-1].project_text

# %% [markdown]
# ### Select "green" gtr docs

# %%
# gtr_docs[gtr_docs.project_id.isin(green_gtr_projects)]

# %%
keywords = get_all_green_keywords(save=True)

# %%
keywords_clean = sorted(set([iss_text.clean_text(keyword) for keyword in keywords]))

# %%
# keywords_clean

# %%
green_ids = []
for keyword in keywords_clean:
    df = iss.is_term_present_in_sentences(
        keyword, gtr_docs.project_text.to_list(), min_mentions=2
    )
    green_ids.append(gtr_docs[df].project_id.to_list())

# %%
keyword_project_counts = pd.DataFrame(
    data={"keyword": keywords_clean, "n_projects": [len(s) for s in green_ids]}
).sort_values("n_projects", ascending=False)

# %%
keyword_project_counts.head(5)

# %%
green_ids_ = list(
    set([project_id for project_ids in green_ids for project_id in project_ids])
)

# %%
# How much overalp
len(set(green_gtr_projects).intersection(set(green_ids_))) / len(
    set(green_gtr_projects).union(set(green_ids_))
)

# %%
# Present in the topic set but not in the keyword set
dif_set = set(green_gtr_projects).difference(set(green_ids_))
gtr_project_topics[gtr_project_topics.project_id.isin(dif_set)].groupby("text").agg(
    counts=("project_id", "count")
).sort_values("counts", ascending=False).head(5)


# %%
# Present in keyword set but not in the topic set
dif_set = set(green_ids_).difference(set(green_gtr_projects))
gtr_project_topics[gtr_project_topics.project_id.isin(dif_set)].groupby("text").agg(
    counts=("project_id", "count")
).sort_values("counts", ascending=False).head(9)


# %%
# gtr_projects[gtr_projects.project_id.isin(dif_set)].title.to_list()

# %%
all_green_projects = set(green_ids_ + list(green_gtr_projects))

# %%
len(all_green_projects)

# %%
green_gtr_docs_fin = gtr_docs[gtr_docs.project_id.isin(all_green_projects)]

# %%
# green_gtr_docs_fin

# %%
# cb_green_docs

# %% [markdown]
# ## Topic modelling of green projects

# %%
# importlib.reload(gtr);
# importlib.reload(cb);
importlib.reload(iss)

# %%
from innovation_sweet_spots.hSBM_Topicmodel.sbmtm import sbmtm
import graph_tool.all as gt

# %%
import innovation_sweet_spots.pipeline.topic_modelling as iss_topics


# %%
def merge_tokenized_doc_sentences(sentences):
    doc = []
    for sent in sentences:
        doc += tokenise_text(sent)
    return doc


def tokenise_text(text, min_length=2):
    """"""
    tokens = text.split(" ")
    # Check minimal length and remove tokens that are only numbers
    tokens = [
        token
        for token in tokens
        if ((len(token) >= min_length) and (not token.isdecimal()))
    ]
    return tokens


## TODO: Use spacy to remove special tokens


# %%
gtr_docs_nonempty = green_gtr_docs_fin[
    green_gtr_docs_fin.project_text.apply(lambda x: len(x) > 0)
]

# %%
cb_green_docs_nonempty = cb_green_docs[
    cb_green_docs.project_text.apply(lambda x: len(x) > 0)
]

# %%
gtr_titles = [
    "GTR_" + "_".join(t[0].split()) for t in gtr_docs_nonempty.project_text.to_list()
]
gtr_texts = [
    merge_tokenized_doc_sentences(s) for s in gtr_docs_nonempty.project_text.to_list()
]

# %%
cb_titles = [
    "CB_" + "_".join(t[0].split())
    for t in cb_green_docs_nonempty.project_text.to_list()
]
cb_texts = [
    merge_tokenized_doc_sentences(s)
    for s in cb_green_docs_nonempty.project_text.to_list()
]

# %%
titles = gtr_titles + cb_titles
texts = gtr_texts + cb_texts

# %%
ids = (
    gtr_docs_nonempty.project_id.to_list() + cb_green_docs_nonempty.project_id.to_list()
)

# %%
all_green_docs = pd.DataFrame(data={"doc_id": ids, "titles": titles, "text": texts})

# %%
all_green_docs.to_csv("green_documents_June29.csv", index=False)

# %%

# %% [markdown]
# ### Alternative document prep

# %%
green_orgs = cb_orgs[
    cb_orgs.id.isin(cb_green_docs.project_id.to_list())
].drop_duplicates()

# %%
green_proj = gtr_projects[gtr_projects.project_id.isin(all_green_projects)]

# %%
cb_cols = ["short_description", "long_description"]
green_orgs_doc = iss.create_documents_from_dataframe(green_orgs, cb_cols, (lambda x: x))

# %%
gtr_cols = ["title", "abstractText", "techAbstractText"]
green_proj_doc = iss.create_documents_from_dataframe(
    green_proj, gtr_cols, (lambda x: x)
)

# %%
all_green_docs = pd.DataFrame(
    data={
        "source": ["CB"] * len(green_orgs) + ["GTR"] * len(green_proj),
        "doc_id": green_orgs.id.to_list() + green_proj.project_id.to_list(),
        "doc_title": green_orgs.name.to_list() + green_proj.title.to_list(),
        "text": green_orgs_doc + green_proj_doc,
    }
)
all_green_docs.to_csv("green_documents_June29.csv", index=False)

# %% [markdown]
# ## Visualise

# %%
emb = np.load(
    PROJECT_DIR
    / "outputs/data/embeddings/green_documents_June29_paraphrase-distilroberta-base-v1.npy"
)

# %%
emb.shape

# %%
gtr_indexes = all_green_docs[all_green_docs.source == "GTR"].index.to_list()

# %%
uk_orgs_ids = cb_orgs[cb_orgs.country == "United Kingdom"].id.to_list()

# %%
cb_indexes = all_green_docs[
    (all_green_docs.source == "CB") & (all_green_docs.doc_id.isin(uk_orgs_ids))
].index.to_list()

# %%
cb_indexes = all_green_docs[(all_green_docs.source == "CB")].index.to_list()

# %%
len(cb_indexes)

# %%
importlib.reload(iss_viz)

# %%
reducer = umap.UMAP(random_state=111, n_neighbors=20, min_dist=0.01, n_components=2)
xy = reducer.fit_transform(emb[gtr_indexes, :])


# %%
all_green_docs_xy = iss_viz.viz_dataframe(
    all_green_docs.loc[gtr_indexes], emb[gtr_indexes, :], xy
)

# %%
all_green_docs_xy["highlight_topic"] = "other"
topic_to_check = "Climate & Climate Change"
topic_proj = gtr_project_topics[
    gtr_project_topics.text == topic_to_check
].project_id.unique()
all_green_docs_xy.loc[
    all_green_docs_xy.doc_id.isin(topic_proj), "highlight_topic"
] = topic_to_check

# %%
cb_xy = reducer.transform(emb[cb_indexes, :])

# %%
all_green_cb_docs_xy = iss_viz.viz_dataframe(
    all_green_docs.loc[cb_indexes], None, cb_xy
)

# %%
# all_green_docs_xy.head(1)

# %%
alt.data_transformers.disable_max_rows()
all_green_docs_xy["size"] = 1
# alt_scatter(all_green_docs_xy, size_column='size', color_column='source', tooltip=['source', 'doc_title'])
#
fig = (
    alt.Chart(
        pd.concat(
            [
                all_green_cb_docs_xy,
                all_green_docs_xy,
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
        color="source",
        tooltip=["source", "doc_title", "text"],
    )
    .interactive()
)
# fig

# %% [markdown]
# ### TopSBM

# %%
gtr_docs_nonempty = green_gtr_docs_fin[
    green_gtr_docs_fin.project_text.apply(lambda x: len(x) > 0)
]

# %%
topix = ["Climate & Climate Change"]
green_topics_x = [t for t in green_topics if t in topix]

# %%
# keywords_clean

# %%
not_green = set(
    gtr_project_topics[
        gtr_project_topics.text.isin(green_topics_x) == False
    ].project_id.to_list()
)
topix_proj = set(
    gtr_project_topics[gtr_project_topics.text == topix].project_id.to_list()
)

# %%
proj_id_to_remove = topix_proj.intersection(not_green)

# %%
gtr_docs_nonempty_nonclimate = gtr_docs_nonempty[
    gtr_docs_nonempty.project_id.isin(proj_id_to_remove) == False
]

# %%
df = gtr_project_topics[
    gtr_project_topics.project_id.isin(
        gtr_docs_nonempty_nonclimate.project_id.to_list()
    )
]

# %%
# df.groupby('text').agg(counts=('project_id', 'count')).sort_values('counts', ascending=False).head(25)

# %%
# df[df.text=="Conservation Ecology"].abstractText.to_list()

# %%

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
gtr_docs_nonempty_nonclimate.head(1)

# %%
# Remove parenthesis
import re

re_parenthesis = re.compile("[\(\)]")

# %%
prohibited = ["ha"]


def extra_clean(tokens):
    tokens = (re_parenthesis.sub("", t) for t in tokens)
    tokens = [
        t
        for t in tokens
        if ((t not in prohibited) and (t.isdecimal() == False) and (len(t) >= 2))
    ]
    return tokens


# %%
gtr_texts_ = [
    merge_tokenized_doc_sentences(s)
    for s in gtr_docs_nonempty_nonclimate.project_text.to_list()
]
gtr_texts_ = [extra_clean(t) for t in gtr_texts_]

# %%
gtr_titles_ = [
    "GTR_" + "_".join(t[0].split())
    for t in gtr_docs_nonempty_nonclimate.project_text.to_list()
]

# %%
texts_to_count = [" ".join(t) for t in gtr_texts_]

# %%
c = CountVectorizer()
c.fit_transform(texts_to_count)

# %%
len(c.vocabulary_)

# %%
len(gtr_titles_), len(gtr_texts_)

# %%
topic_model = iss_topics.train_model(gtr_texts_, gtr_titles_)

# %%
# import pickle
# pickle.dump(topic_model, open('topic_model.p', 'wb'))

# %%
import pickle

topic_model = pickle.load(open("topic_model.p", "rb"))

# %%
topic_model.plot()

# %%
# topic_model.topics(l=1, n=20)
len(topic_model.topics(l=2))

# %%
# i_doc = 2
# print(topic_model.documents[i_doc])
## get a list of tuples (topic-index, probability)
# topic_model.topicdist(i_doc,l=1)

# %%
# topic_model.documents

# %%

# %%
topic_df = iss_topics.post_process_model(topic_model, 2)

# %%
# topic_df.columns

# %%
# topic_df.sort_values('process_using_sustainable_resource_heat')

# %%
# import seaborn as sns

# %%
# topic_df[topic_df.fuel_battery_hydrogen_vehicle_polymer>0.1].sort_values('process_using_sustainable_resource_heat')


# %%
# topic_model.clusters_query(32,l=0,)

# %%
# topic_model.documents()

# %%
# cl = topic_model.clusters(l=1,n=len(topic_model.documents))

# %%
topic_df[topic_df["fuel"] > 0.1]

# %%
for p in topic_df.columns:
    print(p)

# %%

# %%
cl_2 = topic_model.clusters(l=1, n=len(topic_model.documents))

# %%
cl_sizes = [len(cl_2[x]) for x in cl_2]

# %%
df_clustered = gtr_docs_nonempty_nonclimate.copy()
df_clustered["title"] = gtr_titles_

# %%
cl_2_doc_titles = [[d[0] for d in cl_2[c]] for c in cl_2]

# %%
cl_2_doc_titles_ = [
    (i, doc_title)
    for i, doc_titles in enumerate(cl_2_doc_titles)
    for doc_title in doc_titles
]

# %%
clust_df = pd.DataFrame(cl_2_doc_titles_, columns=["cluster", "title"])

# %%
df_clustered_ = (
    df_clustered.merge(clust_df)
    .merge(topic_df.reset_index().rename(columns={"index": "title"}))
    .rename(columns={"title": "title_"})
    .merge(
        all_green_docs_xy[["doc_id", "x", "y"]], left_on="project_id", right_on="doc_id"
    )
    .merge(gtr_projects[["project_id", "title"]])
)
# df_clustered_['color'] = df_clustered_['cluster'].apply(lambda x: str(x))

# %%
# df_clustered_viz = df_clustered_[df_clustered_.cluster==6]

# %%
# df_clustered_.fuel_battery_hydrogen_vehicle_polymer.max()

# %%
fig = (
    alt.Chart(
        pd.concat(
            [
                df_clustered_,
                #         all_green_cb_docs_xy
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
        color=alt.Color(
            "solar_pv_panel_photovoltaic_module",
            scale=alt.Scale(scheme="lightgreyred", domain=[0, 0.4]),
        ),
        tooltip=["title", "cluster", "project_text"],
    )
    .interactive()
)

# %%
# fig

# %%
df_clustered_[df_clustered_.cluster == 26].title.to_list()

# %% [markdown]
# # Check

# %%
sns.displot([len(s) for s in gtr_texts_])

# %%
sns.displot([len(s) for s in cb_texts])

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-distilroberta-base-v1")

# %%
model.tokenize(" ".join(gtr_texts_[2]))["input_ids"].shape

# %%
len(gtr_texts_[2])

# %%
proj_id = df_clustered_[df_clustered_.cluster == 8].sample().iloc[0]["project_id"]
gtr_projects[gtr_projects.project_id == proj_id].abstractText.iloc[0]

# %%
len(gtr_docs_nonempty_nonclimate)

# %%
# gtr_docs_nonempty_nonclimate.project_id.to_list()

# %%
io.save_list_of_terms(
    gtr_docs_nonempty_nonclimate.project_id.to_list(),
    DATA_OUTPUTS / "gtr/green_gtr_project_ids.txt",
)

# %%

# %%

# %%

# %%
