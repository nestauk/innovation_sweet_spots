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
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.utils.embeddings_utils import QueryEmbeddings
import sentence_transformers
import itertools


# %%
import innovation_sweet_spots.utils.embeddings_utils as eu

# %%
CB = CrunchbaseWrangler()

# %% [markdown]
# ## Explore relevant CB industries

# %%
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# %%
embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)

# %%
industry_vectors = embedding_model.encode(CB.industries)
group_vectors = embedding_model.encode(CB.industry_groups)

# %%
q_industries = QueryEmbeddings(industry_vectors, CB.industries, embedding_model)
q_groups = QueryEmbeddings(group_vectors, CB.industry_groups, embedding_model)

# %%
# q_groups.find_most_similar("parenting").head(30).text.to_list()

# %%
q_industries.find_most_similar("children").head(30).text.to_list()

# %%
CB.group_to_industries["education"]
# CB.industry_to_group["virtual world"]

# %%
# CB.get_companies_in_industries([])

# %%
USER_INDUSTRIES = [
    "parenting",
    "child care",
    "children",
    "underserved children",
    "family",
    "baby",
]

# %%
EDUCATION_INDUSTRIES = [
    "education",
    "edtech",
    "e-learning",
    "edutainment",
    "language learning",
    "mooc",
    "music education",
    "personal development",
    "skill assessment",
    "stem education",
    "tutoring",
    "training",
    "primary education",
    "continuing education",
    "charter schools",
]

# %%
INDUSTRIES_TO_REMOVE = [
    "secondary education",
    "higher education",
    "universities",
    "vocational education",
    "corporate training",
    "college recruiting",
]

# %%
DIGITAL_INDUSTRY_GROUPS = [
    "information technology",
    "hardware",
    "software",
    "mobile",
    "consumer electronics",
    "music and audio",
    "gaming",
    "design",
    "privacy and security",
    "messaging and telecommunications",
    "internet services",
    "artificial intelligence",
    "media and entertainment",
    "platforms",
    "data and analytics",
    "apps",
    "video",
    "content and publishing",
    "advertising",
    "consumer electronics",
]

# %%
# for p in DIGITAL_INDUSTRY_GROUPS:
#     print(p)

# %%
DIGITAL_INDUSTRIES = sorted(
    list(
        set(
            itertools.chain(
                *[CB.group_to_industries[group] for group in DIGITAL_INDUSTRY_GROUPS]
            )
        )
    )
    + ["toys"]
)

# %%
len(DIGITAL_INDUSTRIES)

# %% [markdown]
# ## Check CB companies
#

# %% [markdown]
# ### Keywords

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus

# %%
import importlib
import innovation_sweet_spots.analysis.analysis_utils as au

importlib.reload(au)
import pandas as pd

pd.set_option("max_colwidth", 400)

# %%
corpus_full = get_full_crunchbase_corpus()

# %%
Query = QueryTerms(corpus=corpus_full)

# %%
USER_TERMS = [
    [" parent"],
    [" mother"],
    [" mom "],
    [" moms "],
    [" father"],
    [" dad "],
    [" dads "],
    [" baby"],
    [" babies"],
    [" infant"],
    [" child"],
    [" toddler"],
    [" kid "],
    [" kids "],
    [" son "],
    [" sons "],
    [" daughter"],
    [" boy"],
    [" girl"],
]

LEARN_TERMS = [
    [" learn"],
    [" educat"],
    [" develop"],
    [" study"],
]

PRESCHOOL_TERMS = [
    [" preschool"],
    [" pre school"],
    [" kindergarten"],
    [" pre k "],
    [" montessori"],
    [" literacy"],
    [" numeracy"],
    [" math"],
    [" phonics"],
    [" early year"],
]

ALL_LEARN_TERMS = LEARN_TERMS + PRESCHOOL_TERMS

# %%
query_df_users = Query.find_matches(USER_TERMS, return_only_matches=False)
query_df_learning = Query.find_matches(ALL_LEARN_TERMS, return_only_matches=False)
query_df_learning_standalone = Query.find_matches(
    PRESCHOOL_TERMS, return_only_matches=False
)

# %%
keyword_matches = (
    query_df_users["has_any_terms"] & query_df_learning["has_any_terms"]
) | query_df_learning_standalone["has_any_terms"]

# %%
# Number of hits
keyword_matches.sum()

# %%
# Get all companies with keyword hits
keyword_ids = query_df_users[keyword_matches].id.to_list()
cb_orgs = CB.cb_organisations.query("id in @keyword_ids")

# %%
# Filter company by industry

# %%
allowed_industries = USER_INDUSTRIES + EDUCATION_INDUSTRIES
not_allowed_industries = INDUSTRIES_TO_REMOVE
filtered_ids = set(
    CB.get_company_industries(cb_orgs)
    .query("industry in @allowed_industries")
    .id.to_list()
)
ids_to_remove = set(
    CB.get_company_industries(cb_orgs)
    .query("industry in @not_allowed_industries")
    .id.to_list()
)
keyword_ids_filtered = list(filtered_ids.difference(ids_to_remove))

# %%
len(CB.cb_organisations.query("id in @keyword_ids_filtered"))

# %%
cb_orgs_with_funds = au.get_companies_with_funds(
    CB.cb_organisations.query("id in @keyword_ids_filtered")
)

# %%
# Number of hits that have funding
len(cb_orgs_with_funds)

# %%
# cb_orgs_industries = CB.get_company_industries(cb_orgs_with_funds)

# %%
# (
#     cb_orgs_industries
#     .groupby('industry').agg(counts=('id', 'count')).sort_values('counts', ascending=False).head(50)
# )

# %%
cb_orgs_with_funds[["name", "short_description", "long_description", "id"]].sample(5)

# %%
orgs_keyword_hits = cb_orgs_with_funds

# %% [markdown]
# ## Industries

# %%
from innovation_sweet_spots.analysis.query_categories import query_cb_categories

# %%
query_df_user_industry = query_cb_categories(
    USER_INDUSTRIES, CB, return_only_matches=True, verbose=False
)
query_df_education_industry = query_cb_categories(
    EDUCATION_INDUSTRIES, CB, return_only_matches=True, verbose=False
)
query_df_remove_industry = query_cb_categories(
    INDUSTRIES_TO_REMOVE, CB, return_only_matches=True, verbose=False
)
query_df_digital_industry = query_cb_categories(
    DIGITAL_INDUSTRIES, CB, return_only_matches=True, verbose=False
)

# %%
user_industry_ids = set(query_df_user_industry.id.to_list())
education_industry_ids = set(query_df_education_industry.id.to_list())
digital_industry_ids = set(query_df_digital_industry.id.to_list())
remove_industry_ids = set(query_df_remove_industry.id.to_list())

industry_ids = user_industry_ids.difference(remove_industry_ids)

# industry_ids = set(query_df_relevant_industry.id.to_list()).intersection(set(query_df_digital_industry.id.to_list()) | set(query_df_standalone_industry.id.to_list()))
# industry_ids = industry_ids.difference(set(query_df_remove_industry.id.to_list()))
# ids_in_both_relevant_and_digital = set(relevant_industry_ids).intersection(set(digital_industry_ids))

# %%
len(industry_ids)

# %%
orgs_industry_hits = au.get_companies_with_funds(
    CB.cb_organisations.query("id in @industry_ids")
)

# %%
len(orgs_industry_hits)

# %%
orgs_industry_hits[["name", "short_description", "long_description", "id"]].sample(5)

# %% [markdown]
# ## Join up keywords and industry results

# %%
final_industry_ids = set(orgs_industry_hits.id.to_list())
final_keyword_ids = set(orgs_keyword_hits.id.to_list())

# %%
print(len(final_industry_ids), len(final_keyword_ids))
print(len(final_industry_ids.intersection(set(final_keyword_ids))))
print(len(final_industry_ids.union(final_keyword_ids)))

# %%
all_ids = final_industry_ids.union(final_keyword_ids)

# %%
cb_hits = CB.cb_organisations.query("id in @all_ids")

# %% [markdown]
# ### Company embeddings and Clustering

# %%
import umap
import hdbscan
import altair as alt

alt.data_transformers.disable_max_rows()

# %%
from innovation_sweet_spots.utils import text_processing_utils as tpu

# %%
company_docs = tpu.create_documents_from_dataframe(
    cb_hits, ["short_description", "long_description"]
)

# %%
# company_vectors = embedding_model.encode(docs)

# %%
vector_filename = "vectors_2022_03_02"
embedding_model = EMBEDDING_MODEL
PARENTING_DIR = PROJECT_DIR / "outputs/finals/parenting"
EMBEDINGS_DIR = PARENTING_DIR / "embeddings"

# %%
cb_hits.to_csv(PARENTING_DIR / "cb_parenting_companies.csv", index=False)

# %%
cb_hits = pd.read_csv(PARENTING_DIR / "cb_parenting_companies.csv")
len(cb_hits)

# %%
cb_hits.groupby("country").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(5)

# %%
# v1 = Vectors(EMBEDDING_MODEL, company_vector_ids, company_vectors)
# v1.save_vectors('vectors_2022_03_02', PARENTING_DIR / 'embeddings')
v = eu.Vectors(
    filename=vector_filename, model_name=EMBEDDING_MODEL, folder=EMBEDINGS_DIR
)
v.vectors.shape

# %%
v.get_missing_ids(cb_hits.id.to_list())

# %%
v.generate_new_vectors(
    new_document_ids=cb_hits.id.to_list(), texts=company_docs, force_update=False
)

# %%
v.save_vectors(vector_filename, EMBEDINGS_DIR)

# %%
len(v.vector_ids)

# %%
# ids_to_cluster = final_industry_ids
ids_to_cluster = cb_hits.id.to_list()
# ids_to_cluster = final_keyword_ids

# %%
vectors = v.select_vectors(ids_to_cluster)

# %%
vectors.shape

# %%
# list(ids_to_cluster)[0]

# %%
UMAP_PARAMS = {
    "n_neighbors": 10,
    "min_dist": 0.0,
}

# %%
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=1, **UMAP_PARAMS)
embedding = reducer.fit_transform(vectors)

# %%
# Check the shape of the reduced embedding array
embedding.shape

# %%
# %%
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=50, random_state=1, **UMAP_PARAMS)
embedding_clustering = reducer_clustering.fit_transform(vectors)


# %%
# %%
# Clustering with hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=1,
    cluster_selection_method="leaf",
    prediction_data=True,
)
clusterer.fit(embedding_clustering)


# %%
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

# %%
soft_cluster = [np.argmax(x) for x in soft_clusters]

# %%
len(np.unique(soft_cluster))

# %%
# %%
# Prepare dataframe for visualisation
df = (
    cb_hits.set_index("id")
    .loc[ids_to_cluster, :]
    .reset_index()[["id", "name", "short_description", "long_description", "country"]]
    .copy()
)
df = df.merge(CB.get_company_industries(df, return_lists=True), on=["id", "name"])
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]
df["soft_cluster"] = [str(x) for x in soft_cluster]

# %%
# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df, width=500, height=500)
    .mark_circle(size=60)
    .encode(
        x="x",
        y="y",
        tooltip=[
            "soft_cluster",
            "name",
            "short_description",
            "long_description",
            "country",
            "industry",
        ],
        color="soft_cluster",
    )
).interactive()

fig

# %% [markdown]
# ## Characterise clusters

# %%
# importlib.reload(tpu)

# %%
from innovation_sweet_spots.utils import cluster_analysis_utils

# %%
corpus_all_ids = np.array(Query.document_ids)
ids_to_cluster = list(ids_to_cluster)

# %%
corpus_ids = [np.where(doc_id == corpus_all_ids)[0][0] for doc_id in ids_to_cluster]

# %%
cluster_texts = [Query.text_corpus[i] for i in corpus_ids]

# %%
# cluster_texts

# %%
list(ids_to_cluster)[0]

# %%
cluster_keywords = cluster_analysis_utils.cluster_keywords(cluster_texts, soft_cluster)

# %%
for i in list(range(len(cluster_keywords))):
    print(i, cluster_keywords[i])

# %%
