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
from innovation_sweet_spots import PROJECT_DIR

# %%
CB = CrunchbaseWrangler()

# %%
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# %% [markdown]
# ## Explore relevant CB industries

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
CB.group_to_industries["design"]
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
for t in INDUSTRIES_TO_REMOVE:
    print(t)

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
import pandas as pd
import numpy as np

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
# cb_hits.to_csv(PARENTING_DIR / "cb_parenting_companies.csv", index=False)

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
    "min_dist": 0.01,
}

# %%
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=21, **UMAP_PARAMS)
embedding = reducer.fit_transform(vectors)

# %%
# Check the shape of the reduced embedding array
embedding.shape

# %%
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=25, random_state=1, **UMAP_PARAMS)
embedding_clustering = reducer_clustering.fit_transform(vectors)


# %%
# Clustering with hdbscan
np.random.seed(11)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
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
# print(hdbscan.validity.validity_index(embedding_clustering.astype(np.float64), np.array(soft_cluster)))
# print(hdbscan.validity.validity_index(embedding_clustering.astype(np.float64), np.array(clusterer.labels_)))

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

# %%
df["cluster"] = [str(x) for x in clusterer.labels_]
df["soft_cluster"] = [str(x) for x in soft_cluster]

# %%
cluster_col = df.columns.get_loc("cluster")
cluster_col

# %%
# from sklearn.neighbors import KNeighborsClassifier

# radius = 10
# neigh = KNeighborsClassifier(n_neighbors=radius)
# current_unassigned = len(df[df.cluster == -1])

# cluster_list = df.cluster.to_list()

# unassigned_indices = []
# unassigned_indices.extend([i for i, x in enumerate(cluster_list) if x == -1])

# X_list = [
#     x for i, x in enumerate(vectors) if i not in unassigned_indices
# ]
# y_list = [i for i in cluster_list if i != -1]

# neigh.fit(X_list, y_list)

# for x in unassigned_indices:
#     s_df.iat[x, s_cluster_col] = neigh.predict([s_embedding_clustering[x]])

# print(f"{len(s_df[s_df.s_cluster == -1].s_cluster.to_list())} unassigned apps remain.")

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
            "cluster",
            "name",
            "short_description",
            "long_description",
            "country",
            "industry",
        ],
        color="soft_cluster",
    )
).interactive()

# fig

# %% [markdown]
# ## Characterise clusters

# %%
cb_hits_ = cb_hits.copy()
cb_hits_["soft_cluster"] = soft_cluster
cb_hits_["cluster_prob"] = clusterer.probabilities_

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus

corpus_full = get_full_crunchbase_corpus()
Query = QueryTerms(corpus=corpus_full)

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
# list(ids_to_cluster)[0]

# %%
cluster_keywords = cluster_analysis_utils.cluster_keywords(
    cluster_texts, soft_cluster, 10
)

# %%
for i in list(range(len(cluster_keywords))):
    print(i, cluster_keywords[i])

# %%
pd.set_option("max_colwidth", 200)


def show_cluster_hits(cluster):
    return cb_hits_.query("soft_cluster == @cluster").sort_values(
        "cluster_prob", ascending=False
    )[["name", "short_description", "long_description", "cluster_prob"]]


# %%
clusters_labels = {
    0: "sports",
    1: "financial, charitable support",
    2: "children and money",
    3: "sharing media",
    4: "social support",
    5: "developmental needs",
    6: "special needs",
    7: "childcare",
    8: "coding",
    9: "books and reading",
    10: "reading, dyslexia",
    11: "parent support",
    12: "social platforms for kids",
    13: "apps",
    14: "games",
    15: "chinese, children education",
    16: "robots",
    17: "smart toys",
    18: "maths",
    19: "baby food",
    21: "health",
    22: "school tech",
    23: "tutoring",
    24: "education platforms",
    25: "learning management platform",
    26: "clothing",
    27: "baby care",
}

# %%
relevant_clusters = [
    3,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
]

# %%
for i in relevant_clusters:
    print(clusters_labels[i])

# %%
show_cluster_hits(27)

# %%
fig

# %% [markdown]
# ## Topic modelling

# %%
corpus_ids[0]

# %%
# topic_modelling_corpus = [corpus_full[id_] for id_ in corpus_ids]

# %% [markdown]
# # Simple

# %%
CB.cb_organisations.foundend

# %%
benchmarks = au.cb_get_all_timeseries(
    CB.cb_organisations,
    CB.get_funding_rounds(CB.cb_organisations),
    period="year",
    min_year=2005,
    max_year=2020,
)

# %%

# %%
cb_orgs_parenting = CB.get_companies_in_industries(["parenting"])
cb_orgs_child_care = CB.get_companies_in_industries(["child care"])

# %%
cb_orgs_parenting_ = au.get_companies_with_funds(cb_orgs_parenting)
cb_orgs_child_care_ = au.get_companies_with_funds(cb_orgs_child_care)

# %%
len(cb_orgs_parenting_), len(cb_orgs_child_care_)

# %%
cb_orgs_child_care_.groupby("country").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(10) / len(cb_orgs_child_care_)

# %%
cb_orgs_parenting_.groupby("country").agg(counts=("id", "count")).sort_values(
    "counts", ascending=False
).head(10) / len(cb_orgs_parenting_)

# %%
cb_orgs_parenting_funds = CB.get_funding_rounds(cb_orgs_parenting_)
cb_orgs_child_care_funds = CB.get_funding_rounds(cb_orgs_child_care_)

# %%
# cb_orgs_parenting_.groupby('country').sum()

# %%
cb_orgs_parenting_timeseries = au.cb_get_all_timeseries(
    cb_orgs_parenting_, cb_orgs_parenting_funds, "year", 2005, 2020
)
cb_orgs_child_care_timeseries = au.cb_get_all_timeseries(
    cb_orgs_child_care_, cb_orgs_child_care_funds, "year", 2005, 2020
)
cb_orgs_parenting_timeseries.head(1)

# %%
pu.time_series(cb_orgs_parenting_timeseries, y_column="raised_amount_gbp_total")

# %%
pu.time_series(cb_orgs_parenting_timeseries, y_column="no_of_rounds")

# %%
pu.time_series(cb_orgs_child_care_timeseries, y_column="raised_amount_gbp_total")

# %%
pu.time_series(cb_orgs_child_care_timeseries, y_column="no_of_rounds")

# %%

# %% [markdown]
# # Analyse companies

# %%
len(cb_hits)

# %%
sum(CB.cb_organisations.id.duplicated())

# %%
benchmark_country = (
    CB.cb_organisations[-CB.cb_organisations.country.isnull()]
    .groupby("country")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)


# %%
(benchmark_country / benchmark_country.sum()).head(10)

# %%
len(cb_hits_digital)

# %%
cb_hits_country = (
    cb_hits_digital.groupby("country")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)
cb_hits_country = cb_hits_country / cb_hits_country.sum()
cb_hits_country.head(6)

# %%
cb_hits_digital = CB.select_companies_by_industries(cb_hits, DIGITAL_INDUSTRIES)

# %%
df_rounds = (
    CB.get_funding_rounds(cb_hits_digital)
    .drop("country", axis=1)
    .merge(
        cb_hits_digital[["id", "country"]], left_on="org_id", right_on="id", how="left"
    )
)

# %%
df_rounds.query('announced_on >= "2015"').groupby("country").sum().sort_values(
    "raised_amount_gbp", ascending=False
).head(10)[["raised_amount_gbp"]]

# %%
digital_timeseries = au.cb_get_all_timeseries(
    cb_hits_digital, df_rounds, "year", 2005, 2020
)

# %%
pu.time_series(digital_timeseries, y_column="no_of_rounds")

# %%
CB.cb_organisations.founded_on.iloc[0].isdecimal()

# %%
cb_ = CB.cb_organisations.loc[
    -CB.cb_organisations.founded_on.isnull(),
    ["id", "name", "founded_on", "num_funding_rounds", "total_funding_usd"],
]

# %%
cb_ = cb_[cb_.founded_on.apply(lambda x: str(x)[0] in ["1", "2"])]

# %%
cb_with_funds_ = au.get_companies_with_funds(cb_)

# %%
benchmark_df = au.cb_get_all_timeseries(
    cb_with_funds_,
    CB.get_funding_rounds(cb_with_funds_),
    "year",
    min_year=2005,
    max_year=2020,
)

# %%
no_of_orgs_benchmark = au.cb_orgs_founded_per_period(
    cb_with_funds_, period="Y", min_year=2005, max_year=2020
)

# %%
MIN_YEAR = 2005
df_general = (
    au.cb_orgs_founded_per_period(cb_hits, period="Y", min_year=MIN_YEAR, max_year=2020)
    .set_index("time_period")
    .assign(
        digital_orgs=au.cb_orgs_founded_per_period(
            cb_hits_digital, period="Y", min_year=MIN_YEAR, max_year=2020
        ).set_index("time_period")["no_of_orgs_founded"]
    )
    .assign(digital_fraction=lambda x: x.digital_orgs / x.no_of_orgs_founded)
    .assign(
        total_cb_orgs=au.cb_orgs_founded_per_period(
            cb_with_funds_, period="Y", min_year=MIN_YEAR, max_year=2020
        ).set_index("time_period")["no_of_orgs_founded"]
    )
    .assign(digital_of_all=lambda x: x.digital_orgs / x.total_cb_orgs)
    .assign(orgs_of_all=lambda x: x.no_of_orgs_founded / x.total_cb_orgs)
    .reset_index()
)

# %%
df_general.digital_fraction.mean()

# %%
pu.time_series(df_general, y_column="digital_fraction")

# %%
pu.time_series(df_general, y_column="digital_of_all")

# %%
benchmark_df.head(1)

# %%
pu.time_series(benchmark_df, y_column="raised_amount_gbp_total")

# %% [markdown]
# How is "digital" disrupting parenting and education sectors

# %%
df = (
    CB.get_company_industries(cb_hits_digital)
    .query("industry in @DIGITAL_INDUSTRIES")
    .merge(cb_hits_digital[["id", "founded_on"]])
)

# %%
digital_counts = (
    df.groupby("industry")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
    .query("counts >= 10")
    .reset_index()
)

# %%
digital_counts.head(20)

# %%
industry_orgs = (
    au.cb_orgs_founded_per_period(
        df.query("industry == 'edtech'"), period="Y", min_year=MIN_YEAR, max_year=2020
    )
    .set_index("time_period")
    .drop("no_of_orgs_founded", axis=1)
)

for i, row in digital_counts.iterrows():
    industry = row.industry
    industry_orgs[industry] = (
        au.cb_orgs_founded_per_period(
            df.query("industry == @industry"),
            period="Y",
            min_year=MIN_YEAR,
            max_year=2020,
        ).set_index("time_period")["no_of_orgs_founded"]
        / df_general.set_index("time_period")["digital_orgs"]
    )

# %%
pu.time_series(industry_orgs.reset_index(), y_column="music education")

# %%
df_ma = (
    industry_orgs.reset_index()
    .assign(year=lambda x: x.time_period.dt.year)
    .pipe(au.moving_average, replace_columns=True)
    .assign(time_period=industry_orgs.reset_index()["time_period"])
    #     .set_index('year')
)

# %%
pu.time_series(df_ma, y_column="robotics")

# %%
# pd.DataFrame([
#     au.magnitude(
#         industry_orgs.reset_index().assign(year=lambda x: x.time_period.dt.year).drop('time_period', axis=1), 2016, 2020
#     ),
#     au.smoothed_growth(df_ma, 2015, 2020)
# ], columns=['magnitude', 'growth'])

# %%
magnitude_growth = (
    au.magnitude(
        industry_orgs.reset_index()
        .assign(year=lambda x: x.time_period.dt.year)
        .drop("time_period", axis=1),
        2016,
        2020,
    )
    .to_frame("magnitude")
    .assign(growth=au.smoothed_growth(df_ma, 2015, 2020))
)

# %%
alt.Chart(magnitude_growth.reset_index()).mark_circle(size=60).encode(
    x="magnitude", y="growth", tooltip=["index", "growth"]
).interactive()

# %%
industries = ["information services"]
df_funds = au.get_companies_with_funds(
    CB.select_companies_by_industries(cb_hits_digital, industries)
)

# %%
# df_funds

# %%
df_funds[
    [
        "name",
        "short_description",
        "long_description",
        "homepage_url",
        "country",
        "founded_on",
        "total_funding_usd",
        "num_funding_rounds",
        "num_exits",
    ]
].sort_values("total_funding_usd", ascending=False).head(15)


# %% [markdown]
# Need to get benchmarks for these...

# %% [markdown]
# ### Investment amounts

# %%
len(cb_hits_digital)

# %%
benchmark_df.head(1)

# %%
df_funds = CB.get_funding_rounds(cb_hits)
df_funds_digital = CB.get_funding_rounds(cb_hits_digital)

# %%
cb_hits_timeseries = au.cb_get_all_timeseries(cb_hits, df_funds, "year", 2005, 2020)
cb_hits_digital_timeseries = au.cb_get_all_timeseries(
    cb_hits_digital, df_funds_digital, "year", 2005, 2020
)


# %%
pu.time_series(cb_hits_timeseries, y_column="no_of_rounds")

# %%
pu.time_series(cb_hits_timeseries, y_column="raised_amount_gbp_total")

# %%
cb_hits_timeseries_norm = cb_hits_timeseries.copy()
cb_hits_timeseries_norm.raised_amount_gbp_total = (
    cb_hits_timeseries_norm.raised_amount_gbp_total
    / benchmark_df.raised_amount_gbp_total
)
cb_hits_timeseries_norm.no_of_rounds = (
    cb_hits_timeseries_norm.no_of_rounds / benchmark_df.no_of_rounds
)

# %%
pu.time_series(cb_hits_timeseries_norm, y_column="raised_amount_gbp_total")

# %%
pu.time_series(cb_hits_timeseries_norm, y_column="no_of_rounds")

# %%
df_funds = CB.get_funding_rounds(cb_hits)
df_funds_digital = CB.get_funding_rounds(cb_hits_digital)
cb_hits_timeseries = au.cb_get_all_timeseries(cb_hits, df_funds, "year", 2005, 2020)
cb_hits_digital_timeseries = au.cb_get_all_timeseries(
    cb_hits_digital, df_funds_digital, "year", 2005, 2020
)

pu.time_series(cb_hits_timeseries, y_column="no_of_rounds")

# %%
hit_funding_rounds = (
    CB.get_company_industries(cb_hits_digital)
    .query("industry in @DIGITAL_INDUSTRIES")
    .pipe(CB.get_funding_rounds)
)

# %%
hit_funding_rounds_ = hit_funding_rounds[hit_funding_rounds.raised_amount_gbp < 100000]

# %%
industry_investment = au.cb_get_all_timeseries(
    cb_hits_digital, hit_funding_rounds, period="year", min_year=MIN_YEAR, max_year=2020
).set_index("time_period")

# %%
# industry_investment

# %%
hit_company_industries = CB.get_company_industries(cb_hits_digital)

# %%
industry_investment = au.cb_get_all_timeseries(
    cb_hits_digital, hit_funding_rounds, period="year", min_year=MIN_YEAR, max_year=2020
).set_index("time_period")

for i, row in digital_counts.iterrows():
    industry = row.industry

    hit_ids = hit_company_industries.query("industry == @industry").id.to_list()
    df = cb_hits_digital.query("id in @hit_ids")

    industry_investment[industry] = au.cb_get_all_timeseries(
        df, CB.get_funding_rounds(df), period="year", min_year=MIN_YEAR, max_year=2020
    ).set_index("time_period")["raised_amount_gbp_total"]

# %%
cols_to_drop = [
    "no_of_rounds",
    "raised_amount_usd_total",
    "raised_amount_gbp_total",
    "no_of_orgs_founded",
]

# %%
industry_investment = industry_investment.drop(cols_to_drop, axis=1)

# %%
industry_deals = au.cb_get_all_timeseries(
    cb_hits_digital, hit_funding_rounds, period="year", min_year=MIN_YEAR, max_year=2020
).set_index("time_period")

for i, row in digital_counts.iterrows():
    industry = row.industry

    hit_ids = hit_company_industries.query("industry == @industry").id.to_list()
    df = cb_hits_digital.query("id in @hit_ids")

    industry_deals[industry] = au.cb_get_all_timeseries(
        df, CB.get_funding_rounds(df), period="year", min_year=MIN_YEAR, max_year=2020
    ).set_index("time_period")["no_of_rounds"]

# %%
industry_deals = industry_deals.drop(cols_to_drop, axis=1)

# %%
df_ma = (
    industry_investment.reset_index()
    .assign(year=lambda x: x.time_period.dt.year)
    .pipe(au.moving_average, replace_columns=True)
    .assign(time_period=industry_deals.reset_index()["time_period"])
)

# %%
magnitude_growth = (
    au.magnitude(
        industry_investment.reset_index()
        .assign(year=lambda x: x.time_period.dt.year)
        .drop("time_period", axis=1),
        2016,
        2020,
    )
    .to_frame("magnitude")
    .assign(growth=au.smoothed_growth(df_ma, 2015, 2020))
)

# %%
pu.time_series(df_ma, y_column="online portals")

# %%
alt.Chart(magnitude_growth.reset_index()).mark_circle(size=60).encode(
    x="magnitude", y="growth", tooltip=["index", "magnitude", "growth"]
).interactive()

# %%
magnitude_growth.sort_values("growth", ascending=False).head(10)

# %%
industries = ["media and entertainment"]
df_funds = au.get_companies_with_funds(
    CB.select_companies_by_industries(cb_hits_digital, industries)
)

df_funds[
    [
        "name",
        "short_description",
        "long_description",
        "homepage_url",
        "country",
        "founded_on",
        "total_funding_usd",
        "num_funding_rounds",
        "num_exits",
    ]
].sort_values("total_funding_usd", ascending=False).head(15)


# %% [markdown]
# ### Investment over time
#
# - Use CB total as baseline

# %%
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu

# %%
cb_hits_relevant = cb_hits_.query("soft_cluster in @relevant_clusters").query(
    'country == "United States"'
)
funding_df = CB.get_funding_rounds(cb_hits_relevant)

# %%
funding_ts = au.cb_get_all_timeseries(cb_hits_relevant, funding_df, "year", 2010, 2021)

# %%
funding_ts.head(2)

# %%
pu.time_series(funding_ts, y_column="raised_amount_gbp_total")

# %%
pu.time_series(funding_ts, y_column="no_of_rounds")

# %%
pu.time_series(funding_ts, y_column="no_of_orgs_founded")

# %%
cb_hits_relevant.sort_values("total_funding_usd", ascending=False).head(15)[
    ["name", "short_description", "long_description", "total_funding_usd"]
]

# %%
cb_hits_relevant.groupby("country").sum().sort_values(
    "total_funding_usd", ascending=False
).head(15)

# %%
cb_hits_relevant.groupby("soft_cluster").median().sort_values(
    "total_funding_usd", ascending=False
).head(15)

# %%
for i in relevant_clusters:
    print(i, clusters_labels[i])

# %%
# cb_hits_.groupby('soft_cluster').sum()

# %%
