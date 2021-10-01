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
# # Inspect top2vec clusters
#
# - Check average cluster size
# - Try reducing the number of clusters using top2vec API
# - Map businesses to these clusters; which number of epochs is better for this purpose?

# %%
from innovation_sweet_spots.utils.io import load_pickle, save_pickle
from innovation_sweet_spots.analysis import top2vec
import innovation_sweet_spots.analysis.analysis_utils as iss
import pandas as pd
import numpy as np
from innovation_sweet_spots import logging, PROJECT_DIR
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
from innovation_sweet_spots.getters.green_docs import (
    get_green_gtr_docs,
    get_green_cb_docs,
    get_green_cb_docs_by_country,
)
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.green_document_utils import (
    find_green_gtr_projects,
    find_green_cb_companies,
    cb_categories_for_group,
)
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import adjusted_mutual_info_score as ami_score
import altair as alt

alt.data_transformers.disable_max_rows()
from tqdm.notebook import tqdm
import innovation_sweet_spots.utils.io as iss_io

# Import Crunchbase data
from innovation_sweet_spots.getters import crunchbase

# %%
import importlib

# %%
import innovation_sweet_spots.utils.altair_save_utils as altair_save

importlib.reload(altair_save)
driver = altair_save.google_chrome_driver_setup()


# %%
def topic_keywords(documents, clusters, topic_words, n=10, Vectorizer=TfidfVectorizer):
    # Create large "cluster documents" for finding best topic words based on tf-idf scores
    document_cluster_memberships = clusters
    cluster_ids = sorted(np.unique(document_cluster_memberships))
    cluster_docs = {i: [] for i in cluster_ids}
    for i, clust in enumerate(document_cluster_memberships):
        cluster_docs[clust] += documents[i]

    vectorizer = Vectorizer(  # CountVectorizer(
        analyzer="word",
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(list(cluster_docs.values()))

    id_to_token = dict(
        zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
    )

    clust_words = []
    for i in range(X.shape[0]):
        x = X[i, :].todense()
        topic_word_counts = [
            X[i, vectorizer.vocabulary_[token]] for token in topic_words[i]
        ]
        best_i = np.flip(np.argsort(topic_word_counts))
        top_n = best_i[0:n]
        words = [topic_words[i][t] for t in top_n]
        clust_words.append(words)
    logging.info(f"Generated keywords for {len(cluster_ids)} topics")
    return clust_words


def reduce_clustering(clustering, top2vec_model, reductions):
    clustering_reduced = clustering.copy()
    # Reduce clustering and generate labels
    for reduction in reductions:
        level = reduction["level"]
        n_clusters = reduction["n_clusters"]
        clustering_reduced = reduce_topics(
            n_clusters, top2vec_model, clustering_reduced
        ).rename(
            columns={
                "reduced_cluster_id": f"cluster_id_level_{level}",
                "reduced_cluster_keywords": f"keywords_level_{level}",
            }
        )
    # Rename the original labels
    clustering_reduced = clustering_reduced.rename(
        columns={
            "cluster_id": f"cluster_id_level_{level+1}",
            "cluster_keywords": f"keywords_level_{level+1}",
            "wiki_labels": f"wiki_labels_level_{level+1}",
        }
    )
    return clustering_reduced


def get_cluster_topics_table(clustering_reduced, level):
    return (
        clustering_reduced.sort_values(f"cluster_id_level_{level}")
        .drop_duplicates(f"cluster_id_level_{level}")[
            [f"cluster_id_level_{level}", f"keywords_level_{level}"]
        ]
        .reset_index(drop=True)
    )


def reduce_topics(n, top2vec_model, clustering):
    # Reduce the number of topics
    topic_hierarchy = top2vec_model.hierarchical_topic_reduction(n)
    # Create new labels
    cluster_labels_reduced = dict(zip(range(len(topic_hierarchy)), topic_hierarchy))
    base_labels_to_reduced = {
        c: key for key in cluster_labels_reduced for c in cluster_labels_reduced[key]
    }
    clustering_reduced = clustering.copy()
    clustering_reduced["reduced_cluster_id"] = clustering_reduced.cluster_id.apply(
        lambda x: base_labels_to_reduced[x]
    )
    topic_words, _, _ = top2vec_model.get_topics(reduced=True)
    clust_words = topic_keywords(
        top2vec_model.documents,
        clustering_reduced["reduced_cluster_id"].to_list(),
        topic_words,
        n=10,
    )
    clust_words = {i: str(keywords) for i, keywords in enumerate(clust_words)}
    clustering_reduced["reduced_cluster_keywords"] = clustering_reduced[
        "reduced_cluster_id"
    ].apply(lambda x: clust_words[x])
    return clustering_reduced


# Change the tfidf keywords
def recheck_original_cluster_topic_words(clustering_reduced, top2vec_model, level=3):
    topic_words, _, _ = top2vec_model.get_topics(reduced=False)
    keywords = topic_keywords(
        top2vec_model.documents, top2vec_model.doc_top, topic_words, n=10
    )
    keywords = {i: str(k) for i, k in enumerate(keywords)}
    clustering_reduced[f"keywords_level_{level}"] = clustering_reduced[
        f"cluster_id_level_{level}"
    ].apply(lambda x: keywords[x])
    return clustering_reduced


def add_wiki_labels(run, clustering_docs, cluster_id_col="cluster_id"):
    # Get topic labels
    wiki_labels = iss_topics.get_wiki_topic_labels(run)
    # Add wiki labels
    clustering_docs["wiki_labels"] = clustering_docs[cluster_id_col].apply(
        lambda x: wiki_labels[x]
    )
    return clustering_docs


def collect_cluster_prob(clust, top2vec_model, clustering):
    clust_docs = np.where(top2vec_model.doc_top == clust)[0]
    clust_probs = top2vec_model.cluster.probabilities_[clust_docs]
    clustering_ = clustering.iloc[clust_docs].copy()
    clustering_["probability"] = clust_probs
    clustering_ = clustering_.sort_values("probability", ascending=False)
    return clustering_


def get_cluster_counts(
    clustering_reduced, cluster_col="cluster_id_level_3", sources=["cb", "gtr"]
):
    cluster_topics_levels = {
        i: get_cluster_topics_table(clustering_reduced, i) for i in range(1, 4)
    }
    df = (
        clustering_reduced[clustering_reduced.source.isin(sources)]
        .groupby(cluster_col)
        .agg(counts=("doc_id", "count"))
        .sort_values("counts", ascending=False)
        .reset_index()
        .merge(cluster_topics_levels[int(cluster_col[-1])])
    )
    return df


# %%

# %%
def fetch_project_data(clustering_reduced):
    green_projects_ = find_green_gtr_projects()
    green_projects = clustering_reduced[clustering_reduced.source == "gtr"].copy()
    green_projects["project_id"] = green_projects["doc_id"]
    green_projects = green_projects.merge(
        green_projects_[["project_id", "start"]], how="left"
    )
    return green_projects


def get_cluster_funding_level(clust, level, clustering, funded_projects, min_year=2010):
    clust_proj_ids = clustering[
        clustering[f"cluster_id_level_{level}"] == clust
    ].doc_id.to_list()
    df = funded_projects[funded_projects.project_id.isin(clust_proj_ids)].copy()
    cluster_funding = iss.gtr_funding_per_year(df, min_year=min_year)
    return cluster_funding


def cluster_col_name(level):
    return f"cluster_id_level_{level}"


def keywords_col_name(level):
    return f"keywords_level_{level}"


def describe_clusters(clustering_reduced, funded_projects, level=3):
    # Columns we will be creating
    data_cols = [
        "funding_2020_sma5",
        "funding_2020_cum5",
        "funding_growth",
        "funding_growth_abs",
        "projects_2020_sma5",
        "projects_2020_cum5",
        "projects_growth",
        "projects_growth_abs",
    ]
    # Dataframe with the unique topics
    cluster_topics = get_cluster_topics_table(clustering_reduced, level)
    for col in data_cols:
        cluster_topics[col] = 0

    logging.info(f"Assessing {len(cluster_topics)} level {level} clusters")
    for i, c in enumerate(cluster_topics[cluster_col_name(level)].to_list()):
        cluster_funding = get_cluster_funding_level(
            c, level, clustering_reduced, funded_projects
        )
        cluster_topics.loc[i, "funding_2020_sma5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].amount_total.mean()
        cluster_topics.loc[i, "funding_2020_cum5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].amount_total.sum()
        cluster_topics.loc[i, "funding_growth"] = iss.estimate_growth_level(
            cluster_funding, growth_rate=True
        )
        cluster_topics.loc[i, "funding_growth_abs"] = compare_year_stats(
            get_moving_average(cluster_funding, window=5), column="amount_total_sma5"
        )

        cluster_topics.loc[i, "projects_2020_sma5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].no_of_projects.mean()
        cluster_topics.loc[i, "projects_2020_cum5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].no_of_projects.sum()
        cluster_topics.loc[i, "projects_growth"] = iss.estimate_growth_level(
            cluster_funding, column="no_of_projects", growth_rate=True
        )
        cluster_topics.loc[i, "projects_growth_abs"] = compare_year_stats(
            get_moving_average(cluster_funding, window=5), column="no_of_projects_sma5"
        )

    return cluster_topics


def fetch_cb_company_data(clustering_reduced):
    green_cb = clustering_reduced[clustering_reduced.source == "cb"].copy()
    green_cb["id"] = green_cb["doc_id"]
    green_cb["name"] = green_cb["title"]
    return green_cb


def get_cluster_investment_level(
    clust, level, clustering, cb_funding_rounds, min_year=2010
):
    clust_proj_ids = clustering[
        clustering[f"cluster_id_level_{level}"] == clust
    ].doc_id.to_list()
    df = clustering[
        clustering.doc_id.isin(clust_proj_ids) & (clustering.source == "cb")
    ].copy()
    fund_rounds = iss.get_cb_org_funding_rounds(df, cb_funding_rounds)
    funding_per_year = iss.get_cb_funding_per_year(fund_rounds, min_year=min_year)
    return funding_per_year


def describe_clusters_cb(clustering_reduced, funded_projects, level=3):
    # Columns we will be creating
    data_cols = [
        "rounds_2020_sma5",
        "rounds_2020_cum5",
        "rounds_growth",
        "rounds_growth_abs",
        "investment_usd_2020_sma5",
        "investment_usd_2020_cum5",
        "investment_growth",
        "investment_growth_abs",
    ]
    # Dataframe with the unique topics
    cluster_topics = get_cluster_topics_table(clustering_reduced, level)
    for col in data_cols:
        cluster_topics[col] = 0
    logging.info(f"Assessing {len(cluster_topics)} level {level} clusters")
    for i, c in enumerate(cluster_topics[cluster_col_name(level)].to_list()):
        cluster_funding = get_cluster_investment_level(
            c, level, clustering_reduced, funded_projects
        )
        cluster_topics.loc[i, "rounds_growth"] = iss.estimate_growth_level(
            cluster_funding, column="no_of_rounds", growth_rate=True
        )
        cluster_topics.loc[i, "rounds_growth_abs"] = compare_year_stats(
            get_moving_average(cluster_funding, window=5), column="no_of_rounds_sma5"
        )
        cluster_topics.loc[i, "rounds_2020_cum5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].no_of_rounds.sum()
        cluster_topics.loc[i, "rounds_2020_sma5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].no_of_rounds.mean()

        cluster_topics.loc[i, "investment_growth"] = iss.estimate_growth_level(
            cluster_funding, column="raised_amount_usd_total", growth_rate=True
        )
        cluster_topics.loc[i, "investment_growth_abs"] = compare_year_stats(
            get_moving_average(cluster_funding, window=5),
            column="raised_amount_usd_total_sma5",
        )
        cluster_topics.loc[i, "investment_usd_2020_cum5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].raised_amount_usd_total.sum()
        cluster_topics.loc[i, "investment_usd_2020_sma5"] = cluster_funding[
            cluster_funding.year.isin(range(2016, 2021))
        ].raised_amount_usd_total.mean()

    return cluster_topics


def get_funded_projects(green_projects):
    gtr_funds = gtr.get_gtr_funds()
    link_gtr_funds = gtr.get_link_table("gtr_funds")
    gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
    funded_projects = iss.get_gtr_project_funds(green_projects, gtr_project_funds)
    del link_gtr_funds, gtr_project_funds
    return funded_projects


def get_cluster_stats(clustering_reduced):
    # Prepare tables for funding data
    green_projects = fetch_project_data(clustering_reduced)
    funded_projects = get_funded_projects(green_projects)

    # Prepare tables for crunchbase data
    cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()
    green_cb = fetch_cb_company_data(clustering_reduced)

    cluster_stats = {}
    for level in range(1, 4):
        clusters_research = describe_clusters(green_projects, funded_projects, level)
        clusters_business = describe_clusters_cb(green_cb, cb_funding_rounds, level)
        cluster_stats[level] = clusters_research.merge(
            clusters_business,
            on=[cluster_col_name(level), keywords_col_name(level)],
            how="left",
        )

    return cluster_stats


def get_higher_level_topics(clustering_reduced_stats, level):
    """Adds higher level cluster topics"""
    all_clust_columns = [cluster_col_name(level) for level in range(1, level + 1)] + [
        keywords_col_name(level) for level in range(1, level + 1)
    ]
    all_clust_topics = (
        clustering_reduced[all_clust_columns]
        .drop_duplicates(all_clust_columns)
        .sort_values(all_clust_columns)
    )
    clustering_level_stats = (
        clustering_reduced_stats[level]
        .merge(
            all_clust_topics[all_clust_columns],
            on=[cluster_col_name(level), keywords_col_name(level)],
            how="left",
        )
        .sort_values(all_clust_columns)
    )
    stat_cols = list(
        set(list(clustering_level_stats.columns)).difference(set(all_clust_columns))
    )
    clustering_level_stats = clustering_level_stats[
        all_clust_columns + sorted(stat_cols)
    ]
    return clustering_level_stats


def get_moving_average(clust_funding, window=5):
    df = (
        clust_funding.rolling(window, min_periods=1)
        .mean()
        .drop("year", axis=1)
        .rename(
            columns=dict(
                zip(
                    clust_funding.drop("year", axis=1).columns,
                    [
                        f"{s}_sma{window}"
                        for s in clust_funding.drop("year", axis=1).columns
                    ],
                )
            )
        )
    )
    return pd.concat([clust_funding, df], axis=1)


def get_moving_sum(clust_funding, window=5):
    df = (
        clust_funding.rolling(window, min_periods=1)
        .sum()
        .drop("year", axis=1)
        .rename(
            columns=dict(
                zip(
                    clust_funding.drop("year", axis=1).columns,
                    [
                        f"{s}_cum{window}"
                        for s in clust_funding.drop("year", axis=1).columns
                    ],
                )
            )
        )
    )
    return pd.concat([clust_funding, df], axis=1)


def get_cluster_source_counts(clustering_reduced, level):
    cluster_source_counts = (
        pd.concat(
            [
                get_cluster_counts(
                    clustering_reduced,
                    sources=["gtr"],
                    cluster_col=f"cluster_id_level_{level}",
                )
                .rename(columns={"counts": "gtr_counts"})
                .merge(
                    get_cluster_counts(
                        clustering_reduced,
                        sources=["cb"],
                        cluster_col=f"cluster_id_level_{level}",
                    ).rename(columns={"counts": "cb_counts"}),
                    on=[f"cluster_id_level_{level}", f"keywords_level_{level}"],
                    how="left",
                )
            ],
            axis=1,
        )
        .sort_values(f"cluster_id_level_{level}")
        .fillna(0)[
            [
                f"cluster_id_level_{level}",
                f"keywords_level_{level}",
                "gtr_counts",
                "cb_counts",
            ]
        ]
    )
    cluster_source_counts["total_counts"] = (
        cluster_source_counts["cb_counts"] + cluster_source_counts["gtr_counts"]
    )
    return cluster_source_counts


# Get full cluster keywords for all three levels
def get_full_keywords_table(clustering_reduced, level=3):
    all_clust_columns = [cluster_col_name(level) for level in range(1, level + 1)] + [
        keywords_col_name(level) for level in range(1, level + 1)
    ]
    clusters = (
        clustering_reduced.sort_values(all_clust_columns)
        .drop_duplicates(all_clust_columns)
        .merge(
            get_cluster_source_counts(clustering_reduced, level=3),
            on=["cluster_id_level_3", "keywords_level_3"],
        )
    )
    return clusters[
        all_clust_columns
        + ["wiki_labels_level_3", "gtr_counts", "cb_counts", "total_counts"]
    ]


def prep_cluster_table(clustering_reduced_stats, clustering_reduced, level):
    cluster_source_counts = get_cluster_source_counts(clustering_reduced, level)
    df = (
        get_higher_level_topics(clustering_reduced_stats, level=level)
        .drop([keywords_col_name(i) for i in range(1, level)], axis=1)
        .merge(
            cluster_source_counts,
            on=[cluster_col_name(level), keywords_col_name(level)],
        )
    )
    return df


def compare_year_stats(df, column, first_year=2015, second_year=2020, absolute=True):
    first_year_stat = df.loc[df.year == first_year, column].iloc[0]
    second_year_stat = df.loc[df.year == second_year, column].iloc[0]
    if absolute:
        return second_year_stat - first_year_stat
    else:
        return second_year_stat / first_year_stat


def prepare_cluster_item_list(clust, top2vec_model, clustering_reduced):
    df = collect_cluster_prob(clust, top2vec_model, clustering_reduced)
    _, document_scores, document_ids = top2vec_model.search_documents_by_topic(
        topic_num=clust, num_docs=len(df)
    )
    df_scores = pd.DataFrame(index=document_ids, data={"topic_score": document_scores})
    df["topic_score"] = df_scores["topic_score"]
    df = df.sort_values(["probability", "topic_score"], ascending=False)
    df = (
        df[
            [
                "cluster_id_level_3",
                "keywords_level_3",
                "source",
                "title",
                "description",
                "year",
                "probability",
                "topic_score",
            ]
        ]
        .reset_index()
        .rename(columns={"index": "item_number"})
    )
    df.description = df.description.apply(lambda x: x[0:150] + "...")
    return df


# %%
RESULTS_DIR = PROJECT_DIR / "outputs/data/results_july"

# %%
run = "July2021"
run = "July2021_projects_orgs"
run = "July2021_projects_orgs_stopwords"
run = "July2021_projects_orgs_stopwords_e100"
run = "July2021_projects_orgs_stopwords_e400"
clustering = iss_topics.get_clustering(run)
clustering.head(1)

# %% [markdown]
# ## Characterise basic stats

# %%
counts = iss_topics.get_cluster_counts(clustering)
iss_topics.plot_histogram(counts)

# %%
clustering_docs = iss_topics.get_doc_details(clustering)

# %%
# Import top2vec model
top2vec_model = iss_topics.get_top2vec_model(run)

# %%
# Generate visualisation embeddings
xy, umap_model = iss_topics.umap_document_vectors(top2vec_model)

# %%
clustering_docs = add_wiki_labels(run, clustering_docs)
# Generate visualisation embeddings
clustering_docs["x"] = xy[:, 0]
clustering_docs["y"] = xy[:, 1]

# %%
reductions = [
    {"level": 1, "n_clusters": 7},
    {"level": 2, "n_clusters": 75},
]

clustering_reduced = reduce_clustering(clustering_docs, top2vec_model, reductions)
clustering_reduced = recheck_original_cluster_topic_words(
    clustering_reduced, top2vec_model
)

# %%
clustering_reduced_stats = get_cluster_stats(clustering_reduced)

# %% [markdown]
# ### Add extra info

# %%
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()
green_cb = fetch_cb_company_data(clustering_reduced)


# %%
def add_years_to_docs(clustering_reduced):
    # Extra information
    green_projects = fetch_project_data(clustering_reduced)
    cb_orgs = crunchbase.get_crunchbase_orgs()
    green_companies = fetch_cb_company_data(clustering_reduced).merge(
        cb_orgs[["id", "founded_on"]], on="id", how="left"
    )
    green_companies["year"] = green_companies.founded_on.apply(iss.convert_date_to_year)
    green_projects["year"] = green_projects["start"].apply(iss.convert_date_to_year)
    doc_years = pd.concat(
        [green_companies[["doc_id", "year"]], green_projects[["doc_id", "year"]]],
        axis=0,
        ignore_index=True,
    )
    clustering_reduced_year = clustering_reduced.merge(doc_years, how="left")
    return clustering_reduced_year


# %%
def add_amounts_to_docs(clustering_reduced):
    green_projects = fetch_project_data(clustering_reduced)

    gtr_funds = gtr.get_gtr_funds()
    gtr_organisations = gtr.get_gtr_organisations()
    # Links tables
    link_gtr_funds = gtr.get_link_table("gtr_funds")
    link_gtr_organisations = gtr.get_link_table("gtr_organisations")
    link_gtr_topics = gtr.get_link_table("gtr_topic")

    gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
    funded_projects = iss.get_gtr_project_funds(green_projects, gtr_project_funds)
    del link_gtr_funds

    return funded_projects


# %%
green_cb_amount = iss.get_cb_org_funding_rounds(green_cb, cb_funding_rounds)
green_cb_amount["amount"] = green_cb_amount["raised_amount_usd"] * 0.72
green_cb_deals = green_cb.merge(
    green_cb_amount[["org_id", "announced_on", "investment_type", "amount"]],
    left_on="doc_id",
    right_on="org_id",
).drop(["org_id", "id", "name"], axis=1)


# %%
green_projects_funded = add_amounts_to_docs(clustering_reduced)

# %%
green_projects_funded["year"] = green_projects_funded["start"].apply(
    iss.convert_date_to_year
)

# %%
# iss.cb_orgs_with_most_funding(orgs_with_term).sort_values("founded_on", ascending=False)

# %%
# funded_projects = get_funded_projects(green_projects)

# %%
# # Find organisations and subgroups within the 'green' category
# cb_categories = crunchbase.get_crunchbase_category_groups()
# cb_org_categories = crunchbase.get_crunchbase_organizations_categories()
# green_tags = cb_categories_for_group(cb_categories, "Sustainability")
# green_orgs = cb_org_categories[cb_org_categories.category_name.isin(green_tags)]

# %%
# green_orgs.head(1)

# %%
# df = clustering_reduced[['cluster_id_level_3', 'keywords_level_3', 'doc_id']].merge(green_orgs[['organization_id', 'category_name']], left_on='doc_id', right_on='organization_id')

# %%
# df = df.groupby(['cluster_id_level_3','category_name']).agg(counts=('doc_id', 'count')).reset_index().sort_values(['cluster_id_level_3','counts'])
#

# %%
# cb_categories = cb_org_categories[cb_org_categories.organization_id.isin(green_companies.doc_id.to_list())]

# %% [markdown]
# ### Output tables
# - Table with cluster keywords and labels
# - Table with cluster, level_3 stats and higher level cluster numbersclustering_reduced_stats

# %%
# Get stats for each cluster, at all three levels
clustering_level_stats = {
    level: prep_cluster_table(clustering_reduced_stats, clustering_reduced, level=level)
    for level in range(1, 4)
}
# Full keywords and labels table
clusters = get_full_keywords_table(clustering_reduced)

# %%
for level in clustering_level_stats:
    clustering_level_stats[level].to_csv(
        RESULTS_DIR / f"clusters_level_{level}_metrics.csv", index=False
    )
clusters.to_csv(RESULTS_DIR / f"clusters.csv", index=False)

# %%
clustering_reduced_year = add_years_to_docs(clustering_reduced)

# %%
for i in range(0, top2vec_model.get_num_topics()):
    df = prepare_cluster_item_list(i, top2vec_model, clustering_reduced_year)
    df.to_csv(RESULTS_DIR / f"clusters_level_3/items_cluster_{i}.csv", index=False)

# %%
clusters_with_deals = green_cb_deals.cluster_id_level_3.unique()
for c in clusters_with_deals:
    df = green_cb_deals[green_cb_deals.cluster_id_level_3 == c].sort_values(
        "announced_on"
    )
    df = df[
        [
            "cluster_id_level_3",
            "keywords_level_3",
            "title",
            "description",
            "announced_on",
            "investment_type",
            "amount",
        ]
    ]
    df.to_csv(
        RESULTS_DIR / f"clusters_level_3_deals/deals_cluster_{c}.csv", index=False
    )

# %%
# ' \n'.join(collect_cluster_prob(0, top2vec_model, clustering_reduced).title.iloc[0:5].to_list())

# %% [markdown]
# ### Double check the soft clusters

# %%
clusters_kk = pd.read_excel(RESULTS_DIR / "clusters_KK_checked.xlsx")


# %%
# soft_clusters = clusters_kk[clusters_kk.is_soft.isin([0.5,1])].cluster_id_level_3.to_list()
# tech_clusters = clusters_kk[clusters_kk.is_tech==1].cluster_id_level_3.to_list()

# %%
# soft_clusters

# %%
# soft_docs = prepare_cluster_item_list(43, top2vec_model, clustering_reduced_year)

# %%
# clustering_reduced_year.loc[10675].description

# %%
# i=95
# soft_docs.iloc[i]

# %%
# top_topics = top2vec_model.get_documents_topics([soft_docs.iloc[i].item_number], reduced=False, num_topics=10)

# %%
# tt = [t for t in top_topics[0][0] if t not in soft_clusters]
# get_cluster_topics_table(clustering_reduced, level=3).loc[tt]

# %%
# clust tt

# %%
# _, doc_scores, doc_ids = top2vec_model.search_documents_by_documents([soft_docs.iloc[i].item_number], num_docs=100)

# %%
# df = clustering_reduced_year.loc[doc_ids].copy()
# df['scores'] = doc_scores
# df = df[df.cluster_id_level_3.isin(soft_clusters) == False]
# df.sort_values('scores', ascending=False).head(20).groupby('keywords_level_3').agg(counts=('doc_id', 'count'))
# #     clustering_reduced_year.item_number.isin(doc_ids) &
# #     (clustering_reduced_year.cluster_id_level_3.isin(soft_clusters) == False)]

# %%
# doc_scores

# %% [markdown]
# ## Visualisations

# %%
def plot_clustering(
    clustering, selection, colour_col="cluster_keywords", tooltip=None, shape="source"
):
    if tooltip is None:
        tooltip = ["title", colour_col]
    return (
        alt.Chart(
            clustering,
            width=750,
            height=750,
        )
        .mark_circle(size=25)
        .encode(
            x=alt.X("x", axis=alt.Axis(grid=False)),
            y=alt.Y("y", axis=alt.Axis(grid=False)),
            #         size='size',
            #         color='cluster',
            shape=shape,
            color=alt.Color(
                colour_col,
                scale=alt.Scale(scheme="category20"),
                legend=alt.Legend(symbolLimit=150, labelLimit=250),
            ),
            tooltip=tooltip,
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.05)),
        )
        .interactive()
    )


# %% [markdown]
# ### Visuals: Landscape visual
# - All documents
# - Topic centres

# %%
importlib.reload(iss_topics)

# %%
viz_landscape = clustering_reduced_year.copy().reset_index()
viz_landscape["keywords_level_3"] = [
    f"{i}_{j}"
    for (i, j) in zip(
        clustering_reduced_year["cluster_id_level_3"].astype(str),
        clustering_reduced_year["keywords_level_3"],
    )
]
viz_landscape["keywords_level_2"] = [
    f"{i}_{j}"
    for (i, j) in zip(
        clustering_reduced_year["cluster_id_level_2"].astype(str),
        clustering_reduced_year["keywords_level_2"],
    )
]

viz_landscape["x"] = xy[:, 0]
viz_landscape["y"] = xy[:, 1]

# %%
level = 3

selection = alt.selection_multi(fields=[f"keywords_level_{level}"], bind="legend")

fig = plot_clustering(
    #     clustering_reduced[clustering_reduced.source=='cb'].reset_index(),
    viz_landscape,
    selection,
    colour_col=f"keywords_level_{level}",
    shape="source",
    tooltip=[
        "index",
        "title",
        "description",
        "source",
        "cluster_id_level_1",
        "cluster_id_level_2",
        "keywords_level_3",
    ],
).add_selection(selection)

# %%
level = 3
selection = alt.selection_multi(fields=[f"keywords_level_{level}"], bind="legend")

base = (
    (
        alt.Chart(
            viz_landscape,
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
                f"keywords_level_{level}",
                scale=alt.Scale(scheme="category20"),
                legend=alt.Legend(symbolLimit=140, labelLimit=250),
            ),
            tooltip=[
                "index",
                "title",
                "description",
                "source",
                "cluster_id_level_1",
                "cluster_id_level_2",
                "keywords_level_3",
            ],
            opacity=alt.condition(selection, alt.value(0.99), alt.value(0.05)),
        )
    )
    .interactive()
    .add_selection(selection)
)

# %%
# base

# %%
# year_slider = alt.binding_range(min=2007, max=2020, step=1)
# slider_selection = alt.selection_single(bind=year_slider, fields=['year'], name="year_")

# (
#     base
#     .add_selection(
#         slider_selection
#     ).transform_filter(
#         slider_selection
#     ).properties(title="Slider Filtering")
# )

# %%
# fig

# %%
# filter_year = base.add_selection(
#     slider_selection
# ).transform_filter(
#     slider_selection
# ).properties(title="Slider Filtering")

# %%
# fig

# %%
altair_save.save_altair_to_path(
    base, f"explorer_Innovation_landscape_Level{level}", driver, path=RESULTS_DIR
)

# %% [markdown]
# ### Topics

# %%
import ast

# %%
clusters_kk = pd.read_excel(RESULTS_DIR / "clusters_KK_checked.xlsx")
df = clusters_kk.sort_values("cluster_id_level_3")
xy_topics = umap_model.transform(top2vec_model.topic_vectors)
df["x"] = xy_topics[:, 0]
df["y"] = xy_topics[:, 1]
df["text"] = df["keywords_level_3"].apply(lambda x: ", ".join(ast.literal_eval(x)[0:3]))

# %%
fig = (
    alt.Chart(
        df,
        width=750,
        height=750,
    )
    .mark_circle(size=25)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        size="total_counts",
        color=alt.Color("is_tech:N", scale=alt.Scale(scheme="dark2")),
        tooltip=["keywords_level_3", "cluster_id_level_3"],
    )
)

text = (
    alt.Chart(df, width=750, height=750)
    .mark_circle(size=25, color="w")
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
    )
    .mark_text(align="left", baseline="middle", dx=7, fontSize=10, fontWeight=300)
    .encode(text="text")
    .interactive()
)

# text = (
#     fig.mark_text(align="left", baseline="middle", dx=7, fontSize=10, fontWeight=300)
#     .encode(text="keywords_level_3")
#     .interactive()
# )

# %%
altair_save.save_altair_to_path(
    (fig + text), f"explorer_Innovation_topics_Level{3}", driver, path=RESULTS_DIR
)

# %% [markdown]
# ### Visuals: Matrix diagrams

# %%
clustering_level_stats[3].info()

# %% [markdown]
# ### Compare private and public

# %%
level = 2
window = 5
n_clusters = len(clustering_level_stats[level])
funding_all_clusters = pd.DataFrame()
for c in range(0, n_clusters):
    clust_funding = get_cluster_funding_level(
        c, level, green_projects, funded_projects, min_year=2007
    )
    clust_funding_sma = get_moving_average(clust_funding, window=window)
    clust_funding_sma[cluster_col_name(level)] = c
    clust_funding_sma = clust_funding_sma[clust_funding_sma.year >= 2007]
    funding_all_clusters = pd.concat([funding_all_clusters, clust_funding_sma])

n_clusters = len(clustering_level_stats[level])
investment_all_clusters = pd.DataFrame()
for c in range(0, n_clusters):
    clust_funding = get_cluster_investment_level(
        c, level, green_cb, cb_funding_rounds, min_year=2007
    )
    clust_funding_sma = get_moving_average(clust_funding, window=window)
    clust_funding_sma[cluster_col_name(level)] = c
    clust_funding_sma = clust_funding_sma[clust_funding_sma.year >= 2007]
    investment_all_clusters = pd.concat([investment_all_clusters, clust_funding_sma])

df_cb = investment_all_clusters[
    [f"cluster_id_level_{level}", "year", "raised_amount_usd_total_sma5"]
].rename(columns={"raised_amount_usd_total_sma5": "amount"})
df_cb["source"] = "cb"
df_cb["amount"] = df_cb["amount"] * 0.73 / 1000
df_gtr = funding_all_clusters[
    [f"cluster_id_level_{level}", "year", "amount_total_sma5"]
].rename(columns={"amount_total_sma5": "amount"})
df_gtr["source"] = "gtr"
combined_funding_investment = pd.concat([df_gtr, df_cb], axis=0, ignore_index=True)

# %%
tech_clusters = clusters_kk[clusters_kk.is_tech == 1][
    ["cluster_id_level_3", f"cluster_id_level_{level}"]
]
tech_clusters_level = tech_clusters[f"cluster_id_level_{level}"].to_list()


# %%
def explorer_time_series(dot_x, dot_y, times_y, level=2):
    df = (
        get_higher_level_topics(clustering_reduced_stats, level=level).merge(
            get_cluster_source_counts(clustering_reduced, level=level), how="left"
        )
        #     .merge(clusters_kk[['cluster_id_level_3', 'is_tech']], how='left')
    )
    df["is_tech"] = "not_tech"
    df.loc[
        df[f"cluster_id_level_{level}"].isin(tech_clusters_level), "is_tech"
    ] = "is_tech"

    selector = alt.selection_single(empty="all", fields=[cluster_col_name(level)])

    fig = (
        alt.Chart(df, width=500, height=300)
        .add_selection(selector)
        .mark_circle(size=60)
        .encode(
            y=dot_y,
            x=dot_x,
            size="total_counts",
            #         color=alt.Color(keywords_col_name(1)+":N"),
            #         color=alt.Color('is_tech:N',scale=alt.Scale(scheme="dark2")),
            color=alt.condition(selector, "is_tech:N", alt.value("lightgray")),
            tooltip=[
                cluster_col_name(level),
                keywords_col_name(level),
                "cb_counts",
                "gtr_counts",
            ],
        )
    ).interactive()

    timeseries = (
        alt.Chart(timeseries_df)
        .properties(width=500, height=250)
        .add_selection(selector)
        .mark_line()
        .encode(
            x="year:O",
            y=alt.Y(times_y),
            #         color=alt.Color(f'{cluster_col_name(level)}:O', legend=None)
            color="source",
        )
        .transform_filter(selector)
    )

    fig = alt.vconcat(
        fig,
        timeseries,
        title=f"Research funding and investment comparison (level {level} categories)",
    )
    return fig


# %%
dot_x = "funding_growth"
dot_y = "investment_growth"

dot_x = "funding_growth_abs"
dot_y = "investment_growth_abs"
# dot_y = 'projects_2020_sma5'
# timeseries_df = funding_all_clusters
# times_y = 'amount_total_sma5'

# dot_x = 'investment_growth_abs'
# dot_y = 'rounds_2020_sma5'
# times_y = 'no_of_rounds_sma5'
# times_y = 'raised_amount_usd_total_sma5'
timeseries_df = combined_funding_investment
times_y = "amount"

# %%
fig = explorer_time_series(dot_x, dot_y, times_y, level=2)
altair_save.save_altair_to_path(
    fig,
    "explorer_Research_funding_vs_Private_investment_abs_Level2",
    driver,
    path=RESULTS_DIR,
)

# %% [markdown]
# ### Research project explorer

# %%
dot_x = "funding_growth"
dot_y = "projects_2020_sma5"
# dot_y = 'projects_2020_sma5'
# timeseries_df = funding_all_clusters
# times_y = 'amount_total_sma5'

# dot_x = 'investment_growth_abs'
# dot_y = 'rounds_2020_sma5'
# times_y = 'no_of_rounds_sma5'
# times_y = 'raised_amount_usd_total_sma5'
# timeseries_df = combined_funding_investment
# times_y = 'amount'

green_projects_funded_ = green_projects_funded.copy()
green_projects_funded_["description"] = green_projects_funded["description"].apply(
    lambda x: x[0:150] + "..."
)

level = 3
df = (
    get_higher_level_topics(clustering_reduced_stats, level=level).merge(
        get_cluster_source_counts(clustering_reduced, level=level), how="left"
    )
    #     .merge(clusters_kk[['cluster_id_level_3', 'is_tech']], how='left')
)
df["is_tech"] = "not_tech"
df.loc[
    df[f"cluster_id_level_{level}"].isin(tech_clusters[f"cluster_id_level_{level}"]),
    "is_tech",
] = "is_tech"

selector = alt.selection_single(empty="all", fields=[cluster_col_name(level)])

fig = (
    alt.Chart(df, width=500, height=300)
    .add_selection(selector)
    .mark_circle(size=60)
    .encode(
        y=dot_y,
        x=dot_x,
        size="total_counts",
        #         color=alt.Color(keywords_col_name(1)+":N"),
        #         color=alt.Color('is_tech:N',scale=alt.Scale(scheme="dark2")),
        color=alt.condition(selector, "is_tech:N", alt.value("lightgray")),
        tooltip=[
            cluster_col_name(level),
            keywords_col_name(level),
            "cb_counts",
            "gtr_counts",
        ],
    )
).interactive()


timeseries = (
    alt.Chart(green_projects_funded_, width=500, height=250)
    .mark_point(opacity=0.8, size=20, clip=True, color="#6295c4")
    .encode(
        alt.X("year:O", scale=alt.Scale(domain=list(range(2007, 2021)))),
        alt.Y("amount:Q", scale=alt.Scale(domain=(0, 10e6))),
        tooltip=["title", "description", "year", "amount", f"cluster_id_level_{level}"],
    )
    .transform_filter(selector)
).interactive()


fig = alt.vconcat(
    fig,
    timeseries,
    title=f"Research projects and funding growth (level {level} categories)",
)


# %%
altair_save.save_altair_to_path(
    fig, f"explorer_Research_projects_Level{level}", driver, path=RESULTS_DIR
)

# %% [markdown]
# ### Deal explorer

# %%
dot_x = "investment_growth_abs"
dot_y = "rounds_2020_sma5"
# dot_y = 'projects_2020_sma5'
# timeseries_df = funding_all_clusters
# times_y = 'amount_total_sma5'

# dot_x = 'investment_growth_abs'
# dot_y = 'rounds_2020_sma5'
# times_y = 'no_of_rounds_sma5'
# times_y = 'raised_amount_usd_total_sma5'
# timeseries_df = combined_funding_investment
# times_y = 'amount'

green_cb_deals["year"] = green_cb_deals.announced_on.apply(iss.convert_date_to_year)

level = 2
df = (
    get_higher_level_topics(clustering_reduced_stats, level=level).merge(
        get_cluster_source_counts(clustering_reduced, level=level), how="left"
    )
    #     .merge(clusters_kk[['cluster_id_level_3', 'is_tech']], how='left')
)
df["is_tech"] = "not_tech"
df.loc[df[f"cluster_id_level_{level}"].isin(tech_clusters_level), "is_tech"] = "is_tech"

selector = alt.selection_single(empty="all", fields=[cluster_col_name(level)])

fig = (
    alt.Chart(df, width=500, height=300)
    .add_selection(selector)
    .mark_circle(size=60)
    .encode(
        y=dot_y,
        x=dot_x,
        size="total_counts",
        #         color=alt.Color(keywords_col_name(1)+":N"),
        #         color=alt.Color('is_tech:N',scale=alt.Scale(scheme="dark2")),
        color=alt.condition(selector, "is_tech:N", alt.value("lightgray")),
        tooltip=[
            cluster_col_name(level),
            keywords_col_name(level),
            "cb_counts",
            "gtr_counts",
        ],
    )
).interactive()


timeseries = (
    alt.Chart(green_cb_deals, width=500, height=250)
    .mark_point(opacity=0.8, size=20, clip=True, color="#6295c4")
    .encode(
        alt.X("year:O", scale=alt.Scale(domain=list(range(2007, 2021)))),
        alt.Y("amount:Q", scale=alt.Scale(domain=(0, 10e6))),
        tooltip=[
            "title",
            "description",
            "year",
            "investment_type",
            "amount",
            f"cluster_id_level_{level}",
        ],
    )
    .transform_filter(selector)
).interactive()


fig = alt.vconcat(fig, timeseries, title=f"Investment deals (level {level} categories)")


# %%
altair_save.save_altair_to_path(
    fig, f"explorer_Investment_deals_abs_Level{level}", driver, path=RESULTS_DIR
)

# %% [markdown]
# ## Subcluster analysis

# %%
# Extract cluster-specific vectors
# Repeat topic modelling?

# %%
df = clustering_reduced[clustering_reduced.cluster_id_level_3 == 0]

# %%
xy, umap_model = iss_topics.umap_document_vectors(top2vec_model)

# %%
# import umap
# umap_model = umap.UMAP(**iss_topics.umap_args_plotting).fit(
#     top2vec_model._get_document_vectors(norm=False)
# )
# xy = umap_model.transform(top2vec_model._get_document_vectors(norm=False))

# %%
# top2vec_model._get_document_vectors(norm=False)[]

# %%