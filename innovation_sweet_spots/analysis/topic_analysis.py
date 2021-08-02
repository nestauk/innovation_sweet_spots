## topic analysis_utils
import pandas as pd
import numpy as np
import altair as alt
from innovation_sweet_spots import logging, PROJECT_DIR
from innovation_sweet_spots.analysis import top2vec
import umap
import json
from innovation_sweet_spots.getters import gtr, crunchbase

##
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

from tqdm.notebook import tqdm
import innovation_sweet_spots.utils.io as iss_io

# Import Crunchbase data
from innovation_sweet_spots.getters import crunchbase

##

umap_args_plotting = {
    "n_neighbors": 15,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 112,
}

#### general processing


def get_doc_details(clustering: pd.DataFrame) -> pd.DataFrame:
    """
    Adds data about the clustered GTR and CB documents (titles and descriptions)

    Parameters
    clustering :
        Dataframe with columns 'doc_id', 'cluster_id', 'source' and 'cluster_keywords'

    Returns
    -------
    pd.DataFrame

    """

    gtr_projects = gtr.get_gtr_projects()[
        ["project_id", "title", "abstractText"]
    ].rename(
        columns={
            "project_id": "doc_id",
            "title": "title",
            "abstractText": "description",
        }
    )
    ##### To do: Add long description in there as well
    cb_orgs = (
        crunchbase.get_crunchbase_orgs_full()[["id", "name", "short_description"]]
        .drop_duplicates("id")
        .rename(
            columns={
                "id": "doc_id",
                "name": "title",
                "short_description": "description",
            }
        )
    )

    # if "source" not in clustering.columns:
    #     green_country_orgs["source"] = "cb"
    #     gtr_projects["source"] = "gtr"
    combined_df = pd.concat([gtr_projects, cb_orgs], axis=0, ignore_index=True)
    clustering_details = clustering.merge(combined_df, on="doc_id", how="left")
    del combined_df
    return clustering_details


def get_wiki_topic_labels(run):
    """Loads in wiki topic labels"""
    clust_labels = json.load(
        open(
            PROJECT_DIR
            / f"outputs/gtr_green_project_cluster_words_{run}_wiki_labels.json",
            "r",
        )
    )
    clust_labels = {d["id"]: d["labels"] for d in clust_labels}
    return clust_labels


def get_clustering(run):
    return pd.read_csv(PROJECT_DIR / f"outputs/data/gtr/top2vec_clusters_{run}.csv")


# def get_cluster_counts(clusterings, cluster_col="cluster_id"):
#     counts = (
#         clusterings.groupby(cluster_col)
#         .agg(counts=("doc_id", "count"))
#         .reset_index()
#         # .merge(
#         #     clusterings[["cluster_id", "cluster_keywords"]].drop_duplicates(
#         #         "cluster_id"
#         #     ),
#         #     how="left",
#         # )
#     )
#     return counts


def plot_histogram(df, x="counts", bin_step=10):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(f"{x}:Q", bin=alt.Bin(step=bin_step)),
            y="count()",
        )
    )


def plot_clustering(
    clustering, colour_col="cluster_keywords", tooltip=None, shape="source"
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
            color=alt.Color(colour_col, scale=alt.Scale(scheme="category20")),
            tooltip=tooltip,
        )
        .interactive()
    )


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


def get_cluster_topics_table(clustering_reduced, level):
    return (
        clustering_reduced.sort_values(f"cluster_id_level_{level}")
        .drop_duplicates(f"cluster_id_level_{level}")[
            [f"cluster_id_level_{level}", f"keywords_level_{level}"]
        ]
        .reset_index(drop=True)
    )


def add_wiki_labels(run, clustering_docs, cluster_id_col="cluster_id"):
    # Get topic labels
    wiki_labels = iss_topics.get_wiki_topic_labels(run)
    # Add wiki labels
    clustering_docs["wiki_labels"] = clustering_docs[cluster_id_col].apply(
        lambda x: wiki_labels[x]
    )
    return clustering_docs


def get_cluster_counts(
    clustering_reduced,
    cluster_col="cluster_id_level_3",
    sources=["cb", "gtr"],
    ii=1,
    jj=4,
):
    cluster_topics_levels = {
        i: get_cluster_topics_table(clustering_reduced, i) for i in range(ii, jj)
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


def cluster_col_name(level):
    return f"cluster_id_level_{level}"


def keywords_col_name(level):
    return f"keywords_level_{level}"


####################################
#### top2vec outputs processing ####
####################################


def get_top2vec_model(run):
    """Loads in top2vec model"""
    return top2vec.Top2Vec.load(
        PROJECT_DIR / f"outputs/models/top2vec_green_projects_{run}.p"
    )


def umap_document_vectors(top2vec_model):
    """Create a umap embedding to visualise top2vec model vectors"""
    umap_model = umap.UMAP(**umap_args_plotting).fit(
        top2vec_model._get_document_vectors(norm=False)
    )
    xy = umap_model.transform(top2vec_model._get_document_vectors(norm=False))
    return xy, umap_model


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


def collect_cluster_prob(clust, top2vec_model, clustering):
    clust_docs = np.where(top2vec_model.doc_top == clust)[0]
    clust_probs = top2vec_model.cluster.probabilities_[clust_docs]
    clustering_ = clustering.iloc[clust_docs].copy()
    clustering_["probability"] = clust_probs
    clustering_ = clustering_.sort_values("probability", ascending=False)
    return clustering_


##########################################
#### add funding stats and other data ####
##########################################


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
    """Checks how many data points come from gtr vs cb"""
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
