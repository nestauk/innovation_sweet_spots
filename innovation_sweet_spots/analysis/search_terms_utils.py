from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
import numpy as np
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.search_terms as search_terms

CATEGORIES = list(search_terms.categories_keyphrases.keys())


def get_categories_to_top2vec_clusters(cats=CATEGORIES):
    clusters_categories = pd.read_excel(
        PROJECT_DIR / "outputs/data/results_august/top2vec_clusters_categories.xlsx"
    )
    clusters_categories = clusters_categories[
        -clusters_categories.manual_category.isnull()
    ]
    categories_to_top2vec = {}
    for cat in cats:
        categories_to_top2vec[cat] = clusters_categories[
            clusters_categories.manual_category == cat
        ].cluster_id.to_list()
    return categories_to_top2vec


def get_categories_to_tomotopy_narrow_topics(cats=CATEGORIES):
    narrow_set_categories = pd.read_excel(
        PROJECT_DIR
        / "outputs/data/results_august/narrow_set_top2vec_cluster_counts_checked.xlsx"
    )
    narrow_set_categories = narrow_set_categories[-narrow_set_categories.label.isnull()]
    categories_to_narrow_topics = {}
    for cat in cats:
        categories_to_narrow_topics[cat] = narrow_set_categories[
            narrow_set_categories.label == cat
        ].topic.to_list()
    return categories_to_narrow_topics


# Find documents using substring matching
def find_docs_with_terms(terms, corpus_texts, corpus_df, return_dataframe=True):
    x = np.array([False] * len(corpus_texts))
    for term in terms:
        x = x | np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def find_docs_with_all_terms(terms, corpus_texts, corpus_df, return_dataframe=True):
    x = np.array([True] * len(corpus_texts))
    for term in terms:
        x = x & np.array(iss.is_term_present(term, corpus_texts))
    if return_dataframe:
        return corpus_df.iloc[x]
    else:
        return x


def get_docs_with_keyphrases(keyphrases, full_corpus_document_texts, doc_df):
    x = np.array([False] * len(full_corpus_document_texts))
    for terms in keyphrases:
        print(terms)
        x = x | find_docs_with_all_terms(
            terms,
            corpus_texts=full_corpus_document_texts,
            corpus_df=doc_df,
            return_dataframe=False,
        )
    return doc_df.iloc[x]


def add_info(df, clustering, doc_df, topic, topic_probabilities):
    df = df.merge(clustering, how="left")
    # Add topic probs
    df_ = doc_df[["doc_id"]]
    df_["topic_prob"] = np.max(topic_probabilities[:, topic], axis=1)
    df = df.merge(df_, how="left")
    # Determine best cluster
    c = (
        df.groupby("cluster_keywords")
        .agg(counts=("doc_id", "count"))
        #     .sort_values('counts', ascending=False)
        .reset_index()
        .sort_values("cluster_keywords")
        .sort_values("counts", ascending=False)
    )
    df["cluster_keywords"] = pd.Categorical(
        df["cluster_keywords"],
        categories=reversed(c.cluster_keywords.to_list()),
        ordered=True,
    )
    return df.sort_values(["cluster_keywords", "topic_prob"], ascending=False)
